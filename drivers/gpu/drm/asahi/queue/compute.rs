// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! Compute work queue.
//!
//! A compute queue consists of one underlying WorkQueue.
//! This module is in charge of creating all of the firmware structures required to submit compute
//! work to the GPU, based on the userspace command buffer.

use super::common;
use crate::alloc::Allocator;
use crate::debug::*;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::{
    file,
    fw,
    gpu,
    microseq, //
};
use crate::{
    inner_ptr,
    inner_weak_ptr, //
};
use core::sync::atomic::Ordering;
use kernel::dma_fence::RawDmaFence;
use kernel::drm::sched::Job;
use kernel::prelude::*;
use kernel::sync::Arc;
use kernel::uapi;
use kernel::xarray;

const DEBUG_CLASS: DebugFlags = DebugFlags::Compute;

#[versions(AGX)]
impl super::QueueInner::ver {
    /// Build one typed Compute command without adding it to a WorkQueue.
    /// E181 extracts this byte-for-byte construction body so the dormant G15
    /// stock-empty transaction and the existing submit path cannot drift into
    /// separate command/inner-allocation implementations.
    fn build_compute_command(
        &self,
        gpu: &gpu::GpuManager::ver,
        cmdbuf: &uapi::drm_asahi_cmd_compute,
        attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
        vm_bind: crate::mmu::VmBind,
        ev_comp: crate::workqueue::QueueEventInfo::ver,
    ) -> Result<GpuObject<fw::compute::RunCompute::ver>> {
        let mut alloc = gpu.alloc();
        let kalloc = &mut *alloc;

        mod_dev_dbg!(self.dev, "[Submission {}] Compute!\n", id);

        if cmdbuf.flags != 0 {
            return Err(EINVAL);
        }

        let mut user_timestamps: fw::job::UserTimestamps = Default::default();
        user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts.start)?;
        user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts.end)?;

        // This sequence number increases per new client/VM? assigned to some slot,
        // but it's unclear *which* slot...
        let slot_client_seq: u8 = (self.id & 0xff) as u8;
        #[ver(G == G15)]
        let _ = slot_client_seq; // G15 +0x85f is a context-ID generation, not this queue sequence

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] VM slot = {}\n",
            id,
            vm_bind.slot()
        );

        let notifier = self.notifier.clone();

        let preempt2_off = gpu.get_cfg().compute_preempt1_size;
        let preempt3_off = preempt2_off + 8;
        let preempt4_off = preempt3_off + 8;
        let preempt5_off = preempt4_off + 8;
        let preempt_size = preempt5_off + 8;

        #[ver(G == G15)]
        if preempt2_off != 0x1480 || preempt_size != 0x14a0 {
            return Err(EIO);
        }

        let preempt_buf = self
            .ualloc
            .lock()
            .array_empty_tagged(preempt_size, b"CPMT")?;

        // E319 exact 23J220 application-output parent: E318 traced the
        // Shared Metal buffer through AGXBuffer/IOGPUMetalBuffer to an
        // ordinary range-5 uncached bank-0 mapping. Change only the writable
        // result backing class; executable body/profile storage stays code.
        #[ver(G == G15)]
        let mut g15_result = self
            ._g15_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged::<u8>(0x100, b"CDR1")?;
        #[ver(G == G15)]
        for byte in g15_result.as_mut_slice().iter_mut() {
            *byte = 0xff;
        }
        #[ver(G == G15)]
        let g15_result_addr = g15_result.gpu_va().get();
        // E310 exact 23J220 UserBuffer argument table: a separate 35-entry,
        // 8-byte-aligned command-buffer copy in pool 3. E309 proved that
        // pool 3 is range-5 uncached (0x0080_0000_0000_0008 protection bits).
        // E318 separately closes the application result to the same range-5
        // uncached PTE class; keep the argument table a distinct allocation.
        #[ver(G == G15)]
        let mut g15_argument_table = self
            ._g15_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged::<u8>(0x118, b"AT03")?;
        #[ver(G == G15)]
        let g15_args_addr = g15_argument_table.gpu_va().get();
        #[ver(G == G15)]
        if g15_args_addr & 7 != 0 {
            return Err(EIO);
        }
        #[ver(G == G15)]
        {
            let entries = g15_argument_table.as_mut_slice();
            entries.fill(0);
            entries[..8].copy_from_slice(&g15_result_addr.to_le_bytes());
        }

        // E299 mirrors the exact 23J220 per-variant Statics allocation. The
        // direct emitter takes the exact 8-byte zero statics vector, allocates
        // it from DataBuffer pool 0x0a with 32-byte alignment, and the ESL
        // loadAbsolute record targets that allocation base. Pool 0x0a shares
        // the exact range-5 uncached PTE class already proven for pool 5.
        #[ver(G == G15)]
        let mut g15_statics = self
            ._g15_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged::<u8>(8, b"ST0A")?;
        #[ver(G == G15)]
        let g15_statics_addr = g15_statics.gpu_va().get();
        #[ver(G == G15)]
        if g15_statics_addr & 0x1f != 0 {
            return Err(EIO);
        }
        #[ver(G == G15)]
        g15_statics.as_mut_slice().fill(0);

        // E297 mirrors the exact 23J220 enqueue-time ownership transition:
        // the finalized direct ESL is copied into command-buffer DataBuffer
        // pool 5, whose exact G15 UAT class is range-5 uncached
        // (0x0080_0000_0000_0008 protection bits). Keep the actual shader
        // body and persistent profile helper in the independently proven code
        // heap; this experiment changes only the entry-copy backing class.
        #[ver(G == G15)]
        let mut g15_entry = self
            ._g15_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged::<u8>(0x100, b"ESL5")?;
        #[ver(G == G15)]
        let g15_entry_addr = g15_entry.gpu_va().get();
        #[ver(G == G15)]
        if g15_entry_addr & 0x3f != 0 {
            return Err(EIO);
        }

        // E298 mirrors exact DataBuffer pool 0x16 ownership for the direct CDM
        // stream. The preserved 23J220 setupDeferred/resource/PTE chain gives
        // pool 0x16 the same range-5 uncached bank-0 protection class as pool 5.
        #[ver(G == G15)]
        let mut g15_cdm_root = self
            ._g15_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged::<u8>(0x100, b"CD16")?;
        #[ver(G == G15)]
        let g15_cdm_addr = g15_cdm_root.gpu_va().get();

        #[ver(G == G15)]
        let (g15_body, g15_profile_helper) = {
            const CDM_LEN: usize = 0x30;
            const BODY_OFF: usize = 0x200;
            const BLOB_LEN: usize = 0x400;

            // macOS 14.8.3 / 23J220 G15 uses the older two-LDIMM pointer
            // grammar here, not the newer generator form used by E283. E284
            // recovered the exact bit-preserving pointer fixup performed by
            // loadBufferPointer(); keep it isolated so runtime GPU VAs remain
            // dynamic while the instruction grammar stays byte-exact.
            fn exact_pointer(addr: u64) -> [u8; 16] {
                const TEMPLATE: [u8; 16] = [
                    0x1c, 0x80, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
                ];
                const INDEX: [usize; 16] = [
                    0, 0, 0, 3, 8, 8, 9, 10, 4, 4, 4, 7, 12, 12, 13, 14,
                ];
                const RSHIFT: [u8; 16] = [
                    0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0,
                ];
                const PRESERVE: [u8; 16] = [
                    0xff, 0x80, 0xff, 0x01, 0xe1, 0xf3, 0x00, 0xf0,
                    0xff, 0x80, 0xff, 0x01, 0xe1, 0xf3, 0x00, 0xf0,
                ];

                let mut table = [0u8; 16];
                table[..8].copy_from_slice(&addr.to_le_bytes());
                table[8..].copy_from_slice(&(addr >> 5).to_le_bytes());

                let mut out = TEMPLATE;
                let mut i = 0;
                while i < out.len() {
                    let shifted = table[INDEX[i]] >> RSHIFT[i];
                    out[i] = shifted ^ ((shifted ^ TEMPLATE[i]) & PRESERVE[i]);
                    i += 1;
                }
                out
            }

            // E323 proves the exact 23J220 direct-Compute __TEXT body is
            // allocated from AGXMetal.CodeHeap, whose bank-0 protection is the
            // range-5 uncached class. E289 independently proves the persistent
            // inactive-profile helper uses that same Device Heap<true> at
            // Device +0x158. E325 therefore places both allocations in the
            // exact heap mapping class while changing only the helper from E324.
            let mut blob = self
                ._g15_range5_uncached
                .as_ref()
                .ok_or(EINVAL)?
                .lock()
                .array_empty_tagged::<u8>(BLOB_LEN, b"CDM1")?;
            let mut profile_helper = self
                ._g15_range5_uncached
                .as_ref()
                .ok_or(EINVAL)?
                .lock()
                .array_empty_tagged::<u8>(0x10, b"PRFL")?;
            let body_base = blob.gpu_va().get();
            let entry_addr = g15_entry_addr;
            let profile_helper_addr = profile_helper.gpu_va().get();

            // Exact USCProfileControlStateLoaderGen1<G15> helper: two CNDRET_0
            // entry points at offsets 0 and 8, each instruction stored at +4.
            const CNDRET: [u8; 4] = [0x8f, 0x02, 0x54, 0x01];
            let helper = profile_helper.as_mut_slice();
            helper[4..8].copy_from_slice(&CNDRET);
            helper[12..16].copy_from_slice(&CNDRET);

            let body_addr = body_base.checked_add(BODY_OFF as u64).ok_or(EOVERFLOW)?;
            let result_addr = g15_result_addr;
            let body_bytes = blob.as_mut_slice();
            let entry_bytes = g15_entry.as_mut_slice();

            // E321 carries E291's independently validated shader STOP
            // discriminator onto the fully corrected E319 mapping baseline.
            // This is the only GPU-facing delta from E319.
            const STOP_BODY: [u8; 4] = [0x0e, 0x00, 0x00, 0x00];
            body_bytes[BODY_OFF..BODY_OFF + STOP_BODY.len()].copy_from_slice(&STOP_BODY);

            // Exact inactive-profile BL_0 emitted after the final load round and
            // before finish() appends STOP. The target is helper entry 0.
            fn profile_bl(helper_addr: u64) -> [u8; 10] {
                let value = ((helper_addr & 0xffff_ffff_ffff) << 1) | 1;
                let lo = value as u32;
                let hi = (value >> 32) as u32;
                let words = [
                    (((hi >> 6) & 0x800) | 0x8f) as u16,
                    ((((lo & 0x01ff_fffc) << 7) | ((lo & 1) << 8) | 0x54)
                        & 0xffff) as u16,
                    ((value >> 9) & 0xffff) as u16,
                    ((value >> 25) & 0xffff) as u16,
                    ((hi >> 9) & 0xff) as u16,
                ];
                let mut out = [0u8; 10];
                let mut i = 0;
                while i < words.len() {
                    let bytes = words[i].to_le_bytes();
                    out[i * 2] = bytes[0];
                    out[i * 2 + 1] = bytes[1];
                    i += 1;
                }
                out
            }

            // Exact 23J220 direct state-loader entry. E289 adds the production
            // inactive-profile BL_0 that E285/E286 omitted.
            const ENTRY_LEN: usize = 126;
            let mut pos = 0usize;

            for ptr in [exact_pointer(g15_args_addr), exact_pointer(g15_statics_addr)] {
                entry_bytes[pos..pos + ptr.len()].copy_from_slice(&ptr);
                pos += ptr.len();
            }

            const LOADS: [u8; 28] = [
                // BufferBindings/UserBuffer -> r0/r1.
                0x67, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x59, 0x00, 0x00, 0x50, 0x26, 0x00,
                // Statics -> r2/r3.
                0x67, 0x10, 0x54, 0x04, 0x02, 0x00, 0x00, 0x00,
                0x59, 0x00, 0x00, 0x40, 0x26, 0x00,
            ];
            entry_bytes[pos..pos + LOADS.len()].copy_from_slice(&LOADS);
            pos += LOADS.len();

            // appendLdshdr() reserves exactly 0x1c bytes. finish() later patches
            // them with this exact LoadShader encoding; only the body VA varies.
            entry_bytes[pos..pos + 8]
                .copy_from_slice(&[0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00]);
            pos += 8;
            entry_bytes[pos..pos + 3].copy_from_slice(&[0x77, 0x01, 0x2a]);
            pos += 3;
            let shifted_body = (body_addr >> 7).to_le_bytes();
            entry_bytes[pos..pos + 5].copy_from_slice(&shifted_body[..5]);
            pos += 5;
            const LOAD_SHADER_TAIL: [u8; 12] = [
                0x04, 0x00, 0xf7, 0x00, 0x2a, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];
            entry_bytes[pos..pos + LOAD_SHADER_TAIL.len()].copy_from_slice(&LOAD_SHADER_TAIL);
            pos += LOAD_SHADER_TAIL.len();

            // Both inflight loads carry dependency class 1, so finishRound()
            // reduces them to aggregate mask 2: NOP(2), then three NOP(0).
            // COPYs hand the exact r0..r3 load results to u0..u3.
            const ROUND_AND_COPIES: [u8; 24] = [
                0x06, 0x08, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
                0x0b, 0x00, 0x09, 0x24,
                0x1b, 0x02, 0x09, 0x24,
                0x2b, 0x04, 0x09, 0x24,
                0x3b, 0x06, 0x09, 0x24,
            ];
            entry_bytes[pos..pos + ROUND_AND_COPIES.len()].copy_from_slice(&ROUND_AND_COPIES);
            pos += ROUND_AND_COPIES.len();

            let profile = profile_bl(profile_helper_addr);
            entry_bytes[pos..pos + profile.len()].copy_from_slice(&profile);
            pos += profile.len();

            const STOP: [u8; 4] = [0x0e, 0x00, 0x00, 0x00];
            entry_bytes[pos..pos + STOP.len()].copy_from_slice(&STOP);
            pos += STOP.len();

            if pos != ENTRY_LEN {
                return Err(EIO);
            }
            entry_bytes[pos..].fill(0);

            // The 0x30-byte G15 direct stream independently appears in Alyssa's
            // working m1n1 path and pac85's standalone macOS AGX demo.
            fn put32(bytes: &mut [u8], pos: &mut usize, value: u32) {
                bytes[*pos..*pos + 4].copy_from_slice(&value.to_le_bytes());
                *pos += 4;
            }

            let cdm_bytes = g15_cdm_root.as_mut_slice();
            let mut pos = 0usize;
            put32(cdm_bytes, &mut pos, 0x0008_0000);
            let shifted_entry = ((entry_addr & 0xffff_ffff) << 26)
                | ((entry_addr >> 32) << 16);
            cdm_bytes[pos..pos + 8].copy_from_slice(&shifted_entry.to_le_bytes());
            pos += 8;
            // Exact 23J220 ordinary direct-launch dword 3 (E241/E242/E244).
            // The older m1n1 bring-up stream used 0x40 here, but the target
            // producer derives this from ComputeProgramVariant +0xa24 and the
            // default distribution mode, yielding bit 30 set and no low bits.
            put32(cdm_bytes, &mut pos, 0x4000_0000);
            for _ in 0..6 {
                put32(cdm_bytes, &mut pos, 1);
            }
            put32(cdm_bytes, &mut pos, 0x6000_0160);
            put32(cdm_bytes, &mut pos, 0x4000_0000);
            if pos != CDM_LEN {
                return Err(EIO);
            }

            core::sync::atomic::fence(Ordering::SeqCst);
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 E325 exact-CodeHeap body+profile class CDM={:#x} entry={:#x} body={:#x} profile={:#x} result={:#x} args={:#x} statics={:#x}\n",
                g15_cdm_addr,
                entry_addr,
                body_addr,
                profile_helper_addr,
                result_addr,
                g15_args_addr,
                g15_statics_addr
            );
            (blob, profile_helper)
        };
        #[ver(G == G15)]
        let cdm_ctrl_stream_end = g15_cdm_root
            .gpu_va()
            .get()
            // Exact 23J220 endComputePass() publishes raw +0x60 from the
            // pre-increment write pointer: the Stream Terminate token itself,
            // not the allocator's one-past pointer. E248's 0x30 was off by 4.
            .checked_add(0x2c)
            .ok_or(EOVERFLOW)?;
        #[ver(G != G15)]
        let cdm_ctrl_stream_end = cmdbuf.cdm_ctrl_stream_end;

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Event #{} {:#x?} -> {:#x?}\n",
            id,
            ev_comp.slot,
            ev_comp.value,
            ev_comp.value.next(),
        );

        // G15's RCE/CDM path directly touches the command timestamps. The
        // dynamically working M3 m1n1 path faults when these live in the
        // firmware-only shared heap and moves both start/end timestamps to
        // GPU/FW shared RW memory. Keep the legacy placement on older GPUs.
        #[ver(G != G15)]
        let timestamps = Arc::new(
            kalloc.shared.new_default::<fw::job::JobTimestamps>()?,
            GFP_KERNEL,
        )?;
        #[ver(G == G15)]
        let timestamps = Arc::new(
            kalloc.gpu.new_default::<fw::job::JobTimestamps>()?,
            GFP_KERNEL,
        )?;

        #[ver(G != G15)]
        let uuid = 0;
        #[ver(G == G15)]
        let uuid = common::next_g15_command_uuid();
        mod_dev_dbg!(self.dev, "[Submission {}] UUID = {:#x?}\n", id, uuid);

        // TODO: check
        #[ver(V >= V13_0B4)]
        // G15 mirrors AGXCLChannelSKU::submitBuffer(): consume one value from
        // the command-queue-wide sequence used at command +0x04.
        let count = self.counter.fetch_add(1, Ordering::Relaxed);

        // G15 uses the same GPU/FW shared RW command placement as the
        // independently working M3 m1n1 Compute path. In particular, RCE/CDM
        // must be able to access command-side state that is GPU-read-only on
        // older generations.
        #[ver(G != G15)]
        let comp_alloc = kalloc.gpu_ro.alloc_object()?;
        #[ver(G == G15)]
        let comp_alloc = kalloc.gpu.alloc_object()?;

        let comp = GpuObject::new_init_prealloc(
            comp_alloc,
            |ptr: GpuWeakPointer<fw::compute::RunCompute::ver>| {
                let notifier = notifier.clone();
                let vm_bind = vm_bind.clone();
                try_init!(fw::compute::RunCompute::ver {
                    preempt_buf: preempt_buf,
                    #[ver(G == G15)]
                    g15_result: g15_result,
                    #[ver(G == G15)]
                    g15_argument_table: g15_argument_table,
                    #[ver(G == G15)]
                    g15_statics: g15_statics,
                    #[ver(G == G15)]
                    g15_cdm_root: g15_cdm_root,
                    #[ver(G == G15)]
                    g15_body: g15_body,
                    #[ver(G == G15)]
                    g15_entry: g15_entry,
                    #[ver(G == G15)]
                    g15_profile_helper: g15_profile_helper,
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = gpu.initdata.runtime_pointers.stats.comp.weak_pointer();

                        let start_comp = builder.add(microseq::StartCompute::ver {
                            header: microseq::op::StartCompute::HEADER,
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(G < G14X && G != G15)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X || G == G15)]
                            job_params1: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            stats,
                            work_queue: ev_comp.info_ptr,
                            vm_slot: vm_bind.slot(),
                            unk_28: 0x1,
                            event_generation: self.id as u32,
                            event_seq: U64(ev_comp.event_seq),
                            unk_38: 0x0,
                            job_params2: inner_weak_ptr!(ptr, job_params2),
                            unk_44: 0x0,
                            uuid,
                            attachments: *attachments,
                            padding: Default::default(),
                            #[ver(V >= V13_0B4 && G != G15)]
                            unk_flag: inner_weak_ptr!(ptr, unk_flag),
                            #[ver(G == G15)]
                            unk_flag: inner_weak_ptr!(ptr, g15_recovery_marker_878),
                            #[ver(V >= V13_0B4)]
                            counter: U64(count),
                            #[ver(V >= V13_0B4)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                        })?;

                        if user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_comp.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Compute),
                        })?;
                        #[ver(G >= G14X)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;

                        if user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_comp.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_comp);
                        builder.add(microseq::FinalizeCompute::ver {
                            header: microseq::op::FinalizeCompute::HEADER,
                            stats,
                            work_queue: ev_comp.info_ptr,
                            vm_slot: vm_bind.slot(),
                            #[ver(V < V13_0B4)]
                            unk_18: 0,
                            job_params2: inner_weak_ptr!(ptr, job_params2),
                            unk_24: 0,
                            uuid,
                            fw_stamp: ev_comp.fw_stamp_pointer,
                            stamp_value: ev_comp.value.next(),
                            unk_38: 0,
                            unk_3c: 0,
                            unk_40: 0,
                            unk_44: 0,
                            unk_48: 0,
                            unk_4c: 0,
                            unk_50: 0,
                            unk_54: 0,
                            unk_58: 0,
                            #[ver(G == G14 && V < V13_0B4)]
                            unk_5c_g14: U64(0),
                            restart_branch_offset: off,
                            has_attachments: (attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4)]
                            unk_64: Default::default(),
                            #[ver(V >= V13_0B4 && G != G15)]
                            unk_flag: inner_weak_ptr!(ptr, unk_flag),
                            #[ver(G == G15)]
                            unk_flag: inner_weak_ptr!(ptr, g15_recovery_marker_878),
                            #[ver(V >= V13_0B4)]
                            unk_79: Default::default(),
                        })?;

                        builder.add(microseq::RetireStamp {
                            header: microseq::op::RetireStamp::HEADER,
                        })?;
                        builder.build(&mut kalloc.private)?
                    },
                    notifier,
                    vm_bind,
                    timestamps,
                    user_timestamps,
                })
            },
            |inner, _ptr| {
                let vm_slot = vm_bind.slot();
                #[ver(G == G15)]
                let context_generation = vm_bind.generation();
                try_init!(fw::compute::raw::RunCompute::ver {
                    tag: fw::workqueue::CommandType::RunCompute,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count),
                    unk_4: 0,
                    #[ver(G != G15)]
                    vm_slot,
                    #[ver(G == G15)]
                    g15_context_id_10: vm_slot,
                    #[ver(G != G15)]
                    notifier: inner.notifier.gpu_pointer(),
                    #[ver(G == G15)]
                    g15_event_control_fwva_14: U64(0),
                    unk_pointee: Default::default(),
                    #[ver(G < G14X && G != G15)]
                    __pad0: Default::default(),
                    #[ver(G < G14X && G != G15)]
                    job_params1 <- try_init!(fw::compute::raw::JobParameters1 {
                        preempt_buf1: inner.preempt_buf.gpu_pointer(),
                        cdm_ctrl_stream_base: U64(cmdbuf.cdm_ctrl_stream_base),
                        // buf2-5 Only if internal program is used
                        preempt_buf2: inner.preempt_buf.gpu_offset_pointer(preempt2_off),
                        preempt_buf3: inner.preempt_buf.gpu_offset_pointer(preempt3_off),
                        preempt_buf4: inner.preempt_buf.gpu_offset_pointer(preempt4_off),
                        preempt_buf5: inner.preempt_buf.gpu_offset_pointer(preempt5_off),
                        usc_exec_base_cp: U64(self.usc_exec_base),
                        unk_38: U64(0x8c60),
                        helper_program: cmdbuf.helper.binary, // Internal program addr | 1
                        unk_44: 0,
                        helper_arg: U64(cmdbuf.helper.data), // Only if internal program used
                        helper_cfg: cmdbuf.helper.cfg, // 0x40 if internal program used
                        unk_54: 0,
                        unk_58: 1,
                        unk_5c: 0,
                        iogpu_unk_40: 0, // 0x1c if internal program used
                        __pad: Default::default(),
                    }),
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            r.add(0x1a510, inner.preempt_buf.gpu_pointer().into());
                            r.add(0x1a420, cmdbuf.cdm_ctrl_stream_base);
                            // buf2-5 Only if internal program is used
                            r.add(0x1a4d0, inner.preempt_buf.gpu_offset_pointer(preempt2_off).into());
                            r.add(0x1a4d8, inner.preempt_buf.gpu_offset_pointer(preempt3_off).into());
                            r.add(0x1a4e0, inner.preempt_buf.gpu_offset_pointer(preempt4_off).into());
                            r.add(0x1a4e8, inner.preempt_buf.gpu_offset_pointer(preempt5_off).into());
                            r.add(0x10071, self.usc_exec_base); // USC_EXEC_BASE_CP
                            r.add(0x11841, cmdbuf.helper.binary.into());
                            r.add(0x11849, cmdbuf.helper.data);
                            r.add(0x11f81, cmdbuf.helper.cfg.into());
                            r.add(0x1a440, 0x24201);
                            r.add(0x12091, 0 /* iogpu_unk_40 */);
                            /*
                            r.add(0x10201, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x10428, 0x100); // Some kind of counter?? Does this matter?
                            */
                        }
                    ),
                    #[ver(G == G15)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            // Keep the legacy per-queue USC base consumed on the G15
                            // specialization even though Apple does not encode it in
                            // this exact empty-Compute RegisterArray.
                            let _ = self.usc_exec_base;

                            // E067 fixes the exact 23J220 register/source map.
                            // E132 then proved the older E068 cross-build empty oracle
                            // was stale for hardware-facing data-buffer state: stock
                            // 23J220 owns a 0x1480 primary Compute backing, four real
                            // 8-byte tail slots, and raw +0xa4 == 0x1c. The same-build
                            // begin/end pair also owns a terminate-only CDM root below.
                            //
                            // Form-1 Apple register IDs are encoded with bit 0 set in
                            // the 12-byte RegisterArray entry (e.g. 0x12090 -> 0x12091).
                            r.add(0x1a510, inner.preempt_buf.gpu_pointer().into());
                            r.add(0x1a420, inner.g15_cdm_root.gpu_pointer().into());
                            r.add(
                                0x1a4d0,
                                inner.preempt_buf.gpu_offset_pointer(preempt2_off).into(),
                            );
                            r.add(
                                0x1a4d8,
                                inner.preempt_buf.gpu_offset_pointer(preempt3_off).into(),
                            );
                            r.add(
                                0x1a4e0,
                                inner.preempt_buf.gpu_offset_pointer(preempt4_off).into(),
                            );
                            r.add(
                                0x1a4e8,
                                inner.preempt_buf.gpu_offset_pointer(preempt5_off).into(),
                            );
                            r.add(0x1a440, 0x154024201);
                            r.add(0x1a458, 0x10c08860);
                            r.add(0x12091, 0x1c);
                            r.add(0x101d9, 0x1c);
                            r.add(0x1a089, 0);
                            r.add(0x1a091, 0);
                            r.add(0x1a059, 0);
                            r.add(0x1a061, 0);
                            r.add(0x1a0b9, 0);
                            r.add(0x1a0c1, 0);
                            r.add(0x101d1, 0);
                            r.add(0x0d479, 0);
                            r.add(0x1a0e9, 0);
                            // E260 exact 23J220 control-flow closure: with the
                            // ordinary non-ray default requirement tuple, context
                            // +0x658/+0x65c remain zero. endComputePass() therefore
                            // skips the late real-dispatch resource synthesis block;
                            // raw +0x158/+0x1b0 stay zero and 0x107a0 uses Apple's
                            // zero-resource fallback.
                            r.add(0x107a1, 0x00ff_0000);
                        },
                    ),
                    #[ver(G != G15)]
                    __pad1: Default::default(),
                    #[ver(G == G15)]
                    g15_pre_micro_730: Default::default(),
                    // E133 closes the exact 23J220 stock-empty sources:
                    // raw Compute +0xc0/+0xd8 are explicitly initialized to zero
                    // by beginComputePass() and are not rewritten on empty close.
                    #[ver(G == G15)]
                    g15_raw_compute_c0_740: U64(0),
                    #[ver(G == G15)]
                    g15_raw_compute_d8_lo_748: U32(0),
                    #[ver(G == G15)]
                    g15_pre_micro_74c: Default::default(),
                    #[ver(G == G15)]
                    g15_raw_compute_d8_hi_750: U32(0),
                    #[ver(G == G15)]
                    g15_pre_micro_754: Default::default(),
                    #[ver(G != G15)]
                    microsequence: inner.micro_seq.gpu_pointer(),
                    #[ver(G == G15)]
                    // E176 fail-closed phase-0 image. E069/E070 prove +0x760
                    // is a selected G15 SKU-slot FWVA; E175 has not yet called
                    // phase 1/finalize, so there is no legitimate slot to publish.
                    g15_sku_stream_fwva_760: U64(0),
                    #[ver(G != G15)]
                    microsequence_size: inner.micro_seq.len() as u32,
                    #[ver(G == G15)]
                    g15_sku_stream_size_768: 0,
                    job_params2 <- try_init!(fw::compute::raw::JobParameters2::ver {
                        #[ver(V >= V13_0B4)]
                        unk_0_0: 0,
                        #[ver(G != G15)]
                        unk_0: Default::default(),
                        #[ver(G == G15)]
                        g15_unk_770: U32(0),
                        #[ver(G == G15)]
                        g15_state_774: U64(0),
                        #[ver(G == G15)]
                        g15_state_77c: U64(0),
                        #[ver(G == G15)]
                        g15_state_784: U64(0),
                        #[ver(G == G15)]
                        g15_state_78c: U64(0),
                        preempt_buf1: inner.preempt_buf.gpu_pointer(),
                        cdm_ctrl_stream_end: U64(cdm_ctrl_stream_end),
                        #[ver(G != G15)]
                        unk_34: Default::default(),
                        #[ver(G == G15)]
                        g15_state_7a4: U64(0),
                        #[ver(G == G15)]
                        g15_state_7ac: U32(0),
                        #[ver(G == G15)]
                        g15_unk_7b0: U32(0),
                        #[ver(G == G15)]
                        g15_state_7b4: U64(0),
                        #[ver(G == G15)]
                        g15_state_7bc: U32(0),
                        #[ver(G == G15)]
                        g15_unk_7c0: U32(0),
                        #[ver(G < G14X && G != G15)]
                        unk_g14x: 0,
                        #[ver(G >= G14X && G != G15)]
                        unk_g14x: 0x24201,
                        #[ver(G != G15)]
                        unk_58: 0,
                        #[ver(G == G15)]
                        // Apple mirrors the exact emitted 0x1a440 value here.
                        // E068 empty Compute has raw +0x170 == 0 on J615/G15G.
                        g15_reg_1a440_value: U64(0x154024201),
                        #[ver(V < V13_0B4)]
                        unk_5c: 0,
                    }),
                    #[ver(G != G15)]
                    encoder_params <- try_init!(fw::job::raw::EncoderParams {
                        unk_8: 0x0,     // fixed
                        sync_grow: 0x0, // check!
                        unk_10: 0x0,    // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffff,
                        sampler_array: U64(cmdbuf.sampler_heap),
                        sampler_count: cmdbuf.sampler_count as u32,
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    #[ver(G == G15)]
                    g15_raw_compute_50_lo_7cc: U32(0),
                    #[ver(G == G15)]
                    g15_encoder_pad_7d0: Default::default(),
                    #[ver(G == G15)]
                    g15_raw_compute_ac_7d4: 0,
                    #[ver(G == G15)]
                    g15_encoder_pad_7d5: Default::default(),
                    #[ver(G == G15)]
                    g15_raw_compute_a8_b0_lo_7d8: U64(0),
                    #[ver(G == G15)]
                    g15_raw_compute_b0_hi_7e0: U32(0),
                    #[ver(G != G15)]
                    meta <- try_init!(fw::job::raw::JobMeta {
                        unk_0: 0,
                        unk_2: 0,
                        no_preemption: 0,
                        stamp: ev_comp.stamp_pointer,
                        fw_stamp: ev_comp.fw_stamp_pointer,
                        stamp_value: ev_comp.value.next(),
                        stamp_slot: ev_comp.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid,
                        event_seq: ev_comp.event_seq as u32,
                    }),
                    #[ver(G == G15)]
                    meta <- try_init!(fw::job::raw::G15JobMeta {
                        engine_state: U32(0),
                        stamp: ev_comp.stamp_pointer,
                        fw_stamp: ev_comp.fw_stamp_pointer,
                        stamp_value: ev_comp.value.next(),
                        stamp_slot: ev_comp.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid,
                        event_seq: ev_comp.event_seq as u32,
                    }),
                    command_time: U64(0),
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), end)),
                    }),
                    user_timestamp_pointers: inner.user_timestamps.pointers()?,
                    #[ver(G != G15)]
                    client_sequence: slot_client_seq,
                    #[ver(G != G15)]
                    pad_2d1: Default::default(),
                    #[ver(G != G15)]
                    unk_2d4: 0,
                    #[ver(G != G15)]
                    unk_2d8: 0,
                    #[ver(G == G15)]
                    g15_pad_838: Default::default(),
                    #[ver(G == G15)]
                    g15_uma_page_pool_state_fwva_83e: U64(0),
                    #[ver(G == G15)]
                    // AGXUMAPool::prepareLocked() sets the descriptor prepared
                    // state before submitBuffer() copies descriptor +0x624 here.
                    // The exact stock 23J220 empty-Compute path therefore uses 1.
                    g15_uma_prepared_846: 1,
                    #[ver(G == G15)]
                    g15_uma_min_pool_size_847: U64(0),
                    #[ver(G == G15)]
                    g15_uma_ideal_pool_size_84f: U64(0),
                    #[ver(G == G15)]
                    g15_uma_metrics_fwva_857: U64(0),
                    #[ver(G == G15)]
                    // 23J220 exports AGXCLCommandDescriptor +0x494 here.
                    // That byte is the generation paired with the managed
                    // context ID at RunCompute +0x10, not a constant.
                    g15_context_id_generation_85f: context_generation,
                    #[ver(V >= V13_0B4)]
                    context_store_req: U64(0),
                    #[ver(G == G15)]
                    g15_tail_868: Default::default(),
                    #[ver(V >= V13_0B4)]
                    context_store_compl: U64(0),
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_2e9: Default::default(),
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_flag: U32(0),
                    #[ver(G == G15)]
                    g15_recovery_marker_878: U32(0),
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_pad: Default::default(),
                    #[ver(G == G15)]
                    g15_tail_87c: Default::default(),
                })
            },
        )?;

        core::mem::drop(alloc);

        Ok(comp)
    }

    /// Submit work to a compute queue.
    pub(super) fn submit_compute(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_compute,
        attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        let gpu = match (*self.dev)
            .gpu
            .as_any()
            .downcast_ref::<gpu::GpuManager::ver>()
        {
            Some(gpu) => gpu,
            None => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with Queue!\n");
                return Err(EIO);
            }
        };

        let vm_bind = job.vm_bind.clone();
        let fence = job.fence.clone();
        let ev_comp = job.get_comp()?.event_info();
        let comp = self.build_compute_command(
            gpu,
            cmdbuf,
            attachments,
            objects,
            id,
            flush_stamps,
            vm_bind.clone(),
            ev_comp,
        )?;
        let comp_job = job.get_comp()?;

        fence.add_command();
        comp_job.add_cb(comp, vm_bind.slot(), move |error| {
            if let Some(err) = error {
                fence.set_error(err.into())
            }

            fence.command_complete();
        })?;

        comp_job.next_seq();

        Ok(())
    }

    /// Stock-empty command construction route. The normal UAPI
    /// fields are validated before E180 is entered; E180 then arms phase 1 before
    /// invoking the exact shared builder above. The fully finalized command stays
    /// trapped in the QueueJob RAII owner and is never added to a WorkQueue here.
    fn construct_g15_stock_empty_phase0_unpublished(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_compute,
        attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = (job, cmdbuf, attachments, objects, id, flush_stamps);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if cmdbuf.flags != 0
                || cmdbuf.sampler_count != 0
                || cmdbuf.cdm_ctrl_stream_base != 0
                || cmdbuf.cdm_ctrl_stream_end != 0
                || cmdbuf.sampler_heap != 0
                || cmdbuf.helper.binary != 0
                || cmdbuf.helper.cfg != 0
                || cmdbuf.helper.data != 0
            {
                return Err(EINVAL);
            }

            // Claim the bounded execution slot only after the normal UAPI
            // command has been copied and mechanically proven stock-empty.
            file::claim_g15_stock_empty_compute()?;
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 normal-UAPI stock-empty Compute accepted; bounded slot claimed\n"
            );

            let gpu = match (*self.dev)
                .gpu
                .as_any()
                .downcast_ref::<gpu::GpuManager::ver>()
            {
                Some(gpu) => gpu,
                None => {
                    dev_crit!(self.dev.as_ref(), "GpuManager mismatched with Queue!\n");
                    return Err(EIO);
                }
            };
            let vm_bind = job.vm_bind.clone();
            let ev_comp = job.get_comp()?.event_info();

            job.construct_g15_stock_empty_unpublished(|| {
                self.build_compute_command(
                    gpu,
                    cmdbuf,
                    attachments,
                    objects,
                    id,
                    flush_stamps,
                    vm_bind,
                    ev_comp,
                )
            })
        }
    }


    /// Stock-empty composition boundary. This produces one fully-owned pending
    /// Compute WorkQueue command but deliberately stops before Job::commit(),
    /// Job::submit(), QueueInfo publication, pipe transport, or GPU execution.
    pub(super) fn prepare_g15_stock_empty_workqueue_unpublished(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_compute,
        attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = (job, cmdbuf, attachments, objects, id, flush_stamps);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            self.construct_g15_stock_empty_phase0_unpublished(
                job,
                cmdbuf,
                attachments,
                objects,
                id,
                flush_stamps,
            )?;
            job.transfer_g15_stock_empty_to_workqueue()
        }
    }

}
