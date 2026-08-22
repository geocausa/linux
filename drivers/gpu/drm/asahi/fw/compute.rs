// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU compute job firmware structures

use super::types::*;
use super::{
    event,
    job,
    workqueue, //
};
use crate::{
    microseq,
    mmu, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) cdm_ctrl_stream_base: U64,
        pub(crate) preempt_buf2: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf3: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf4: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf5: GpuPointer<'a, &'a [u8]>,
        pub(crate) usc_exec_base_cp: U64,
        pub(crate) unk_38: U64,
        pub(crate) helper_program: u32,
        pub(crate) unk_44: u32,
        pub(crate) helper_arg: U64,
        pub(crate) helper_cfg: u32,
        pub(crate) unk_54: u32,
        pub(crate) unk_58: u32,
        pub(crate) unk_5c: u32,
        pub(crate) iogpu_unk_40: u32,
        pub(crate) __pad: Pad<0xfc>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2<'a> {
        #[ver(V >= V13_0B4)]
        pub(crate) unk_0_0: u32,
        #[ver(G != G15)]
        pub(crate) unk_0: Array<0x24, u8>,
        // G15 submitBuffer() copies four consecutive descriptor qwords into
        // command +0x774..+0x793, preceded by one still-unresolved dword.
        #[ver(G == G15)]
        pub(crate) g15_unk_770: U32,
        #[ver(G == G15)]
        pub(crate) g15_state_774: U64,
        #[ver(G == G15)]
        pub(crate) g15_state_77c: U64,
        #[ver(G == G15)]
        pub(crate) g15_state_784: U64,
        #[ver(G == G15)]
        pub(crate) g15_state_78c: U64,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) cdm_ctrl_stream_end: U64,
        #[ver(G != G15)]
        pub(crate) unk_34: Array<0x20, u8>,
        // G15 submitBuffer() writes qword/dword pairs at +0x7a4/+0x7ac and
        // +0x7b4/+0x7bc, leaving the intervening dwords untouched here.
        #[ver(G == G15)]
        pub(crate) g15_state_7a4: U64,
        #[ver(G == G15)]
        pub(crate) g15_state_7ac: U32,
        #[ver(G == G15)]
        pub(crate) g15_unk_7b0: U32,
        #[ver(G == G15)]
        pub(crate) g15_state_7b4: U64,
        #[ver(G == G15)]
        pub(crate) g15_state_7bc: U32,
        #[ver(G == G15)]
        pub(crate) g15_unk_7c0: U32,
        #[ver(G != G15)]
        pub(crate) unk_g14x: u32,
        #[ver(G != G15)]
        pub(crate) unk_58: u32,
        // G15 generateRegisterList() stores the exact synthesized value used
        // for register 0x1a440 into command +0x7c4 (JobParameters2 +0x58).
        #[ver(G == G15)]
        pub(crate) g15_reg_1a440_value: U64,
        #[ver(V < V13_0B4)]
        pub(crate) unk_5c: u32,
    }

    const _: [(); 0x60] = [(); core::mem::size_of::<JobParameters2G15V14_7<'static>>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_unk_770)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_774)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_77c)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_784)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_78c)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, preempt_buf1)];
    const _: [(); 0x30] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, cdm_ctrl_stream_end)];
    const _: [(); 0x38] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_7a4)];
    const _: [(); 0x40] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_7ac)];
    const _: [(); 0x44] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_unk_7b0)];
    const _: [(); 0x48] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_7b4)];
    const _: [(); 0x50] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_state_7bc)];
    const _: [(); 0x54] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_unk_7c0)];
    const _: [(); 0x58] = [(); core::mem::offset_of!(JobParameters2G15V14_7<'static>, g15_reg_1a440_value)];

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunCompute<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) unk_4: u32,
        pub(crate) vm_slot: u32,
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        pub(crate) unk_pointee: u32,
        #[ver(G < G14X && G != G15)]
        pub(crate) __pad0: Array<0x50, u8>,
        #[ver(G < G14X && G != G15)]
        pub(crate) job_params1: JobParameters1<'a>,
        #[ver(G >= G14X || G == G15)]
        pub(crate) registers: job::raw::RegisterArray,
        #[ver(G != G15)]
        pub(crate) __pad1: Array<0x20, u8>,
        // Apple parseAndValidate() maps raw Compute payload +0xc0 -> descriptor
        // +0x3f0 and raw +0xd8 -> descriptor +0x3f8. submitBuffer() exports the
        // qword at +0xc0 and splits raw +0xd8 around the command's 4-byte gap.
        #[ver(G == G15)]
        pub(crate) g15_pre_micro_730: Array<0x10, u8>,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_c0_740: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_d8_lo_748: U32,
        #[ver(G == G15)]
        pub(crate) g15_pre_micro_74c: Array<0x04, u8>,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_d8_hi_750: U32,
        #[ver(G == G15)]
        pub(crate) g15_pre_micro_754: Array<0x0c, u8>,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) job_params2: JobParameters2::ver<'a>,
        #[ver(G != G15)]
        pub(crate) encoder_params: job::raw::EncoderParams,
        // Compact G15 encoder metadata is a direct raw-Compute repack:
        // +0x7cc <- raw +0x50 low32; +0x7d4 <- raw +0xac byte;
        // +0x7d8 = {raw +0xb0 low32, raw +0xa8}; +0x7e0 <- raw +0xb4.
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_50_lo_7cc: U32,
        #[ver(G == G15)]
        pub(crate) g15_encoder_pad_7d0: Array<0x04, u8>,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_ac_7d4: u8,
        #[ver(G == G15)]
        pub(crate) g15_encoder_pad_7d5: Array<0x03, u8>,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_a8_b0_lo_7d8: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_compute_b0_hi_7e0: U32,
        pub(crate) meta: job::raw::JobMeta,
        pub(crate) command_time: U64,
        pub(crate) timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,
        #[ver(G != G15)]
        pub(crate) client_sequence: u8,
        #[ver(G != G15)]
        pub(crate) pad_2d1: Array<3, u8>,
        #[ver(G != G15)]
        pub(crate) unk_2d4: u32,
        #[ver(G != G15)]
        pub(crate) unk_2d8: u8,
        // Apple G15 submission leaves +0x838..+0x83d unwritten; the first
        // explicit late-tail store begins at +0x83e. The same six-byte
        // packed-ABI gap appears in Compute, TA, and 3D immediately before an
        // unaligned U64. AGXCLCommandDescriptor embeds AGXUMAData at +0x5e0;
        // prepare/complete pass that exact record to AGXUMAPool. submitBuffer()
        // then exports the same UMA lifecycle fields used by RunFragment.
        #[ver(G == G15)]
        pub(crate) g15_pad_838: Array<0x06, u8>,
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_state_fwva_83e: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_prepared_846: u8,
        #[ver(G == G15)]
        pub(crate) g15_uma_min_pool_size_847: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_ideal_pool_size_84f: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_metrics_fwva_857: U64,
        // Low byte of AGXContextIDManager::alloc()'s generation out-parameter.
        // Apple increments this byte whenever the context-ID slot is newly
        // allocated, keeping stale/reused IDs distinguishable.
        #[ver(G == G15)]
        pub(crate) g15_context_id_generation_85f: u8,
        // Firmware explicitly treats these as CDM context-store request and
        // completion timestamps and checks their latency.
        #[ver(V >= V13_0B4)]
        pub(crate) context_store_req: U64,
        #[ver(G == G15)]
        pub(crate) g15_tail_868: Array<0x08, u8>,
        #[ver(V >= V13_0B4)]
        pub(crate) context_store_compl: U64,
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_2e9: Array<0x14, u8>,
        // G15's encoder explicitly forms command_fwva + 0x878 and generic
        // submission zeroes that dword, matching the old command flag role.
        #[ver(V >= V13_0B4)]
        pub(crate) unk_flag: U32,
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_pad: Array<0x10, u8>,
        #[ver(G == G15)]
        pub(crate) g15_tail_87c: Array<0x04, u8>,
    }

    const _: [(); 0x880] = [(); core::mem::size_of::<RunComputeG15V14_7<'static>>()];
    // AGXCLChannelSKU::submitBuffer() writes the G15 context ID at +0x10,
    // matching the existing VM-slot field, then the notifier at +0x14.
    const _: [(); 0x10] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, vm_slot)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, notifier)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, unk_pointee)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, registers)];
    const _: [(); 0x730] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pre_micro_730)];
    const _: [(); 0x740] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_c0_740)];
    const _: [(); 0x748] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_d8_lo_748)];
    const _: [(); 0x74c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pre_micro_74c)];
    const _: [(); 0x750] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_d8_hi_750)];
    const _: [(); 0x754] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pre_micro_754)];
    const _: [(); 0x760] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, microsequence)];
    const _: [(); 0x768] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, microsequence_size)];
    const _: [(); 0x76c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, job_params2)];
    const _: [(); 0x7cc] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_50_lo_7cc)];
    const _: [(); 0x7d0] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_encoder_pad_7d0)];
    const _: [(); 0x7d4] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_ac_7d4)];
    const _: [(); 0x7d5] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_encoder_pad_7d5)];
    const _: [(); 0x7d8] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_a8_b0_lo_7d8)];
    const _: [(); 0x7e0] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_raw_compute_b0_hi_7e0)];
    const _: [(); 0x7e4] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)];
    const _: [(); 0x810] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, command_time)];
    const _: [(); 0x818] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, timestamp_pointers)];
    const _: [(); 0x828] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, user_timestamp_pointers)];
    const _: [(); 0x838] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pad_838)];
    const _: [(); 0x83e] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_page_pool_state_fwva_83e)];
    const _: [(); 0x846] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_prepared_846)];
    const _: [(); 0x847] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_min_pool_size_847)];
    const _: [(); 0x84f] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_ideal_pool_size_84f)];
    const _: [(); 0x857] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_metrics_fwva_857)];
    const _: [(); 0x85f] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_context_id_generation_85f)];
    const _: [(); 0x860] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, context_store_req)];
    const _: [(); 0x870] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, context_store_compl)];
    const _: [(); 0x878] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, unk_flag)];
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunCompute {
    pub(crate) notifier: Arc<GpuObject<event::Notifier::ver>>,
    pub(crate) preempt_buf: GpuArray<u8>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VmBind,
    pub(crate) timestamps: Arc<GpuObject<job::JobTimestamps>>,
    pub(crate) user_timestamps: job::UserTimestamps,
}

#[versions(AGX)]
impl GpuStruct for RunCompute::ver {
    type Raw<'a> = raw::RunCompute::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunCompute::ver {}
