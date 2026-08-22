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
        pub(crate) unk_0: Array<0x24, u8>,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) cdm_ctrl_stream_end: U64,
        pub(crate) unk_34: Array<0x20, u8>,
        pub(crate) unk_g14x: u32,
        pub(crate) unk_58: u32,
        #[ver(V < V13_0B4)]
        pub(crate) unk_5c: u32,
    }

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
        // G15 keeps the register-list command generation used by G14X, but
        // inserts 0x10 bytes of host-populated state before the microsequence.
        // The exact semantics of this active region are not yet closed, so it
        // remains opaque while G15 runtime submission stays fail-closed.
        #[ver(G == G15)]
        pub(crate) g15_pre_micro: Array<0x30, u8>,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) job_params2: JobParameters2::ver<'a>,
        #[ver(G != G15)]
        pub(crate) encoder_params: job::raw::EncoderParams,
        // Apple G15 compresses the post-JobParameters2 encoder metadata from
        // 0x28 to 0x18 bytes, returning JobMeta to the same +0x7e4 offset as
        // G14X. Keep it opaque until the individual host fields are named.
        #[ver(G == G15)]
        pub(crate) g15_encoder_meta: Array<0x18, u8>,
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
        // G15 has active unaligned state/pointers in +0x838..+0x85f. Their
        // exact individual meanings are still under reconstruction; preserve
        // the proven geometry and keep runtime submission disabled.
        #[ver(G == G15)]
        pub(crate) g15_tail_838: Array<0x28, u8>,
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
    const _: [(); 0x20] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, registers)];
    const _: [(); 0x760] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, microsequence)];
    const _: [(); 0x768] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, microsequence_size)];
    const _: [(); 0x76c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, job_params2)];
    const _: [(); 0x7e4] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)];
    const _: [(); 0x810] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, command_time)];
    const _: [(); 0x818] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, timestamp_pointers)];
    const _: [(); 0x828] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, user_timestamp_pointers)];
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
