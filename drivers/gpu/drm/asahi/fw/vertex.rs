// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU vertex job firmware structures

use super::types::*;
use super::{
    event,
    job,
    workqueue, //
};
use crate::{
    buffer,
    fw,
    microseq,
    mmu, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default, Copy, Clone)]
    #[repr(C)]
    pub(crate) struct TilingParameters {
        pub(crate) rgn_size: u32,
        pub(crate) unk_4: u32,
        pub(crate) ppp_ctrl: u32,
        pub(crate) x_max: u16,
        pub(crate) y_max: u16,
        pub(crate) te_screen: u32,
        pub(crate) te_mtile1: u32,
        pub(crate) te_mtile2: u32,
        pub(crate) tiles_per_mtile: u32,
        pub(crate) tpc_stride: u32,
        pub(crate) unk_24: u32,
        pub(crate) unk_28: u32,
        pub(crate) helper_cfg: u32,
        pub(crate) __pad: Pad<0x70>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: F32,
        pub(crate) unk_c: F32,
        pub(crate) tvb_tilemap: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_tilemaps: Option<GpuPointer<'a, &'a [u8]>>,
        pub(crate) tpc: GpuPointer<'a, &'a [u8]>,
        pub(crate) tvb_heapmeta: GpuPointer<'a, &'a [u8]>,
        pub(crate) iogpu_unk_54: U64,
        pub(crate) iogpu_unk_56: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta1: Option<GpuPointer<'a, &'a [u8]>>,
        pub(crate) utile_config: u32,
        pub(crate) unk_4c: u32,
        pub(crate) ppp_multisamplectl: U64,
        pub(crate) tvb_layermeta: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_layermeta: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) core_mask: Array<2, u32>,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf2: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_80: U64,
        pub(crate) preempt_buf3: GpuPointer<'a, &'a [u8]>,
        pub(crate) vdm_ctrl_stream_base: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta2: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta3: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) tiling_control: u32,
        #[ver(G < G14)]
        pub(crate) unk_ac: u32,
        pub(crate) unk_b0: Array<6, U64>,
        pub(crate) usc_exec_base_ta: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta4: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) unk_f0: U64,
        pub(crate) unk_f8: U64,
        pub(crate) helper_program: u32,
        pub(crate) unk_104: u32,
        pub(crate) helper_arg: U64,
        pub(crate) unk_110: U64,
        pub(crate) unk_118: u32,
        #[ver(G >= G14)]
        pub(crate) __pad: Pad<{ 8 * 9 + 0x268 }>,
        #[ver(G < G14)]
        pub(crate) __pad: Pad<0x268>,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2<'a> {
        pub(crate) unk_480: Array<4, u32>,
        pub(crate) unk_498: U64,
        pub(crate) unk_4a0: u32,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_4ac: u32,
        pub(crate) unk_4b0: U64,
        pub(crate) unk_4b8: u32,
        pub(crate) unk_4bc: U64,
        pub(crate) unk_4c4_padding: Array<0x48, u8>,
        pub(crate) unk_50c: u32,
        pub(crate) unk_510: U64,
        pub(crate) unk_518: U64,
        pub(crate) unk_520: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunVertex<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) vm_slot: u32,
        pub(crate) unk_8: u32,
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        pub(crate) buffer_slot: u32,
        pub(crate) unk_1c: u32,
        pub(crate) buffer: GpuPointer<'a, fw::buffer::Info::ver>,
        pub(crate) scene: GpuPointer<'a, fw::buffer::Scene::ver>,
        pub(crate) unk_buffer_buf: GpuWeakPointer<[u8]>,
        pub(crate) unk_34: u32,

        #[ver(G < G14X && G != G15)]
        pub(crate) job_params1: JobParameters1::ver<'a>,
        #[ver(G < G14X && G != G15)]
        pub(crate) tiling_params: TilingParameters,
        #[ver(G >= G14X || G == G15)]
        pub(crate) registers: job::raw::RegisterArray,

        // G15 keeps the register-list command generation, with 0x10 bytes of
        // command state inserted after the register array.
        #[ver(G == G15)]
        pub(crate) g15_pre_tpc_750: Array<0x10, u8>,
        pub(crate) tpc: GpuPointer<'a, &'a [u8]>,
        pub(crate) tpc_size: U64,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) fragment_stamp_slot: u32,
        pub(crate) fragment_stamp_value: EventValue,
        pub(crate) unk_pointee: u32,
        #[ver(G != G15)]
        pub(crate) unk_pad: u32,
        #[ver(G != G15)]
        pub(crate) job_params2: JobParameters2<'a>,
        #[ver(G != G15)]
        pub(crate) encoder_params: job::raw::EncoderParams,
        #[ver(G != G15)]
        pub(crate) unk_55c: u32,
        #[ver(G != G15)]
        pub(crate) unk_560: u32,
        #[ver(G != G15)]
        pub(crate) sync_grow: u32,
        #[ver(G != G15)]
        pub(crate) unk_568: u32,
        #[ver(G != G15)]
        pub(crate) uses_scratch: u32,
        // Apple generic TA submission exports raw Render payload state here:
        // +0x788/+0x790/+0x798/+0x7a0/+0x7a8 <- raw +0x10/+0x18/+0x20/+0x28/+0x60.
        // Later +0x83c/+0x840/+0x848 pack raw +0x60c/+0x608/+0x610/+0x614.
        #[ver(G == G15)]
        pub(crate) g15_raw_render_10_788: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_18_790: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_20_798: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_28_7a0: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_60_7a8: U64,
        #[ver(G == G15)]
        pub(crate) g15_zero_7b0: U64,
        #[ver(G == G15)]
        pub(crate) g15_zero_7b8: U64,
        #[ver(G == G15)]
        pub(crate) g15_zero_7c0: U64,
        // Apple TA submit leaves +0x7c8..+0x80f unwritten, explicitly zeros
        // +0x810..+0x82f with two 16-byte stores, then leaves +0x830..+0x83b
        // unwritten before the next scalar at +0x83c.
        #[ver(G == G15)]
        pub(crate) g15_pad_7c8: Array<0x48, u8>,
        #[ver(G == G15)]
        pub(crate) g15_zero_810: Array<0x20, u8>,
        #[ver(G == G15)]
        pub(crate) g15_pad_830: Array<0x0c, u8>,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_60c_83c: U32,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_608_610_lo_840: U64,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_614_848: U32,
        // processRenderSetup() writes descriptor +0xe08 from Render wrapper +0x230,
        // which parseAndValidate() maps from raw Render byte +0x1bc. Apple TA
        // submission zero-extends the resulting nonzero boolean to +0x84c.
        #[ver(G == G15)]
        pub(crate) g15_raw_render_1bc_nonzero_84c: U32,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_1bf_850: U32,
        // bool(AGXSegmentKernelCommand byte +0x198).
        #[ver(G == G15)]
        pub(crate) g15_segment_flag_198_854: U32,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_1c1_858: U32,
        #[ver(G == G15)]
        pub(crate) g15_raw_render_1c2_85c: U32,
        #[ver(G != G15)]
        pub(crate) meta: job::raw::JobMeta,
        #[ver(G == G15)]
        pub(crate) meta: job::raw::G15JobMeta,
        #[ver(G != G15)]
        pub(crate) unk_after_meta: u32,
        // Apple G15 processRenderSetup() copies raw Render byte +0x619 through
        // AGXRenderHardwareKernelCommand wrapper +0x1b8 into TA descriptor +0x7b0;
        // submitBuffer() writes (descriptor[0x7b0] == 2) here.
        #[ver(G == G15)]
        pub(crate) g15_raw_render_619_eq_2_88c: U32,
        #[ver(G != G15)]
        pub(crate) unk_buf_0: U64,
        #[ver(G != G15)]
        pub(crate) unk_buf_8: U64,
        #[ver(G != G15)]
        pub(crate) unk_buf_10: U64,
        #[ver(G != G15)]
        pub(crate) command_time: U64,
        #[ver(G != G15)]
        pub(crate) timestamp_pointers: job::raw::TimestampPointers<'a>,
        #[ver(G != G15)]
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,
        #[ver(G != G15)]
        pub(crate) client_sequence: u8,
        #[ver(G != G15)]
        pub(crate) pad_5d5: Array<3, u8>,
        #[ver(G != G15)]
        pub(crate) unk_5d8: u32,
        #[ver(G != G15)]
        pub(crate) unk_5dc: u8,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_ts: U64,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_5dd_8: Array<0x1b, u8>,

        // Apple passes command +0x890 as AGFIBarrierState to G15 vslot +0x218.
        // That slot resolves to generatePreparseBarrierRegister(), whose entire
        // body is `bti; ret`; submitBuffer() itself does not write +0x890..+0x8a7.
        // Keep the first qword addressable for the compile-only scaffold and
        // model the following 0x10 bytes as unwritten packed padding.
        #[ver(G == G15)]
        pub(crate) g15_barrier_state_890: U64,
        #[ver(G == G15)]
        pub(crate) g15_pad_898: Array<0x10, u8>,
        #[ver(G == G15)]
        pub(crate) g15_zero_8a8: U64,
        // processRenderSetup() stores the IOGPUSegmentResourceList GPUVA at
        // descriptor +0x6d0 (unless feature-gated to zero); TA submit converts
        // it through AGXArmFirmware::convertGPUVAToFWVA() into command +0x8b0.
        #[ver(G == G15)]
        pub(crate) g15_segment_resource_list_fwva_8b0: U64,
        // Apple copies IOGPUBlockFence +0x108 (the first shared time slot
        // consumed by AGXBlockFence::getHostTime()) into the TA descriptor,
        // then ChinookV9 convertGPUVAToFWVA() exports it here.
        #[ver(G == G15)]
        pub(crate) g15_block_fence_time0_fwva_8b8: U64,
        // AGXMTLCounterSampler::fwTokenEncode() writes its two U64 outputs to
        // TA descriptor +0x1080/+0x1088; submitBuffer() copies them unchanged.
        #[ver(G == G15)]
        pub(crate) g15_mtl_counter_fw_token_0_8c0: U64,
        #[ver(G == G15)]
        pub(crate) g15_mtl_counter_fw_token_1_8c8: U64,
        // Apple G15 submission leaves +0x8d0..+0x8d5 unwritten; the first explicit
        // late-tail store begins at +0x8d6. The same six-byte packed-ABI gap
        // appears in Compute, TA, and 3D immediately before an unaligned U64.
        #[ver(G == G15)]
        pub(crate) g15_pad_8d0: Array<0x06, u8>,
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_state_fwva_8d6: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_prepared_8de: u8,
        #[ver(G == G15)]
        pub(crate) g15_uma_min_pool_size_8df: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_ideal_pool_size_8e7: U64,
        #[ver(G == G15)]
        pub(crate) g15_uma_metrics_fwva_8ef: U64,
        #[ver(G == G15)]
        pub(crate) g15_context_id_generation_8f7: u8,
        // The G15 TA encoder embeds command_fwva +0x8f8. RTKit writes the
        // dispatch/start timestamp at +0x8f8; normal TA completion requires
        // +0x8f8 < +0x908 and measures that interval. The shared timing state
        // occupies +0x8f8..+0x917; the final command qword remains opaque.
        #[ver(G == G15)]
        pub(crate) g15_sku_timing_8f8: job::raw::G15SkuTimingState,
        #[ver(G == G15)]
        pub(crate) g15_tail_918: Array<0x08, u8>,
    }

    const _: [(); 0x920] = [(); core::mem::size_of::<RunVertexG15V14_7<'static>>()];
    const _: [(); 0x40] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, registers)];
    const _: [(); 0x760] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, tpc)];
    const _: [(); 0x768] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, tpc_size)];
    const _: [(); 0x770] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, microsequence)];
    const _: [(); 0x778] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, microsequence_size)];
    const _: [(); 0x77c] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, fragment_stamp_slot)];
    const _: [(); 0x780] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, fragment_stamp_value)];
    const _: [(); 0x784] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, unk_pointee)];
    const _: [(); 0x788] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_10_788)];
    const _: [(); 0x790] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_18_790)];
    const _: [(); 0x798] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_20_798)];
    const _: [(); 0x7a0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_28_7a0)];
    const _: [(); 0x7a8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_60_7a8)];
    const _: [(); 0x7b0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_zero_7b0)];
    const _: [(); 0x7b8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_zero_7b8)];
    const _: [(); 0x7c0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_zero_7c0)];
    const _: [(); 0x7c8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_pad_7c8)];
    const _: [(); 0x810] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_zero_810)];
    const _: [(); 0x830] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_pad_830)];
    const _: [(); 0x83c] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_60c_83c)];
    const _: [(); 0x840] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_608_610_lo_840)];
    const _: [(); 0x848] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_614_848)];
    const _: [(); 0x84c] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_1bc_nonzero_84c)];
    const _: [(); 0x850] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_1bf_850)];
    const _: [(); 0x854] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_segment_flag_198_854)];
    const _: [(); 0x858] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_1c1_858)];
    const _: [(); 0x85c] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_1c2_85c)];
    const _: [(); 0x860] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, meta)];
    // Generic Apple TA submission writes its queue sequence directly to +0x888,
    // independently confirming JobMeta::event_seq within the G15 command.
    const _: [(); 0x888] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, meta)
        + core::mem::offset_of!(job::raw::G15JobMeta, event_seq)];
    const _: [(); 0x88c] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_raw_render_619_eq_2_88c)];
    const _: [(); 0x890] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_barrier_state_890)];
    const _: [(); 0x898] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_pad_898)];
    const _: [(); 0x8a8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_zero_8a8)];
    const _: [(); 0x8b0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_segment_resource_list_fwva_8b0)];
    const _: [(); 0x8b8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_block_fence_time0_fwva_8b8)];
    const _: [(); 0x8c0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_mtl_counter_fw_token_0_8c0)];
    const _: [(); 0x8c8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_mtl_counter_fw_token_1_8c8)];
    const _: [(); 0x8d0] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_pad_8d0)];
    const _: [(); 0x8d6] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_uma_page_pool_state_fwva_8d6)];
    const _: [(); 0x8de] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_uma_prepared_8de)];
    const _: [(); 0x8df] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_uma_min_pool_size_8df)];
    const _: [(); 0x8e7] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_uma_ideal_pool_size_8e7)];
    const _: [(); 0x8ef] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_uma_metrics_fwva_8ef)];
    const _: [(); 0x8f7] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_context_id_generation_8f7)];
    const _: [(); 0x8f8] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_sku_timing_8f8)];
    const _: [(); 0x908] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_sku_timing_8f8)
        + core::mem::offset_of!(job::raw::G15SkuTimingState, complete_timestamp)];
    const _: [(); 0x918] = [(); core::mem::offset_of!(RunVertexG15V14_7<'static>, g15_tail_918)];
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunVertex {
    pub(crate) notifier: Arc<GpuObject<event::Notifier::ver>>,
    pub(crate) scene: Arc<buffer::Scene::ver>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VmBind,
    pub(crate) timestamps: Arc<GpuObject<job::RenderTimestamps>>,
    pub(crate) user_timestamps: job::UserTimestamps,
}

#[versions(AGX)]
impl GpuStruct for RunVertex::ver {
    type Raw<'a> = raw::RunVertex::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunVertex::ver {}
