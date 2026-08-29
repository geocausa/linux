// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU compute job firmware structures

use super::types::*;
use super::{
    event,
    job,
    workqueue, //
};
use crate::{
    alloc,
    microseq,
    mmu, //
};
use kernel::{
    prelude::*,
    sync::{Arc, Mutex}, //
};

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
        #[ver(G != G15)]
        pub(crate) vm_slot: u32,
        // Exact 23J220 submitBuffer() copies the managed context ID acquired by
        // AGXContextIDManager::alloc() to RunCompute +0x10.
        #[ver(G == G15)]
        pub(crate) g15_context_id_10: u32,
        #[ver(G != G15)]
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        // Exact 23J220: descriptor +0x148 is the selected GPU address from
        // AGXCommandBuffer's 36-entry event-control pool. submitBuffer()
        // converts it to FWVA and writes it at RunCompute +0x14. Linux does
        // not yet own the matching per-command-buffer G15 pool, so keep this
        // pointer zero/fail-closed instead of exporting the legacy queue-wide
        // notifier object.
        #[ver(G == G15)]
        pub(crate) g15_event_control_fwva_14: U64,
        pub(crate) unk_pointee: u32,
        #[ver(G < G14X && G != G15)]
        pub(crate) __pad0: Array<0x50, u8>,
        #[ver(G < G14X && G != G15)]
        pub(crate) job_params1: JobParameters1<'a>,
        #[ver(G >= G14X || G == G15)]
        pub(crate) registers: job::raw::RegisterArray,
        #[ver(G != G15)]
        pub(crate) __pad1: Array<0x20, u8>,
        // processComputeSetup() snapshots AGXCommandQueue state into a local
        // record. In the active branch descriptor +0x3f0 receives queue +0x20
        // and descriptor +0x3f8 receives queue +0x38; submitBuffer() exports the
        // latter qword around the command's 4-byte gap.
        #[ver(G == G15)]
        pub(crate) g15_pre_micro_730: Array<0x10, u8>,
        #[ver(G == G15)]
        pub(crate) g15_queue_state_20_740: U64,
        #[ver(G == G15)]
        pub(crate) g15_queue_state_38_lo_748: U32,
        #[ver(G == G15)]
        pub(crate) g15_pre_micro_74c: Array<0x04, u8>,
        #[ver(G == G15)]
        pub(crate) g15_queue_state_38_hi_750: U32,
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
        #[ver(G != G15)]
        pub(crate) meta: job::raw::JobMeta,
        #[ver(G == G15)]
        pub(crate) meta: job::raw::G15JobMeta,
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
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_flag: U32,
        // RTKit's engine-2 Interrupt Post recovery callback sets this to 1
        // before reconciling this command's UMA Page Pool State.
        #[ver(G == G15)]
        pub(crate) g15_recovery_marker_878: U32,
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_pad: Array<0x10, u8>,
        #[ver(G == G15)]
        pub(crate) g15_tail_87c: Array<0x04, u8>,
    }

    const _: [(); 0x880] = [(); core::mem::size_of::<RunComputeG15V14_7<'static>>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, counter)];
    // AGXCLChannelSKU::submitBuffer() writes the managed G15 context ID at
    // +0x10, then the event-control FWVA at +0x14.
    const _: [(); 0x10] =
        [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_context_id_10)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(
        RunComputeG15V14_7<'static>,
        g15_event_control_fwva_14
    )];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, unk_pointee)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, registers)];
    const _: [(); 0x730] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pre_micro_730)];
    const _: [(); 0x740] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_queue_state_20_740)];
    const _: [(); 0x748] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_queue_state_38_lo_748)];
    const _: [(); 0x74c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pre_micro_74c)];
    const _: [(); 0x750] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_queue_state_38_hi_750)];
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
    // E131 exact SKU source loads: +0x7f0 fw_stamp, +0x7f8 stamp_value,
    // +0x808 UUID and +0x80c queue-local event sequence.
    const _: [(); 0x7f0] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)
        + core::mem::offset_of!(job::raw::G15JobMeta, fw_stamp)];
    const _: [(); 0x7f8] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)
        + core::mem::offset_of!(job::raw::G15JobMeta, stamp_value)];
    const _: [(); 0x808] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)
        + core::mem::offset_of!(job::raw::G15JobMeta, uuid)];
    const _: [(); 0x80c] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, meta)
        + core::mem::offset_of!(job::raw::G15JobMeta, event_seq)];
    const _: [(); 0x810] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, command_time)];
    const _: [(); 0x818] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, timestamp_pointers)];
    const _: [(); 0x828] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, user_timestamp_pointers)];
    const _: [(); 0x830] = [();
        core::mem::offset_of!(RunComputeG15V14_7<'static>, user_timestamp_pointers)
            + core::mem::offset_of!(job::raw::TimestampPointers<'static>, end_addr)
    ];
    const _: [(); 0x838] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_pad_838)];
    const _: [(); 0x83e] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_page_pool_state_fwva_83e)];
    const _: [(); 0x846] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_prepared_846)];
    const _: [(); 0x847] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_min_pool_size_847)];
    const _: [(); 0x84f] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_ideal_pool_size_84f)];
    const _: [(); 0x857] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_uma_metrics_fwva_857)];
    const _: [(); 0x85f] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_context_id_generation_85f)];
    const _: [(); 0x860] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, context_store_req)];
    const _: [(); 0x870] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, context_store_compl)];
    const _: [(); 0x878] = [(); core::mem::offset_of!(RunComputeG15V14_7<'static>, g15_recovery_marker_878)];
}


/// Exact inactive/stock-empty J615 G15 Compute SKU stream size.
///
/// This is the 23J220 no-PerfCtr path: type-0xb setup, start timestamp,
/// Compute WFI, end timestamp, type-0xc retirement packet, finish word, and
/// zero padding to Apple's 0x40-byte reported alignment.
pub(crate) const G15_STOCK_EMPTY_SKU_STREAM_SIZE: usize = 0x2c0;
/// Exact 23J220 Compute SKU encoder slot count on normal J615.
pub(crate) const G15_SKU_SLOT_COUNT: usize = 0xf0;
/// Exact per-slot allocation stride. The stock-empty stream uses 0x2c0 bytes
/// and therefore leaves the final 0x40 bytes of each slot unused.
pub(crate) const G15_SKU_SLOT_STRIDE: usize = 0x300;
/// Exact mapped SKU backing after rounding 0xf0 * 0x300 to the 16-KiB host page.
pub(crate) const G15_SKU_BACKING_BYTES: usize = 0x30000;
/// Exact host-only IOGPUEvent storage owned by Apple's SKU encoder. Linux does
/// not reproduce this raw event array in E104; slot retirement stays separate.
#[allow(dead_code)]
pub(crate) const G15_SKU_HOST_EVENT_BYTES: usize = 0x3c00;
const G15_STOCK_EMPTY_SKU_PRE_ROUND_SIZE: usize = 0x2b8;
const G15_SKU_TYPE0B_PACKET_SIZE: usize = 0x1bc;
const G15_SKU_TIMESTAMP_SIZE: usize = 0x3c;
const G15_SKU_TYPE0C_PACKET_SIZE: usize = 0x7c;
const G15_SKU_START_TIMESTAMP_OFFSET: usize = 0x1bc;
const G15_SKU_WFI_OFFSET: usize = 0x1f8;
const G15_SKU_END_TIMESTAMP_OFFSET: usize = 0x1fc;
const G15_SKU_TYPE0C_OFFSET: usize = 0x238;
const G15_SKU_FINISH_OFFSET: usize = 0x2b4;

// Exact J615 CL-channel resource geometry from AGXCLChannel::init():
// aligned `(num_cores * 0x1800 + num_mgpus * 0x40)` with 10 cores / 1 MGPU,
// then one 0x800-byte MGPU span.
pub(crate) const G15_J615_CL_SKU_REGION_STRIDE: u64 = 0xf400;
pub(crate) const G15_J615_CL_SKU_MGPU_SPAN: u64 = 0x800;
/// Exact logical size of the persistent J615 CL-channel command-resource
/// backing: two 0xf400 regions, one 0x800 MGPU span, and Apple's fixed 0x400
/// low-bit term. E114 independently recovers the same geometry from 23J220.
pub(crate) const G15_J615_CL_COMMAND_RESOURCE_BYTES: usize = 0x1f400;

const _: [(); 0x1f400] = [();
    ((2 * G15_J615_CL_SKU_REGION_STRIDE + G15_J615_CL_SKU_MGPU_SPAN) as usize) | 0x400
];
const _: [(); G15_J615_CL_COMMAND_RESOURCE_BYTES] = [(); 0x1f400];

const _: [(); G15_STOCK_EMPTY_SKU_PRE_ROUND_SIZE] = [();
    G15_SKU_TYPE0B_PACKET_SIZE
        + G15_SKU_TIMESTAMP_SIZE
        + 4
        + G15_SKU_TIMESTAMP_SIZE
        + G15_SKU_TYPE0C_PACKET_SIZE
        + 4
];
const _: [(); G15_STOCK_EMPTY_SKU_STREAM_SIZE] = [(); 0x2c0];
const _: [(); 0x2d000] = [(); G15_SKU_SLOT_COUNT * G15_SKU_SLOT_STRIDE];
const _: [(); G15_SKU_BACKING_BYTES] = [();
    ((G15_SKU_SLOT_COUNT * G15_SKU_SLOT_STRIDE + mmu::UAT_PGSZ - 1)
        & !mmu::UAT_PGMSK)
];
const _: [(); G15_SKU_HOST_EVENT_BYTES] = [(); G15_SKU_SLOT_COUNT * 0x40];
const _: [(); G15_SKU_START_TIMESTAMP_OFFSET] = [(); G15_SKU_TYPE0B_PACKET_SIZE];
const _: [(); G15_SKU_WFI_OFFSET] =
    [(); G15_SKU_START_TIMESTAMP_OFFSET + G15_SKU_TIMESTAMP_SIZE];
const _: [(); G15_SKU_END_TIMESTAMP_OFFSET] = [(); G15_SKU_WFI_OFFSET + 4];
const _: [(); G15_SKU_TYPE0C_OFFSET] =
    [(); G15_SKU_END_TIMESTAMP_OFFSET + G15_SKU_TIMESTAMP_SIZE];
const _: [(); G15_SKU_FINISH_OFFSET] = [(); G15_SKU_TYPE0C_OFFSET + G15_SKU_TYPE0C_PACKET_SIZE];

/// Runtime-owned sources required to serialize Apple's exact stock-empty G15
/// Compute SKU stream. E101 closes every byte producer; this structure keeps
/// dynamic addresses/state explicit instead of capturing or guessing them.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15StockEmptySkuInputs {
    pub(crate) command_fwva: u64,
    pub(crate) stream_fwva: u64,
    /// Converted firmware-object +0x268 value used in both type-0xb/0xc.
    pub(crate) firmware_state_fwva: u64,
    /// Converted AGXChannel +0x90 state address.
    pub(crate) channel_state_fwva: u64,
    /// Base of the four CL channel command-resource regions (channel +0x1c0).
    pub(crate) channel_command_region_base_fwva: u64,
    pub(crate) event_control_fwva: u64,
    pub(crate) page_pool_state_fwva: u64,
    pub(crate) hwmetrics_fwva: u64,
    pub(crate) fw_stamp_fwva: u64,
    /// Exact encodeTimeStamp() predicate: either command +0x828/+0x830 is present.
    pub(crate) user_timestamps_present: bool,
    pub(crate) command_counter: u64,
    pub(crate) context_id: u32,
    pub(crate) state_sequence: u32,
    pub(crate) queue_event_sequence: u32,
    pub(crate) evctl_index: u32,
    pub(crate) uuid: u32,
    pub(crate) stamp_value: u32,
    pub(crate) gart_soft_fault_enabled: bool,
    pub(crate) accelerator_654_bit7: bool,
}

/// Byte-exact serializer for the inactive stock-empty 23J220 G15 Compute SKU
/// stream. It owns no GPU memory and is deliberately unused by RunCompute.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct G15StockEmptySkuStream {
    bytes: [u8; G15_STOCK_EMPTY_SKU_STREAM_SIZE],
}

#[allow(dead_code)]
impl G15StockEmptySkuStream {
    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn add_addr(base: u64, offset: u64) -> Result<u64> {
        base.checked_add(offset).ok_or(EOVERFLOW)
    }

    fn write_timestamp(
        bytes: &mut [u8],
        offset: usize,
        start: bool,
        input: &G15StockEmptySkuInputs,
    ) -> Result {
        Self::put_u32(bytes, offset, if start { 0x8000_0003 } else { 0x0000_0003 });
        Self::put_u64(bytes, offset + 0x04, Self::add_addr(input.command_fwva, 0x810)?);
        Self::put_u64(bytes, offset + 0x0c, Self::add_addr(input.command_fwva, 0x818)?);
        Self::put_u64(
            bytes,
            offset + 0x14,
            Self::add_addr(input.command_fwva, if start { 0x818 } else { 0x820 })?,
        );
        Self::put_u64(bytes, offset + 0x1c, input.channel_state_fwva);
        let user_ts = if input.user_timestamps_present {
            Self::add_addr(input.command_fwva, 0x828)?
        } else {
            0
        };
        Self::put_u64(bytes, offset + 0x24, user_ts);
        Self::put_u64(bytes, offset + 0x2c, Self::add_addr(input.command_fwva, 0x868)?);
        // Exact encodeTimeStamp() stores its first argument here. The G15 CL
        // caller passes command +0x808 (UUID); command type 3 only selects the
        // offset table above.
        Self::put_u32(bytes, offset + 0x34, input.uuid);
        Ok(())
    }

    pub(crate) fn new(input: G15StockEmptySkuInputs) -> Result<Self> {
        if input.command_fwva == 0
            || input.stream_fwva == 0
            || input.firmware_state_fwva == 0
            || input.channel_state_fwva == 0
            || input.channel_command_region_base_fwva == 0
            || input.event_control_fwva == 0
            || input.page_pool_state_fwva == 0
            || input.hwmetrics_fwva == 0
            || input.fw_stamp_fwva == 0
        {
            return Err(EINVAL);
        }

        let mut bytes = [0u8; G15_STOCK_EMPTY_SKU_STREAM_SIZE];

        // Type-0xb packet. The exact host zeroes all 0x1b8 payload bytes first.
        Self::put_u32(&mut bytes, 0x000, 0x0000_000b);
        let p = 0x004;
        Self::put_u64(&mut bytes, p + 0x10, Self::add_addr(input.command_fwva, 0x20)?);
        Self::put_u64(&mut bytes, p + 0x18, input.firmware_state_fwva);
        Self::put_u64(&mut bytes, p + 0x20, input.channel_state_fwva);
        Self::put_u32(&mut bytes, p + 0x28, input.context_id);
        Self::put_u32(&mut bytes, p + 0x2c, input.gart_soft_fault_enabled as u32);
        Self::put_u32(&mut bytes, p + 0x30, input.state_sequence);
        Self::put_u32(&mut bytes, p + 0x34, input.queue_event_sequence);
        Self::put_u32(&mut bytes, p + 0x38, input.evctl_index);
        // p+0x3c is exact `(descriptor+0x460 != 0)`, zero on stock-empty.
        Self::put_u64(&mut bytes, p + 0x40, Self::add_addr(input.command_fwva, 0x76c)?);
        Self::put_u32(&mut bytes, p + 0x4c, input.uuid);
        // p+0x50..+0x150 are zero: no stock-empty per-counter subrecords/count.
        Self::put_u64(&mut bytes, p + 0x158, input.page_pool_state_fwva);
        // p+0x160/+0x168 are zero on the shared/async stock-empty UMA path.
        Self::put_u64(&mut bytes, p + 0x170, input.hwmetrics_fwva);
        bytes[p + 0x178] = input.accelerator_654_bit7 as u8;

        let region0 = input.channel_command_region_base_fwva;
        let region1 = Self::add_addr(region0, G15_J615_CL_SKU_REGION_STRIDE)?;
        let region2 = Self::add_addr(region1, G15_J615_CL_SKU_REGION_STRIDE)?;
        let region3 = Self::add_addr(region2, G15_J615_CL_SKU_MGPU_SPAN)?;
        Self::put_u64(&mut bytes, p + 0x180, region0);
        Self::put_u64(&mut bytes, p + 0x188, region1);
        Self::put_u64(&mut bytes, p + 0x190, region2);
        Self::put_u64(&mut bytes, p + 0x198, region3);
        Self::put_u64(&mut bytes, p + 0x1a0, Self::add_addr(input.command_fwva, 0x878)?);
        Self::put_u64(&mut bytes, p + 0x1a8, input.command_counter);
        Self::put_u64(&mut bytes, p + 0x1b0, Self::add_addr(input.event_control_fwva, 0xa8)?);

        Self::write_timestamp(&mut bytes, G15_SKU_START_TIMESTAMP_OFFSET, true, &input)?;
        Self::put_u32(&mut bytes, G15_SKU_WFI_OFFSET, 1);
        Self::write_timestamp(&mut bytes, G15_SKU_END_TIMESTAMP_OFFSET, false, &input)?;

        // Exact stock-empty type-0xc retirement packet.
        let c = G15_SKU_TYPE0C_OFFSET;
        Self::put_u32(&mut bytes, c, 0x0000_000c);
        Self::put_u64(&mut bytes, c + 0x04, input.firmware_state_fwva);
        Self::put_u64(&mut bytes, c + 0x0c, input.channel_state_fwva);
        Self::put_u32(&mut bytes, c + 0x14, input.context_id);
        Self::put_u64(&mut bytes, c + 0x18, Self::add_addr(input.command_fwva, 0x76c)?);
        // c+0x20 is zero; c+0x24 is the command UUID.
        Self::put_u32(&mut bytes, c + 0x24, input.uuid);
        Self::put_u64(&mut bytes, c + 0x28, input.fw_stamp_fwva);
        Self::put_u32(&mut bytes, c + 0x30, input.stamp_value);
        // c+0x34..+0x54 are the absent stock-empty per-counter material.
        Self::put_u64(&mut bytes, c + 0x58, Self::add_addr(input.stream_fwva, 0x15c)?);
        // Encoder bookkeeping is the negated packet-start offset. The inactive
        // stream's type-0xc packet begins at 0x238, yielding -0x238 in u32.
        Self::put_u32(&mut bytes, c + 0x60, 0xffff_fdc8);
        // c+0x64 is zero: no counter subrecord and accelerator +0x2464 == 0.
        Self::put_u64(&mut bytes, c + 0x65, Self::add_addr(input.command_fwva, 0x878)?);
        Self::put_u64(&mut bytes, c + 0x6d, Self::add_addr(input.command_fwva, 0x860)?);
        // c+0x75..+0x7b remain exact zero tail bytes.

        Self::put_u32(&mut bytes, G15_SKU_FINISH_OFFSET, 0x4000_0002);
        // 0x2b8..0x2bf is report-size alignment padding and remains zero.
        Ok(Self { bytes })
    }

    pub(crate) fn as_bytes(&self) -> &[u8; G15_STOCK_EMPTY_SKU_STREAM_SIZE] {
        &self.bytes
    }
}

/// One selected/retired SKU slot whose address is known before bytes are
/// serialized. E113 separates this reservation from the later command-aware
/// serializer because the exact SKU payload embeds both command and stream
/// addresses. The token has no RunCompute consumer.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15ReservedSkuSlot {
    index: u16,
    fwva: u64,
}

#[allow(dead_code)]
impl G15ReservedSkuSlot {
    pub(crate) fn index(&self) -> usize {
        self.index as usize
    }

    pub(crate) fn fwva(&self) -> u64 {
        self.fwva
    }
}

/// One fully serialized but still-unpublished SKU slot. E107 allows the FWVA
/// to exist only in this definition-only token; RunCompute has no consumer.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15PreparedSkuSlot {
    index: u16,
    fwva: u64,
    size: u32,
}

#[allow(dead_code)]
impl G15PreparedSkuSlot {
    pub(crate) fn index(&self) -> usize {
        self.index as usize
    }

    pub(crate) fn fwva(&self) -> u64 {
        self.fwva
    }

    pub(crate) fn size(&self) -> u32 {
        self.size
    }
}

/// Persistent exact 23J220 Compute SKU backing owner.
///
/// E103 proves one accelerator-owned special-range-8 mapping of exactly
/// 0x30000 bytes, divided into 0xf0 slots at 0x300-byte stride. The E104 owner
/// is deliberately unreachable. E107 adds only a controlled retired-slot copy
/// method; RunCompute still has no consumer for the returned prepared token.
/// Apple's separate 0xf0 x 0x40 host event array is not reproduced here.
#[allow(dead_code)]
pub(crate) struct G15SkuBacking {
    backing: alloc::G15SharedGpuArray<u8>,
}

#[allow(dead_code)]
impl G15SkuBacking {
    pub(crate) fn new(
        dev: &crate::driver::AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut allocator = alloc::G15SharedBank1Allocator::new_range8(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let backing = allocator.array_empty_shared_data::<u8>(G15_SKU_BACKING_BYTES)?;
        let base: u64 = backing.weak_pointer().into();
        if backing.len() != G15_SKU_BACKING_BYTES
            || base & mmu::UAT_PGMSK as u64 != 0
        {
            return Err(EIO);
        }
        if backing.as_slice().iter().any(|byte| *byte != 0) {
            return Err(EIO);
        }

        Ok(Self { backing })
    }

    /// Resolve one already-retired slot address without writing the backing.
    /// This is the first phase required by E112: the selected stream FWVA must
    /// be known before E102 can serialize command-relative SKU bytes.
    pub(crate) fn reserve_retired_slot(&self, index: usize) -> Result<G15ReservedSkuSlot> {
        if index >= G15_SKU_SLOT_COUNT {
            return Err(EINVAL);
        }
        let start = index.checked_mul(G15_SKU_SLOT_STRIDE).ok_or(EOVERFLOW)?;
        let end = start.checked_add(G15_SKU_SLOT_STRIDE).ok_or(EOVERFLOW)?;
        if end > self.backing.len() {
            return Err(EIO);
        }
        let base: u64 = self.backing.weak_pointer().into();
        let fwva = base.checked_add(start as u64).ok_or(EOVERFLOW)?;
        Ok(G15ReservedSkuSlot {
            index: index.try_into().map_err(|_| EOVERFLOW)?,
            fwva,
        })
    }

    /// Copy a finalized E102 stock-empty stream into a previously reserved
    /// slot. The reservation is re-derived before the write so an index/FWVA
    /// mismatch fails closed. The complete 0x300 bytes are cleared first.
    pub(crate) fn write_reserved_stock_empty_slot(
        &mut self,
        reserved: G15ReservedSkuSlot,
        stream: &G15StockEmptySkuStream,
    ) -> Result<G15PreparedSkuSlot> {
        let expected = self.reserve_retired_slot(reserved.index())?;
        if expected.fwva() != reserved.fwva() {
            return Err(EIO);
        }
        let start = reserved
            .index()
            .checked_mul(G15_SKU_SLOT_STRIDE)
            .ok_or(EOVERFLOW)?;
        let end = start.checked_add(G15_SKU_SLOT_STRIDE).ok_or(EOVERFLOW)?;
        let slot = &mut self.backing.as_mut_slice()[start..end];
        for byte in slot.iter_mut() {
            *byte = 0;
        }
        slot[..G15_STOCK_EMPTY_SKU_STREAM_SIZE].copy_from_slice(stream.as_bytes());
        if slot[G15_STOCK_EMPTY_SKU_STREAM_SIZE..]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(EIO);
        }
        Ok(G15PreparedSkuSlot {
            index: reserved.index().try_into().map_err(|_| EOVERFLOW)?,
            fwva: reserved.fwva(),
            size: G15_STOCK_EMPTY_SKU_STREAM_SIZE as u32,
        })
    }

    /// Copy one E102 stock-empty stream into a slot that the caller has already
    /// proved retired/bound through the E106 guard. The entire 0x300-byte slot
    /// is cleared first, preserving Apple's 0x40 bytes of stock-empty slack.
    /// The returned FWVA is intentionally trapped in an unpublished token.
    pub(crate) fn write_retired_stock_empty_slot(
        &mut self,
        index: usize,
        stream: &G15StockEmptySkuStream,
    ) -> Result<G15PreparedSkuSlot> {
        let reserved = self.reserve_retired_slot(index)?;
        self.write_reserved_stock_empty_slot(reserved, stream)
    }

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
