// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU work queue firmware structes

use super::event;
use super::types::*;
use crate::event::EventValue;
use crate::{
    default_zeroed,
    trivial_gpustruct, //
};
use kernel::sync::Arc;

#[derive(Debug)]
#[repr(u32)]
pub(crate) enum CommandType {
    RunVertex = 0,
    RunFragment = 1,
    #[allow(dead_code)]
    RunBlitter = 2,
    RunCompute = 3,
    Barrier = 4,
    InitBuffer = 6,
}

pub(crate) trait Command: GpuStruct + Send + Sync {}

pub(crate) mod raw {
    use super::*;

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Barrier {
        pub(crate) tag: CommandType,
        pub(crate) wait_stamp: GpuWeakPointer<FwStamp>,
        // G15 setupBarrierCommand() writes a second unaligned stamp pointer
        // at +0x0c before the value/slot fields. In the common case Apple
        // writes the same FW stamp address into both pointers; alternate
        // event mappings may select different backing stamp spaces.
        #[ver(G == G15)]
        pub(crate) wait_stamp_2: GpuWeakPointer<FwStamp>,
        pub(crate) wait_value: EventValue,
        pub(crate) wait_slot: u32,
        pub(crate) stamp_self: EventValue,
        pub(crate) uuid: u32,
        pub(crate) external_barrier: u32,
        // G14X/G15 use this final control word for internal barrier state.
        pub(crate) internal_barrier_type: u32,
        #[ver(G != G15)]
        pub(crate) padding: Pad<0x1c>,
        #[ver(G == G15)]
        pub(crate) padding: Pad<0x04>,
    }

    const _: [(); 0x30] = [(); core::mem::size_of::<BarrierG15V14_7>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(BarrierG15V14_7, wait_stamp)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(BarrierG15V14_7, wait_stamp_2)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(BarrierG15V14_7, wait_value)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(BarrierG15V14_7, wait_slot)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(BarrierG15V14_7, stamp_self)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(BarrierG15V14_7, uuid)];
    const _: [(); 0x24] = [(); core::mem::offset_of!(BarrierG15V14_7, external_barrier)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(BarrierG15V14_7, internal_barrier_type)];

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct GpuContextData {
        pub(crate) unk_0: u8,
        pub(crate) unk_1: u8,
        unk_2: Array<0x2, u8>,
        pub(crate) unk_4: u8,
        pub(crate) unk_5: u8,
        unk_6: Array<0x18, u8>,
        pub(crate) unk_1e: u8,
        pub(crate) unk_1f: u8,
        unk_20: Array<0x3, u8>,
        pub(crate) unk_23: u8,
        unk_24: Array<0x1c, u8>,
    }

    impl Default for GpuContextData {
        fn default() -> Self {
            Self {
                unk_0: 0xff,
                unk_1: 0xff,
                unk_2: Default::default(),
                unk_4: 0,
                unk_5: 1,
                unk_6: Default::default(),
                unk_1e: 0xff,
                unk_1f: 0,
                unk_20: Default::default(),
                unk_23: 2,
                unk_24: Default::default(),
            }
        }
    }

    impl GpuContextData {
        /// Exact G15 scheduler/context resource bootstrap reconstructed from
        /// AGXCommandQueue::init(). The G15 firmware-visible object is 0x38
        /// bytes; this shared raw type retains the inherited 0x40 allocation
        /// size for older generations, but the first 0x38 bytes match Apple.
        pub(crate) fn g15() -> Self {
            // Apple zeroes exactly 0x38 bytes, then writes:
            //   +0x00/+0x01 = 0xff, +0x05 = 1, +0x22 = 0xff,
            //   +0x23..+0x26 = 0, +0x27 = AGXShared+0x100 = 2.
            let mut s = Self {
                unk_0: 0xff,
                unk_1: 0xff,
                unk_2: Default::default(),
                unk_4: 0,
                unk_5: 1,
                unk_6: Default::default(),
                unk_1e: 0,
                unk_1f: 0,
                unk_20: Default::default(),
                unk_23: 0,
                unk_24: Default::default(),
            };
            s.unk_20[2] = 0xff; // +0x22
            s.unk_24[3] = 2; // +0x27
            s
        }

        /// Fields consumed by G15 DeviceControl opcode 0x11
        /// (AGXArmFirmware::submitReleaseResource()).
        pub(crate) fn g15_release_resource_fields(&self) -> (u8, u8, u8, u8) {
            (self.unk_24[3], self.unk_0, self.unk_1, self.unk_4)
        }
    }

    const _: [(); 0x40] = [(); core::mem::size_of::<GpuContextData>()];
    const _: [(); 0x20] = [(); core::mem::offset_of!(GpuContextData, unk_20)];
    const _: [(); 0x24] = [(); core::mem::offset_of!(GpuContextData, unk_24)];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RingState {
        pub(crate) gpu_doneptr: AtomicU32,
        __pad0: Pad<0xc>,
        pub(crate) unk_10: AtomicU32,
        __pad1: Pad<0xc>,
        pub(crate) unk_20: AtomicU32,
        __pad2: Pad<0xc>,
        pub(crate) gpu_rptr: AtomicU32,
        __pad3: Pad<0xc>,
        pub(crate) cpu_wptr: AtomicU32,
        __pad4: Pad<0xc>,
        pub(crate) rb_size: u32,
        __pad5: Pad<0xc>,
        // This isn't part of the structure, but it's here as a
        // debugging hack so we can inspect what ring position
        // the driver considered complete and freeable.
        pub(crate) cpu_freeptr: AtomicU32,
        __pad6: Pad<0xc>,
    }
    default_zeroed!(RingState);

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct Priority(
        pub(crate) u32,
        pub(crate) u32,
        pub(crate) U64,
        pub(crate) u32,
        pub(crate) u32,
        pub(crate) u32,
    );

    pub(crate) const PRIORITY: [Priority; 4] = [
        Priority(0, 0, U64(0xffff_ffff_ffff_0000), 1, 0, 1),
        Priority(1, 1, U64(0xffff_ffff_0000_0000), 0, 0, 0),
        Priority(2, 2, U64(0xffff_0000_0000_0000), 0, 0, 2),
        Priority(3, 3, U64(0x0000_0000_0000_0000), 0, 0, 3),
    ];

    impl Default for Priority {
        fn default() -> Priority {
            PRIORITY[2]
        }
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct QueueInfo<'a> {
        pub(crate) state: GpuPointer<'a, super::RingState>,
        pub(crate) ring: GpuPointer<'a, &'a [u64]>,
        // Historical inherited name. E125 proves exact G15 QueueInfo +0x10 is
        // the selected AGXTimeStampQueue state FWVA, not the queue-wide
        // NotifierList. The live WorkQueue constructor is deliberately left
        // unchanged/fail-closed until that owner is integrated.
        pub(crate) notifier_list: GpuPointer<'a, event::NotifierList>,
        pub(crate) gpu_buf: GpuPointer<'a, &'a [u8]>,
        pub(crate) gpu_rptr1: AtomicU32,
        pub(crate) gpu_rptr2: AtomicU32,
        pub(crate) gpu_rptr3: AtomicU32,
        pub(crate) event_id: AtomicI32,
        pub(crate) priority: Priority,
        pub(crate) unk_4c: i32,
        pub(crate) uuid: u32,
        pub(crate) unk_54: i32,
        pub(crate) unk_58: U64,
        pub(crate) busy: AtomicU32,
        pub(crate) __pad: Pad<0x20>,
        #[ver(V >= V13_2 && G < G14X)]
        pub(crate) unk_84_0: u32,
        pub(crate) unk_84_state: AtomicU32,
        pub(crate) error_count: AtomicU32,
        pub(crate) unk_8c: u32,
        pub(crate) unk_90: u32,
        pub(crate) unk_94: u32,
        pub(crate) pending: AtomicU32,
        pub(crate) unk_9c: u32,
        pub(crate) gpu_context: GpuPointer<'a, super::GpuContextData>,
        #[ver(G != G15)]
        pub(crate) unk_a8: U64,
        // Exact G15 QueueInfo tail. Apple zeroes the whole 0x24c0 channel-state
        // block, writes the GpuContext pointer at +0xa4, then stores the byte
        // returned by AGXArmFirmware::getCDMBackoffTimeout() at +0xac. The
        // getter reads q4 +0x9bc, whose exact J615 bootstrap value is 4.
        #[ver(G == G15)]
        pub(crate) cdm_backoff_timeout_ac: u8,
        #[ver(G == G15)]
        pub(crate) pad_ad: Pad<0x03>,
        #[ver(V >= V13_2 && G < G14X && G != G15)]
        pub(crate) unk_b0: u32,
    }

    const _: [(); 0xb0] = [(); core::mem::size_of::<QueueInfoG15V14_7<'static>>()];
    const _: [(); 0xa4] = [(); core::mem::offset_of!(QueueInfoG15V14_7<'static>, gpu_context)];
    const _: [(); 0xac] = [(); core::mem::offset_of!(QueueInfoG15V14_7<'static>, cdm_backoff_timeout_ac)];
}

/// Exact 23J220 G15 `_AGFIChannelState` resource-stack geometry.
///
/// E116 proves firmware owns range-8 backing blocks of 0x8000 bytes and
/// hands AGXChannel one selected 0x24c0-byte slot from each block. Three
/// complete slots fit in one block; selection/reset ownership remains host
/// state and is deliberately not modeled by these constants.
pub(crate) const G15_CHANNEL_STATE_BYTES: usize = 0x24c0;
pub(crate) const G15_CHANNEL_STATE_BACKING_BYTES: usize = 0x8000;
pub(crate) const G15_CHANNEL_STATE_SLOTS_PER_BACKING: usize = 3;
pub(crate) const G15_CHANNEL_STATE_BACKING_SLACK_BYTES: usize =
    G15_CHANNEL_STATE_BACKING_BYTES
        - G15_CHANNEL_STATE_SLOTS_PER_BACKING * G15_CHANNEL_STATE_BYTES;

/// Exact in-slot pointer used by G15 QueueInfo `gpu_buf`. E121 proves
/// AGXChannel::init() stores selected-state GPUVA + 0xb0 at channel +0x88,
/// and resetChannelState() converts that value into QueueInfo +0x18.
/// This is an offset inside the selected channel-state slot, not a separate
/// G15 queue allocation.
pub(crate) const G15_CHANNEL_STATE_GPU_BUF_OFFSET: usize = 0xb0;

const _: [(); 0x8000] = [(); G15_CHANNEL_STATE_BACKING_BYTES];
const _: [(); 0x24c0] = [(); G15_CHANNEL_STATE_BYTES];
const _: [(); 3] = [(); G15_CHANNEL_STATE_SLOTS_PER_BACKING];
const _: [(); 0x11c0] = [(); G15_CHANNEL_STATE_BACKING_SLACK_BYTES];
const _: [(); 0xb0] = [(); G15_CHANNEL_STATE_GPU_BUF_OFFSET];
const _: [(); G15_CHANNEL_STATE_GPU_BUF_OFFSET] =
    [(); core::mem::size_of::<raw::QueueInfoG15V14_7<'static>>()];

/// Exact 23J220 normal-J615 cached/uncached firmware channel-memory stack
/// geometry closed by E122. Both stacks use the same element size, derived
/// from the exact firmware queue count 0x50, but they live in distinct normal
/// range-7 / special range-8 mapping classes. Allocation blocks are page
/// rounded to 0x8000 and contain three complete 0x2860-byte elements.
pub(crate) const G15_J615_FW_QUEUE_COUNT: usize = 0x50;
pub(crate) const G15_J615_CHANNEL_MEMORY_BYTES: usize =
    0x60 | ((G15_J615_FW_QUEUE_COUNT & 0x0fff_ffff) << 7);
pub(crate) const G15_J615_CHANNEL_MEMORY_BACKING_BYTES: usize = 0x8000;
pub(crate) const G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING: usize = 3;
pub(crate) const G15_J615_CHANNEL_MEMORY_BACKING_SLACK_BYTES: usize =
    G15_J615_CHANNEL_MEMORY_BACKING_BYTES
        - G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING * G15_J615_CHANNEL_MEMORY_BYTES;

const _: [(); 0x50] = [(); G15_J615_FW_QUEUE_COUNT];
const _: [(); 0x2860] = [(); G15_J615_CHANNEL_MEMORY_BYTES];
const _: [(); 0x8000] = [(); G15_J615_CHANNEL_MEMORY_BACKING_BYTES];
const _: [(); 3] = [(); G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING];
const _: [(); 0x6e0] = [(); G15_J615_CHANNEL_MEMORY_BACKING_SLACK_BYTES];

/// Exact 23J220 `_AGFITimeStampQueue` firmware resource-stack geometry.
/// E125 proves a 0x18-byte state in normal range 7; the generic stack rounds
/// two elements to one exact 0x4000 J615 host page, yielding 0x2aa complete
/// states and 0x10 bytes trailing slack.
pub(crate) const G15_TIMESTAMP_QUEUE_STATE_BYTES: usize = 0x18;
pub(crate) const G15_TIMESTAMP_QUEUE_BACKING_BYTES: usize = 0x4000;
pub(crate) const G15_TIMESTAMP_QUEUE_STATES_PER_BACKING: usize =
    G15_TIMESTAMP_QUEUE_BACKING_BYTES / G15_TIMESTAMP_QUEUE_STATE_BYTES;
pub(crate) const G15_TIMESTAMP_QUEUE_BACKING_SLACK_BYTES: usize =
    G15_TIMESTAMP_QUEUE_BACKING_BYTES
        - G15_TIMESTAMP_QUEUE_STATES_PER_BACKING * G15_TIMESTAMP_QUEUE_STATE_BYTES;

const _: [(); 0x18] = [(); G15_TIMESTAMP_QUEUE_STATE_BYTES];
const _: [(); 0x4000] = [(); G15_TIMESTAMP_QUEUE_BACKING_BYTES];
const _: [(); 0x2aa] = [(); G15_TIMESTAMP_QUEUE_STATES_PER_BACKING];
const _: [(); 0x10] = [(); G15_TIMESTAMP_QUEUE_BACKING_SLACK_BYTES];

/// Exact normal-J615 CL-channel constructor values closed by E119.
#[allow(dead_code)]
pub(crate) const G15_J615_FIRST_CL_EVCTL_INDEX: u32 = 0;
#[allow(dead_code)]
pub(crate) const G15_J615_CL_SECOND_CONSTRUCTOR_INTEGER: u32 = 0x50;
/// E124 closes AGXChannel::init()'s derived channel +0x54 value for the
/// normal J615 CL constructor: min(0x50, 0x80) << 4 = 0x500. Exact
/// resetChannelState() publishes this into selected uncached channel memory
/// +0x50 while zeroing four sibling header words.
pub(crate) const G15_J615_CL_UNCACHED_CHANNEL_VALUE_50: u32 =
    G15_J615_CL_SECOND_CONSTRUCTOR_INTEGER << 4;
const _: [(); 0x500] = [(); G15_J615_CL_UNCACHED_CHANNEL_VALUE_50 as usize];
pub(crate) const G15_J615_CL_PRIORITY_INTEGER_ARGUMENT: u32 = 2;
pub(crate) const G15_J615_CDM_BACKOFF_TIMEOUT: u8 = 4;

/// Exact six-field priority image written by
/// AGXArmFirmware::setChannelPriority() into selected `_AGFIChannelState`
/// QueueInfo +0x30..+0x4b. This is deliberately separate from the inherited
/// `raw::PRIORITY` table: E118 proves that table is not a direct G15 mapping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct G15ClChannelPriorityImage {
    pub(crate) class_30: u32,
    pub(crate) mask_38: u64,
    pub(crate) control_40: u32,
    pub(crate) integer_arg_44: u32,
    pub(crate) qos_value_48: u32,
}

/// Resolve the exact normal-J615 CL priority image from IOGPU-owned runtime
/// priority/QoS state. E119 proves the integer argument is always 2 here and
/// that effective priority is normally 1 (foreground branch) or 2 (alternate
/// branch). Other values are rejected instead of inheriting guessed tables.
pub(crate) const fn g15_j615_cl_priority_image(
    effective_priority: u32,
    queue_qos: u32,
) -> Option<G15ClChannelPriorityImage> {
    if queue_qos > 4 {
        return None;
    }

    match effective_priority {
        1 => {
            let (class_30, mask_38, qos_value_48) = match queue_qos {
                0 => (2, 0xffff_ffff_0000_0000, 0),
                1 => (2, 0xffff_0000_0000_0000, 1),
                2 => (2, 0xffff_0000_0000_0000, 2),
                3 => (2, 0xffff_0000_0000_0000, 3),
                4 => (3, 0x0000_0000_0000_0000, 4),
                _ => return None,
            };
            Some(G15ClChannelPriorityImage {
                class_30,
                mask_38,
                control_40: 0,
                integer_arg_44: G15_J615_CL_PRIORITY_INTEGER_ARGUMENT,
                qos_value_48,
            })
        }
        2 => Some(G15ClChannelPriorityImage {
            class_30: 3,
            mask_38: 0,
            control_40: 0,
            integer_arg_44: G15_J615_CL_PRIORITY_INTEGER_ARGUMENT,
            qos_value_48: 0,
        }),
        _ => None,
    }
}

trivial_gpustruct!(RingState);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Barrier {}

#[versions(AGX)]
default_zeroed!(Barrier::ver);

#[versions(AGX)]
impl GpuStruct for Barrier::ver {
    type Raw<'a> = raw::Barrier::ver;
}

#[versions(AGX)]
impl Command for Barrier::ver {}

pub(crate) struct GpuContextData {
    pub(crate) _buffer: Arc<dyn core::any::Any + Send + Sync>,
}
impl GpuStruct for GpuContextData {
    type Raw<'a> = raw::GpuContextData;
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct QueueInfo {
    pub(crate) state: GpuObject<RingState>,
    pub(crate) ring: GpuArray<u64>,
    pub(crate) gpu_buf: GpuArray<u8>,
    pub(crate) notifier_list: Arc<GpuObject<event::NotifierList>>,
    pub(crate) gpu_context: Arc<crate::workqueue::GpuContext>,
}

#[versions(AGX)]
impl GpuStruct for QueueInfo::ver {
    type Raw<'a> = raw::QueueInfo::ver<'a>;
}
