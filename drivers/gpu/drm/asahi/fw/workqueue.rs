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
