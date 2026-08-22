// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU tiled vertex buffer control firmware structures

use super::types::*;
use super::workqueue;
use crate::{
    default_zeroed,
    no_debug,
    trivial_gpustruct, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct BlockControl {
        pub(crate) total: AtomicU32,
        pub(crate) wptr: AtomicU32,
        pub(crate) unk: AtomicU32,
        pub(crate) pad: Pad<0x34>,
    }
    default_zeroed!(BlockControl);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Counter {
        pub(crate) count: AtomicU32,
        #[ver(G != G15)]
        __pad: Pad<0x3c>,
    }

    #[versions(AGX)]
    default_zeroed!(Counter::ver);

    /// G15 UMA Page Pool State. Apple allocates this as an exact 0x70-byte
    /// GPU-mapped object (AGXUMAFList +0x1c0) and passes its FWVA in the late
    /// Compute/TA/3D command tails. The deliberately unaligned U64 fields are
    /// part of the Apple ABI, so this structure must remain packed.
    #[derive(Clone, Copy)]
    #[repr(C, packed)]
    pub(crate) struct G15UMAPagePoolState {
        pub(crate) pool_id: U64,                    // +0x00: AGXUMAPool global ID
        pub(crate) descriptor_index: U32,           // +0x08: host initializes 0xffffffff; RTKit accepts 0..255
        pub(crate) config_flag_0c: U32,             // +0x0c: AGXUMAPool::init() final bool
        pub(crate) priority: U32,                   // +0x10: _AGFIUMAPoolPriorityType
        pub(crate) page_pool_list_fwva: U64,        // +0x14: Apple "UMA Page Pool List"
        pub(crate) page_pool_list_capacity: U32,    // +0x1c: backing allocation size / 8
        pub(crate) dynamic_20: U32,                 // +0x20: FW/table mirrored, semantics pending
        pub(crate) dynamic_24: U32,                 // +0x24: FW/table mirrored, semantics pending
        pub(crate) dynamic_28: U32,                 // +0x28: FW/table mirrored single-bit field
        pub(crate) page_count: U32,                 // +0x2c: current allocated bytes >> 12; 22-bit in table
        // +0x30 is a small RTKit/page-pool lifecycle state: host initializes
        // it to 0, interrupt-post recovery can write 1, and cleanup treats
        // value 2 specially. Exact value names remain intentionally unknown.
        pub(crate) lifecycle_state_30: U32,
        pub(crate) backup_page_list_fwva: U64,      // +0x34: Apple "UMA Backup Page List"
        pub(crate) unk_3c: Pad<0x08>,               // +0x3c..+0x43: not written by host helper
        pub(crate) backup_page_list_count: U32,     // +0x44: compact-list count, rounded to 8
        pub(crate) fw_uncached_state_fwva: U64,     // +0x48: Apple "UMA FW Uncached State"
        pub(crate) fw_uncached_state_mirror: U64,   // +0x50: cached qword compared by FW
        pub(crate) shared_pool_mode: U32,           // +0x58: zero for normal non-shared channels
        pub(crate) fw_state_5c: U32,                // +0x5c: FW mutates low byte while active
        pub(crate) zero_60: U64,                    // +0x60
        pub(crate) zero_68: U64,                    // +0x68
    }
    default_zeroed!(G15UMAPagePoolState);
    const _: [(); 0x70] = [(); core::mem::size_of::<G15UMAPagePoolState>()];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15UMAPagePoolState, descriptor_index)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15UMAPagePoolState, page_pool_list_fwva)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(G15UMAPagePoolState, page_pool_list_capacity)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(G15UMAPagePoolState, dynamic_20)];
    const _: [(); 0x30] = [(); core::mem::offset_of!(G15UMAPagePoolState, lifecycle_state_30)];
    const _: [(); 0x34] = [(); core::mem::offset_of!(G15UMAPagePoolState, backup_page_list_fwva)];
    const _: [(); 0x44] = [(); core::mem::offset_of!(G15UMAPagePoolState, backup_page_list_count)];
    const _: [(); 0x48] = [(); core::mem::offset_of!(G15UMAPagePoolState, fw_uncached_state_fwva)];
    const _: [(); 0x50] = [(); core::mem::offset_of!(G15UMAPagePoolState, fw_uncached_state_mirror)];
    const _: [(); 0x58] = [(); core::mem::offset_of!(G15UMAPagePoolState, shared_pool_mode)];

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct Stats {
        pub(crate) max_pages: AtomicU32,
        pub(crate) max_b: AtomicU32,
        pub(crate) overflow_count: AtomicU32,
        pub(crate) gpu_c: AtomicU32,
        pub(crate) __pad0: Pad<0x10>,
        pub(crate) reset: AtomicU32,
        pub(crate) __pad1: Pad<0x1c>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Info<'a> {
        pub(crate) gpu_counter: u32,
        pub(crate) unk_4: u32,
        pub(crate) last_id: i32,
        pub(crate) cur_id: i32,
        pub(crate) unk_10: u32,
        pub(crate) gpu_counter2: u32,
        pub(crate) unk_18: u32,

        #[ver(V < V13_0B4 || G >= G14X)]
        pub(crate) unk_1c: u32,

        pub(crate) page_list: GpuPointer<'a, &'a [u32]>,
        pub(crate) page_list_size: u32,
        pub(crate) page_count: AtomicU32,
        pub(crate) max_blocks: u32,
        pub(crate) block_count: AtomicU32,
        pub(crate) unk_38: u32,
        pub(crate) block_list: GpuPointer<'a, &'a [u32]>,
        pub(crate) block_ctl: GpuPointer<'a, super::BlockControl>,
        pub(crate) last_page: AtomicU32,

        // G15 keeps the legacy-compatible prefix through +0x48, then switches
        // to a compact 0x80-byte parameter-buffer state. Apple allocates the
        // external control word as a separate 4-byte object and stores its FW
        // address at +0x58.
        #[ver(G == G15)]
        pub(crate) g15_unk_4c: u32,
        #[ver(G == G15)]
        pub(crate) g15_unk_50: U64,
        #[ver(G == G15)]
        pub(crate) counter: GpuPointer<'a, super::Counter::ver>,
        #[ver(G == G15)]
        pub(crate) g15_unk_60: U64,
        #[ver(G == G15)]
        pub(crate) g15_unk_68: U64,
        #[ver(G == G15)]
        pub(crate) g15_unk_70: u32,
        #[ver(G == G15)]
        pub(crate) g15_unk_74: u32,
        #[ver(G == G15)]
        pub(crate) g15_unk_78: U64,

        #[ver(G != G15)]
        pub(crate) gpu_page_ptr1: u32,
        #[ver(G != G15)]
        pub(crate) gpu_page_ptr2: u32,
        #[ver(G != G15)]
        pub(crate) unk_58: u32,
        #[ver(G != G15)]
        pub(crate) block_size: u32,
        #[ver(G != G15)]
        pub(crate) unk_60: U64,
        #[ver(G != G15)]
        pub(crate) counter: GpuPointer<'a, super::Counter::ver>,
        #[ver(G != G15)]
        pub(crate) unk_70: u32,
        #[ver(G != G15)]
        pub(crate) unk_74: u32,
        #[ver(G != G15)]
        pub(crate) unk_78: u32,
        #[ver(G != G15)]
        pub(crate) unk_7c: u32,
        #[ver(G != G15)]
        pub(crate) unk_80: u32,
        #[ver(G != G15)]
        pub(crate) max_pages: u32,
        #[ver(G != G15)]
        pub(crate) max_pages_nomemless: u32,
        #[ver(G != G15)]
        pub(crate) unk_8c: u32,
        #[ver(G != G15)]
        pub(crate) unk_90: Array<0x30, u8>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Scene<'a> {
        #[ver(G >= G14X || G == G15)]
        pub(crate) control_word: GpuPointer<'a, &'a [u32]>,
        #[ver(G >= G14X || G == G15)]
        pub(crate) control_word2: GpuPointer<'a, &'a [u32]>,
        pub(crate) pass_page_count: AtomicU32,
        pub(crate) unk_4: u32,
        pub(crate) unk_8: U64,
        pub(crate) unk_10: U64,
        pub(crate) user_buffer: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_20: u32,
        // G15 has one extra dword here. This moves the following U64/pointers
        // to the exact firmware-observed +0x38/+0x40/+0x48 anchors.
        #[ver(G == G15)]
        pub(crate) g15_unk_34: u32,
        #[ver(V >= V13_3)]
        pub(crate) unk_28: U64,
        pub(crate) stats: GpuWeakPointer<super::Stats>,
        pub(crate) total_page_count: AtomicU32,
        #[ver(G < G14X && G != G15)]
        pub(crate) unk_30: U64, // pad
        #[ver(G < G14X && G != G15)]
        pub(crate) unk_38: U64, // pad
        // Apple backs each G15 scene with 0x80 bytes. RTKit actively
        // invalidates/uses the first 0x50; the remaining bytes stay opaque.
        #[ver(G == G15)]
        pub(crate) g15_tail: Pad<0x34>,
    }

    // Exact G15 parameter-buffer backing recovered from Apple host allocation
    // and RTKit command handling.
    const _: [(); 0x04] = [(); core::mem::size_of::<CounterG15V14_7>()];
    const _: [(); 0x80] = [(); core::mem::size_of::<InfoG15V14_7<'static>>()];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, page_list)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, page_count)];
    const _: [(); 0x30] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, block_count)];
    const _: [(); 0x38] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, block_list)];
    const _: [(); 0x40] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, block_ctl)];
    const _: [(); 0x48] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, last_page)];
    const _: [(); 0x58] = [(); core::mem::offset_of!(InfoG15V14_7<'static>, counter)];
    const _: [(); 0x80] = [(); core::mem::size_of::<SceneG15V14_7<'static>>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, control_word)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, control_word2)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, pass_page_count)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, user_buffer)];
    const _: [(); 0x34] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, g15_unk_34)];
    const _: [(); 0x38] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, unk_28)];
    const _: [(); 0x40] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, stats)];
    const _: [(); 0x48] = [(); core::mem::offset_of!(SceneG15V14_7<'static>, total_page_count)];

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct InitBuffer<'a> {
        pub(crate) tag: workqueue::CommandType,
        pub(crate) vm_slot: u32,
        pub(crate) buffer_slot: u32,
        pub(crate) unk_c: u32,
        pub(crate) block_count: u32,
        pub(crate) buffer: GpuPointer<'a, super::Info::ver>,
        pub(crate) stamp_value: EventValue,
    }

    // AGXTAChannelSKU::submitBuffer() constructs the G15 tag-6 record as
    // exactly 0x20 bytes with the same packed field geometry, including the
    // deliberately unaligned FW-visible buffer pointer at +0x14.
    const _: [(); 0x20] = [(); core::mem::size_of::<InitBufferG15V14_7<'static>>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, tag)];
    const _: [(); 0x04] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, vm_slot)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, buffer_slot)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, unk_c)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, block_count)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, buffer)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(InitBufferG15V14_7<'static>, stamp_value)];
}

trivial_gpustruct!(BlockControl);
#[versions(AGX)]
trivial_gpustruct!(Counter::ver);
trivial_gpustruct!(Stats);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Info {
    pub(crate) block_ctl: GpuObject<BlockControl>,
    pub(crate) counter: GpuObject<Counter::ver>,
    pub(crate) page_list: GpuArray<u32>,
    pub(crate) block_list: GpuArray<u32>,
}

#[versions(AGX)]
impl GpuStruct for Info::ver {
    type Raw<'a> = raw::Info::ver<'a>;
}

pub(crate) struct ClusterBuffers {
    pub(crate) tilemaps: GpuArray<u8>,
    pub(crate) meta: GpuArray<u8>,
}

#[versions(AGX)]
pub(crate) struct Scene {
    pub(crate) user_buffer: GpuArray<u8>,
    pub(crate) buffer: crate::buffer::Buffer::ver,
    pub(crate) tvb_heapmeta: GpuArray<u8>,
    pub(crate) tvb_tilemap: GpuArray<u8>,
    pub(crate) tpc: Arc<GpuArray<u8>>,
    pub(crate) clustering: Option<ClusterBuffers>,
    pub(crate) preempt_buf: GpuArray<u8>,
    #[ver(G >= G14X || G == G15)]
    pub(crate) control_word: GpuArray<u32>,
}

#[versions(AGX)]
no_debug!(Scene::ver);

#[versions(AGX)]
impl GpuStruct for Scene::ver {
    type Raw<'a> = raw::Scene::ver<'a>;
}

#[versions(AGX)]
pub(crate) struct InitBuffer {
    pub(crate) scene: Arc<crate::buffer::Scene::ver>,
}

#[versions(AGX)]
no_debug!(InitBuffer::ver);

#[versions(AGX)]
impl workqueue::Command for InitBuffer::ver {}

#[versions(AGX)]
impl GpuStruct for InitBuffer::ver {
    type Raw<'a> = raw::InitBuffer::ver<'a>;
}
