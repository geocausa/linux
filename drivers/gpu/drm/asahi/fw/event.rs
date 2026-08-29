// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU events control structures & stamps

use super::types::*;
use crate::{
    default_zeroed,
    trivial_gpustruct, //
};
use core::sync::atomic::Ordering;

/// Exact 23J220 AGXCommandBuffer stamp/event-control geometry for G15.
///
/// E092 closes two contiguous shared-data backings: 36 four-byte stamps in
/// normal eGartRange 7 and 36 0xc0-byte event-control states in eGartRange 8.
pub(crate) const G15_EVENT_CONTROL_STATE_COUNT: usize = 36;
pub(crate) const G15_EVENT_CONTROL_STAMP_SIZE: usize = core::mem::size_of::<Stamp>();
pub(crate) const G15_EVENT_CONTROL_STAMP_POOL_SIZE: usize =
    G15_EVENT_CONTROL_STATE_COUNT * G15_EVENT_CONTROL_STAMP_SIZE;
pub(crate) const G15_EVENT_CONTROL_BLOCK_SIZE: usize = 0xc0;
pub(crate) const G15_EVENT_CONTROL_POOL_SIZE: usize =
    G15_EVENT_CONTROL_STATE_COUNT * G15_EVENT_CONTROL_BLOCK_SIZE;

const _: [(); 0x90] = [(); G15_EVENT_CONTROL_STAMP_POOL_SIZE];
const _: [(); 0x1b00] = [(); G15_EVENT_CONTROL_POOL_SIZE];

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    pub(crate) struct LinkedListHead {
        pub(crate) prev: Option<GpuWeakPointer<LinkedListHead>>,
        pub(crate) next: Option<GpuWeakPointer<LinkedListHead>>,
    }

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct NotifierList {
        pub(crate) list_head: LinkedListHead,
        pub(crate) unkptr_10: U64,
    }
    default_zeroed!(NotifierList);

    /// Partial exact 23J220 G15 command-buffer event-control state.
    ///
    /// E092 closes the construction/rotation writes below while leaving all
    /// still-unnamed bytes as padding. In particular, `config_10` is named only
    /// by producer/offset until its J615 value and semantic are independently
    /// recovered. This must not be confused with the legacy Notifier layout.
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15EventControlBlock {
        pub(crate) stamp_fwva: U64,
        pub(crate) stamp_index_08: u32,
        zero_0c: u32,
        config_10: u32,
        zero_14: u32,
        zero_18: U64,
        __pad_20: Pad<0x88>,
        sentinel_a8: U64,
        __pad_b0: Pad<0x10>,
    }
    default_zeroed!(G15EventControlBlock);

    /// Exact contiguous 36-state command-buffer event-control pool layout.
    /// E092 proves the pool maps through shared eGartRange 8, but the compile-
    /// only E093 owner uses `GpuArray<G15EventControlBlock>` directly rather
    /// than imposing an unrelated inherited Notifier `GpuStruct`.
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15EventControlPool {
        blocks: Array<G15_EVENT_CONTROL_STATE_COUNT, G15EventControlBlock>,
    }
    default_zeroed!(G15EventControlPool);

    const _: [(); G15_EVENT_CONTROL_BLOCK_SIZE] =
        [(); core::mem::size_of::<G15EventControlBlock>()];
    const _: [(); G15_EVENT_CONTROL_POOL_SIZE] =
        [(); core::mem::size_of::<G15EventControlPool>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(G15EventControlBlock, stamp_fwva)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15EventControlBlock, stamp_index_08)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15EventControlBlock, zero_0c)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15EventControlBlock, config_10)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15EventControlBlock, zero_14)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(G15EventControlBlock, zero_18)];
    const _: [(); 0xa8] = [(); core::mem::offset_of!(G15EventControlBlock, sentinel_a8)];

    #[versions(AGX)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct NotifierState {
        unk_14: u32,
        unk_18: U64,
        unk_20: u32,
        vm_slot: u32,
        has_vtx: u32,
        pstamp_vtx: Array<4, U64>,
        has_frag: u32,
        pstamp_frag: Array<4, U64>,
        has_comp: u32,
        pstamp_comp: Array<4, U64>,
        #[ver(G >= G14 && V < V13_0B4)]
        unk_98_g14_0: Array<0x14, u8>,
        in_list: u32,
        list_head: LinkedListHead,
        #[ver(G >= G14 && V < V13_0B4)]
        unk_a8_g14_0: Pad<4>,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_buf: Array<0x8, u8>, // Init to all-ff
    }

    #[versions(AGX)]
    impl Default for NotifierState::ver {
        fn default() -> Self {
            #[allow(unused_mut)]
            // SAFETY: All bit patterns are valid for this type.
            let mut s: Self = unsafe { core::mem::zeroed() };
            #[ver(V >= V13_0B4)]
            s.unk_buf = Array::new([0xff; 0x8]);
            s
        }
    }

    #[derive(Debug)]
    #[repr(transparent)]
    pub(crate) struct Threshold(AtomicU64);
    default_zeroed!(Threshold);

    impl Threshold {
        pub(crate) fn increase(&self, amount: u32) {
            // We could use fetch_add, but the non-LSE atomic
            // sequence Rust produces confuses the hypervisor.
            let v = self.0.load(Ordering::Relaxed);
            self.0.store(v + (amount as u64), Ordering::Relaxed);
        }
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Notifier<'a> {
        pub(crate) threshold: GpuPointer<'a, super::Threshold>,
        pub(crate) generation: AtomicU32,
        pub(crate) cur_count: AtomicU32,
        pub(crate) unk_10: AtomicU32,
        pub(crate) state: NotifierState::ver,
    }
}

trivial_gpustruct!(Threshold);
trivial_gpustruct!(NotifierList);

/// Host-side selector for the exact G15 command-buffer event-control ring.
///
/// `AGXCommandBuffer::init()` initializes the selected state to zero and
/// `nextCommandBufferState()` increments before selecting modulo 36. This type
/// intentionally owns no GPU memory; it only pins that rotation contract.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub(crate) struct G15EventControlSelector {
    current: u32,
}

#[allow(dead_code)]
impl G15EventControlSelector {
    pub(crate) fn advance(&mut self) -> usize {
        self.current = (self.current + 1) % G15_EVENT_CONTROL_STATE_COUNT as u32;
        self.current as usize
    }

    pub(crate) fn current(&self) -> usize {
        self.current as usize
    }

    pub(crate) const fn state_offset(index: usize) -> Option<usize> {
        if index < G15_EVENT_CONTROL_STATE_COUNT {
            Some(index * G15_EVENT_CONTROL_BLOCK_SIZE)
        } else {
            None
        }
    }
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Notifier {
    pub(crate) threshold: GpuObject<Threshold>,
}

#[versions(AGX)]
impl GpuStruct for Notifier::ver {
    type Raw<'a> = raw::Notifier::ver<'a>;
}
