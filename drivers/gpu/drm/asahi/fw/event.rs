// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU events control structures & stamps

use super::types::*;
use crate::{
    default_zeroed,
    trivial_gpustruct, //
};
use core::sync::atomic::Ordering;

/// Exact 23J220 AGXCommandBuffer event-control geometry for G15.
///
/// The backing mapping class is intentionally not modeled yet. E088 proves only
/// the contiguous 36-state/0xc0-byte ownership and selection contract.
pub(crate) const G15_EVENT_CONTROL_STATE_COUNT: usize = 36;
pub(crate) const G15_EVENT_CONTROL_BLOCK_SIZE: usize = 0xc0;
pub(crate) const G15_EVENT_CONTROL_POOL_SIZE: usize =
    G15_EVENT_CONTROL_STATE_COUNT * G15_EVENT_CONTROL_BLOCK_SIZE;

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

    /// Opaque exact-size G15 command-buffer event-control state.
    ///
    /// E088 closes the 0xc0-byte stride and CPU/GPU paired ownership, but not
    /// the complete field semantics or mapping class. Keep the bytes opaque so
    /// no inherited Notifier layout can be accidentally imposed on G15.
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15EventControlBlock {
        opaque: Pad<G15_EVENT_CONTROL_BLOCK_SIZE>,
    }
    default_zeroed!(G15EventControlBlock);

    /// Exact contiguous 36-state command-buffer event-control pool layout.
    /// This deliberately does not implement `GpuStruct`: E091 models geometry
    /// only and cannot allocate or publish this pool before its mapping class is
    /// independently recovered.
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
