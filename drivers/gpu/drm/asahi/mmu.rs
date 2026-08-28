// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU UAT (MMU) management
//!
//! AGX GPUs use an MMU called the UAT, which is largely compatible with the ARM64 page table
//! format. This module manages the global MMU structures, including a shared handoff structure
//! that is used to coordinate VM management operations with the firmware, the TTBAT which points
//! to currently active GPU VM contexts, as well as the individual `Vm` operations to map and
//! unmap buffer objects into a single user or kernel address space.
//!
//! The actual page table management is in the `pt` module.

use core::fmt::Debug;
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::ops::Range;
use core::sync::atomic::{
    fence,
    AtomicU32,
    AtomicU64,
    AtomicU8,
    Ordering, //
};

use core::cmp;

use kernel::{
    addr::PhysicalAddr,
    bindings::drm_gpuvm_flags_DRM_GPUVM_IMMEDIATE_MODE,
    c_str,
    device,
    drm::{
        gem::shmem,
        gpuvm,
        mm, //
    },
    error::Result,
    io,
    new_mutex,
    page::Page,
    prelude::*,
    types::Owned,
    static_lock_class,
    sync::{
        aref::ARef,
        lock::{
            mutex::MutexBackend,
            Guard, //
        },
        Arc,
        Mutex, //
    },
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    }, //
};

use crate::debug::*;
use crate::module_parameters;
use crate::no_debug;
use crate::{
    alloc,
    driver,
    fw,
    gem,
    hw,
    mem,
    pgtable,
    slotalloc,
    util::RangeExt, //
};

// KernelMapping protection types
pub(crate) use crate::pgtable::Prot;
pub(crate) use pgtable::prot::*;
pub(crate) use pgtable::{
    UatPageTable,
    UAT_PGBIT,
    UAT_PGMSK,
    UAT_PGSZ, //
};

use pin_init;

const DEBUG_CLASS: DebugFlags = DebugFlags::Mmu;

/// PPL magic number for the handoff region
const PPL_MAGIC: u64 = 0x4b1d000000000002;

/// Number of supported context entries in the TTBAT
const UAT_NUM_CTX: usize = 64;
/// First context available for users
const UAT_USER_CTX_START: usize = 1;
/// Number of available user contexts
const UAT_USER_CTX: usize = UAT_NUM_CTX - UAT_USER_CTX_START;

/// Lower/user base VA
pub(crate) const IOVA_USER_BASE: u64 = UAT_PGSZ as u64;
/// Current userspace ABI aperture. Keep this independent from the hardware UAT IAS
/// until the G15 userspace VA contract is characterized.
const UAT_USER_IAS: u32 = 39;

/// Exact Apple G15 non-legacy GART geometry uses 16 KiB pages, two full
/// 11-bit lower indices, and six significant bits in the top index
/// (`0x3f << 36`). This gives a 42-bit translated input address space.
///
/// Keep this compile-only and distinct from UAT_USER_IAS: generation-7 runtime
/// matching and the userspace VA contract are still deliberately disabled.
const G15_HW_UAT_IAS: u32 = 42;

/// Exact `AGXGart::returnGartRange(u64)` classifier from the G15 Apple host
/// driver. These range IDs feed the Apple memory-mapping API and are part of
/// the G15 VA contract; they are not Linux allocator IDs.
const fn g15_apple_gart_range(addr: u64) -> u8 {
    if addr >> 36 != 0 {
        if addr >> 34 < 0x1b {
            return 1;
        }
        if addr >> 33 > 0x36 {
            if addr >> 32 < 0x6f {
                return 2;
            }
            if addr < 0x6fff_c00000 {
                return 3;
            }
            if addr >> 36 < 7 {
                return 4;
            }
            if addr >> 40 != 0 {
                if addr >> 40 < 3 {
                    return 5;
                }
                if addr > 0xffff_fc1f_ffdf_ffff {
                    if addr < 0xffff_fc20_0000_0000 {
                        return 6;
                    }
                    if addr < 0xffff_fc20_0c00_0000 {
                        return 7;
                    }
                    if addr < 0xffff_fc20_1000_0000 {
                        return 8;
                    }
                    if addr < 0xffff_fc20_1140_0000 {
                        return 9;
                    }
                    if addr < 0xffff_fc20_1180_0000 {
                        return 10;
                    }
                    if addr < 0xffff_fc20_1580_0000 {
                        return 11;
                    }
                    if addr < 0xffff_fc20_1980_0000 {
                        return 12;
                    }
                }
            }
        }
    }
    0
}

// Pin the exact non-legacy G15 top-level mask: bits 36..41 are significant.
const _: [(); 42] = [(); G15_HW_UAT_IAS as usize];
const _: [(); 0x3f] = [(); (0x3f0_0000_0000u64 >> 36) as usize];
// PM/GTP scene resources use Apple eGartRange 5.
const _: [(); 5] = [(); g15_apple_gart_range(0x100_0000_0000) as usize];
const _: [(); 5] = [(); g15_apple_gart_range(0x2ff_ffff_ffff) as usize];
const _: [(); 0] = [(); g15_apple_gart_range(0x300_0000_0000) as usize];
// PMPageMetricsBuffer uses Apple eGartRange 7.
pub(crate) const G15_GART_RANGE7: Range<u64> =
    0xffff_fc20_0000_0000..0xffff_fc20_0c00_0000;
const _: [(); 7] = [(); g15_apple_gart_range(G15_GART_RANGE7.start) as usize];
const _: [(); 7] = [(); g15_apple_gart_range(G15_GART_RANGE7.end - 1) as usize];
const _: [(); 8] = [(); g15_apple_gart_range(G15_GART_RANGE7.end) as usize];

/// Apple G15 eGartRange 8. Exact 23J220 AGXUMAFList uses this 64-MiB
/// bank-1 aperture for its 0x70-byte UMA Page-Pool State object.
pub(crate) const G15_GART_RANGE8: Range<u64> =
    G15_GART_RANGE7.end..0xffff_fc20_1000_0000;
const _: [(); 8] = [(); g15_apple_gart_range(G15_GART_RANGE8.start) as usize];
const _: [(); 8] = [(); g15_apple_gart_range(G15_GART_RANGE8.end - 1) as usize];
const _: [(); 9] = [(); g15_apple_gart_range(G15_GART_RANGE8.end) as usize];

/// Apple G15 eGartRange 5. Parameter Scene Allocations and GTP/TPC use this
/// per-client lower-address-space aperture. It is intentionally outside the
/// current 39-bit DRM userspace ABI while remaining inside the 42-bit G15 UAT.
pub(crate) const G15_GART_RANGE5: Range<u64> = 0x100_0000_0000..0x300_0000_0000;
const _: [(); 1] = [(); (G15_GART_RANGE5.start >= (1u64 << UAT_USER_IAS)) as usize];
const _: [(); 1] = [(); (G15_GART_RANGE5.end <= (1u64 << G15_HW_UAT_IAS)) as usize];

/// Linux-internal non-overlapping sub-arenas within Apple eGartRange 5. Apple
/// uses one range with per-mapping PTE attributes; DefaultAllocator fixes one
/// protection class per heap, so keep the exact G15 0x308 and 0x303 classes in
/// separate halves. This split is an implementation detail, not an Apple ABI.
pub(crate) const G15_GART_RANGE5_UNCACHED: Range<u64> =
    G15_GART_RANGE5.start..0x200_0000_0000;
pub(crate) const G15_GART_RANGE5_CACHED: Range<u64> =
    G15_GART_RANGE5_UNCACHED.end..G15_GART_RANGE5.end;
const _: [(); 1] = [(); (G15_GART_RANGE5_UNCACHED.end == G15_GART_RANGE5_CACHED.start) as usize];
const _: [(); 5] = [(); g15_apple_gart_range(G15_GART_RANGE5_UNCACHED.start) as usize];
const _: [(); 5] = [(); g15_apple_gart_range(G15_GART_RANGE5_CACHED.start) as usize];

/// G15's UnifiedAddressTranslator has two bank-local page-table state blocks.
/// Apple selects the bank with VA bit 42, then indexes the top-level table with
/// VA bits 36..41. This is distinct from the 42 translated bits within a bank.
const fn g15_uat_bank(addr: u64) -> usize {
    ((addr >> 42) & 1) as usize
}

const fn g15_uat_top_index(addr: u64) -> usize {
    ((addr >> 36) & 0x3f) as usize
}

/// Address presented to one bank-local 42-bit page-table walker after G15 has
/// selected the bank with VA bit 42. Canonical high addresses therefore keep
/// only their low 42 translated bits inside the selected bank.
const fn g15_uat_bank_iova(addr: u64) -> u64 {
    addr & ((1u64 << G15_HW_UAT_IAS) - 1)
}

// Apple AGXUnifiedAddressTranslator stores two 0x90-byte bank-local state
// blocks. Bank 0 begins at +0x28 and bank 1 at +0xb8. Normal client init copies
// the accelerator-owned bank-1 block into +0xb8, then allocateGart() allocates
// only bank 0 locally. getPageTablePhysicalBaseAddress(0/1) reads the roots
// from these two blocks, and setClientContextID() publishes them as adjacent
// GPTBAT qwords. Keep this compile-only until G15 TTB publication is enabled.
const G15_UAT_BANK_STATE_BYTES: usize = 0x90;
const G15_UAT_BANK0_STATE_OFFSET: usize = 0x28;
const G15_UAT_BANK1_STATE_OFFSET: usize =
    G15_UAT_BANK0_STATE_OFFSET + G15_UAT_BANK_STATE_BYTES;
const G15_GPTBAT_ROOT_MASK: u64 = 0xffff_ffff_ffff_c000;
const G15_GPTBAT_PHYS_ROOT_MASK: u64 = ((1u64 << G15_HW_UAT_IAS) - 1) & !0x3fff;
const G15_GPTBAT_BANK0_HIGH_MASK: u64 = 0xffff_fc00_0000_0000;

const fn g15_gptbat_bank0(root: u64, context_id: u8) -> u64 {
    (root & G15_GPTBAT_ROOT_MASK) | ((context_id as u64) << 48) | TTBR_VALID
}

const fn g15_gptbat_bank1(bank0_root: u64, bank1_root: u64, context_id: u8) -> u64 {
    (bank1_root & G15_GPTBAT_ROOT_MASK)
        | (bank0_root & G15_GPTBAT_BANK0_HIGH_MASK)
        | ((context_id as u64) << 48)
        | TTBR_VALID
}

/// setClientContextID() rejects either bank root when physical address bits
/// above the 42-bit G15 UAT output-address limit are set.
const fn g15_gptbat_roots_fit(bank0_root: u64, bank1_root: u64) -> bool {
    ((bank0_root | bank1_root) >> G15_HW_UAT_IAS) == 0
}

// Range 5 (PM scene / GTP) is bank 0; range 7 (PM page metrics) is bank 1.
const _: [(); 0] = [(); g15_uat_bank(G15_GART_RANGE5.start)];
const _: [(); 0x10] = [(); g15_uat_top_index(G15_GART_RANGE5.start)];
const _: [(); 1] = [(); g15_uat_bank(G15_GART_RANGE7.start)];
const _: [(); 0x2] = [(); g15_uat_top_index(G15_GART_RANGE7.start)];
const _: [(); 1] = [(); (g15_uat_bank_iova(G15_GART_RANGE7.start) == 0x20_0000_0000) as usize];
const _: [(); 0xb8] = [(); G15_UAT_BANK1_STATE_OFFSET];
const G15_GPTBAT_SAMPLE_BANK0_ROOT: u64 = 0x0000_0001_2345_4000;
const G15_GPTBAT_SAMPLE_BANK1_ROOT: u64 = 0x0000_0020_5678_8000;
const G15_GPTBAT_SAMPLE_CONTEXT: u8 = 0x3f;
const G15_GPTBAT_SAMPLE0: u64 =
    g15_gptbat_bank0(G15_GPTBAT_SAMPLE_BANK0_ROOT, G15_GPTBAT_SAMPLE_CONTEXT);
const G15_GPTBAT_SAMPLE1: u64 = g15_gptbat_bank1(
    G15_GPTBAT_SAMPLE_BANK0_ROOT,
    G15_GPTBAT_SAMPLE_BANK1_ROOT,
    G15_GPTBAT_SAMPLE_CONTEXT,
);
const _: [(); 1] = [(); ((G15_GPTBAT_SAMPLE0 & G15_GPTBAT_PHYS_ROOT_MASK)
    == G15_GPTBAT_SAMPLE_BANK0_ROOT) as usize];
const _: [(); 1] = [(); ((G15_GPTBAT_SAMPLE1 & G15_GPTBAT_PHYS_ROOT_MASK)
    == G15_GPTBAT_SAMPLE_BANK1_ROOT) as usize];
const _: [(); 0x3f] = [(); ((G15_GPTBAT_SAMPLE0 >> 48) & 0xff) as usize];
const _: [(); 0x3f] = [(); ((G15_GPTBAT_SAMPLE1 >> 48) & 0xff) as usize];
const _: [(); 1] = [(); g15_gptbat_roots_fit(
    G15_GPTBAT_SAMPLE_BANK0_ROOT,
    G15_GPTBAT_SAMPLE_BANK1_ROOT,
) as usize];
const _: [(); 0] = [(); g15_gptbat_roots_fit(1u64 << 42, G15_GPTBAT_SAMPLE_BANK1_ROOT) as usize];
/// Lower/user top VA.
pub(crate) const IOVA_USER_TOP: u64 = 1 << UAT_USER_IAS;
/// Lower/user VA range
pub(crate) const IOVA_USER_RANGE: Range<u64> = IOVA_USER_BASE..IOVA_USER_TOP;

/// Upper/kernel base VA
#[cfg(CONFIG_DEV_COREDUMP)]
const IOVA_TTBR1_BASE: u64 = 0xffffff8000000000;
/// Driver-managed kernel base VA
const IOVA_KERN_BASE: u64 = 0xffffffa000000000;
/// Driver-managed kernel top VA
const IOVA_KERN_TOP: u64 = 0xffffffb000000000;
/// Driver-managed kernel VA range
const IOVA_KERN_RANGE: Range<u64> = IOVA_KERN_BASE..IOVA_KERN_TOP;
/// Full kernel VA range
#[cfg(CONFIG_DEV_COREDUMP)]
const IOVA_KERN_FULL_RANGE: Range<u64> = IOVA_TTBR1_BASE..(!UAT_PGMSK as u64);

const TTBR_VALID: u64 = 0x1; // BIT(0)
const TTBR_ASID_SHIFT: usize = 48;

/// Address of a special dummy page?
//const IOVA_UNK_PAGE: u64 = 0x6f_ffff8000;
pub(crate) const IOVA_UNK_PAGE: u64 = IOVA_USER_TOP - 2 * UAT_PGSZ as u64;
/// User VA range excluding the unk page
pub(crate) const IOVA_USER_USABLE_RANGE: Range<u64> = IOVA_USER_BASE..IOVA_UNK_PAGE;

/// A pre-allocated memory region for UAT management
struct UatRegion {
    base: PhysicalAddr,
    map: io::mem::Mem,
}

/// SAFETY: It's safe to share UAT region records across threads.
unsafe impl Send for UatRegion {}
/// SAFETY: It's safe to share UAT region records across threads.
unsafe impl Sync for UatRegion {}

/// Handoff region flush info structure
#[repr(C)]
struct FlushInfo {
    state: AtomicU64,
    addr: AtomicU64,
    size: AtomicU64,
}

/// UAT Handoff region layout
#[repr(C)]
struct Handoff {
    magic_ap: AtomicU64,
    magic_fw: AtomicU64,

    lock_ap: AtomicU8,
    lock_fw: AtomicU8,
    // Implicit padding: 2 bytes
    turn: AtomicU32,
    cur_slot: AtomicU32,
    // Implicit padding: 4 bytes
    flush: [FlushInfo; UAT_NUM_CTX + 1],

    unk2: AtomicU8,
    // Implicit padding: 7 bytes
    unk3: AtomicU64,
}

const HANDOFF_SIZE: usize = size_of::<Handoff>();

/// One VM slot in the TTBAT
#[repr(C)]
struct SlotTTBS {
    ttb0: AtomicU64,
    ttb1: AtomicU64,
}

const SLOTS_SIZE: usize = UAT_NUM_CTX * size_of::<SlotTTBS>();

// We need at least page 0 (ttb0)
const PAGETABLES_SIZE: usize = UAT_PGSZ;

/// Inner data for a Vm instance. This is reference-counted by the outer Vm object.
struct VmInner {
    dev: driver::AsahiDevRef,
    is_kernel: bool,
    va_range: Range<u64>,
    page_table: UatPageTable,
    mm: mm::Allocator<(), KernelMappingInner>,
    uat_inner: Arc<UatInner>,
    binding: Arc<Mutex<VmBinding>>,
    id: u64,
}

/// Slot binding-related inner data for a Vm instance.
struct VmBinding {
    active_users: usize,
    binding: Option<slotalloc::Guard<SlotInner>>,
    bind_token: Option<slotalloc::SlotToken>,
    ttb: u64,
}

struct VmBoInner {
    sgt: Option<shmem::SGTable<gem::AsahiObject>>,
    sg_vec: Option<KVVec<(usize, Range<usize>)>>,
}

/// Data associated with a VM <=> BO pairing
#[pin_data]
struct VmBo {
    #[pin]
    inner: Mutex<VmBoInner>,
}

impl gpuvm::DriverGpuVmBo for VmBo {
    fn new() -> impl PinInit<Self> {
        pin_init!(VmBo {
            inner <- new_mutex!(VmBoInner {
                sgt: None,
                sg_vec: None,
            }, "VmBinding"),
        })
    }
}

#[derive(Default)]
struct StepContext {
    new_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    prev_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    next_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    vm_bo: Option<ARef<gpuvm::GpuVmBo<VmInner>>>,
    prot: Prot,
}

impl gpuvm::DriverGpuVm for VmInner {
    type Driver = driver::AsahiDriver;
    type GpuVmBo = VmBo;
    type StepContext = StepContext;

    fn step_map(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpMap<Self>,
        ctx: &mut Self::StepContext,
    ) -> Result {
        let mut iova = op.addr();
        let mut left = op.range() as usize;
        let mut offset = op.offset() as usize;

        let bo = ctx.vm_bo.as_ref().expect("step_map with no BO");

        let one_page = op.flags().contains(gpuvm::GpuVaFlags::REPEAT);

        let mut do_map = |mut addr: usize, mut len: usize, offset: &mut usize| -> Result<bool> {
            if left == 0 {
                return Ok(false);
            }

            if *offset > 0 {
                let skip = len.min(*offset);
                addr += skip;
                len -= skip;
                *offset -= skip;
            }
            if len == 0 {
                return Ok(true);
            }
            assert!(*offset == 0);

            if one_page {
                len = left;
            } else {
                len = len.min(left);
            }

            mod_dev_dbg!(
                self.dev,
                "MMU: map: {:#x}:{:#x} -> {:#x} [OP={}]\n",
                addr,
                len,
                iova,
                one_page
            );

            self.page_table.map_pages(
                iova..(iova + len as u64),
                addr as PhysicalAddr,
                ctx.prot,
                one_page,
            )?;

            left -= len;
            iova += len as u64;
            Ok(true)
        };

        let guard = bo.inner().inner.lock();
        if let Some(sg_vec) = guard.sg_vec.as_ref() {
            let start_idx = sg_vec
                .binary_search_by(|range| {
                    if range.0 > offset {
                        cmp::Ordering::Greater
                    } else if (range.0 + range.1.len()) <= offset {
                        cmp::Ordering::Less
                    } else {
                        cmp::Ordering::Equal
                    }
                })
                .expect("sg_vec does not contain offset???");

            offset -= sg_vec[start_idx].0 as usize;

            for cur in start_idx..sg_vec.len() {
                let addr = sg_vec[cur].1.start as usize;
                let len: usize = sg_vec[cur].1.len() as usize;
                if do_map(addr, len, &mut offset)? == false {
                    break;
                }
            }
        } else {
            for range in guard.sgt.as_ref().expect("step_map with no SGT").iter() {
                // TODO: proper DMA address/length handling
                let addr = range.dma_address() as usize;
                let len: usize = range.dma_len() as usize;
                if do_map(addr, len, &mut offset)? == false {
                    break;
                }
            }
        }

        let gpuva = ctx.new_va.take().expect("Multiple step_map calls");

        if op
            .map_and_link_va(
                self,
                gpuva,
                ctx.vm_bo.as_ref().expect("step_map with no BO"),
            )
            .is_err()
        {
            dev_err!(
                self.dev.as_ref(),
                "map_and_link_va failed: {:#x} [{:#x}] -> {:#x}\n",
                op.offset(),
                op.range(),
                op.addr()
            );
            return Err(EINVAL);
        }
        Ok(())
    }
    fn step_unmap(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpUnMap<Self>,
        _ctx: &mut Self::StepContext,
    ) -> Result {
        let va = op.va().expect("step_unmap: missing VA");

        mod_dev_dbg!(self.dev, "MMU: unmap: {:#x}:{:#x}\n", va.addr(), va.range());

        self.page_table
            .unmap_pages(va.addr()..(va.addr() + va.range()))?;

        if let Some(asid) = self.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, va.addr() as usize, va.range() as usize);
            mod_dev_dbg!(
                self.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                va.addr(),
                va.range(),
            );
            mem::sync();
        }

        if op.unmap_and_unlink_va_defer().is_none() {
            dev_err!(self.dev.as_ref(), "step_unmap: could not unlink gpuva");
        }
        Ok(())
    }
    fn step_remap(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpReMap<Self>,
        vm_bo: &gpuvm::GpuVmBo<Self>,
        ctx: &mut Self::StepContext,
    ) -> Result {
        let va = op.unmap().va().expect("No previous VA");
        let orig_addr = va.addr();
        let orig_range = va.range();

        // Only unmap the hole between prev/next, if they exist
        let unmap_start = if let Some(op) = op.prev_map() {
            op.addr() + op.range()
        } else {
            orig_addr
        };

        let unmap_end = if let Some(op) = op.next_map() {
            op.addr()
        } else {
            orig_addr + orig_range
        };

        mod_dev_dbg!(
            self.dev,
            "MMU: unmap for remap: {:#x}..{:#x} (from {:#x}:{:#x})\n",
            unmap_start,
            unmap_end,
            orig_addr,
            orig_range
        );

        let unmap_range = unmap_end - unmap_start;

        self.page_table.unmap_pages(unmap_start..unmap_end)?;

        if let Some(asid) = self.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, unmap_start as usize, unmap_range as usize);
            mod_dev_dbg!(
                self.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                unmap_start,
                unmap_range,
            );
            mem::sync();
        }

        if op.unmap().unmap_and_unlink_va_defer().is_none() {
            dev_err!(self.dev.as_ref(), "step_unmap: could not unlink gpuva");
        }

        if let Some(prev_op) = op.prev_map() {
            let prev_gpuva = ctx
                .prev_va
                .take()
                .expect("Multiple step_remap calls with prev_op");
            if prev_op.map_and_link_va(self, prev_gpuva, vm_bo).is_err() {
                dev_err!(self.dev.as_ref(), "step_remap: could not relink prev gpuva");
                return Err(EINVAL);
            }
        }

        if let Some(next_op) = op.next_map() {
            let next_gpuva = ctx
                .next_va
                .take()
                .expect("Multiple step_remap calls with next_op");
            if next_op.map_and_link_va(self, next_gpuva, vm_bo).is_err() {
                dev_err!(self.dev.as_ref(), "step_remap: could not relink next gpuva");
                return Err(EINVAL);
            }
        }

        Ok(())
    }
}

impl VmInner {
    /// Returns the slot index, if this VM is bound.
    fn slot(&self) -> Option<u32> {
        if self.is_kernel {
            // The GFX ASC does not care about the ASID. Pick an arbitrary one.
            // TODO: This needs to be a persistently reserved ASID once we integrate
            // with the ARM64 kernel ASID machinery to avoid overlap.
            Some(0)
        } else {
            // We don't check whether we lost the slot, which could cause unnecessary
            // invalidations against another Vm. However, this situation should be very
            // rare (e.g. a Vm lost its slot, which means 63 other Vms bound in the
            // interim, and then it gets killed / drops its mappings without doing any
            // final rendering). Anything doing active maps/unmaps is probably also
            // rendering and therefore likely bound.
            self.binding
                .lock()
                .bind_token
                .as_ref()
                .map(|token| token.last_slot() + UAT_USER_CTX_START as u32)
        }
    }

    /// Returns the translation table base for this Vm
    fn ttb(&self) -> u64 {
        self.page_table.ttb()
    }

    /// Map an `mm::Node` representing an mapping in VA space.
    fn map_node(&mut self, node: &mm::Node<(), KernelMappingInner>, prot: Prot) -> Result {
        let mut iova = node.start();
        let guard = node.bo.as_ref().ok_or(EINVAL)?.inner().inner.lock();
        let sgt = guard.sgt.as_ref().ok_or(EINVAL)?;
        let mut offset = node.offset;
        let mut left = node.mapped_size;

        for range in sgt.iter() {
            if left == 0 {
                break;
            }

            // TODO: proper DMA address/length handling
            let mut addr = range.dma_address() as usize;
            let mut len: usize = range.dma_len() as usize;

            if (offset | addr | len | iova as usize) & UAT_PGMSK != 0 {
                dev_err!(
                    self.dev.as_ref(),
                    "MMU: KernelMapping {:#x}:{:#x} -> {:#x} is not page-aligned\n",
                    addr,
                    len,
                    iova
                );
                return Err(EINVAL);
            }

            if offset > 0 {
                let skip = len.min(offset);
                addr += skip;
                len -= skip;
                offset -= skip;
            }

            len = len.min(left);

            if len == 0 {
                continue;
            }

            mod_dev_dbg!(
                self.dev,
                "MMU: map: {:#x}:{:#x} -> {:#x}\n",
                addr,
                len,
                iova
            );

            self.page_table.map_pages(
                iova..(iova + len as u64),
                addr as PhysicalAddr,
                prot,
                false,
            )?;

            iova += len as u64;
            left -= len;
        }
        Ok(())
    }
}

/// Shared reference to a virtual memory address space ([`Vm`]).
#[derive(Clone)]
pub(crate) struct Vm {
    id: u64,
    inner: ARef<gpuvm::GpuVm<VmInner>>,
    dummy_obj: ARef<gem::Object>,
    binding: Arc<Mutex<VmBinding>>,
}
no_debug!(Vm);

/// Slot data for a [`Vm`] slot.
///
/// G15 firmware pairs the context ID with an 8-bit generation. Apple advances
/// that byte only when an ID is assigned to a different GART; sticky reuse of
/// the same slot preserves it. Keep the generation with the slot so Linux's
/// existing sticky allocator has the same lifetime semantics.
pub(crate) struct SlotInner {
    generation: u8,
}

impl slotalloc::SlotItem for SlotInner {
    type Data = ();
}

/// Represents a single user of a binding of a [`Vm`] to a slot.
///
/// The number of users is counted, and the slot will be freed when it drops to 0.
#[derive(Debug)]
pub(crate) struct VmBind(Vm, u32, u8);

impl VmBind {
    /// Returns the slot that this `Vm` is bound to.
    pub(crate) fn slot(&self) -> u32 {
        self.1
    }

    /// Returns the generation paired with this G15 context ID.
    pub(crate) fn generation(&self) -> u8 {
        self.2
    }
}

impl Drop for VmBind {
    fn drop(&mut self) {
        let mut binding = self.0.binding.lock();

        assert_ne!(binding.active_users, 0);
        binding.active_users -= 1;
        mod_pr_debug!(
            "MMU: slot {} active users {}\n",
            self.1,
            binding.active_users
        );
        if binding.active_users == 0 {
            binding.binding = None;
        }
    }
}

impl Clone for VmBind {
    fn clone(&self) -> VmBind {
        let mut binding = self.0.binding.lock();

        binding.active_users += 1;
        mod_pr_debug!(
            "MMU: slot {} active users {}\n",
            self.1,
            binding.active_users
        );
        VmBind(self.0.clone(), self.1, self.2)
    }
}

/// Inner data required for an object mapping into a [`Vm`].
pub(crate) struct KernelMappingInner {
    // Drop order matters:
    // - Drop the GpuVmBo first, which resv locks its BO and drops a GpuVm reference
    // - Drop the GEM BO next, since BO free can take the resv lock itself
    // - Drop the owner GpuVm last, since that again can take resv locks when the refcount drops to 0
    bo: Option<ARef<gpuvm::GpuVmBo<VmInner>>>,
    _gem: Option<ARef<gem::Object>>,
    owner: ARef<gpuvm::GpuVm<VmInner>>,
    uat_inner: Arc<UatInner>,
    prot: Prot,
    offset: usize,
    mapped_size: usize,
}

/// An object mapping into a [`Vm`], which reserves the address range from use by other mappings.
pub(crate) struct KernelMapping(mm::Node<(), KernelMappingInner>);

impl KernelMapping {
    /// Returns the IOVA base of this mapping
    pub(crate) fn iova(&self) -> u64 {
        self.0.start()
    }

    /// Returns the size of this mapping in bytes
    pub(crate) fn size(&self) -> usize {
        self.0.mapped_size
    }

    /// Returns the IOVA base of this mapping
    pub(crate) fn iova_range(&self) -> Range<u64> {
        self.0.start()..(self.0.start() + self.0.mapped_size as u64)
    }

    /// Remap a cached mapping as uncached, then synchronously flush that range of VAs from the
    /// coprocessor cache. This is required to safely unmap cached/private mappings.
    fn remap_uncached_and_flush(&mut self) {
        let mut owner = self
            .0
            .owner
            .exec_lock(None, false)
            .expect("Failed to exec_lock in remap_uncached_and_flush");

        mod_dev_dbg!(
            owner.dev,
            "MMU: remap as uncached {:#x}:{:#x}\n",
            self.iova(),
            self.size()
        );

        // Remap in-place as uncached.
        // Do not try to unmap the guard page (-1)
        let prot = self.0.prot.as_uncached();
        if owner
            .page_table
            .reprot_pages(self.iova_range(), prot)
            .is_err()
        {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: remap {:#x}:{:#x} failed\n",
                self.iova(),
                self.size()
            );
        }
        fence(Ordering::SeqCst);

        // If we don't have (and have never had) a VM slot, just return
        let slot = match owner.slot() {
            None => return,
            Some(slot) => slot,
        };

        let flush_slot = if owner.is_kernel {
            // If this is a kernel mapping, always flush on index 64
            UAT_NUM_CTX as u32
        } else {
            // Otherwise, check if this slot is the active one, otherwise return
            // Also check that we actually own this slot
            let ttb = owner.ttb() | TTBR_VALID | (slot as u64) << TTBR_ASID_SHIFT;

            let uat_inner = self.0.uat_inner.lock();
            uat_inner.handoff().lock();
            let cur_slot = uat_inner.handoff().current_slot();
            let ttb_cur = uat_inner.ttbs()[slot as usize].ttb0.load(Ordering::Relaxed);
            uat_inner.handoff().unlock();
            if cur_slot == Some(slot) && ttb_cur == ttb {
                slot
            } else {
                return;
            }
        };

        // FIXME: There is a race here, though it'll probably never happen in practice.
        // In theory, it's possible for the ASC to finish using our slot, whatever command
        // it was processing to complete, the slot to be lost to another context, and the ASC
        // to begin using it again with a different page table, thus faulting when it gets a
        // flush request here. In practice, the chance of this happening is probably vanishingly
        // small, as all 62 other slots would have to be recycled or in use before that slot can
        // be reused, and the ASC using user contexts at all is very rare.

        // Still, the locking around UAT/Handoff/TTBs should probably be redesigned to better
        // model the interactions with the firmware and avoid these races.
        // Possibly TTB changes should be tied to slot locks:

        // Flush:
        //  - Can early check handoff here (no need to lock).
        //      If user slot and it doesn't match the active ASC slot,
        //      we can elide the flush as the ASC guarantees it flushes
        //      TLBs/caches when it switches context. We just need a
        //      barrier to ensure ordering.
        //  - Lock TTB slot
        //      - If user ctx:
        //          - Lock handoff AP-side
        //              - Lock handoff dekker
        //                  - Check TTB & handoff cur ctx
        //      - Perform flush if necessary
        //          - This implies taking the fwring lock
        //
        // TTB change:
        //  - lock TTB slot
        //      - lock handoff AP-side
        //          - lock handoff dekker
        //              change TTB

        // Lock this flush slot, and write the range to it
        let flush = self.0.uat_inner.lock_flush(flush_slot);
        let pages = self.size() >> UAT_PGBIT;
        flush.begin_flush(self.iova(), self.size() as u64);
        if pages >= 0x10000 {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: Flush too big ({:#x} pages))\n",
                pages
            );
        }

        let cmd = fw::channels::FwCtlMsg {
            addr: fw::types::U64(self.iova()),
            unk_8: 0,
            slot: flush_slot,
            page_count: pages as u16,
            unk_12: 2, // ?
        };

        // Tell the firmware to do a cache flush
        if let Err(e) = (*owner.dev).gpu.fwctl(cmd) {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: ASC cache flush {:#x}:{:#x} failed (err: {:?})\n",
                self.iova(),
                self.size(),
                e
            );
        }

        // Finish the flush
        flush.end_flush();

        // Slot is unlocked here
    }
}
no_debug!(KernelMapping);

impl Drop for KernelMapping {
    fn drop(&mut self) {
        // This is the main unmap function for UAT mappings.
        // The sequence of operations here is finicky, due to the interaction
        // between cached GFX ASC mappings and the page tables. These mappings
        // always have to be flushed from the cache before being unmapped.

        // For uncached mappings, just unmapping and flushing the TLB is sufficient.

        // For cached mappings, this is the required sequence:
        // 1. Remap it as uncached
        // 2. Flush the TLB range
        // 3. If kernel VA mapping OR user VA mapping and handoff.current_slot() == slot:
        //    a. Take a lock for this slot
        //    b. Write the flush range to the right context slot in handoff area
        //    c. Issue a cache invalidation request via FwCtl queue
        //    d. Poll for completion via queue
        //    e. Check for completion flag in the handoff area
        //    f. Drop the lock
        // 4. Unmap
        // 5. Flush the TLB range again

        if self.0.prot.is_cached_noncoherent() {
            mod_pr_debug!(
                "MMU: remap as uncached {:#x}:{:#x}\n",
                self.iova(),
                self.size()
            );
            self.remap_uncached_and_flush();
        }

        let mut owner = self
            .0
            .owner
            .exec_lock(None, false)
            .expect("exec_lock failed in KernelMapping::drop");
        mod_dev_dbg!(
            owner.dev,
            "MMU: unmap {:#x}:{:#x}\n",
            self.iova(),
            self.size()
        );

        if owner.page_table.unmap_pages(self.iova_range()).is_err() {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: unmap {:#x}:{:#x} failed\n",
                self.iova(),
                self.size()
            );
        }

        if let Some(asid) = owner.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, self.iova() as usize, self.size());
            mod_dev_dbg!(
                owner.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                self.iova(),
                self.size()
            );
            mem::sync();
        }
    }
}


/// Address-space bookkeeping for the accelerator-shared G15 UAT bank 1.
/// Canonical G15 range-7 VAs are retained in the allocator and reduced to the
/// bank-local low-42-bit indices by this generation-specific page-table view.
///
/// The bank-1 root and range-7 L2 table are Apple/firmware carveout pages. Linux
/// owns only the six L3 pages installed beneath the live-proven empty L2 slots.
/// Keeping that ownership split explicit avoids treating reserved firmware DRAM
/// as `struct page` memory while still giving normal map/unmap callers a stable
/// page-table backend for the complete 192-MiB range-7 aperture plus the
/// adjacent 64-MiB range-8 Page-Pool-State aperture.
struct G15SharedBank1PageTable {
    ttb: PhysicalAddr,
    shared_l2_phys: PhysicalAddr,
    oas_mask: u64,
    _root: io::mem::Mem,
    _shared_l2: io::mem::Mem,
    l3: Option<KVec<Owned<Page>>>,
    l3_desc: [u64; Self::BANK1_L2_ENTRIES],
}

impl G15SharedBank1PageTable {
    const RANGE7_TOP_INDEX: usize = 2;
    const RANGE7_L2_ENTRIES: usize = 6;
    const RANGE8_L2_ENTRIES: usize = 2;
    const BANK1_L2_ENTRIES: usize = Self::RANGE7_L2_ENTRIES + Self::RANGE8_L2_ENTRIES;
    const L3_ENTRIES: usize = UAT_PGSZ / core::mem::size_of::<u64>();
    const L2_SPAN: u64 = (UAT_PGSZ as u64) * (Self::L3_ENTRIES as u64);
    const TABLE_TYPE_BITS: u64 = 0x3;
    const RANGE7_LEAF_BITS: u64 = 0x00c0_0000_0000_0447;
    const RANGE7_FLIST_LEAF_BITS: u64 = 0x00c0_0000_0000_044b;
    const RANGE8_LEAF_BITS: u64 = 0x00c0_0000_0000_0443;

    fn new(dev: &driver::AsahiDevice, cfg: &'static hw::HwConfig, ttb: PhysicalAddr) -> Result<Self> {
        if cfg.uat_ias < G15_HW_UAT_IAS || ttb & UAT_PGMSK as u64 != 0 {
            return Err(EINVAL);
        }

        let root = unsafe {
            io::mem::Mem::try_new_phys(ttb, UAT_PGSZ, (io::mem::MemFlag::WB).into())?
        };
        let root_pte = unsafe {
            core::ptr::read_volatile(
                root.ptr()
                    .add(Self::RANGE7_TOP_INDEX * core::mem::size_of::<u64>())
                    .cast::<u64>(),
            )
        };
        let oas_mask = if cfg.uat_oas >= 64 {
            return Err(EINVAL);
        } else {
            (1u64 << cfg.uat_oas) - 1
        };
        let shared_l2_phys = root_pte & oas_mask & !(UAT_PGMSK as u64);
        let expected_l2_phys = ttb.checked_sub(UAT_PGSZ as u64).ok_or(EINVAL)?;

        if root_pte & Self::TABLE_TYPE_BITS != Self::TABLE_TYPE_BITS
            || shared_l2_phys != expected_l2_phys
        {
            dev_err!(
                dev.as_ref(),
                "MMU: G15 bank-1 root[2] mismatch: pte={:#x} child={:#x} expected={:#x}\n",
                root_pte,
                shared_l2_phys,
                expected_l2_phys
            );
            return Err(EIO);
        }

        let shared_l2 = unsafe {
            io::mem::Mem::try_new_phys(
                shared_l2_phys,
                UAT_PGSZ,
                (io::mem::MemFlag::WB).into(),
            )?
        };
        for idx in 0..Self::RANGE7_L2_ENTRIES {
            let pte = unsafe {
                core::ptr::read_volatile(
                    shared_l2
                        .ptr()
                        .add(idx * core::mem::size_of::<u64>())
                        .cast::<u64>(),
                )
            };
            if pte != 0 {
                dev_err!(
                    dev.as_ref(),
                    "MMU: G15 shared range-7 L2[{}] unexpectedly populated: {:#x}\n",
                    idx,
                    pte
                );
                return Err(EIO);
            }
        }

        // E074 read-only ownership preflight. Range 8 immediately follows the
        // six range-7 L2 slots in this same firmware-owned shared-L2 page. Do
        // not publish a Linux child descriptor yet; first require both parent
        // slots to be empty on the exact J615 machine.
        for rel in 0..Self::RANGE8_L2_ENTRIES {
            let idx = Self::RANGE7_L2_ENTRIES + rel;
            let pte = unsafe {
                core::ptr::read_volatile(
                    shared_l2
                        .ptr()
                        .add(idx * core::mem::size_of::<u64>())
                        .cast::<u64>(),
                )
            };
            if pte != 0 {
                dev_err!(
                    dev.as_ref(),
                    "MMU: G15 E074 range-8 L2[{}] unexpectedly populated: {:#x}\n",
                    idx,
                    pte
                );
                return Err(EIO);
            }
        }
        dev_info!(
            dev.as_ref(),
            "MMU: G15 E074 range-8 parent preflight PASS (shared-L2[6..8) empty, read-only)\n"
        );

        // Allocate every Linux-owned child before the first firmware-carveout
        // mutation. This means allocation failure can never leave a dangling
        // parent descriptor in the Apple-owned shared L2.
        let mut l3 = KVec::with_capacity(Self::BANK1_L2_ENTRIES, GFP_KERNEL)?;
        let mut l3_desc = [0u64; Self::BANK1_L2_ENTRIES];
        for idx in 0..Self::BANK1_L2_ENTRIES {
            let page = Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
            let phys = page.phys();
            if phys & UAT_PGMSK as u64 != 0 || phys & !oas_mask != 0 {
                dev_err!(dev.as_ref(), "MMU: G15 invalid bank-1 L3 page {} at {:#x}\n", idx, phys);
                return Err(EIO);
            }
            let zero = page.with_page_mapped(|ptr| {
                for q in 0..Self::L3_ENTRIES {
                    let v = unsafe {
                        core::ptr::read_volatile(
                            ptr.add(q * core::mem::size_of::<u64>()).cast::<u64>(),
                        )
                    };
                    if v != 0 {
                        return false;
                    }
                }
                true
            });
            if !zero {
                dev_err!(dev.as_ref(), "MMU: G15 bank-1 L3 page {} is not zeroed\n", idx);
                return Err(EIO);
            }
            l3_desc[idx] = phys | Self::TABLE_TYPE_BITS;
            l3.push(page, GFP_KERNEL)?;
        }

        // Recheck the live ownership boundary immediately before publication.
        for idx in 0..Self::BANK1_L2_ENTRIES {
            let slot = unsafe {
                shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            if unsafe { core::ptr::read_volatile(slot) } != 0 {
                return Err(EBUSY);
            }
        }
        for idx in 0..Self::BANK1_L2_ENTRIES {
            let slot = unsafe {
                shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            unsafe { core::ptr::write_volatile(slot, l3_desc[idx]) };
        }
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();

        let mut publish_ok = true;
        for idx in 0..Self::BANK1_L2_ENTRIES {
            let slot = unsafe {
                shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            let got = unsafe { core::ptr::read_volatile(slot) };
            if got != l3_desc[idx] {
                publish_ok = false;
                dev_err!(
                    dev.as_ref(),
                    "MMU: G15 shared bank-1 L2[{}] publish mismatch got={:#x} expected={:#x}\n",
                    idx,
                    got,
                    l3_desc[idx]
                );
            }
        }
        if !publish_ok {
            for idx in 0..Self::BANK1_L2_ENTRIES {
                let slot = unsafe {
                    shared_l2
                        .ptr()
                        .add(idx * core::mem::size_of::<u64>())
                        .cast::<u64>()
                };
                if unsafe { core::ptr::read_volatile(slot) } == l3_desc[idx] {
                    unsafe { core::ptr::write_volatile(slot, 0) };
                }
            }
            fence(Ordering::SeqCst);
            mem::tlbi_all();
            mem::sync();
            let detached = (0..Self::BANK1_L2_ENTRIES).all(|idx| {
                let slot = unsafe {
                    shared_l2
                        .ptr()
                        .add(idx * core::mem::size_of::<u64>())
                        .cast::<u64>()
                };
                let got = unsafe { core::ptr::read_volatile(slot) };
                got == 0
            });
            if !detached {
                core::mem::forget(l3);
            }
            return Err(EIO);
        }

        dev_info!(
            dev.as_ref(),
            "MMU: G15 shared bank-1 backend online root={:#x} shared-l2={:#x}, 8 Linux L3 tables published (range7=6, range8=2)\n",
            ttb,
            shared_l2_phys
        );

        Ok(Self {
            ttb,
            shared_l2_phys,
            oas_mask,
            _root: root,
            _shared_l2: shared_l2,
            l3: Some(l3),
            l3_desc,
        })
    }

    fn ttb(&self) -> PhysicalAddr {
        self.ttb
    }

    fn validate_range(&self, iova_range: &Range<u64>) -> Result<usize> {
        if iova_range.start > iova_range.end
            || (iova_range.start | iova_range.end) & UAT_PGMSK as u64 != 0
        {
            return Err(EINVAL);
        }
        let in_range7 = iova_range.start >= G15_GART_RANGE7.start
            && iova_range.end <= G15_GART_RANGE7.end;
        let in_range8 = iova_range.start >= G15_GART_RANGE8.start
            && iova_range.end <= G15_GART_RANGE8.end;
        if !in_range7 && !in_range8 {
            return Err(EINVAL);
        }
        Ok(((iova_range.end - iova_range.start) >> UAT_PGBIT) as usize)
    }

    fn validate_leaf_bits(iova_range: &Range<u64>, leaf_bits: u64) -> Result {
        if iova_range.start >= G15_GART_RANGE7.start && iova_range.end <= G15_GART_RANGE7.end {
            if leaf_bits == Self::RANGE7_LEAF_BITS || leaf_bits == Self::RANGE7_FLIST_LEAF_BITS {
                Ok(())
            } else {
                Err(EINVAL)
            }
        } else if iova_range.start >= G15_GART_RANGE8.start
            && iova_range.end <= G15_GART_RANGE8.end
        {
            if leaf_bits == Self::RANGE8_LEAF_BITS {
                Ok(())
            } else {
                Err(EINVAL)
            }
        } else {
            Err(EINVAL)
        }
    }

    fn leaf_location(iova: u64) -> Result<(usize, usize)> {
        if iova < G15_GART_RANGE7.start || iova >= G15_GART_RANGE8.end {
            return Err(EINVAL);
        }
        let off = iova - G15_GART_RANGE7.start;
        let l2 = (off / Self::L2_SPAN) as usize;
        let l3 = ((off >> UAT_PGBIT) & (Self::L3_ENTRIES as u64 - 1)) as usize;
        if l2 >= Self::BANK1_L2_ENTRIES {
            return Err(EINVAL);
        }
        Ok((l2, l3))
    }

    fn read_leaf(&self, iova: u64) -> Result<u64> {
        let (l2, l3_idx) = Self::leaf_location(iova)?;
        let leaves = self.l3.as_ref().ok_or(EIO)?;
        Ok(leaves[l2].with_page_mapped(|ptr| unsafe {
            core::ptr::read_volatile(
                ptr.add(l3_idx * core::mem::size_of::<u64>()).cast::<u64>(),
            )
        }))
    }

    fn write_leaf(&self, iova: u64, value: u64) -> Result {
        let (l2, l3_idx) = Self::leaf_location(iova)?;
        let leaves = self.l3.as_ref().ok_or(EIO)?;
        leaves[l2].with_page_mapped(|ptr| unsafe {
            core::ptr::write_volatile(
                ptr.add(l3_idx * core::mem::size_of::<u64>()).cast::<u64>(),
                value,
            )
        });
        Ok(())
    }

    fn validate_parents(&self) -> Result {
        for idx in 0..Self::BANK1_L2_ENTRIES {
            let slot = unsafe {
                self._shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            let got = unsafe { core::ptr::read_volatile(slot) };
            if got != self.l3_desc[idx] {
                pr_err!(
                    "MMU: G15 shared bank-1 parent {} changed got={:#x} expected={:#x}\n",
                    idx,
                    got,
                    self.l3_desc[idx]
                );
                return Err(EIO);
            }
        }
        Ok(())
    }

    fn alloc_pages(&mut self, iova_range: Range<u64>) -> Result {
        self.validate_range(&iova_range)?;
        self.validate_parents()
    }

    fn map_pages(
        &mut self,
        iova_range: Range<u64>,
        phys: PhysicalAddr,
        prot: Prot,
        one_page: bool,
    ) -> Result {
        let count = self.validate_range(&iova_range)?;
        self.validate_parents()?;
        if count == 0 {
            return Ok(());
        }
        if phys & UAT_PGMSK as u64 != 0 || phys & !self.oas_mask != 0 {
            return Err(EINVAL);
        }
        let leaf_bits = prot.as_pte() | Self::TABLE_TYPE_BITS;
        if Self::validate_leaf_bits(&iova_range, leaf_bits).is_err() {
            pr_err!(
                "MMU: G15 bank-1 rejected PTE protection bits {:#x} for {:#x}..{:#x}\n",
                leaf_bits,
                iova_range.start,
                iova_range.end
            );
            return Err(EINVAL);
        }

        // First validate the full operation so a collision or bad physical page
        // cannot leave a partially mapped range.
        for page in 0..count {
            let iova = iova_range.start + (page * UAT_PGSZ) as u64;
            let paddr = if one_page {
                phys
            } else {
                phys.checked_add((page * UAT_PGSZ) as u64).ok_or(EINVAL)?
            };
            if paddr & UAT_PGMSK as u64 != 0 || paddr & !self.oas_mask != 0 {
                return Err(EINVAL);
            }
            if self.read_leaf(iova)? != 0 {
                pr_err!("MMU: G15 bank-1 IOVA {:#x} already mapped\n", iova);
                return Err(EBUSY);
            }
        }

        for page in 0..count {
            let iova = iova_range.start + (page * UAT_PGSZ) as u64;
            let paddr = if one_page {
                phys
            } else {
                phys + (page * UAT_PGSZ) as u64
            };
            self.write_leaf(iova, paddr | leaf_bits)?;
        }
        Ok(())
    }

    fn unmap_pages(&mut self, iova_range: Range<u64>) -> Result {
        let count = self.validate_range(&iova_range)?;
        self.validate_parents()?;
        for page in 0..count {
            let iova = iova_range.start + (page * UAT_PGSZ) as u64;
            let pte = self.read_leaf(iova)?;
            if pte != 0 && pte & Self::TABLE_TYPE_BITS != Self::TABLE_TYPE_BITS {
                pr_err!("MMU: G15 bank-1 invalid leaf at {:#x}: {:#x}\n", iova, pte);
                return Err(EIO);
            }
        }
        for page in 0..count {
            let iova = iova_range.start + (page * UAT_PGSZ) as u64;
            self.write_leaf(iova, 0)?;
        }
        Ok(())
    }

    /// E029 exercises the exact persistent backend entry points while still
    /// below InitData/RTKit. Parent L3 descriptors remain installed until UAT
    /// teardown, unlike E028's test-local tables.
    fn e029_preflight_backend(&mut self, dev: &driver::AsahiDevice) -> Result {
        let data = Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
        let phys = data.phys();
        if phys & UAT_PGMSK as u64 != 0 || phys & !self.oas_mask != 0 {
            return Err(EIO);
        }
        let range = G15_GART_RANGE7.start..(G15_GART_RANGE7.start + UAT_PGSZ as u64);
        self.alloc_pages(range.clone())?;
        self.map_pages(range.clone(), phys, PROT_G15_RANGE7_FW, false)?;
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();

        let published = self.read_leaf(range.start)?;
        let expected = phys | Self::RANGE7_LEAF_BITS;
        let unmap_result = self.unmap_pages(range.clone());
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();
        let cleared = self.read_leaf(range.start)? == 0;

        if unmap_result.is_err() || !cleared {
            // Never release data backing while a possibly-live leaf can still
            // reference it. This branch is terminal for the one-shot test.
            core::mem::forget(data);
            return Err(EIO);
        }
        if published != expected {
            dev_err!(
                dev.as_ref(),
                "MMU: E029 backend leaf mismatch got={:#x} expected={:#x}\n",
                published,
                expected
            );
            return Err(EIO);
        }

        dev_info!(
            dev.as_ref(),
            "MMU: G15 E029 shared bank-1 backend PASS (real alloc/map/unmap, clean leaf)\n"
        );
        Ok(())
    }

    /// E075 validates the exact FList range-8 Page-Pool-State leaf class while
    /// still below InitData/RTKit. Only one temporary leaf is published; the
    /// normal shared-bank allocator remains range-7-only in this checkpoint.
    fn e075_preflight_range8_leaf(&mut self, dev: &driver::AsahiDevice) -> Result {
        let data = Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
        let phys = data.phys();
        if phys & UAT_PGMSK as u64 != 0 || phys & !self.oas_mask != 0 {
            return Err(EIO);
        }
        let range = G15_GART_RANGE8.start..(G15_GART_RANGE8.start + UAT_PGSZ as u64);
        self.alloc_pages(range.clone())?;
        self.map_pages(range.clone(), phys, PROT_G15_RANGE8_FW, false)?;
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();

        let published = self.read_leaf(range.start)?;
        let expected = phys | Self::RANGE8_LEAF_BITS;
        let unmap_result = self.unmap_pages(range.clone());
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();
        let cleared = self.read_leaf(range.start)? == 0;

        if unmap_result.is_err() || !cleared {
            core::mem::forget(data);
            return Err(EIO);
        }
        if published != expected {
            dev_err!(
                dev.as_ref(),
                "MMU: E075 range-8 leaf mismatch got={:#x} expected={:#x}\n",
                published,
                expected
            );
            return Err(EIO);
        }

        dev_info!(
            dev.as_ref(),
            "MMU: G15 E075 range-8 leaf PTE PASS (VA {:#x}, bits {:#018x}, clean leaf)\n",
            range.start,
            Self::RANGE8_LEAF_BITS
        );
        Ok(())
    }
}

impl Drop for G15SharedBank1PageTable {
    fn drop(&mut self) {
        let leaves = match self.l3.as_ref() {
            Some(leaves) => leaves,
            None => return,
        };

        // An occupied leaf means external lifetime ordering failed. Keep every
        // L3 page alive rather than letting Apple/firmware retain a pointer into
        // freed Linux memory.
        let all_leaves_zero = (0..Self::BANK1_L2_ENTRIES).all(|l2| {
            leaves[l2].with_page_mapped(|ptr| {
                for idx in 0..Self::L3_ENTRIES {
                    let pte = unsafe {
                        core::ptr::read_volatile(
                            ptr.add(idx * core::mem::size_of::<u64>()).cast::<u64>(),
                        )
                    };
                    if pte != 0 {
                        return false;
                    }
                }
                true
            })
        });
        if !all_leaves_zero || self.validate_parents().is_err() {
            pr_err!(
                "MMU: G15 shared bank-1 teardown unsafe; retaining Linux L3 backing until reboot\n"
            );
            if let Some(leaves) = self.l3.take() {
                core::mem::forget(leaves);
            }
            return;
        }

        for idx in 0..Self::BANK1_L2_ENTRIES {
            let slot = unsafe {
                self._shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            if unsafe { core::ptr::read_volatile(slot) } == self.l3_desc[idx] {
                unsafe { core::ptr::write_volatile(slot, 0) };
            }
        }
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();

        let detached = (0..Self::BANK1_L2_ENTRIES).all(|idx| {
            let slot = unsafe {
                self._shared_l2
                    .ptr()
                    .add(idx * core::mem::size_of::<u64>())
                    .cast::<u64>()
            };
            let got = unsafe { core::ptr::read_volatile(slot) };
            got == 0
        });
        if !detached {
            pr_err!(
                "MMU: G15 shared bank-1 parent detach failed; retaining Linux L3 backing until reboot\n"
            );
            if let Some(leaves) = self.l3.take() {
                core::mem::forget(leaves);
            }
            return;
        }

        // Taking and dropping only after exact parent-zero proof releases the
        // eight Linux pages with no remaining firmware-carveout references.
        core::mem::drop(self.l3.take());
        pr_info!(
            "MMU: G15 shared bank-1 backend teardown PASS (8 parents detached, leaves clean)\n"
        );
    }
}

struct G15SharedBank1State {
    page_table: G15SharedBank1PageTable,
    range7_mm: mm::Allocator<(), G15SharedBank1MappingInner>,
    range8_mm: mm::Allocator<(), G15SharedBank1MappingInner>,
}

impl G15SharedBank1State {
    fn new(
        dev: &driver::AsahiDevice,
        cfg: &'static hw::HwConfig,
        ttb: PhysicalAddr,
    ) -> Result<Self> {
        Ok(Self {
            page_table: G15SharedBank1PageTable::new(dev, cfg, ttb)?,
            range7_mm: mm::Allocator::new(G15_GART_RANGE7.start, G15_GART_RANGE7.range(), ())?,
            range8_mm: mm::Allocator::new(G15_GART_RANGE8.start, G15_GART_RANGE8.range(), ())?,
        })
    }

    fn ttb(&self) -> PhysicalAddr {
        self.page_table.ttb()
    }
}

/// Host producer for the native G15 q22 mapping-notification ring.
///
/// The state and ring themselves are bootstrap range-7 mappings created before
/// this producer is handed to any later range-7 allocation. Holding this object
/// strongly therefore keeps both q22 backings alive for every mapping that may
/// need to publish its eventual unmap record.
pub(crate) struct G15MappingNotifier {
    dev: driver::AsahiDevRef,
    backing: alloc::G15SharedGpuObject<fw::initdata::G15MappingRingBacking>,
}

/// Cloneable, debug-safe strong owner for the q22 producer. Kernel Mutex does
/// not implement Debug, while InitData does; keep formatting intentionally
/// opaque rather than exposing synchronization internals or raw pointers.
pub(crate) struct G15MappingNotifierHandle(Arc<Mutex<G15MappingNotifier>>);

impl G15MappingNotifierHandle {
    pub(crate) fn new(inner: Arc<Mutex<G15MappingNotifier>>) -> Self {
        Self(inner)
    }

    pub(crate) fn arc(&self) -> Arc<Mutex<G15MappingNotifier>> {
        self.0.clone()
    }
}

impl Clone for G15MappingNotifierHandle {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl core::ops::Deref for G15MappingNotifierHandle {
    type Target = Arc<Mutex<G15MappingNotifier>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::fmt::Debug for G15MappingNotifierHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("G15MappingNotifierHandle")
    }
}

impl G15MappingNotifier {
    const RING_LEN: u32 = 0x100;
    const PRESSURE_MASK: u32 = 0xc0;

    pub(crate) fn new(
        dev: &driver::AsahiDevice,
        backing: alloc::G15SharedGpuObject<fw::initdata::G15MappingRingBacking>,
    ) -> Self {
        Self { dev: dev.into(), backing }
    }

    pub(crate) fn state_gpu_va(&self) -> u64 {
        self.backing.gpu_va().get()
    }

    pub(crate) fn ring_gpu_va(&self) -> u64 {
        self.backing.gpu_va().get() + 0x40
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.backing.with(|raw, _inner| {
            raw.state.read_idx.load(Ordering::Relaxed) == 0
                && raw.state.write_idx.load(Ordering::Relaxed) == 0
        })
    }

    fn publish_pages(&mut self, base: u64, phys_pages: &[u64], mapping: bool) -> Result {
        // The DRM device reference pins the same device/manager lifetime as every
        // queue-owned range-7 mapping. No manager backlink is stored in q22, so
        // this does not introduce another ownership edge into InitData.
        let dev = self.dev.clone();
        self.backing.with_mut(|raw, _inner| -> Result {
            let mut write = raw.state.write_idx.load(Ordering::Relaxed);
            if write >= Self::RING_LEN {
                dev_err!(
                    dev.as_ref(),
                    "MMU: invalid G15 q22 write cursor {}\n",
                    write
                );
                return Err(EIO);
            }

            for (i, phys) in phys_pages.iter().enumerate() {
                let mut read = raw.state.read_idx.load(Ordering::Relaxed);
                if read >= Self::RING_LEN {
                    dev_err!(
                        dev.as_ref(),
                        "MMU: invalid G15 q22 read cursor {}\n",
                        read
                    );
                    return Err(EIO);
                }

                // Exact AGXArmFirmware::insertNewMappingEntry() threshold:
                // `bics wzr, 0xc0, write-read` falls through to the RTBuddy
                // async note only when both occupancy bits 0x80 and 0x40 are
                // set. Apple sends the note before attempting this insertion
                // and may therefore send it repeatedly while occupancy stays
                // in the 0xc0..0xff modulo-256 range.
                let occupancy = write.wrapping_sub(read) & 0xff;
                if occupancy & Self::PRESSURE_MASK == Self::PRESSURE_MASK {
                    dev.gpu.g15_mapping_pressure_kick()?;
                }

                let next = (write + 1) & 0xff;
                while next == read {
                    // Apple calls its 10-ms sleep wrapper and reloads read_idx
                    // until firmware consumes at least one q22 entry. Preserve
                    // that blocking producer behavior, but fail closed if the
                    // firmware has crashed instead of sleeping forever.
                    if dev.gpu.is_crashed() {
                        return Err(ENODEV);
                    }
                    fsleep(Delta::from_millis(10));
                    read = raw.state.read_idx.load(Ordering::Relaxed);
                    if read >= Self::RING_LEN {
                        dev_err!(
                            dev.as_ref(),
                            "MMU: invalid G15 q22 read cursor {} while waiting for ring space\n",
                            read
                        );
                        return Err(EIO);
                    }
                }

                raw.ring[write as usize] = fw::initdata::raw::G15CacheFlushEntry {
                    addr: fw::types::U64(base + (i * UAT_PGSZ) as u64),
                    phys_page_4k: fw::types::U32((phys >> 12).try_into()?),
                    secure_context_id: fw::types::U32(if mapping { 0 } else { u32::MAX }),
                    fw_page_count: 1,
                    mapping_flags: if mapping { fw::initdata::raw::G15_MAP_FLAG_MAP } else { 0 },
                    reserved_14: fw::types::U32(0),
                };
                // Exact Apple publication order: copy the complete 0x18-byte entry,
                // DMB ISH, then publish state+0x10. SeqCst fence is the kernel-Rust
                // ARM64 DMB class used elsewhere in this driver and is at least as strong.
                fence(Ordering::SeqCst);
                raw.state.write_idx.store(next, Ordering::Relaxed);
                write = next;
            }
            Ok(())
        })
    }

    /// CPU-only reversible check of the exact q22 encoder and publication
    /// cursors. This is called only by the manager preflight before RTKit or
    /// firmware construction, and restores both touched slots and cursors to
    /// their allocation-zero bootstrap state before returning.
    pub(crate) fn preflight_roundtrip(&mut self) -> Result {
        if !self.is_empty() {
            return Err(EBUSY);
        }

        let base = G15_GART_RANGE7.start + 0x0040_0000;
        let phys = 0x0000_0008_1234_0000u64;
        let pages = [phys];

        let result = (|| -> Result {
            self.publish_mapping(base, &pages)?;
            let map_ok = self.backing.with(|raw, _inner| {
                let e = raw.ring[0];
                raw.state.read_idx.load(Ordering::Relaxed) == 0
                    && raw.state.write_idx.load(Ordering::Relaxed) == 1
                    && e.addr.0 == base
                    && e.phys_page_4k.0 == (phys >> 12) as u32
                    && e.secure_context_id.0 == 0
                    && e.fw_page_count == 1
                    && e.mapping_flags == fw::initdata::raw::G15_MAP_FLAG_MAP
                    && e.reserved_14.0 == 0
            });
            if !map_ok {
                return Err(EIO);
            }

            self.publish_unmapping(base, &pages)?;
            let unmap_ok = self.backing.with(|raw, _inner| {
                let e = raw.ring[1];
                raw.state.read_idx.load(Ordering::Relaxed) == 0
                    && raw.state.write_idx.load(Ordering::Relaxed) == 2
                    && e.addr.0 == base
                    && e.phys_page_4k.0 == (phys >> 12) as u32
                    && e.secure_context_id.0 == u32::MAX
                    && e.fw_page_count == 1
                    && e.mapping_flags == 0
                    && e.reserved_14.0 == 0
            });
            if !unmap_ok {
                return Err(EIO);
            }
            Ok(())
        })();

        self.backing.with_mut(|raw, _inner| {
            raw.ring[0] = Default::default();
            raw.ring[1] = Default::default();
            raw.state.read_idx.store(0, Ordering::Relaxed);
            raw.state.write_idx.store(0, Ordering::Relaxed);
        });
        fence(Ordering::SeqCst);

        result?;
        if !self.is_empty() {
            return Err(EIO);
        }
        Ok(())
    }

    fn publish_mapping(&mut self, base: u64, phys_pages: &[u64]) -> Result {
        self.publish_pages(base, phys_pages, true)
    }

    fn publish_unmapping(&mut self, base: u64, phys_pages: &[u64]) -> Result {
        self.publish_pages(base, phys_pages, false)
    }
}

struct G15SharedBank1MappingInner {
    _gem: ARef<gem::Object>,
    mapped_size: usize,
}

/// One object mapping in the accelerator-shared G15 bank 1.
pub(crate) struct G15SharedBank1Mapping {
    node: Option<mm::Node<(), G15SharedBank1MappingInner>>,
    inner: Arc<UatInner>,
    notifier: Option<Arc<Mutex<G15MappingNotifier>>>,
    phys_pages: KVec<u64>,
}

impl G15SharedBank1Mapping {
    pub(crate) fn iova(&self) -> u64 {
        self.node.as_ref().unwrap().start()
    }

}

impl Drop for G15SharedBank1Mapping {
    fn drop(&mut self) {
        let node = self.node.take().unwrap();
        let iova = node.start();
        let size = node.mapped_size;
        if let Some(notifier) = self.notifier.as_ref() {
            if notifier.lock().publish_unmapping(iova, &self.phys_pages).is_err() {
                pr_err!(
                    "MMU: failed to publish G15 shared bank-1 unmapping {:#x}:{:#x}; preserving PTE and VA reservation\n",
                    iova,
                    size
                );
                // Apple publishes the q22 unmap record before AGXSecureGart::unmap().
                // If that publication cannot be completed, removing the PTE would
                // leave firmware with stale mapping state. Leak the reservation and
                // its GEM reference instead of making the VA reusable underneath
                // firmware. This is a terminal fail-closed path (RTKit loss/crash).
                core::mem::forget(node);
                return;
            }
        }
        {
            let mut shared = self.inner.lock();
            if let Some(bank1) = shared.g15_shared_bank1.as_mut() {
                if bank1
                    .page_table
                    .unmap_pages(iova..(iova + size as u64))
                    .is_err()
                {
                    pr_err!(
                        "MMU: failed to unmap G15 shared bank-1 range {:#x}:{:#x}\n",
                        iova,
                        size
                    );
                }
            }
        }
        // Bank 1 is shared by every G15 client context. A full invalidate is
        // conservative until the exact G15 bank-scoped invalidation command is
        // wired into the runtime path.
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();
        core::mem::drop(node);
    }
}

/// Disjoint Apple G15 bank-1 VA classes. Keeping the allocator arenas
/// separate prevents a range-8 Page-Pool-State allocation from consuming a
/// PM/range-7 VA (or vice versa) even though both share one page-table root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum G15SharedBank1Aperture {
    /// PM/q22 resource class, compact SecureGart option 0x007.
    Range7,
    /// AGXUMAFList FW-Uncached-State class, compact option 0x00b.
    Range7FList,
    /// AGXUMAFList Page-Pool-State class, compact option 0x003.
    Range8,
}

/// Cloneable mapping handle for the accelerator-shared G15 UAT bank 1.
#[derive(Clone)]
pub(crate) struct G15SharedBank1 {
    dev: driver::AsahiDevRef,
    inner: Arc<UatInner>,
}

impl G15SharedBank1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map(
        &self,
        aperture: G15SharedBank1Aperture,
        gem: &gem::Object,
        size: usize,
        alignment: u64,
        prot: Prot,
        guard: bool,
        notifier: Option<Arc<Mutex<G15MappingNotifier>>>,
    ) -> Result<G15SharedBank1Mapping> {
        let sgt = gem.owned_sg_table()?;
        if size & UAT_PGMSK != 0 {
            return Err(EINVAL);
        }
        let reserve_size = size + if guard { UAT_PGSZ } else { 0 };
        let mut phys_pages = KVec::with_capacity(size / UAT_PGSZ, GFP_KERNEL)?;
        let mut shared = self.inner.lock();
        let bank1 = shared.g15_shared_bank1.as_mut().ok_or(EINVAL)?;
        let (arena, start, end, expected_prot) = match aperture {
            G15SharedBank1Aperture::Range7 => (
                &mut bank1.range7_mm,
                G15_GART_RANGE7.start,
                G15_GART_RANGE7.end,
                PROT_G15_RANGE7_FW,
            ),
            G15SharedBank1Aperture::Range7FList => (
                &mut bank1.range7_mm,
                G15_GART_RANGE7.start,
                G15_GART_RANGE7.end,
                PROT_G15_RANGE7_FLIST_FW,
            ),
            G15SharedBank1Aperture::Range8 => (
                &mut bank1.range8_mm,
                G15_GART_RANGE8.start,
                G15_GART_RANGE8.end,
                PROT_G15_RANGE8_FW,
            ),
        };
        if prot.as_pte() != expected_prot.as_pte() {
            dev_err!(
                self.dev.as_ref(),
                "MMU: G15 shared bank-1 aperture/protection mismatch {:?} got={:#x} expected={:#x}\n",
                aperture,
                prot.as_pte(),
                expected_prot.as_pte()
            );
            return Err(EINVAL);
        }
        let node = arena.insert_node_in_range(
            G15SharedBank1MappingInner {
                _gem: gem.into(),
                mapped_size: size,
            },
            reserve_size as u64,
            alignment,
            0,
            start,
            end,
            mm::InsertMode::Best,
        )?;

        let base = node.start();
        bank1
            .page_table
            .alloc_pages(base..(base + size as u64))?;

        let mut iova = base;
        let mut left = size;
        for range in sgt.iter() {
            if left == 0 {
                break;
            }
            let addr = range.dma_address() as usize;
            let len = (range.dma_len() as usize).min(left);
            if (addr | len | iova as usize) & UAT_PGMSK != 0 {
                dev_err!(
                    self.dev.as_ref(),
                    "MMU: G15 shared bank-1 mapping is not page-aligned\n"
                );
                return Err(EINVAL);
            }
            for page_off in (0..len).step_by(UAT_PGSZ) {
                phys_pages
                    .push((addr + page_off) as u64, GFP_KERNEL)
                    .expect("G15 phys_pages push failed after reserve");
            }
            bank1.page_table.map_pages(
                iova..(iova + len as u64),
                addr as PhysicalAddr,
                prot,
                false,
            )?;
            iova += len as u64;
            left -= len;
        }
        if left != 0 {
            dev_err!(
                self.dev.as_ref(),
                "MMU: G15 shared bank-1 SG table is shorter than mapping\n"
            );
            return Err(EINVAL);
        }
        core::mem::drop(shared);

        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();

        if let Some(q22) = notifier.as_ref() {
            if let Err(err) = q22.lock().publish_mapping(base, &phys_pages) {
                // Native G15 ordering is secure-GART map first, q22 publication
                // second. If publication fails, undo the just-created PTEs before
                // allowing the VA reservation to be released/reused.
                let rollback = {
                    let mut shared = self.inner.lock();
                    match shared.g15_shared_bank1.as_mut() {
                        Some(bank1) => bank1
                            .page_table
                            .unmap_pages(base..(base + size as u64)),
                        None => Err(EINVAL),
                    }
                };
                fence(Ordering::SeqCst);
                mem::tlbi_all();
                mem::sync();
                if rollback.is_err() {
                    dev_err!(
                        self.dev.as_ref(),
                        "MMU: failed to roll back unpublished G15 shared bank-1 mapping {:#x}:{:#x}; leaking VA reservation\n",
                        base,
                        size
                    );
                    core::mem::forget(node);
                }
                return Err(err);
            }
        }

        Ok(G15SharedBank1Mapping {
            node: Some(node),
            inner: self.inner.clone(),
            notifier,
            phys_pages,
        })
    }
}

/// Shared UAT global data structures
struct UatShared {
    kernel_ttb1: u64,
    map_kernel_to_user: bool,
    // G15 client contexts import accelerator-owned bank 1 while allocating
    // bank 0 privately. This root is driver-owned but its leaf PTEs are global.
    // User-slot publication is latent behind this root existing; no current
    // G13/G14 configuration creates it and G15 runtime remains fail-closed.
    g15_shared_bank1: Option<G15SharedBank1State>,
    handoff_rgn: UatRegion,
    ttbs_rgn: UatRegion,
}

impl UatShared {
    /// Returns the handoff region area
    fn handoff(&self) -> &Handoff {
        // SAFETY: pointer is non-null per the type invariant
        unsafe { (self.handoff_rgn.map.ptr() as *mut Handoff).as_ref() }.unwrap()
    }

    /// Returns the TTBAT area
    fn ttbs(&self) -> &[SlotTTBS; UAT_NUM_CTX] {
        // SAFETY: pointer is non-null per the type invariant
        unsafe { (self.ttbs_rgn.map.ptr() as *mut [SlotTTBS; UAT_NUM_CTX]).as_ref() }.unwrap()
    }
}

// SAFETY: Nothing here is unsafe to send across threads.
unsafe impl Send for UatShared {}

/// Inner data for the top-level UAT instance.
#[pin_data]
struct UatInner {
    #[pin]
    shared: Mutex<UatShared>,
    #[pin]
    handoff_flush: [Mutex<HandoffFlush>; UAT_NUM_CTX + 1],
}

impl UatInner {
    /// Take the lock on the shared data and return the guard.
    fn lock(&self) -> Guard<'_, UatShared, MutexBackend> {
        self.shared.lock()
    }

    /// Take a lock on a handoff flush slot and return the guard.
    fn lock_flush(&self, slot: u32) -> Guard<'_, HandoffFlush, MutexBackend> {
        self.handoff_flush[slot as usize].lock()
    }
}

/// Top-level UAT manager object
pub(crate) struct Uat {
    dev: driver::AsahiDevRef,
    cfg: &'static hw::HwConfig,

    inner: Arc<UatInner>,
    slots: slotalloc::SlotAllocator<SlotInner>,

    kernel_vm: Vm,
    kernel_lower_vm: Vm,
}

impl Handoff {
    /// Lock the handoff region from firmware access
    fn lock(&self) {
        self.lock_ap.store(1, Ordering::Relaxed);
        fence(Ordering::SeqCst);

        while self.lock_fw.load(Ordering::Relaxed) != 0 {
            if self.turn.load(Ordering::Relaxed) != 0 {
                self.lock_ap.store(0, Ordering::Relaxed);
                while self.turn.load(Ordering::Relaxed) != 0 {}
                self.lock_ap.store(1, Ordering::Relaxed);
                fence(Ordering::SeqCst);
            }
        }
        fence(Ordering::Acquire);
    }

    /// Unlock the handoff region, allowing firmware access
    fn unlock(&self) {
        self.turn.store(1, Ordering::Relaxed);
        self.lock_ap.store(0, Ordering::Release);
    }

    /// Returns the current Vm slot mapped by the firmware for lower/unprivileged access, if any.
    fn current_slot(&self) -> Option<u32> {
        let slot = self.cur_slot.load(Ordering::Relaxed);
        if slot == 0 || slot == u32::MAX {
            None
        } else {
            Some(slot)
        }
    }

    /// Initialize the handoff region
    fn init(&self, g15: bool) -> Result {
        self.magic_ap.store(PPL_MAGIC, Ordering::Relaxed);
        // Apple G15 AGXUnifiedAddressTranslator::initHandoff() writes
        // 0xffffffff at +0x18 for the no-current-slot sentinel. Older
        // generations use the existing zero sentinel; current_slot() accepts
        // both representations as None.
        self.cur_slot
            .store(if g15 { u32::MAX } else { 0 }, Ordering::Relaxed);
        self.unk3.store(0, Ordering::Relaxed);
        fence(Ordering::SeqCst);

        let start = Instant::<Monotonic>::now();
        const TIMEOUT: Delta = Delta::from_millis(1000);

        self.lock();
        while start.elapsed() < TIMEOUT {
            if self.magic_fw.load(Ordering::Relaxed) == PPL_MAGIC {
                break;
            } else {
                self.unlock();
                fsleep(Delta::from_millis(10));
                self.lock();
            }
        }

        if self.magic_fw.load(Ordering::Relaxed) != PPL_MAGIC {
            self.unlock();
            pr_err!("Handoff: Failed to initialize (firmware not running?)\n");
            return Err(EIO);
        }

        self.unlock();

        for i in 0..=UAT_NUM_CTX {
            self.flush[i].state.store(0, Ordering::Relaxed);
            self.flush[i].addr.store(0, Ordering::Relaxed);
            self.flush[i].size.store(0, Ordering::Relaxed);
        }
        fence(Ordering::SeqCst);
        Ok(())
    }
}

/// Represents a single flush info slot in the handoff region.
///
/// # Invariants
/// The pointer is valid and there is no aliasing HandoffFlush instance.
struct HandoffFlush(*const FlushInfo);

// SAFETY: These pointers are safe to send across threads.
unsafe impl Send for HandoffFlush {}

impl HandoffFlush {
    /// Set up a flush operation for the coprocessor
    fn begin_flush(&self, start: u64, size: u64) {
        // SAFETY: Per the type invariant, this is safe
        let flush = unsafe { self.0.as_ref().unwrap() };

        let state = flush.state.load(Ordering::Relaxed);
        if state != 0 {
            pr_err!("Handoff: expected flush state 0, got {}\n", state);
        }
        flush.addr.store(start, Ordering::Relaxed);
        flush.size.store(size, Ordering::Relaxed);
        flush.state.store(1, Ordering::Relaxed);
    }

    /// Complete a flush operation for the coprocessor
    fn end_flush(&self) {
        // SAFETY: Per the type invariant, this is safe
        let flush = unsafe { self.0.as_ref().unwrap() };
        let state = flush.state.load(Ordering::Relaxed);
        if state != 2 {
            pr_err!("Handoff: expected flush state 2, got {}\n", state);
        }
        flush.state.store(0, Ordering::Relaxed);
    }
}

impl Vm {
    /// Create a new virtual memory address space
    fn new(
        dev: &driver::AsahiDevice,
        uat_inner: Arc<UatInner>,
        kernel_range: Range<u64>,
        cfg: &'static hw::HwConfig,
        ttb: Option<PhysicalAddr>,
        id: u64,
    ) -> Result<Vm> {
        let dummy_obj = gem::new_kernel_object(dev, UAT_PGSZ)?;
        let is_kernel = ttb.is_some();

        let page_table = if let Some(ttb) = ttb {
            UatPageTable::new_with_ttb(ttb, IOVA_KERN_RANGE, cfg.uat_ias, cfg.uat_oas)?
        } else {
            UatPageTable::new(cfg.uat_ias, cfg.uat_oas)?
        };

        let (va_range, gpuvm_range) = if is_kernel {
            (IOVA_KERN_RANGE, kernel_range.clone())
        } else {
            // Keep DRM GPUVM constrained to the stable 39-bit userspace ABI,
            // but let driver-owned KernelMappings use the full hardware TTBR0
            // input range. This is behavior-identical on current G13/G14
            // configs (uat_ias=39) and permits G15's hidden range-5 mappings
            // once a generation-7 HwConfig (uat_ias=42) exists.
            (
                IOVA_USER_BASE..(1u64 << cfg.uat_ias),
                IOVA_USER_USABLE_RANGE,
            )
        };

        let mm = mm::Allocator::new(va_range.start, va_range.range(), ())?;

        let binding = Arc::pin_init(
            new_mutex!(
                VmBinding {
                    binding: None,
                    bind_token: None,
                    active_users: 0,
                    ttb: page_table.ttb(),
                },
                "VmBinding",
            ),
            GFP_KERNEL,
        )?;

        let binding_clone = binding.clone();
        Ok(Vm {
            id,
            dummy_obj: dummy_obj.gem.clone(),
            inner: gpuvm::GpuVm::new(
                c_str!("Asahi::GpuVm"),
                // TODO: should we using DRM_GPUVM_RESV_PROTECTED as well?
                drm_gpuvm_flags_DRM_GPUVM_IMMEDIATE_MODE,
                dev,
                dummy_obj.gem.clone(),
                gpuvm_range,
                kernel_range,
                init!(VmInner {
                    dev: dev.into(),
                    va_range,
                    is_kernel,
                    page_table,
                    mm,
                    uat_inner,
                    binding: binding_clone,
                    id,
                }),
            )?,
            binding,
        })
    }

    /// Get the translation table base for this Vm
    fn ttb(&self) -> u64 {
        self.binding.lock().ttb
    }

    /// Map a GEM object (using its `SGTable`) into this Vm at a free address in a given range.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map_in_range(
        &self,
        gem: &gem::Object,
        object_range: Range<usize>,
        alignment: u64,
        range: Range<u64>,
        prot: Prot,
        guard: bool,
    ) -> Result<KernelMapping> {
        let size = object_range.range();
        let sgt = gem.owned_sg_table()?;
        let mut inner = self.inner.exec_lock(Some(gem), false)?;
        let vm_bo = self.inner.obtain_bo(gem)?;

        let mut vm_bo_guard = vm_bo.inner().inner.lock();
        if vm_bo_guard.sgt.is_none() {
            vm_bo_guard.sgt.replace(sgt);
        }
        core::mem::drop(vm_bo_guard);

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.insert_node_in_range(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: Some(vm_bo),
                _gem: Some(gem.into()),
                offset: object_range.start,
                mapped_size: size,
            },
            (size + if guard { UAT_PGSZ } else { 0 }) as u64, // Add guard page
            alignment,
            0,
            range.start,
            range.end,
            mm::InsertMode::Best,
        )?;

        let ret = inner.map_node(&node, prot);
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Map a GEM object into this Vm at a specific address.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map_at(
        &self,
        addr: u64,
        size: usize,
        gem: ARef<gem::Object>,
        prot: Prot,
        guard: bool,
    ) -> Result<KernelMapping> {
        let sgt = gem.owned_sg_table()?;
        let mut inner = self.inner.exec_lock(Some(&gem), false)?;

        let vm_bo = self.inner.obtain_bo(&gem)?;

        let mut vm_bo_guard = vm_bo.inner().inner.lock();
        if vm_bo_guard.sgt.is_none() {
            vm_bo_guard.sgt.replace(sgt);
        }
        core::mem::drop(vm_bo_guard);

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.reserve_node(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: Some(vm_bo),
                _gem: Some(gem.clone()),
                offset: 0,
                mapped_size: size,
            },
            addr,
            (size + if guard { UAT_PGSZ } else { 0 }) as u64, // Add guard page
            0,
        )?;

        let ret = inner.map_node(&node, prot);
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Map a range of a GEM object into this Vm using GPUVM.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_object(
        &self,
        gem: &gem::Object,
        addr: u64,
        size: u64,
        offset: u64,
        prot: Prot,
        single_page: bool,
    ) -> Result {
        // Mapping needs a complete context
        let mut ctx = StepContext {
            new_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            prev_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            next_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            prot,
            ..Default::default()
        };

        let vm_bo = self.inner.obtain_bo(gem)?;
        {
            let mut vm_bo_guard = vm_bo.inner().inner.lock();
            if vm_bo_guard.sgt.is_none() {
                let sgt = gem.owned_sg_table()?;

                if vm_bo_guard.sg_vec.is_none() {
                    let mut sg_vec = KVVec::new();
                    let mut offset = 0;
                    for range in sgt.iter() {
                        let addr = range.dma_address() as usize;
                        let len = range.dma_len() as usize;
                        sg_vec.push((offset, addr..(addr + len)), GFP_KERNEL)?;
                        offset += len;
                    }
                    vm_bo_guard.sg_vec.replace(sg_vec);
                }
                vm_bo_guard.sgt.replace(sgt);
            }
            core::mem::drop(vm_bo_guard);
        }

        let mut inner = self.inner.exec_lock(Some(gem), true)?;

        // Preallocate the page tables, to fail early if we ENOMEM
        inner.page_table.alloc_pages(addr..(addr + size))?;

        ctx.vm_bo = Some(vm_bo);

        if (addr | size | offset) & (UAT_PGMSK as u64) != 0 {
            dev_err!(
                inner.dev.as_ref(),
                "MMU: Map step {:#x} [{:#x}] -> {:#x} is not page-aligned\n",
                offset,
                size,
                addr
            );
            return Err(EINVAL);
        }

        let (flags, gem_range) = if single_page {
            (gpuvm::GpuVaFlags::REPEAT, UAT_PGSZ as u32)
        } else {
            (gpuvm::GpuVaFlags::NONE, 0u32)
        };

        mod_dev_dbg!(
            inner.dev,
            "MMU: sm_map: {:#x} [{:#x}] -> {:#x}\n",
            offset,
            size,
            addr
        );
        inner.sm_map(&mut ctx, addr, size, offset, gem_range, flags)
    }

    /// Add a direct MMIO mapping to this Vm at a free address.
    pub(crate) fn map_io(
        &self,
        iova: u64,
        phys: usize,
        size: usize,
        prot: Prot,
    ) -> Result<KernelMapping> {
        let mut inner = self.inner.exec_lock(None, false)?;

        if (iova as usize | phys | size) & UAT_PGMSK != 0 {
            dev_err!(
                inner.dev.as_ref(),
                "MMU: KernelMapping {:#x}:{:#x} -> {:#x} is not page-aligned\n",
                phys,
                size,
                iova
            );
            return Err(EINVAL);
        }

        dev_info!(
            inner.dev.as_ref(),
            "MMU: IO map: {:#x}:{:#x} -> {:#x}\n",
            phys,
            size,
            iova
        );

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.reserve_node(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: None,
                _gem: None,
                offset: 0,
                mapped_size: size,
            },
            iova,
            size as u64,
            0,
        )?;

        let ret = inner.page_table.map_pages(
            iova..(iova + size as u64),
            phys as PhysicalAddr,
            prot,
            false,
        );
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Unmap everything in an address range.
    pub(crate) fn unmap_range(&self, iova: u64, size: u64) -> Result {
        // Unmapping a range can only do a single split, so just preallocate
        // the prev and next GpuVas
        let mut ctx = StepContext {
            prev_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            next_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            ..Default::default()
        };

        let mut inner = self.inner.exec_lock(None, false)?;

        mod_dev_dbg!(inner.dev, "MMU: sm_unmap: {:#x}:{:#x}\n", iova, size);
        inner.sm_unmap(&mut ctx, iova, size)
    }

    /// Drop mappings for a given bo.
    pub(crate) fn drop_mappings(&self, gem: &gem::Object) -> Result {
        // Removing whole mappings only does unmaps, so no preallocated VAs
        let mut ctx = Default::default();

        let inner = self.inner.exec_lock(Some(gem), false)?;

        if let Some(bo) = self.inner.find_bo(gem) {
            mod_dev_dbg!(inner.dev, "MMU: bo_unmap\n");
            self.inner.bo_unmap(&mut ctx, &bo)?;
            mod_dev_dbg!(inner.dev, "MMU: bo_unmap done\n");
            // We need to drop the exec_lock first, then the GpuVmBo since that will take the lock itself.
            core::mem::drop(inner);
            core::mem::drop(bo);
        }

        Ok(())
    }

    /// Returns the dummy GEM object used to hold the shared DMA reservation locks
    pub(crate) fn get_resv_obj(&self) -> ARef<gem::Object> {
        self.dummy_obj.clone()
    }

    /// Check whether an object is external to this GpuVm
    pub(crate) fn is_extobj(&self, gem: &gem::Object) -> bool {
        self.inner.is_extobj(gem)
    }

    /// Check whether an object is external to this GpuVm
    pub(crate) fn bo_deferred_cleanup(&self) {
        self.inner.bo_deferred_cleanup()
    }
}

impl Drop for VmInner {
    fn drop(&mut self) {
        let mut binding = self.binding.lock();
        assert_eq!(binding.active_users, 0);

        mod_pr_debug!(
            "VmInner::Drop [{}]: bind_token={:?}\n",
            self.id,
            binding.bind_token
        );

        // Make sure this VM is not mapped to a TTB if it was
        if let Some(token) = binding.bind_token.take() {
            let idx = (token.last_slot() as usize) + UAT_USER_CTX_START;

            let uat_inner = self.uat_inner.lock();
            uat_inner.handoff().lock();
            let handoff_cur = uat_inner.handoff().current_slot();
            let ttb_cur = uat_inner.ttbs()[idx].ttb0.load(Ordering::SeqCst);
            let ttb1_cur = uat_inner.ttbs()[idx].ttb1.load(Ordering::SeqCst);
            let (expected_ttb, expected_ttb1, g15_banked) =
                if let Some(bank1) = uat_inner.g15_shared_bank1.as_ref() {
                    (
                        g15_gptbat_bank0(self.ttb(), idx as u8),
                        g15_gptbat_bank1(self.ttb(), bank1.ttb(), idx as u8),
                        true,
                    )
                } else {
                    (
                        self.ttb() | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT,
                        if uat_inner.map_kernel_to_user {
                            uat_inner.kernel_ttb1 | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT
                        } else {
                            0
                        },
                        false,
                    )
                };
            let inval = if g15_banked {
                ttb_cur == expected_ttb && ttb1_cur == expected_ttb1
            } else {
                ttb_cur == expected_ttb
            };
            if inval {
                if handoff_cur == Some(idx as u32) {
                    pr_err!(
                        "VmInner::drop owning slot {}, but it is currently in use by the ASC?\n",
                        idx
                    );
                }
                if g15_banked {
                    // Apple invalidateGPTBATEntry() preserves both roots/context IDs
                    // and clears only the valid bit in both adjacent qwords.
                    uat_inner.ttbs()[idx]
                        .ttb0
                        .store(ttb_cur & !TTBR_VALID, Ordering::SeqCst);
                    uat_inner.ttbs()[idx]
                        .ttb1
                        .store(ttb1_cur & !TTBR_VALID, Ordering::SeqCst);
                } else {
                    uat_inner.ttbs()[idx].ttb0.store(0, Ordering::SeqCst);
                    uat_inner.ttbs()[idx].ttb1.store(0, Ordering::SeqCst);
                }
            }
            uat_inner.handoff().unlock();
            core::mem::drop(uat_inner);

            // In principle we dropped all the KernelMappings already, but we might as
            // well play it safe and invalidate the whole ASID.
            if inval {
                mod_pr_debug!(
                    "VmInner::Drop [{}]: need inval for ASID {:#x}\n",
                    self.id,
                    idx
                );
                mem::tlbi_asid(idx as u8);
                mem::sync();
            }
        }
    }
}

impl Uat {
    /// Map a bootloader-preallocated memory region
    fn map_region(
        dev: &device::Device,
        name: &CStr,
        size: usize,
        cached: bool,
    ) -> Result<UatRegion> {
        let of_node = dev.of_node().ok_or(EINVAL)?;
        let res = of_node.reserved_mem_region_to_resource_byname(name)?;
        let base = res.start();
        let res_size = res.size().try_into()?;

        if size > res_size {
            dev_err!(
                dev,
                "Region {} is too small (expected {}, got {})\n",
                name,
                size,
                res_size
            );
            return Err(ENOMEM);
        }

        let flags = if cached {
            io::mem::MemFlag::WB
        } else {
            io::mem::MemFlag::WC
        };

        // SAFETY: The safety of this operation hinges on the correctness of
        // much of this file and also the `pgtable` module, so it is difficult
        // to prove in a single safety comment. Such is life with raw GPU
        // page table management...
        let map = unsafe { io::mem::Mem::try_new(res, flags.into()) }.inspect_err(|_| {
            dev_err!(dev, "Failed to remap {} mem resource\n", name);
        })?;

        Ok(UatRegion { base, map })
    }

    /// Returns a reference to the global kernel (upper half) `Vm`
    pub(crate) fn kernel_vm(&self) -> &Vm {
        &self.kernel_vm
    }

    /// Returns a reference to the local kernel (lower half) `Vm`
    pub(crate) fn kernel_lower_vm(&self) -> &Vm {
        &self.kernel_lower_vm
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    pub(crate) fn dump_kernel_pages(&self) -> Result<KVVec<pgtable::DumpedPage>> {
        let mut inner = self.kernel_vm.inner.exec_lock(None, false)?;
        inner.page_table.dump_pages(IOVA_KERN_FULL_RANGE)
    }

    /// Read-only G15 bring-up helper for a single TTBR0 page. This does not
    /// allocate, map, unmap, or alter page-table state.
    #[cfg(CONFIG_DEV_COREDUMP)]
    pub(crate) fn probe_lower_page(&self, iova: u64) -> Result<Option<(u64, u64, bool)>> {
        let start = iova & !(UAT_PGMSK as u64);
        let end = start + UAT_PGSZ as u64;
        let mut inner = self.kernel_lower_vm.inner.exec_lock(None, false)?;
        let pages = inner.page_table.dump_pages(start..end)?;

        for page in pages {
            return Ok(Some((
                page.pte,
                page.pte & pgtable::PTE_ADDR_BITS,
                page.data.is_some(),
            )));
        }
        Ok(None)
    }

    /// Returns the base physical address of the TTBAT region.
    pub(crate) fn ttb_base(&self) -> u64 {
        let inner = self.inner.lock();

        inner.ttbs_rgn.base
    }

    /// Returns a mapping handle for G15's accelerator-shared bank 1 when the
    /// hardware configuration provides the 42-bit banked UAT.
    pub(crate) fn g15_shared_bank1(&self) -> Option<G15SharedBank1> {
        if self.inner.lock().g15_shared_bank1.is_some() {
            Some(G15SharedBank1 {
                dev: self.dev.clone(),
                inner: self.inner.clone(),
            })
        } else {
            None
        }
    }

    /// Binds a `Vm` to a slot, preferring the last used one.
    pub(crate) fn bind(&self, vm: &Vm) -> Result<VmBind> {
        let mut binding = vm.binding.lock();

        if binding.binding.is_none() {
            assert_eq!(binding.active_users, 0);

            let isolation = *module_parameters::robust_isolation.value() != 0;

            self.slots.set_limit(if isolation {
                NonZeroUsize::new(1)
            } else {
                None
            });

            let mut slot = self.slots.get(binding.bind_token)?;
            if slot.changed() {
                mod_pr_debug!("Vm Bind [{}]: bind_token={:?}\n", vm.id, slot.token(),);
                let idx = (slot.slot() as usize) + UAT_USER_CTX_START;
                let uat_inner = self.inner.lock();

                let (ttb, ttb1) = if let Some(bank1) = uat_inner.g15_shared_bank1.as_ref() {
                    let bank1_root = bank1.ttb();
                    if !g15_gptbat_roots_fit(binding.ttb, bank1_root) {
                        dev_err!(
                            self.dev.as_ref(),
                            "MMU: G15 GPTBAT roots exceed 42-bit OAS ({:#x}, {:#x})\n",
                            binding.ttb,
                            bank1_root
                        );
                        return Err(EINVAL);
                    }
                    (
                        g15_gptbat_bank0(binding.ttb, idx as u8),
                        g15_gptbat_bank1(binding.ttb, bank1_root, idx as u8),
                    )
                } else {
                    let ttb0 = binding.ttb | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT;
                    let ttb1 = if uat_inner.map_kernel_to_user {
                        uat_inner.kernel_ttb1 | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT
                    } else {
                        0
                    };
                    (ttb0, ttb1)
                };

                let ttbs = uat_inner.ttbs();
                uat_inner.handoff().lock();
                if uat_inner.handoff().current_slot() == Some(idx as u32) {
                    pr_err!(
                        "Vm::bind to slot {}, but it is currently in use by the ASC?\n",
                        idx
                    );
                }
                ttbs[idx].ttb0.store(ttb, Ordering::Release);
                ttbs[idx].ttb1.store(ttb1, Ordering::Release);
                uat_inner.handoff().unlock();
                core::mem::drop(uat_inner);

                // Make sure all TLB entries from the previous owner of this ASID are gone
                mem::tlbi_asid(idx as u8);
                mem::sync();

                if self.cfg.gpu_gen == hw::GpuGen::G15 {
                    // 23J220 AGXContextIDManager::alloc() increments the
                    // generation only after registerContextID() succeeds.
                    slot.generation = slot.generation.wrapping_add(1);
                }
            }

            binding.bind_token = Some(slot.token());
            binding.binding = Some(slot);
        }

        binding.active_users += 1;

        let slot_guard = binding.binding.as_ref().unwrap();
        let slot = slot_guard.slot() + UAT_USER_CTX_START as u32;
        let generation = slot_guard.generation;
        mod_pr_debug!(
            "MMU: slot {} generation {} active users {}\n",
            slot,
            generation,
            binding.active_users
        );
        Ok(VmBind(vm.clone(), slot, generation))
    }

    /// Creates a new `Vm` linked to this UAT.
    pub(crate) fn new_vm(&self, id: u64, kernel_range: Range<u64>) -> Result<Vm> {
        Vm::new(
            &self.dev,
            self.inner.clone(),
            kernel_range,
            self.cfg,
            None,
            id,
        )
    }

    /// Creates the reference-counted inner data for a new `Uat` instance.
    #[inline(never)]
    fn make_inner(dev: &driver::AsahiDevice) -> Result<Arc<UatInner>> {
        let handoff_rgn = Self::map_region(dev.as_ref(), c_str!("handoff"), HANDOFF_SIZE, true)?;
        let ttbs_rgn = Self::map_region(dev.as_ref(), c_str!("ttbs"), SLOTS_SIZE, true)?;

        // SAFETY: The Handoff struct layout matches the firmware's view of memory at this address,
        // and the region is at least large enough per the size specified above.
        let handoff = unsafe { &(handoff_rgn.map.ptr() as *mut Handoff).as_ref().unwrap() };

        dev_info!(dev.as_ref(), "MMU: Initializing kernel page table\n");

        Arc::pin_init(
            try_pin_init!(UatInner {
                handoff_flush <- pin_init::pin_init_array_from_fn(|i| {
                    new_mutex!(HandoffFlush(&handoff.flush[i]), "handoff_flush")
                }),
                shared <- new_mutex!(
                    UatShared {
                        kernel_ttb1: 0,
                        map_kernel_to_user: false,
                        g15_shared_bank1: None,
                        handoff_rgn,
                        ttbs_rgn,
                    },
                    "uat_shared"
                ),
            }),
            GFP_KERNEL,
        )
    }

    /// Creates a new `Uat` instance given the relevant hardware config.
    #[inline(never)]
    pub(crate) fn new(
        dev: &driver::AsahiDevice,
        cfg: &'static hw::HwConfig,
        map_kernel_to_user: bool,
    ) -> Result<Self> {
        dev_info!(dev.as_ref(), "MMU: Initializing...\n");

        // G15 has two 42-bit bank-local roots.  The accelerator/kernel UAT
        // owns both roots; normal clients import its bank 1.  Construct the
        // shared range-7 view only after the existing kernel TTB1 root is
        // known, so slot 0 and every client use one physical bank-1 root.
        let inner = Self::make_inner(dev)?;

        let of_node = dev.as_ref().of_node().ok_or(EINVAL)?;
        let res = of_node.reserved_mem_region_to_resource_byname(c_str!("pagetables"))?;
        let ttb1 = res.start();
        let ttb1size: usize = res.size().try_into()?;

        if ttb1size < PAGETABLES_SIZE {
            dev_err!(dev.as_ref(), "MMU: Pagetables region is too small\n");
            return Err(ENOMEM);
        }

        dev_info!(dev.as_ref(), "MMU: Creating kernel page tables\n");
        let kernel_lower_vm = Vm::new(dev, inner.clone(), IOVA_USER_RANGE, cfg, None, 1)?;
        let kernel_vm = Vm::new(dev, inner.clone(), IOVA_KERN_RANGE, cfg, Some(ttb1), 0)?;

        dev_info!(dev.as_ref(), "MMU: Kernel page tables created\n");

        let ttb0 = kernel_lower_vm.ttb();

        let uat = Self {
            dev: dev.into(),
            cfg,
            kernel_vm,
            kernel_lower_vm,
            inner,
            slots: slotalloc::SlotAllocator::new(
                UAT_USER_CTX as u32,
                (),
                |_inner, _slot| Some(SlotInner { generation: 0 }),
                c_str!("Uat::SlotAllocator"),
                static_lock_class!(),
                static_lock_class!(),
            )?,
        };

        let mut inner = uat.inner.lock();

        inner.map_kernel_to_user = map_kernel_to_user;
        inner.kernel_ttb1 = ttb1;
        if let Some(bank1) = inner.g15_shared_bank1.as_ref() {
            dev_info!(
                dev.as_ref(),
                "MMU: G15 shared bank-1 root prepared at {:#x}\n",
                bank1.ttb()
            );
        }

        inner.handoff().init(cfg.gpu_gen == hw::GpuGen::G15)?;

        // G15 firmware initializes root[2] of the accelerator-owned bank-1
        // spine during the handoff.  Only after that handshake is complete can
        // Linux safely discover the firmware-carveout shared-L2 page.  E026
        // validates this spine read-only; range-7 leaf mutation remains gated.
        if cfg.uat_ias >= G15_HW_UAT_IAS {
            core::mem::drop(inner);
            let mut bank1 = G15SharedBank1State::new(dev, cfg, ttb1)?;
            // E029 exercises the persistent shared-bank1 backend itself. The
            // Eight Linux L3 parents (six range-7, two range-8) remain installed
            // until UAT teardown; both test leaves are mapped/unmapped through
            // the same bounded backend methods.
            bank1.page_table.e029_preflight_backend(dev)?;
            bank1.page_table.e075_preflight_range8_leaf(dev)?;
            inner = uat.inner.lock();
            inner.g15_shared_bank1 = Some(bank1);
        }

        dev_info!(dev.as_ref(), "MMU: Initializing TTBs\n");

        inner.handoff().lock();

        let ttbs = inner.ttbs();

        ttbs[0].ttb0.store(ttb0 | TTBR_VALID, Ordering::SeqCst);
        ttbs[0].ttb1.store(ttb1 | TTBR_VALID, Ordering::SeqCst);

        for ctx in &ttbs[1..] {
            ctx.ttb0.store(0, Ordering::Relaxed);
            ctx.ttb1.store(0, Ordering::Relaxed);
        }

        inner.handoff().unlock();

        core::mem::drop(inner);

        dev_info!(dev.as_ref(), "MMU: initialized\n");

        Ok(uat)
    }
}

impl Drop for Uat {
    fn drop(&mut self) {
        // Make sure we flush the TLBs
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();
    }
}
