// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Tiled Vertex Buffer management
//!
//! This module manages the Tiled Vertex Buffer, also known as the Parameter Buffer (in imgtec
//! parlance) or the tiler heap (on other architectures). This buffer holds transformed primitive
//! data between the vertex/tiling stage and the fragment stage.
//!
//! On AGX, the buffer is a heap of 128K blocks split into 32K pages (which must be aligned to a
//! multiple of 32K in VA space). The buffer can be shared between multiple render jobs, and each
//! will allocate pages from it during vertex processing and return them during fragment processing.
//!
//! If the buffer runs out of free pages, the vertex pass stops and a partial fragment pass occurs,
//! spilling the intermediate render target state to RAM (a partial render). This is all managed
//! transparently by the firmware. Since partial renders are less efficient, the kernel must grow
//! the heap in response to feedback from the firmware to avoid partial renders in the future.
//! Currently, we only ever grow the heap, and never shrink it.
//!
//! AGX also supports memoryless render targets, which can be used for intermediate results within
//! a render pass. To support partial renders, it seems the GPU/firmware has the ability to borrow
//! pages from the TVB buffer as a temporary render target buffer. Since this happens during a
//! partial render itself, if the buffer runs out of space, it requires synchronous growth in
//! response to a firmware interrupt. This is not currently supported, but may be in the future,
//! though it is unclear whether it is worth the effort.
//!
//! This module is also in charge of managing the temporary objects associated with a single render
//! pass, which includes the top-level tile array, the tail pointer cache, preemption buffers, and
//! other miscellaneous structures collectively managed as a "scene".
//!
//! To avoid runaway memory usage, there is a maximum size for buffers (at that point it's unlikely
//! that partial renders will incur much overhead over the buffer data access itself). This is
//! different depending on whether memoryless render targets are in use, and is currently hardcoded.
//! to the most common value used by macOS.

use crate::debug::*;
use crate::fw::buffer;
use crate::fw::types::*;
use crate::util::*;
use crate::{
    alloc,
    fw,
    gpu,
    hw,
    mmu,
    slotalloc, //
};
use core::sync::atomic::Ordering;
use kernel::new_mutex;
use kernel::prelude::*;
use kernel::sync::{
    Arc,
    Mutex, //
};
use kernel::{
    c_str,
    static_lock_class, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Buffer;

/// There are 127 GPU/firmware-side buffer manager slots (yes, 127, not 128).
const NUM_BUFFERS: u32 = 127;

/// Page size bits for buffer pages (32K). VAs must be aligned to this size.
pub(crate) const PAGE_SHIFT: usize = 15;
/// Page size for buffer pages.
pub(crate) const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
/// Number of pages in a buffer block, which should be contiguous in VA space.
pub(crate) const PAGES_PER_BLOCK: usize = 4;
/// Size of a buffer block.
pub(crate) const BLOCK_SIZE: usize = PAGE_SIZE * PAGES_PER_BLOCK;

/// Exact J615/G15G ContextSwitcherGen3 render scratch allocation. Apple
/// publishes its 32-byte-aligned base and base + 0x280 to the TA command.
pub(crate) const G15_RENDER_CTXSWITCH_BYTES: usize = 0x8e0;

/// Exact G15 AGXHardwareBufferIDManager capacity. Apple G15/G15G start()
/// initializes this manager with 0x100 entries; RTKit reuses the assigned ID
/// directly as the UMA Page-Pool descriptor-table index.
pub(crate) const G15_HARDWARE_BUFFER_ID_COUNT: usize = 0x100;
const G15_HARDWARE_BUFFER_ID_NONE: u32 = u32::MAX;
const G15_HARDWARE_BUFFER_BITMAP_WORDS: usize = G15_HARDWARE_BUFFER_ID_COUNT / 64;

const fn g15_hardware_buffer_initial_free_stack() -> [u16; G15_HARDWARE_BUFFER_ID_COUNT] {
    let mut stack = [0u16; G15_HARDWARE_BUFFER_ID_COUNT];
    let mut i = 0;
    while i < G15_HARDWARE_BUFFER_ID_COUNT {
        stack[i] = (G15_HARDWARE_BUFFER_ID_COUNT - 1 - i) as u16;
        i += 1;
    }
    stack
}

const G15_HARDWARE_BUFFER_INITIAL_FREE_STACK: [u16; G15_HARDWARE_BUFFER_ID_COUNT] =
    g15_hardware_buffer_initial_free_stack();
const _: [(); 0xff] = [(); G15_HARDWARE_BUFFER_INITIAL_FREE_STACK[0] as usize];
const _: [(); 0x00] = [(); G15_HARDWARE_BUFFER_INITIAL_FREE_STACK[0xff] as usize];

/// Per-FList sticky HardwareBuffer-ID state. Apple stores the sticky ID at
/// AGXHardwareBufferBase +0x10; when an inactive ID is stolen, the old object
/// keeps that stale value until its next allocation detects the owner mismatch.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15HardwareBufferBinding {
    owner_cookie: u64,
    hardware_buffer_id: u32,
}

#[allow(dead_code)]
impl G15HardwareBufferBinding {
    pub(crate) fn new(owner_cookie: u64) -> Result<Self> {
        if owner_cookie == 0 {
            return Err(EINVAL);
        }
        Ok(Self {
            owner_cookie,
            hardware_buffer_id: G15_HARDWARE_BUFFER_ID_NONE,
        })
    }

    pub(crate) fn hardware_buffer_id(&self) -> Option<u32> {
        (self.hardware_buffer_id != G15_HARDWARE_BUFFER_ID_NONE)
            .then_some(self.hardware_buffer_id)
    }
}

/// Result of one HardwareBuffer-ID reference acquisition. `first_reference`
/// matches Apple's bool out-parameter and the argument passed to
/// AGXHardwareBufferBase::prepareBufferResources(): it is true exactly on the
/// transition from zero references to one.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15HardwareBufferLease {
    pub(crate) hardware_buffer_id: u32,
    pub(crate) first_reference: bool,
}

/// Compile-only reconstruction of AGXHardwareBufferIDManager's exact G15 state
/// machine. Callers must provide external synchronization before this becomes
/// runtime-active; E079 deliberately does not instantiate or lock this object.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct G15HardwareBufferIdState {
    owners: [u64; G15_HARDWARE_BUFFER_ID_COUNT],
    refs: [u32; G15_HARDWARE_BUFFER_ID_COUNT],
    free_bitmap: [u64; G15_HARDWARE_BUFFER_BITMAP_WORDS],
    free_stack: [u16; G15_HARDWARE_BUFFER_ID_COUNT],
    free_top: i16,
}

#[allow(dead_code)]
impl G15HardwareBufferIdState {
    pub(crate) fn new() -> Self {
        Self {
            owners: [0; G15_HARDWARE_BUFFER_ID_COUNT],
            refs: [0; G15_HARDWARE_BUFFER_ID_COUNT],
            free_bitmap: [u64::MAX; G15_HARDWARE_BUFFER_BITMAP_WORDS],
            free_stack: G15_HARDWARE_BUFFER_INITIAL_FREE_STACK,
            free_top: (G15_HARDWARE_BUFFER_ID_COUNT - 1) as i16,
        }
    }

    fn set_free(&mut self, id: usize, free: bool) {
        let word = id >> 6;
        let bit = 1u64 << (id & 63);
        if free {
            self.free_bitmap[word] |= bit;
        } else {
            self.free_bitmap[word] &= !bit;
        }
    }

    fn pop_free_id(&mut self) -> Option<u32> {
        if self.free_top >= 0 {
            let top = self.free_top as usize;
            let id = self.free_stack[top] as u32;
            self.free_top -= 1;
            return Some(id);
        }

        for (word_index, word) in self.free_bitmap.iter().copied().enumerate() {
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                return Some((word_index * 64 + bit) as u32);
            }
        }
        None
    }

    fn push_free_id(&mut self, id: u32) -> Result {
        let next = self.free_top as i32 + 1;
        if next < 0 || next as usize >= G15_HARDWARE_BUFFER_ID_COUNT {
            return Err(EOVERFLOW);
        }
        self.free_top = next as i16;
        self.free_stack[next as usize] = id as u16;
        Ok(())
    }

    /// Mirrors AGXHardwareBufferIDManager::alloc(). Sticky reuse succeeds when
    /// binding ID and manager owner still match. Otherwise an unowned LIFO ID
    /// is used first; after that initial stack is exhausted, the lowest set bit
    /// in the zero-reference bitmap may steal a dormant sticky ID.
    pub(crate) fn acquire(
        &mut self,
        binding: &mut G15HardwareBufferBinding,
    ) -> Result<G15HardwareBufferLease> {
        if binding.owner_cookie == 0 {
            return Err(EINVAL);
        }

        if let Some(id) = binding.hardware_buffer_id() {
            let idx = id as usize;
            if idx < G15_HARDWARE_BUFFER_ID_COUNT && self.owners[idx] == binding.owner_cookie {
                self.refs[idx] = self.refs[idx].checked_add(1).ok_or(EOVERFLOW)?;
                self.set_free(idx, false);
                return Ok(G15HardwareBufferLease {
                    hardware_buffer_id: id,
                    first_reference: self.refs[idx] == 1,
                });
            }
        }

        let id = self.pop_free_id().ok_or(ENOSPC)?;
        let idx = id as usize;
        if idx >= G15_HARDWARE_BUFFER_ID_COUNT || self.refs[idx] != 0 {
            return Err(EIO);
        }
        self.refs[idx] = 1;
        self.set_free(idx, false);
        self.owners[idx] = binding.owner_cookie;
        binding.hardware_buffer_id = id;
        Ok(G15HardwareBufferLease {
            hardware_buffer_id: id,
            first_reference: true,
        })
    }

    /// Mirrors AGXHardwareBufferIDManager::complete(). The return value is true
    /// exactly when the reference count is zero after completion and therefore
    /// the caller must run completeBufferResources(). A normally sticky owner
    /// retains its ID/owner entry; a mismatched stale owner releases the slot to
    /// the LIFO stack after clearing the manager's owner entry.
    pub(crate) fn complete(
        &mut self,
        binding: &G15HardwareBufferBinding,
        hardware_buffer_id: u32,
    ) -> Result<bool> {
        let idx = hardware_buffer_id as usize;
        if idx >= G15_HARDWARE_BUFFER_ID_COUNT {
            return Err(EINVAL);
        }

        if self.refs[idx] != 0 {
            self.refs[idx] -= 1;
        }
        if self.refs[idx] != 0 {
            return Ok(false);
        }

        self.set_free(idx, true);
        if binding.hardware_buffer_id != hardware_buffer_id {
            self.owners[idx] = 0;
            self.push_free_id(hardware_buffer_id)?;
        }
        Ok(true)
    }
}

/// Synchronized owner for the exact G15 HardwareBuffer-ID state machine.
/// E080 defines the lock/Arc lifetime but deliberately does not instantiate this
/// manager in GpuManager, InitData, a queue, or any FList runtime object.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct G15HardwareBufferIdManager(Arc<Mutex<G15HardwareBufferIdState>>);

#[allow(dead_code)]
impl G15HardwareBufferIdManager {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self(Arc::pin_init(
            new_mutex!(G15HardwareBufferIdState::new(), "g15_hardware_buffer_ids"),
            GFP_KERNEL,
        )?))
    }

    pub(crate) fn acquire(
        &self,
        binding: &mut G15HardwareBufferBinding,
    ) -> Result<G15HardwareBufferLease> {
        self.0.lock().acquire(binding)
    }

    pub(crate) fn complete(
        &self,
        binding: &G15HardwareBufferBinding,
        hardware_buffer_id: u32,
    ) -> Result<bool> {
        self.0.lock().complete(binding, hardware_buffer_id)
    }
}

/// FList-side sticky HardwareBuffer ownership. This is the host object boundary
/// corresponding to AGXHardwareBufferBase +0x10: it retains the binding across
/// zero-reference periods so the manager can reuse the same dormant ID until it
/// is stolen. `prepare_reference()` / `complete_reference()` return the exact
/// callback transition booleans; E080 does not execute those callbacks itself.
#[allow(dead_code)]
pub(crate) struct G15FListHardwareBufferOwner {
    manager: G15HardwareBufferIdManager,
    binding: G15HardwareBufferBinding,
}

#[allow(dead_code)]
impl G15FListHardwareBufferOwner {
    pub(crate) fn new(manager: G15HardwareBufferIdManager, owner_cookie: u64) -> Result<Self> {
        Ok(Self {
            manager,
            binding: G15HardwareBufferBinding::new(owner_cookie)?,
        })
    }

    pub(crate) fn sticky_id(&self) -> Option<u32> {
        self.binding.hardware_buffer_id()
    }

    pub(crate) fn prepare_reference(&mut self) -> Result<G15HardwareBufferLease> {
        self.manager.acquire(&mut self.binding)
    }

    pub(crate) fn complete_reference(&self, hardware_buffer_id: u32) -> Result<bool> {
        self.manager.complete(&self.binding, hardware_buffer_id)
    }
}

/// Apple G15 parameter-management device configuration recovered from
/// `AGXAcceleratorG15::halGetPMConfig()`.
///
/// This is deliberately kept separate from the firmware-visible parameter-buffer
/// ABI above.  The values control host-side PM bookkeeping/allocation geometry;
/// they do not justify enabling G15 PM register emission by themselves.
struct G15PmDeviceConfig {
    usage_page_granule: usize,
    scene_pages_per_entry: usize,
    scene_header_bytes: usize,
    scene_entry_bytes: usize,
    scene_alignment: usize,
}

const G15_PM_DEVICE_CONFIG: G15PmDeviceConfig = G15PmDeviceConfig {
    usage_page_granule: 4,
    scene_pages_per_entry: 0x1800,
    scene_header_bytes: 8,
    scene_entry_bytes: 8,
    scene_alignment: 0x10,
};

/// J615 has one MGPU, so Apple's multi-MGPU extra-entry term is absent from
/// `AGXParameterManagementVirtual::setupSceneState()`.
///
/// Base `AGXAccelerator::configureDevice()` writes the packed dword pair
/// `{0x24, 0x24}` at accelerator +0x678/+0x67c. G15 overwrites only +0x678
/// with 0x50, so the normal J615 PM record count is 80 and scene group count
/// remains 36. The optional +0x2420/+0x2424 overrides stay zero in the
/// analyzed normal path.
pub(crate) const G15_J615_PM_RECORD_COUNT: usize = 0x50;
const G15_J615_PM_SCENE_GROUP_COUNT: usize = 0x24;
/// G15G/C0 sets accelerator +0x1dd8, so Apple reserves one additional
/// shared scene slice after the 36 modulo-selected per-record slices.
const G15_J615_PM_EXTRA_SCENE_SLICES: usize = 1;

/// Total J615 Parameter Scene Allocations backing: 36 modulo-selected 0x30
/// slices plus the one G15G/C0 common slice. Apple allocates this in
/// eGartRange 5. Exact AGXGart::returnGartRange() places that range above the
/// current Linux 39-bit user aperture, so allocation stays blocked until the
/// G15 VA contract is implemented.
pub(crate) const G15_J615_PM_SCENE_ALLOC_BYTES: usize =
    (G15_J615_PM_SCENE_GROUP_COUNT + G15_J615_PM_EXTRA_SCENE_SLICES) * 0x30;

/// G15 TA-channel object-payload resource geometry. Apple's
/// `AGXTAChannelG15::getObjectPayloadBufferSize()` computes
/// `q = (num_gps << 17) / num_mgpus`, returns `num_mgpus * q` bytes, and
/// stores `q >> 10` in the channel for register 0x1ca48. J615 has four GPS
/// and one MGPU, so the exact allocation is 0x80000 bytes and the packed
/// high field is 0x200. The resource is a normal eGartRange-5, option-0x3
/// kernel resource and therefore uses the uncached G15 range-5 PTE class.
const fn g15_ta_object_payload_bytes(num_gps: usize, num_mgpus: usize) -> usize {
    let q = (num_gps << 17) / num_mgpus;
    num_mgpus * q
}

const fn g15_ta_object_payload_units(num_gps: u64, num_mgpus: u64) -> u64 {
    ((num_gps << 17) / num_mgpus) >> 10
}

pub(crate) const G15_J615_TA_OBJECT_PAYLOAD_BYTES: usize =
    g15_ta_object_payload_bytes(4, 1);
pub(crate) const G15_J615_TA_OBJECT_PAYLOAD_UNITS: u64 =
    g15_ta_object_payload_units(4, 1);

/// Exact G15 TA 0x1ca48 value. `bindAndRetainMeshRenderingBuffers()` uses the
/// mapped kernel-resource base plus AGXResource +0x48. For `newKernelResource`
/// the inherited placement wrapper forces that field to zero, so J615 uses the
/// allocation GPUVA directly. The low 10 bits are discarded and the channel's
/// payload-unit field is inserted at bit 48.
pub(crate) const fn g15_ta_object_payload_reg_1ca48(gpuva: u64, units: u64) -> u64 {
    (gpuva & !0x3ff) | (units << 48)
}

/// Host-side G15 parameter-management record layout. Apple allocates these at
/// an exact 0x80-byte stride and publishes one record pointer per PM slot.
///
/// Only mechanically proven fields are named. In particular, +0x30 points to
/// the same extra scene-allocation slice for every record when accelerator
/// +0x1dd8 is set; no stronger semantic role is assumed here.
#[repr(C)]
pub(crate) struct G15PmRecord {
    page_metrics_gpuva: u64,
    page_metrics_fwva: u64,
    // setupSceneState() initially zeroes these qwords, but normal Fragment
    // completion later consumes the low dwords at +0x10 and +0x18. Keep the
    // names direction-neutral until their producer is recovered.
    completion_stat_10: u32,
    opaque_14: u32,
    completion_stat_18: u32,
    opaque_1c: u32,
    zero_20: u64,
    scene_slice_gpuva: u64,
    shared_scene_slice_gpuva: u64,
    opaque_38: u64,
    scene_stats_fwva: u64,
    // RTKit adds +0x10 + +0x18 into this persistent low-dword accumulator on
    // every normal Fragment completion, then max-tracks it in scene statistics.
    completion_accumulator_48: u32,
    opaque_4c: [u8; 0x34],
}

/// The upper 0x40-byte half of Apple's exact 0x80-byte
/// "Firmware Page List Entries, Parameter Scene statistics" resource.
/// Normal Fragment completion accesses it through G15PmRecord +0x40.
#[repr(C)]
struct G15PmSceneStats {
    max_record_accumulator_00: u32,
    max_info_completion_stat_04: u32,
    reset_cleared_08: u32,
    reset_cleared_0c: u32,
    opaque_10: [u8; 0x10],
    // A nonzero value causes RTKit to clear +0/+4/+8/+0xc and this field
    // before applying the current completion update.
    reset_request_20: u32,
    opaque_24: [u8; 0x1c],
}

impl Default for G15PmRecord {
    fn default() -> Self {
        Self {
            page_metrics_gpuva: 0,
            page_metrics_fwva: 0,
            completion_stat_10: 0,
            opaque_14: 0,
            completion_stat_18: 0,
            opaque_1c: 0,
            zero_20: 0,
            scene_slice_gpuva: 0,
            shared_scene_slice_gpuva: 0,
            opaque_38: 0,
            scene_stats_fwva: 0,
            completion_accumulator_48: 0,
            opaque_4c: [0; 0x34],
        }
    }
}

impl G15PmRecord {
    /// Construct one J615 GPU-facing PM record from the mechanically proven
    /// range-5 scene pointers and range-7 PM resources. Apple initializes the
    /// remaining completion/tail fields to zero here; their later producers
    /// remain independently unresolved.
    pub(crate) fn new_with_resources(
        scene_slice_gpuva: u64,
        shared_scene_slice_gpuva: u64,
        page_metrics_gpuva: u64,
        scene_stats_fwva: u64,
    ) -> Self {
        Self {
            page_metrics_gpuva,
            page_metrics_fwva: page_metrics_gpuva,
            scene_slice_gpuva,
            shared_scene_slice_gpuva,
            scene_stats_fwva,
            ..Default::default()
        }
    }
}

const _: [(); 0x80] = [(); core::mem::size_of::<G15PmRecord>()];
const _: [(); 0x00] = [(); core::mem::offset_of!(G15PmRecord, page_metrics_gpuva)];
const _: [(); 0x08] = [(); core::mem::offset_of!(G15PmRecord, page_metrics_fwva)];
const _: [(); 0x10] = [(); core::mem::offset_of!(G15PmRecord, completion_stat_10)];
const _: [(); 0x18] = [(); core::mem::offset_of!(G15PmRecord, completion_stat_18)];
const _: [(); 0x20] = [(); core::mem::offset_of!(G15PmRecord, zero_20)];
const _: [(); 0x28] = [(); core::mem::offset_of!(G15PmRecord, scene_slice_gpuva)];
const _: [(); 0x30] = [(); core::mem::offset_of!(G15PmRecord, shared_scene_slice_gpuva)];
const _: [(); 0x40] = [(); core::mem::offset_of!(G15PmRecord, scene_stats_fwva)];
const _: [(); 0x48] = [(); core::mem::offset_of!(G15PmRecord, completion_accumulator_48)];
const _: [(); 0x40] = [(); core::mem::size_of::<G15PmSceneStats>()];
const _: [(); 0x00] = [(); core::mem::offset_of!(G15PmSceneStats, max_record_accumulator_00)];
const _: [(); 0x04] = [(); core::mem::offset_of!(G15PmSceneStats, max_info_completion_stat_04)];
const _: [(); 0x20] = [(); core::mem::offset_of!(G15PmSceneStats, reset_request_20)];

/// `AGXParameterManagement::init()` allocates the record pool as
/// `(record_count * 0x80) | 0x40`. Since every record starts on a 0x80
/// boundary, the final OR reserves one trailing 0x40-byte state region. The
/// GPUVA immediately after the records is retained at PM +0xb8 and exported
/// through the G15 3D work-command header at +0x38. J615 also
/// allocates a separate eight-byte-per-record "Firmware Page List Entries"
/// resource and an exact 0x80-byte
/// "Firmware Page List Entries, Parameter Scene statistics" resource.
/// Record +0x40 is the FWVA of the latter resource at offset +0x40, i.e. the
/// Parameter Scene statistics half. These remain allocation geometry only.
const G15_PM_RECORD_BYTES: usize = 0x80;
/// Exact GPU-facing range-5 PM record resource: 80 records at a 0x80 stride.
/// This deliberately excludes the separate 0x40 PM tail/state bookkeeping.
pub(crate) const G15_J615_PM_GPU_RECORD_BYTES: usize =
    G15_J615_PM_RECORD_COUNT * G15_PM_RECORD_BYTES;
const G15_PM_STATE_BYTES: usize = 0x40;
const G15_PM_FW_PAGE_LIST_ENTRY_BYTES: usize = 8;
pub(crate) const G15_PM_PAGE_LIST_STATS_BYTES: usize = 0x80;
const G15_PM_SCENE_STATS_OFFSET: usize = 0x40;

const fn g15_pm_state_offset(record_count: usize) -> usize {
    record_count * G15_PM_RECORD_BYTES
}

const fn g15_pm_record_pool_bytes(record_count: usize) -> usize {
    g15_pm_state_offset(record_count) | G15_PM_STATE_BYTES
}

const fn g15_pm_fw_page_list_bytes(record_count: usize) -> usize {
    record_count * G15_PM_FW_PAGE_LIST_ENTRY_BYTES
}

/// Apple keeps the selected PM record index at AGXParameterManagement +0x2c.
/// processRenderSetup() increments it before loading the command descriptor and
/// wraps it modulo the configured record count (0x50 on J615).  The PM object
/// itself is owned by AGX3DWorkQueue (+0x1f8) and may be explicitly shared by
/// another work queue, so this index is queue/PM state rather than a TVB slot.
pub(crate) const fn g15_j615_pm_next_record_index(current: u32) -> u32 {
    let next = current + 1;
    if next >= G15_J615_PM_RECORD_COUNT as u32 {
        0
    } else {
        next
    }
}

/// setupSceneState() precomputes record[i]+0x28 as scene_base +
/// 0x30 * (i % 36) on J615.  This returns only the mechanically proven offset;
/// allocation/base-address wiring remains deliberately separate.
pub(crate) const fn g15_j615_pm_scene_slice_offset(record_index: u32) -> usize {
    (record_index as usize % G15_J615_PM_SCENE_GROUP_COUNT)
        * g15_j615_pm_scene_stride(0x3366_0000)
}

/// Common G15G/C0 scene slice shared by every PM record.
pub(crate) const fn g15_j615_pm_shared_scene_slice_offset() -> usize {
    G15_J615_PM_SCENE_GROUP_COUNT * g15_j615_pm_scene_stride(0x3366_0000)
}

/// Selected GPU record offset inside the exact 0x2800 range-5 record backing.
pub(crate) const fn g15_j615_pm_record_offset(record_index: u32) -> usize {
    (record_index as usize % G15_J615_PM_RECORD_COUNT) * G15_PM_RECORD_BYTES
}

/// `AGXArmFirmware::allocFirmwareData()` sizes one PMPageMetricsBuffer resource
/// element as `align(record_count * 4, 0x40)`. Each PM record then receives a
/// distinct four-byte slot within that element at `base + 4 * record_index`.
/// Linux now models one requested element directly in shared bank 1; Apple's
/// resource-stack backing/growth policy remains a separate pooling detail.
const G15_PM_PAGE_METRICS_SLOT_BYTES: usize = 4;
const G15_PM_PAGE_METRICS_ALIGNMENT: usize = 0x40;

const fn g15_pm_page_metrics_bytes(record_count: usize) -> usize {
    let bytes = record_count * G15_PM_PAGE_METRICS_SLOT_BYTES;
    (bytes + G15_PM_PAGE_METRICS_ALIGNMENT - 1) & !(G15_PM_PAGE_METRICS_ALIGNMENT - 1)
}

/// Exact J615 PMPageMetricsBuffer resource element size.
pub(crate) const G15_J615_PM_PAGE_METRICS_BYTES: usize =
    g15_pm_page_metrics_bytes(G15_J615_PM_RECORD_COUNT);

/// Offset of one record's four-byte PMPageMetricsBuffer slot.
pub(crate) const fn g15_j615_pm_page_metrics_slot_offset(record_index: u32) -> usize {
    (record_index as usize % G15_J615_PM_RECORD_COUNT) * G15_PM_PAGE_METRICS_SLOT_BYTES
}

/// Exact G15 TA register 0x1c910 encoding of a selected PMPageMetricsBuffer
/// slot GPUVA. Apple places this resource in eGartRange 7. The transform folds
/// source address bit 42 into result bit 39 and sets bit 0 as the enable bit.
/// RegisterArray emission remains fail-closed with the rest of the G15 list,
/// but the value is now fully constructible from the compile-only resource.
pub(crate) const fn g15_pm_page_metrics_reg_1c910(gpuva: u64) -> u64 {
    let prefix = if gpuva & 0x400_0000_0000 != 0 {
        0
    } else {
        0x70_0000_0000
    };

    ((gpuva >> 3) & 0x80_0000_0000) | ((prefix + gpuva) & 0x7f_ffff_fffe) | 1
}

const fn g15_j615_pm_scene_stride(pb_max_size: usize) -> usize {
    let pages = (pb_max_size + PAGE_SIZE - 1) / PAGE_SIZE;
    let entries = (pages + G15_PM_DEVICE_CONFIG.scene_pages_per_entry - 1)
        / G15_PM_DEVICE_CONFIG.scene_pages_per_entry;
    let bytes = G15_PM_DEVICE_CONFIG.scene_header_bytes
        + entries * G15_PM_DEVICE_CONFIG.scene_entry_bytes;
    (bytes + G15_PM_DEVICE_CONFIG.scene_alignment - 1)
        & !(G15_PM_DEVICE_CONFIG.scene_alignment - 1)
}

// Apple G15's device config is {4, 0x1800, 8, 8, 0x10}.  Linux's existing
// 16-GiB-class PB maximum is exactly Apple's 0x33660000 default, giving five
// 0x1800-page groups and therefore a 0x30-byte per-scene PM slice.
const _: [(); 4] = [(); G15_PM_DEVICE_CONFIG.usage_page_granule];
const _: [(); 0x80000] = [(); G15_J615_TA_OBJECT_PAYLOAD_BYTES];
const _: [(); 0x200] = [(); G15_J615_TA_OBJECT_PAYLOAD_UNITS as usize];
const _: [(); 0x0200] = [();
    (g15_ta_object_payload_reg_1ca48(0, G15_J615_TA_OBJECT_PAYLOAD_UNITS) >> 48) as usize
];
const _: [(); 0x50] = [(); G15_J615_PM_RECORD_COUNT];
const _: [(); 0x24] = [(); G15_J615_PM_SCENE_GROUP_COUNT];
const _: [(); 0x2800] = [(); g15_pm_state_offset(G15_J615_PM_RECORD_COUNT)];
const _: [(); 0x2800] = [(); G15_J615_PM_GPU_RECORD_BYTES];
const _: [(); 0x2840] = [(); g15_pm_record_pool_bytes(G15_J615_PM_RECORD_COUNT)];
const _: [(); 0x280] = [(); g15_pm_fw_page_list_bytes(G15_J615_PM_RECORD_COUNT)];
const _: [(); 0x80] = [(); G15_PM_PAGE_LIST_STATS_BYTES];
const _: [(); 0x40] = [(); G15_PM_SCENE_STATS_OFFSET];
const _: [(); 0x140] = [(); G15_J615_PM_PAGE_METRICS_BYTES];
const _: [(); 0] = [(); g15_j615_pm_page_metrics_slot_offset(0)];
const _: [(); 0x13c] = [(); g15_j615_pm_page_metrics_slot_offset(79)];
const _: [(); 0] = [(); g15_j615_pm_page_metrics_slot_offset(80)];
const _: [(); 0xa0] = [(); (g15_pm_page_metrics_reg_1c910(0xffff_fc20_0000_0000) >> 32) as usize];
const _: [(); 0x4001] = [(); (g15_pm_page_metrics_reg_1c910(0xffff_fc20_0000_4000) & 0xffff) as usize];
const _: [(); 0x30] = [(); g15_j615_pm_scene_stride(0x3366_0000)];
const _: [(); 0x6c0] = [(); G15_J615_PM_SCENE_GROUP_COUNT
    * g15_j615_pm_scene_stride(0x3366_0000)];
const _: [(); 0x6f0] = [(); G15_J615_PM_SCENE_ALLOC_BYTES];
const _: [(); 0x6f0] = [(); (G15_J615_PM_SCENE_GROUP_COUNT
    + G15_J615_PM_EXTRA_SCENE_SLICES)
    * g15_j615_pm_scene_stride(0x3366_0000)];
// PM +0x2c starts at zero, but processRenderSetup() advances before
// descriptor selection, so the first ordinary record is index 1.
const _: [(); 1] = [(); g15_j615_pm_next_record_index(0) as usize];
const _: [(); 0] = [(); g15_j615_pm_next_record_index(0x4f) as usize];
const _: [(); 0x30] = [(); g15_j615_pm_scene_slice_offset(1)];
const _: [(); 0] = [(); g15_j615_pm_scene_slice_offset(36)];
const _: [(); 0x150] = [(); g15_j615_pm_scene_slice_offset(79)];
const _: [(); 0x6c0] = [(); g15_j615_pm_shared_scene_slice_offset()];
const _: [(); 0x80] = [(); g15_j615_pm_record_offset(1)];
const _: [(); 0x2780] = [(); g15_j615_pm_record_offset(79)];
const _: [(); 0] = [(); g15_j615_pm_record_offset(80)];

/// Metadata about the tiling configuration for a scene. This is computed in the `render` module.
/// based on dimensions, tile size, and other info.
pub(crate) struct TileInfo {
    /// Tile count in the X dimension. Tiles are always 32x32.
    pub(crate) tiles_x: u32,
    /// Tile count in the Y dimension. Tiles are always 32x32.
    pub(crate) tiles_y: u32,
    /// Total tile count.
    pub(crate) tiles: u32,
    /// Micro-tile width (16 or 32).
    pub(crate) utile_width: u32,
    /// Micro-tile height (16 or 32).
    pub(crate) utile_height: u32,
    // Macro-tiles in the X dimension. Always 4.
    //pub(crate) mtiles_x: u32,
    // Macro-tiles in the Y dimension. Always 4.
    //pub(crate) mtiles_y: u32,
    /// Tiles per macro-tile in the X dimension.
    pub(crate) tiles_per_mtile_x: u32,
    /// Tiles per macro-tile in the Y dimension.
    pub(crate) tiles_per_mtile_y: u32,
    // Total tiles per macro-tile.
    //pub(crate) tiles_per_mtile: u32,
    /// Micro-tiles per macro-tile in the X dimension.
    pub(crate) utiles_per_mtile_x: u32,
    /// Micro-tiles per macro-tile in the Y dimension.
    pub(crate) utiles_per_mtile_y: u32,
    // Total micro-tiles per macro-tile.
    //pub(crate) utiles_per_mtile: u32,
    /// Size of the top-level tilemap, in bytes (for all layers, one cluster).
    pub(crate) tilemap_size: usize,
    /// Size of the Tail Pointer Cache, in bytes (for all layers * clusters).
    pub(crate) tpc_size: usize,
    /// Number of blocks in the clustering meta buffer (for clustering) per layer.
    pub(crate) meta1_layer_stride: u32,
    /// Number of blocks in the clustering meta buffer (for clustering).
    pub(crate) meta1_blocks: u32,
    /// Layering metadata size.
    pub(crate) layermeta_size: usize,
    /// Minimum number of TVB blocks for this render.
    pub(crate) min_tvb_blocks: usize,
    /// Tiling parameter structure passed to firmware.
    pub(crate) params: fw::vertex::raw::TilingParameters,
}

/// A single scene, representing a render pass and its required buffers.
#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Scene {
    object: GpuObject<buffer::Scene::ver>,
    slot: u32,
    rebind: bool,
    // Host-owned lifetime anchor for the G15 ContextSwitcherGen3 render scratch
    // allocation. Older generations keep this as None.
    #[allow(dead_code)]
    g15_ctxswitch: Option<GpuArray<u8>>,
    preempt2_off: usize,
    preempt3_off: usize,
    // Note: these are dead code only on some version variants.
    // It's easier to do this than to propagate the version conditionals everywhere.
    #[allow(dead_code)]
    meta1_off: usize,
    #[allow(dead_code)]
    meta2_off: usize,
    #[allow(dead_code)]
    meta3_off: usize,
    #[allow(dead_code)]
    meta4_off: usize,
}

#[versions(AGX)]
impl Scene::ver {
    /// Returns true if the buffer was bound to a fresh manager slot, and therefore needs an init
    /// command before a render.
    pub(crate) fn rebind(&self) -> bool {
        self.rebind
    }

    /// Returns the buffer manager slot this scene's buffer was bound to.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    /// Returns the GPU pointer to the [`buffer::Scene::ver`].
    pub(crate) fn gpu_pointer(&self) -> GpuPointer<'_, buffer::Scene::ver> {
        self.object.gpu_pointer()
    }

    /// Returns the GPU weak pointer to the [`buffer::Scene::ver`].
    pub(crate) fn weak_pointer(&self) -> GpuWeakPointer<buffer::Scene::ver> {
        self.object.weak_pointer()
    }

    /// Returns the GPU weak pointer to the kernel-side temp buffer.
    /// (purpose unknown...)
    pub(crate) fn kernel_buffer_pointer(&self) -> GpuWeakPointer<[u8]> {
        self.object.buffer.inner.lock().kernel_buffer.weak_pointer()
    }

    /// Returns the GPU pointer to the `buffer::Info::ver` object associated with this Scene.
    pub(crate) fn buffer_pointer(&self) -> GpuPointer<'_, buffer::Info::ver> {
        // SAFETY: We can't return the strong pointer directly since its lifetime crosses a lock,
        // but we know its lifetime will be valid as long as &self since we hold a reference to the
        // buffer, so just construct the strong pointer with the right lifetime here.
        unsafe { self.weak_buffer_pointer().upgrade() }
    }

    /// Returns the GPU weak pointer to the `buffer::Info::ver` object associated with this Scene.
    pub(crate) fn weak_buffer_pointer(&self) -> GpuWeakPointer<buffer::Info::ver> {
        self.object.buffer.inner.lock().info.weak_pointer()
    }

    /// Returns the GPU pointer to the TVB heap metadata buffer.
    pub(crate) fn tvb_heapmeta_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tvb_heapmeta.gpu_pointer()
    }

    /// Returns the GPU pointer to the layer metadata buffer.
    pub(crate) fn tvb_layermeta_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tvb_heapmeta.gpu_offset_pointer(0x200)
    }

    /// Returns the GPU pointer to the top-level TVB tilemap buffer.
    pub(crate) fn tvb_tilemap_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tvb_tilemap.gpu_pointer()
    }

    /// Returns the GPU pointer to the Tail Pointer Cache buffer.
    pub(crate) fn tpc_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tpc.gpu_pointer()
    }

    /// Returns the GPU pointer to the G15 ContextSwitcherGen3 render scratch region.
    #[allow(dead_code)]
    pub(crate) fn g15_ctxswitch_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.g15_ctxswitch.as_ref().map(|buf| buf.gpu_pointer())
    }

    /// Returns the GPU pointer to the first preemption scratch buffer.
    pub(crate) fn preempt_buf_1_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.preempt_buf.gpu_pointer()
    }

    /// Returns the GPU pointer to the second preemption scratch buffer.
    pub(crate) fn preempt_buf_2_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object
            .preempt_buf
            .gpu_offset_pointer(self.preempt2_off)
    }

    /// Returns the GPU pointer to the third preemption scratch buffer.
    pub(crate) fn preempt_buf_3_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object
            .preempt_buf
            .gpu_offset_pointer(self.preempt3_off)
    }

    /// Returns the GPU pointer to the per-cluster tilemap buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn cluster_tilemaps_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.tilemaps.gpu_pointer())
    }

    /// Returns the GPU pointer to the clustering layer metadata buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn tvb_cluster_layermeta_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_pointer())
    }

    /// Returns the GPU pointer to the clustering metadata 1 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_1_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta1_off))
    }

    /// Returns the GPU pointer to the clustering metadata 2 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_2_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta2_off))
    }

    /// Returns the GPU pointer to the clustering metadata 3 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_3_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta3_off))
    }

    /// Returns the GPU pointer to the clustering metadata 4 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_4_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta4_off))
    }
}

#[versions(AGX)]
impl Drop for Scene::ver {
    fn drop(&mut self) {
        let mut inner = self.object.buffer.inner.lock();
        assert_ne!(inner.active_scenes, 0);
        inner.active_scenes -= 1;

        if inner.active_scenes == 0 {
            mod_pr_debug!(
                "Buffer: no scenes left, dropping slot {}",
                inner.active_slot.take().unwrap().slot()
            );
            inner.active_slot = None;
        }
    }
}

/// Inner data for a single TVB buffer object.
#[versions(AGX)]
struct BufferInner {
    info: GpuObject<buffer::Info::ver>,
    ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
    ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
    blocks: KVec<GpuOnlyArray<u8>>,
    max_blocks: usize,
    max_blocks_nomemless: usize,
    mgr: BufferManager::ver,
    active_scenes: usize,
    active_slot: Option<slotalloc::Guard<BufferSlotInner::ver>>,
    last_token: Option<slotalloc::SlotToken>,
    tpc: Option<Arc<GpuArray<u8>>>,
    kernel_buffer: GpuArray<u8>,
    stats: GpuObject<buffer::Stats>,
    cfg: &'static hw::HwConfig,
    preempt1_size: usize,
    preempt2_size: usize,
    preempt3_size: usize,
    num_clusters: usize,
}

/// Locked and reference counted TVB buffer.
#[versions(AGX)]
pub(crate) struct Buffer {
    inner: Arc<Mutex<BufferInner::ver>>,
}

#[versions(AGX)]
impl Buffer::ver {
    /// Create a new Buffer for a given VM, given the per-VM allocators.
    pub(crate) fn new(
        gpu: &dyn gpu::GpuManager,
        alloc: &mut gpu::KernelAllocators,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        mgr: &BufferManager::ver,
    ) -> Result<Buffer::ver> {
        // These are the typical max numbers on macOS.
        // 8GB machines have this halved.
        let max_size: usize = 862_322_688; // bytes
        let max_size_nomemless = max_size / 3;

        let max_blocks = max_size / BLOCK_SIZE;
        let max_blocks_nomemless = max_size_nomemless / BLOCK_SIZE;
        let max_pages = max_blocks * PAGES_PER_BLOCK;
        #[allow(unused_variables)] // G15's compact Info has no legacy max-pages tail.
        let max_pages_nomemless = max_blocks_nomemless * PAGES_PER_BLOCK;

        let num_clusters = gpu.get_dyncfg().id.num_clusters as usize;
        let num_clusters_adj = if num_clusters > 1 {
            num_clusters + 1
        } else {
            1
        };

        let preempt1_size = num_clusters_adj * gpu.get_cfg().preempt1_size;
        let preempt2_size = num_clusters_adj * gpu.get_cfg().preempt2_size;
        let preempt3_size = num_clusters_adj * gpu.get_cfg().preempt3_size;

        let shared = &mut alloc.shared;
        let info = alloc.private.new_init(
            {
                let ualloc_priv = &ualloc_priv;
                try_init!(buffer::Info::ver {
                    block_ctl: shared.new_default::<buffer::BlockControl>()?,
                    counter: shared.new_default::<buffer::Counter::ver>()?,
                    page_list: ualloc_priv.lock().array_empty_tagged(max_pages, b"PLST")?,
                    block_list: ualloc_priv
                        .lock()
                        .array_empty_tagged(max_blocks * 2, b"BLST")?,
                })
            },
            |inner, _p| {
                try_init!(buffer::raw::Info::ver {
                    gpu_counter: 0x0,
                    unk_4: 0,
                    last_id: 0x0,
                    cur_id: -1,
                    unk_10: 0x0,
                    gpu_counter2: 0x0,
                    unk_18: 0x0,
                    #[ver(V < V13_0B4 || G >= G14X)]
                    unk_1c: 0x0,
                    page_list: inner.page_list.gpu_pointer(),
                    page_list_size: (4 * max_pages).try_into()?,
                    page_count: AtomicU32::new(0),
                    max_blocks: max_blocks.try_into()?,
                    block_count: AtomicU32::new(0),
                    unk_38: 0x0,
                    block_list: inner.block_list.gpu_pointer(),
                    block_ctl: inner.block_ctl.gpu_pointer(),
                    last_page: AtomicU32::new(0),
                    #[ver(G == G15)]
                    g15_unk_4c: 0,
                    #[ver(G == G15)]
                    g15_completion_stat_50: U64(0),
                    #[ver(G == G15)]
                    counter: inner.counter.gpu_pointer(),
                    #[ver(G == G15)]
                    g15_unk_60: U64(0),
                    #[ver(G == G15)]
                    g15_unk_68: U64(0),
                    #[ver(G == G15)]
                    g15_unk_70: 0,
                    #[ver(G == G15)]
                    g15_unk_74: 0,
                    #[ver(G == G15)]
                    g15_unk_78: U64(0),
                    #[ver(G != G15)]
                    gpu_page_ptr1: 0x0,
                    #[ver(G != G15)]
                    gpu_page_ptr2: 0x0,
                    #[ver(G != G15)]
                    unk_58: 0x0,
                    #[ver(G != G15)]
                    block_size: BLOCK_SIZE as u32,
                    #[ver(G != G15)]
                    unk_60: U64(0x0),
                    #[ver(G != G15)]
                    counter: inner.counter.gpu_pointer(),
                    #[ver(G != G15)]
                    unk_70: 0x0,
                    #[ver(G != G15)]
                    unk_74: 0x0,
                    #[ver(G != G15)]
                    unk_78: 0x0,
                    #[ver(G != G15)]
                    unk_7c: 0x0,
                    #[ver(G != G15)]
                    unk_80: 0x1,
                    #[ver(G != G15)]
                    max_pages: max_pages.try_into()?,
                    #[ver(G != G15)]
                    max_pages_nomemless: max_pages_nomemless.try_into()?,
                    #[ver(G != G15)]
                    unk_8c: 0x0,
                    #[ver(G != G15)]
                    unk_90: Default::default(),
                })
            },
        )?;

        // Technically similar to Scene below, let's play it safe.
        let kernel_buffer = alloc.shared.array_empty_tagged(0x40, b"KBUF")?;
        let stats = alloc
            .shared
            .new_object(Default::default(), |_inner| buffer::raw::Stats {
                reset: AtomicU32::from(1),
                ..Default::default()
            })?;

        Ok(Buffer::ver {
            inner: Arc::pin_init(
                new_mutex!(BufferInner::ver {
                    info,
                    ualloc,
                    ualloc_priv,
                    blocks: KVec::new(),
                    max_blocks,
                    max_blocks_nomemless,
                    mgr: mgr.clone(),
                    active_scenes: 0,
                    active_slot: None,
                    last_token: None,
                    tpc: None,
                    kernel_buffer,
                    stats,
                    cfg: gpu.get_cfg(),
                    preempt1_size,
                    preempt2_size,
                    preempt3_size,
                    num_clusters,
                }),
                GFP_KERNEL,
            )?,
        })
    }

    /// Returns the total block count allocated to this Buffer.
    pub(crate) fn block_count(&self) -> u32 {
        self.inner.lock().blocks.len() as u32
    }

    /// Automatically grow the Buffer based on feedback from the statistics.
    pub(crate) fn auto_grow(&self) -> Result<bool> {
        let inner = self.inner.lock();

        let used_pages = inner.stats.with(|raw, _inner| {
            let used = raw.max_pages.load(Ordering::Relaxed);
            raw.reset.store(1, Ordering::Release);
            used as usize
        });

        let need_blocks = (used_pages * 2)
            .div_ceil(PAGES_PER_BLOCK)
            .min(inner.max_blocks_nomemless);
        let want_blocks = (used_pages * 3)
            .div_ceil(PAGES_PER_BLOCK)
            .min(inner.max_blocks_nomemless);

        let cur_count = inner.blocks.len();

        if need_blocks <= cur_count {
            Ok(false)
        } else {
            // Grow to 3x requested size (same logic as macOS)
            core::mem::drop(inner);
            self.ensure_blocks(want_blocks)?;
            Ok(true)
        }
    }

    /// Synchronously grow the Buffer.
    pub(crate) fn sync_grow(&self) {
        let inner = self.inner.lock();

        let cur_count = inner.blocks.len();
        core::mem::drop(inner);
        if self.ensure_blocks(cur_count + 10).is_err() {
            pr_err!("BufferManager: Failed to grow buffer synchronously\n");
        }
    }

    /// Ensure that the buffer has at least a certain minimum size in blocks.
    pub(crate) fn ensure_blocks(&self, min_blocks: usize) -> Result<bool> {
        let mut inner = self.inner.lock();

        let cur_count = inner.blocks.len();
        if cur_count >= min_blocks {
            return Ok(false);
        }
        if min_blocks > inner.max_blocks {
            return Err(ENOMEM);
        }

        let add_blocks = min_blocks - cur_count;
        let new_count = min_blocks;

        let mut new_blocks: KVec<GpuOnlyArray<u8>> = KVec::new();

        // Allocate the new blocks first, so if it fails they will be dropped
        let mut ualloc = inner.ualloc.lock();
        for _i in 0..add_blocks {
            new_blocks.push(ualloc.array_gpuonly(BLOCK_SIZE)?, GFP_KERNEL)?;
        }
        core::mem::drop(ualloc);

        // Then actually commit them
        inner.blocks.reserve(add_blocks, GFP_KERNEL)?;

        for (i, block) in new_blocks.into_iter().enumerate() {
            let page_num = (block.gpu_va().get() >> PAGE_SHIFT) as u32;

            inner
                .blocks
                .push(block, GFP_KERNEL)
                .expect("push() failed after reserve()");
            inner.info.block_list[2 * (cur_count + i)] = page_num;
            for j in 0..PAGES_PER_BLOCK {
                inner.info.page_list[(cur_count + i) * PAGES_PER_BLOCK + j] = page_num + j as u32;
            }
        }

        inner.info.block_ctl.with(|raw, _inner| {
            raw.total.store(new_count as u32, Ordering::SeqCst);
            raw.wptr.store(new_count as u32, Ordering::SeqCst);
        });

        /* Only do this update if the buffer manager is idle (which means we own it) */
        if inner.active_scenes == 0 {
            let page_count = (new_count * PAGES_PER_BLOCK) as u32;
            inner.info.with(|raw, _inner| {
                raw.page_count.store(page_count, Ordering::Relaxed);
                raw.block_count.store(new_count as u32, Ordering::Relaxed);
                raw.last_page.store(page_count - 1, Ordering::Relaxed);
            });
        }

        Ok(true)
    }

    /// Create a new [`Scene::ver`] (render pass) using this buffer.
    pub(crate) fn new_scene(
        &self,
        alloc: &mut gpu::KernelAllocators,
        tile_info: &TileInfo,
    ) -> Result<Scene::ver> {
        let mut inner = self.inner.lock();

        let tilemap_size = tile_info.tilemap_size;
        let tpc_size = tile_info.tpc_size;

        // TODO: what is this exactly?
        mod_pr_debug!("Buffer: Allocating TVB buffers\n");

        // This seems to be a list, with 4x2 bytes of headers and 8 bytes per entry.
        // On single-cluster devices, the used length always seems to be 1.
        // On M1 Ultra, it can grow and usually doesn't exceed 64 entries.
        // macOS allocates a whole 64K * 0x80 for this, so let's go with
        // that to be safe...
        let user_buffer = inner.ualloc.lock().array_empty_tagged(
            if inner.num_clusters > 1 {
                0x10080
            } else {
                0x80
            },
            b"UBUF",
        )?;

        let tvb_heapmeta = inner
            .ualloc
            .lock()
            .array_empty_tagged(0x200 + tile_info.layermeta_size, b"HMTA")?;
        let tvb_tilemap = inner
            .ualloc
            .lock()
            .array_empty_tagged(tilemap_size, b"TMAP")?;

        mod_pr_debug!("Buffer: Allocating misc buffers\n");
        let preempt_buf = inner.ualloc.lock().array_empty_tagged(
            inner.preempt1_size + inner.preempt2_size + inner.preempt3_size,
            b"PRMT",
        )?;

        // ContextSwitcherGen3::setupRenderCommand() allocates exactly 0x8e0
        // bytes for G15 TA context switching. Keep it Scene-owned so both the
        // register list and command body can reference one allocation for the
        // full render-job lifetime.
        #[ver(G == G15)]
        let g15_ctxswitch = inner
            .ualloc
            .lock()
            .array_empty_tagged(G15_RENDER_CTXSWITCH_BYTES, b"CTSW")?;

        let tpc = match inner.tpc.as_ref() {
            Some(buf) if buf.len() >= tpc_size => buf.clone(),
            _ => {
                // MacOS allocates this as shared GPU+FW, but
                // priv seems to work and might be faster?
                // Needs to be FW-writable anyway, so ualloc
                // won't work.
                let buf = Arc::new(
                    inner.ualloc_priv.lock().array_empty_tagged(
                        (tpc_size + mmu::UAT_PGMSK) & !mmu::UAT_PGMSK,
                        b"TPC ",
                    )?,
                    GFP_KERNEL,
                )?;
                inner.tpc = Some(buf.clone());
                buf
            }
        };

        let mut clmeta_size = 0;
        let mut meta1_size = 0;
        let mut meta2_size = 0;
        let mut meta3_size = 0;

        let clustering = if inner.num_clusters > 1 {
            let cfg = inner.cfg.clustering.as_ref().unwrap();

            clmeta_size = tile_info.layermeta_size * cfg.max_splits;
            // Maybe: (4x4 macro tiles + 1 global page)*n, 32bit each (17*4*n)
            // Unused on t602x?
            meta1_size = align(tile_info.meta1_blocks as usize * cfg.meta1_blocksize, 0x80);
            meta2_size = align(cfg.meta2_size, 0x80);
            meta3_size = align(cfg.meta3_size, 0x80);
            let meta4_size = cfg.meta4_size;

            let meta_size = clmeta_size + meta1_size + meta2_size + meta3_size + meta4_size;

            mod_pr_debug!("Buffer: Allocating clustering buffers\n");
            let tilemaps = inner
                .ualloc
                .lock()
                .array_empty_tagged(cfg.max_splits * tilemap_size, b"CTMP")?;
            let meta = inner.ualloc.lock().array_empty_tagged(meta_size, b"CMTA")?;
            Some(buffer::ClusterBuffers { tilemaps, meta })
        } else {
            None
        };

        // Could be made strong, but we wind up with a deadlock if we try to grab the
        // pointer through the inner.buffer path inside the closure.
        let stats_pointer = inner.stats.weak_pointer();

        let _gpu = &mut alloc.gpu;

        // macOS allocates this as private. However, the firmware does not
        // DC CIVAC this before reading it (like it does most other things),
        // which causes odd cache incoherency bugs when combined with
        // speculation on the firmware side (maybe). This doesn't happen
        // on macOS because these structs are a circular pool that is mapped
        // already initialized. Just mark this shared for now.
        let scene = alloc.shared.new_init(
            try_init!(buffer::Scene::ver {
                user_buffer: user_buffer,
                buffer: self.clone(),
                tvb_heapmeta: tvb_heapmeta,
                tvb_tilemap: tvb_tilemap,
                tpc: tpc,
                clustering: clustering,
                preempt_buf: preempt_buf,
                #[ver(G >= G14X || G == G15)]
                control_word: _gpu.array_empty_tagged(1, b"CWRD")?,
            }),
            |inner, _p| {
                try_init!(buffer::raw::Scene::ver {
                    #[ver(G >= G14X || G == G15)]
                    control_word: inner.control_word.gpu_pointer(),
                    #[ver(G >= G14X || G == G15)]
                    control_word2: inner.control_word.gpu_pointer(),
                    pass_page_count: AtomicU32::new(0),
                    unk_4: 0,
                    unk_8: U64(0),
                    unk_10: U64(0),
                    user_buffer: inner.user_buffer.gpu_pointer(),
                    unk_20: 0,
                    #[ver(G == G15)]
                    g15_unk_34: 0,
                    #[ver(V >= V13_3)]
                    unk_28: U64(0),
                    stats: stats_pointer,
                    total_page_count: AtomicU32::new(0),
                    #[ver(G < G14X && G != G15)]
                    unk_30: U64(0),
                    #[ver(G < G14X && G != G15)]
                    unk_38: U64(0),
                    #[ver(G == G15)]
                    g15_tail: Default::default(),
                })
            },
        )?;

        let mut rebind = false;

        if inner.active_slot.is_none() {
            assert_eq!(inner.active_scenes, 0);

            let slot = inner.mgr.0.get_inner(inner.last_token, |inner, mgr| {
                inner.owners[mgr.slot() as usize] = Some(self.clone());
                Ok(())
            })?;
            rebind = slot.changed();

            mod_pr_debug!("Buffer: assigning slot {} (rebind={})", slot.slot(), rebind);

            inner.last_token = Some(slot.token());
            inner.active_slot = Some(slot);
        }

        inner.active_scenes += 1;

        Ok(Scene::ver {
            object: scene,
            slot: inner.active_slot.as_ref().unwrap().slot(),
            rebind,
            #[ver(G == G15)]
            g15_ctxswitch: Some(g15_ctxswitch),
            #[ver(G != G15)]
            g15_ctxswitch: None,
            preempt2_off: inner.preempt1_size,
            preempt3_off: inner.preempt1_size + inner.preempt2_size,
            meta1_off: clmeta_size,
            meta2_off: clmeta_size + meta1_size,
            meta3_off: clmeta_size + meta1_size + meta2_size,
            meta4_off: clmeta_size + meta1_size + meta2_size + meta3_size,
        })
    }

    /// Increment the buffer manager usage count. Should we done once we know the Scene is ready
    /// to be committed and used in commands submitted to the GPU.
    pub(crate) fn increment(&self) {
        let inner = self.inner.lock();
        inner.info.counter.with(|raw, _inner| {
            // We could use fetch_add, but the non-LSE atomic
            // sequence Rust produces confuses the hypervisor.
            // We have inner locked anyway, so this is not racy.
            let v = raw.count.load(Ordering::Relaxed);
            raw.count.store(v + 1, Ordering::Relaxed);
        });
    }

    pub(crate) fn any_ref(&self) -> Arc<dyn core::any::Any + Send + Sync> {
        self.inner.clone()
    }
}

#[versions(AGX)]
impl Clone for Buffer::ver {
    fn clone(&self) -> Self {
        Buffer::ver {
            inner: self.inner.clone(),
        }
    }
}

#[versions(AGX)]
struct BufferSlotInner();

#[versions(AGX)]
impl slotalloc::SlotItem for BufferSlotInner::ver {
    type Data = BufferManagerInner::ver;

    fn release(&mut self, data: &mut Self::Data, slot: u32) {
        mod_pr_debug!("BufferManager: Released slot {}\n", slot);
        data.owners[slot as usize] = None;
    }
}

/// Inner data for the buffer manager, to be protected by the SlotAllocator lock.
#[versions(AGX)]
pub(crate) struct BufferManagerInner {
    owners: KVec<Option<Buffer::ver>>,
}

/// The GPU-global buffer manager, used to allocate and release buffer slots from the pool.
#[versions(AGX)]
pub(crate) struct BufferManager(slotalloc::SlotAllocator<BufferSlotInner::ver>);

#[versions(AGX)]
impl BufferManager::ver {
    pub(crate) fn new() -> Result<BufferManager::ver> {
        let mut owners = KVec::new();
        for _i in 0..(NUM_BUFFERS as usize) {
            owners.push(None, GFP_KERNEL)?;
        }
        Ok(BufferManager::ver(slotalloc::SlotAllocator::new(
            NUM_BUFFERS,
            BufferManagerInner::ver { owners },
            |_inner, _slot| Some(BufferSlotInner::ver()),
            c_str!("BufferManager::SlotAllocator"),
            static_lock_class!(),
            static_lock_class!(),
        )?))
    }

    /// Signals a Buffer to synchronously grow.
    pub(crate) fn grow(&self, slot: u32) {
        match self
            .0
            .with_inner(|inner| inner.owners[slot as usize].as_ref().cloned())
        {
            Some(owner) => {
                pr_err!(
                    "BufferManager: Unexpected grow request for slot {}. This might deadlock. Please report this bug.\n",
                    slot
                );
                owner.sync_grow();
            }
            None => {
                pr_err!(
                    "BufferManager: Received grow request for empty slot {}\n",
                    slot
                );
            }
        }
    }
}

#[versions(AGX)]
impl Clone for BufferManager::ver {
    fn clone(&self) -> Self {
        BufferManager::ver(self.0.clone())
    }
}
