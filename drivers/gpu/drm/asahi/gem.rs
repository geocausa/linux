// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Asahi driver GEM object implementation
//!
//! Basic wrappers and adaptations between generic GEM shmem objects and this driver's
//! view of what a GPU buffer object is. It is in charge of keeping track of all mappings for
//! each GEM object so we can remove them when a client (File) or a Vm are destroyed, as well as
//! implementing RTKit buffers on top of GEM objects for firmware use.

use kernel::{
    bindings,
    container_of,
    drm,
    drm::gem::{
        shmem,
        shmem::VMap,
        BaseObject,
        DriverObject,
        IntoGEMObject, //
    },
    error::Result,
    prelude::*,
    sync::aref::ARef,
    uapi, //
};

use core::ops::Range;
use core::sync::atomic::{
    AtomicU64,
    Ordering, //
};

use crate::{
    debug::*,
    driver::{
        AsahiDevice,
        AsahiDriver, //
    },
    file,
    mmu,
    util::*, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Gem;

/// Represents the inner data of a GEM object for this driver.
#[pin_data]
pub(crate) struct AsahiObject {
    /// ID for debug
    id: u64,
    /// Object creation flags.
    flags: u32,
    /// Whether this object can be exported.
    exportable: bool,
    /// Whether this is a kernel-created object.
    kernel: bool,
}

/// Type alias for the shmem GEM object type for this driver.
pub(crate) type Object = shmem::Object<AsahiObject>;

unsafe impl Send for AsahiObject {}
unsafe impl Sync for AsahiObject {}

// /// Type alias for the SGTable type for this driver.
// pub(crate) type SGTable = shmem::SGTable<AsahiObject>;

/// A shared reference to a GEM object for this driver.
pub(crate) struct ObjectRef {
    /// The underlying GEM object reference
    pub(crate) gem: ARef<Object>,
    /// The kernel-side VMap of this object, if needed
    vmap: Option<VMap<AsahiObject, u8>>,
}

crate::no_debug!(ObjectRef);

static GEM_ID: AtomicU64 = AtomicU64::new(0);

impl ObjectRef {
    /// Create a new wrapper for a raw GEM object reference.
    pub(crate) fn new(gem: ARef<Object>) -> ObjectRef {
        ObjectRef { gem, vmap: None }
    }

    /// Return the `VMap` for this object, creating it if necessary.
    pub(crate) fn vmap(&mut self) -> Result<shmem::VMapRef<'_, AsahiObject, u8>> {
        if self.vmap.is_none() {
            self.vmap = Some(self.gem.owned_vmap()?);
        }
        self.gem.vmap()
    }

    /// Returns the size of an object in bytes
    pub(crate) fn size(&self) -> usize {
        self.gem.size()
    }

    /// Synchronize this kernel shmem object's streaming DMA mapping for CPU access.
    ///
    /// `drm_gem_shmem_get_pages_sgt()` maps the backing with
    /// `DMA_BIDIRECTIONAL`. J615 is non-coherent under Linux, so cached CPU
    /// construction after that map requires an explicit ownership transition.
    pub(crate) fn dma_sync_for_cpu(&self) -> Result {
        let _ = self.gem.sg_table()?;

        // SAFETY: `self.gem` is a DRM shmem object by construction. `sg_table()`
        // above guarantees that `shmem->sgt` exists and is DMA-mapped for this
        // same DRM device. The mapping direction matches the shmem helper.
        unsafe {
            let shmem = container_of!(
                self.gem.as_raw(),
                bindings::drm_gem_shmem_object,
                base
            );
            let sgt = (*shmem).sgt;
            if sgt.is_null() {
                return Err(EIO);
            }
            let nents = i32::try_from((*sgt).orig_nents).map_err(|_| EOVERFLOW)?;
            bindings::__dma_sync_sg_for_cpu(
                self.gem.dev().as_ref().as_raw(),
                (*sgt).sgl,
                nents,
                bindings::dma_data_direction_DMA_BIDIRECTIONAL,
            );
        }
        Ok(())
    }

    /// Synchronize this kernel shmem object's streaming DMA mapping for device access.
    pub(crate) fn dma_sync_for_device(&self) -> Result {
        let _ = self.gem.sg_table()?;

        // SAFETY: See `dma_sync_for_cpu()`. This transfers ownership back to
        // the device before the GPU consumes CPU-authored command/program data.
        unsafe {
            let shmem = container_of!(
                self.gem.as_raw(),
                bindings::drm_gem_shmem_object,
                base
            );
            let sgt = (*shmem).sgt;
            if sgt.is_null() {
                return Err(EIO);
            }
            let nents = i32::try_from((*sgt).orig_nents).map_err(|_| EOVERFLOW)?;
            bindings::__dma_sync_sg_for_device(
                self.gem.dev().as_ref().as_raw(),
                (*sgt).sgl,
                nents,
                bindings::dma_data_direction_DMA_BIDIRECTIONAL,
            );
        }
        Ok(())
    }

    /// Synchronize a byte range of this object's streaming DMA mapping for CPU access.
    pub(crate) fn dma_sync_range_for_cpu(&self, offset: usize, size: usize) -> Result {
        self.dma_sync_range(offset, size, true)
    }

    /// Synchronize a byte range of this object's streaming DMA mapping for device access.
    pub(crate) fn dma_sync_range_for_device(&self, offset: usize, size: usize) -> Result {
        self.dma_sync_range(offset, size, false)
    }

    fn dma_sync_range(&self, offset: usize, size: usize, for_cpu: bool) -> Result {
        if size == 0 {
            return Ok(());
        }
        let end = offset.checked_add(size).ok_or(EOVERFLOW)?;
        if end > self.size() {
            return Err(EINVAL);
        }

        let _ = self.gem.sg_table()?;

        // SAFETY: `self.gem` is a DRM shmem object by construction and `sg_table()`
        // above guarantees a DMA-mapped SG table for this DRM device. Walk only
        // the mapped DMA segments and synchronize the exact overlapping byte
        // ranges, never unrelated allocations sharing the same GEM backing.
        unsafe {
            let shmem = container_of!(
                self.gem.as_raw(),
                bindings::drm_gem_shmem_object,
                base
            );
            let sgt = (*shmem).sgt;
            if sgt.is_null() {
                return Err(EIO);
            }

            let mut sg = (*sgt).sgl;
            let mut logical = 0usize;
            let mut covered = 0usize;
            let nents = (*sgt).nents as usize;

            for _ in 0..nents {
                if sg.is_null() {
                    return Err(EIO);
                }
                let seg_len = bindings::sg_dma_len(sg) as usize;
                let seg_end = logical.checked_add(seg_len).ok_or(EOVERFLOW)?;

                let overlap_start = core::cmp::max(offset, logical);
                let overlap_end = core::cmp::min(end, seg_end);
                if overlap_start < overlap_end {
                    let in_seg = overlap_start - logical;
                    let sync_len = overlap_end - overlap_start;
                    let dma_addr = bindings::sg_dma_address(sg)
                        .checked_add(in_seg as u64)
                        .ok_or(EOVERFLOW)?;
                    if for_cpu {
                        bindings::__dma_sync_single_for_cpu(
                            self.gem.dev().as_ref().as_raw(),
                            dma_addr,
                            sync_len,
                            bindings::dma_data_direction_DMA_BIDIRECTIONAL,
                        );
                    } else {
                        bindings::__dma_sync_single_for_device(
                            self.gem.dev().as_ref().as_raw(),
                            dma_addr,
                            sync_len,
                            bindings::dma_data_direction_DMA_BIDIRECTIONAL,
                        );
                    }
                    covered = covered.checked_add(sync_len).ok_or(EOVERFLOW)?;
                }

                if seg_end >= end {
                    break;
                }
                logical = seg_end;
                sg = bindings::sg_next(sg);
            }

            if covered != size {
                return Err(EIO);
            }
        }
        Ok(())
    }

    /// Maps an object into a given `Vm` at any free address within a given range.
    pub(crate) fn map_into_range(
        &mut self,
        vm: &crate::mmu::Vm,
        range: Range<u64>,
        alignment: u64,
        prot: mmu::Prot,
        guard: bool,
    ) -> Result<crate::mmu::KernelMapping> {
        // Only used for kernel objects now
        if !self.gem.kernel {
            return Err(EINVAL);
        }
        vm.map_in_range(&self.gem, 0..self.gem.size(), alignment, range, prot, guard)
    }

    /// Maps a range within an object into a given `Vm` at any free address within a given range.
    pub(crate) fn map_range_into_range(
        &mut self,
        vm: &crate::mmu::Vm,
        obj_range: Range<usize>,
        range: Range<u64>,
        alignment: u64,
        prot: mmu::Prot,
        guard: bool,
    ) -> Result<crate::mmu::KernelMapping> {
        if obj_range.end > self.gem.size() {
            return Err(EINVAL);
        }
        if self.gem.flags & uapi::drm_asahi_gem_flags_DRM_ASAHI_GEM_VM_PRIVATE != 0
            && vm.is_extobj(&*self.gem)
        {
            return Err(EINVAL);
        }
        vm.map_in_range(&self.gem, obj_range, alignment, range, prot, guard)
    }

    /// Maps an object into a given `Vm` at a specific address.
    ///
    /// Returns Err(ENOSPC) if the requested address is already busy.
    pub(crate) fn map_at(
        &mut self,
        vm: &crate::mmu::Vm,
        addr: u64,
        prot: mmu::Prot,
        guard: bool,
    ) -> Result<crate::mmu::KernelMapping> {
        if self.gem.flags & uapi::drm_asahi_gem_flags_DRM_ASAHI_GEM_VM_PRIVATE != 0
            && vm.is_extobj(&*self.gem)
        {
            return Err(EINVAL);
        }

        vm.map_at(addr, self.gem.size(), self.gem.clone(), prot, guard)
    }
}

pub(crate) struct AsahiObjConfig {
    flags: u32,
    exportable: bool,
    kernel: bool,
}

/// Create a new kernel-owned GEM object.
pub(crate) fn new_kernel_object(dev: &AsahiDevice, size: usize) -> Result<ObjectRef> {
    let gem = shmem::Object::<AsahiObject>::new(
        dev,
        align(size, mmu::UAT_PGSZ),
        shmem::ObjectConfig::<AsahiObject> {
            map_wc: false,
            parent_resv_obj: None,
        },
        AsahiObjConfig {
            flags: 0,
            exportable: false,
            kernel: true,
        },
    )?;

    mod_pr_debug!("AsahiObject new kernel object id={}\n", gem.id);
    Ok(ObjectRef::new(gem))
}

/// Create a new user-owned GEM object with the given flags.
pub(crate) fn new_object(
    dev: &AsahiDevice,
    size: usize,
    flags: u32,
    parent_object: Option<&shmem::Object<AsahiObject>>,
) -> Result<ARef<Object>> {
    if (flags & uapi::drm_asahi_gem_flags_DRM_ASAHI_GEM_VM_PRIVATE != 0) != parent_object.is_some()
    {
        return Err(EINVAL);
    }

    let gem = shmem::Object::<AsahiObject>::new(
        dev,
        align(size, mmu::UAT_PGSZ),
        shmem::ObjectConfig::<AsahiObject> {
            map_wc: flags & uapi::drm_asahi_gem_flags_DRM_ASAHI_GEM_WRITEBACK == 0,
            parent_resv_obj: parent_object,
        },
        AsahiObjConfig {
            flags,
            exportable: parent_object.is_none(),
            kernel: false,
        },
    )?;

    mod_pr_debug!("AsahiObject new user object: id={}\n", gem.id);
    Ok(gem)
}

#[vtable]
impl DriverObject for AsahiObject {
    type Driver = AsahiDriver;
    type Args = AsahiObjConfig;

    const HAS_EXPORT: bool = true;

    /// Callback to create the inner data of a GEM object
    fn new(_dev: &AsahiDevice, _size: usize, args: Self::Args) -> impl PinInit<Self, Error> {
        let id = GEM_ID.fetch_add(1, Ordering::Relaxed);
        mod_pr_debug!("AsahiObject::new id={}\n", id);
        try_pin_init!(AsahiObject {
            id,
            flags: args.flags,
            exportable: args.exportable,
            kernel: args.kernel,
        })
    }

    /// Callback to drop all mappings for a GEM object owned by a given `File`
    fn close(obj: &<Self::Driver as drm::Driver>::Object, file: &drm::gem::DriverFile<Self>) {
        // fn close(obj: &Object, file: &DrmFile) {
        mod_pr_debug!("AsahiObject::close id={}\n", obj.id);
        if file::File::unbind_gem_object(file, obj).is_err() {
            pr_err!("AsahiObject::close: Failed to unbind GEM object\n");
        }
    }

    /// Optional handle for exporting a gem object.
    fn export(
        obj: &<Self::Driver as drm::Driver>::Object,
        flags: u32,
    ) -> Result<drm::gem::DmaBuf<Object>> {
        if !obj.exportable {
            return Err(EINVAL);
        }

        obj.prime_export(flags)
    }
}
