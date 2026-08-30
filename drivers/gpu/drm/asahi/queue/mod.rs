// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Submission queue management
//!
//! This module implements the userspace view of submission queues and the logic to map userspace
//! submissions to firmware queues.

use kernel::dma_fence::*;
use kernel::prelude::*;
use kernel::{
    c_str,
    dma_fence,
    drm::sched,
    macros::versions,
    sync::{
        Arc,
        LockClassKey,
        Mutex, //
    },
    uapi,
    xarray, //
};

use crate::alloc::Allocator;
use crate::debug::*;
use crate::driver::{AsahiDevRef, AsahiDevice};
use crate::file::MAX_COMMANDS_PER_SUBMISSION;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::inner_weak_ptr;
use crate::microseq;
use crate::module_parameters;
use crate::util::{
    AnyBitPattern,
    Reader, //
};
use crate::{
    alloc,
    buffer,
    channel,
    event,
    file,
    fw,
    gpu,
    mmu,
    workqueue, //
};

use core::sync::atomic::{
    AtomicU64,
    Ordering, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Queue;

const WQ_SIZE: u32 = 0x500;

mod common;
mod compute;
mod render;

/// Trait implemented by all versioned queues.
pub(crate) trait Queue: Send + Sync {
    /// Publish this queue's VM in a UAT user slot without constructing or
    /// submitting any GPU work. This is used only by the bounded G15 bring-up
    /// gate immediately before the real submission path.
    fn preflight_vm_bind(&mut self) -> Result<u32>;

    fn submit(
        &mut self,
        id: u64,
        syncs: KVec<file::SyncItem>,
        in_sync_count: usize,
        cmdbuf_raw: &[u8],
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
    ) -> Result;
}

#[versions(AGX)]
struct SubQueue {
    wq: Arc<workqueue::WorkQueue::ver>,
}

#[versions(AGX)]
impl SubQueue::ver {
    fn new_job(&mut self, fence: dma_fence::Fence) -> SubQueueJob::ver {
        SubQueueJob::ver {
            wq: self.wq.clone(),
            fence: Some(fence),
            job: None,
        }
    }
}

#[versions(AGX)]
struct SubQueueJob {
    wq: Arc<workqueue::WorkQueue::ver>,
    job: Option<workqueue::Job::ver>,
    fence: Option<dma_fence::Fence>,
}

#[versions(AGX)]
impl SubQueueJob::ver {
    fn get(&mut self) -> Result<&mut workqueue::Job::ver> {
        if self.job.is_none() {
            mod_pr_debug!("SubQueueJob: Creating {:?} job\n", self.wq.pipe_type());
            self.job
                .replace(self.wq.new_job(self.fence.take().unwrap())?);
        }
        Ok(self.job.as_mut().expect("expected a Job"))
    }

    fn commit(&mut self) -> Result {
        match self.job.as_mut() {
            Some(job) => job.commit(),
            None => Ok(()),
        }
    }

    fn can_submit(&self) -> Option<Fence> {
        self.job.as_ref().and_then(|job| job.can_submit())
    }
}

/// Compile-only owner for exact 23J220 G15 UMA HWMetrics.
///
/// E097 proves one page-base 0x4000-byte range-7 mapping using compact UAT
/// option 0x30b / leaf 0x00e000000000040b. The backing is zeroed once at
/// channel construction and contains 0x100 records of 0x40 bytes. Apple keeps
/// this mapping prepared for the channel lifetime. This owner remains
/// unreachable and exposes no FWVA to RunCompute.
#[allow(dead_code)]
struct G15HWMetricsBacking {
    _page: alloc::G15SharedGpuArray<u8>,
    record_offset: u32,
}

#[allow(dead_code)]
impl G15HWMetricsBacking {
    const PAGE_BYTES: usize = 0x4000;
    const RECORD_BYTES: u32 = 0x40;
    const RECORD_COUNT: usize = Self::PAGE_BYTES / Self::RECORD_BYTES as usize;

    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut metrics_alloc = alloc::G15SharedBank1Allocator::new_range7_hwmetrics(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let page = metrics_alloc.array_empty_shared_data::<u8>(Self::PAGE_BYTES)?;
        if page.len() != Self::PAGE_BYTES
            || Self::PAGE_BYTES != mmu::UAT_PGSZ
            || Self::RECORD_COUNT != 0x100
        {
            return Err(EIO);
        }
        Ok(Self {
            _page: page,
            record_offset: 0,
        })
    }

    /// Exact host ring update from AGXUMAFList::updateSubmitInfo(): return the
    /// current record offset, then advance by 0x40 modulo one 0x4000 page.
    fn take_record_offset(&mut self) -> u32 {
        let current = self.record_offset;
        self.record_offset = (self.record_offset + Self::RECORD_BYTES) % Self::PAGE_BYTES as u32;
        current
    }

    fn take_record_fwva(&mut self) -> Result<u64> {
        let base: u64 = self._page.weak_pointer().into();
        let offset = self.take_record_offset() as u64;
        base.checked_add(offset).ok_or(EOVERFLOW)
    }
}

const _: [(); 0x100] = [(); G15HWMetricsBacking::RECORD_COUNT];

/// One exact 23J220 `_AGFITimeStampQueue` range-7 backing block.
///
/// E125 proves firmware allocates one page-rounded 0x4000 block containing
/// 0x2aa complete 0x18-byte timestamp states. The global resource stack owns
/// selection/release; this compile-only block has no externally reachable
/// selected-state accessor.
#[allow(dead_code)]
struct G15TimestampQueueBackingBlock {
    block: alloc::G15SharedGpuArray<u8>,
}

/// Private proof token for one caller-selected local timestamp state. Its FWVA
/// is usable only by other definition-only exact-host models; there is no live
/// QueueInfo/RunCompute conversion. The local index is deliberately not claimed
/// to equal Apple's global firmware resource-stack index.
#[derive(Debug)]
#[allow(dead_code)]
struct G15PreparedTimestampQueueState {
    slot_index: usize,
    fwva: u64,
}

#[allow(dead_code)]
impl G15TimestampQueueBackingBlock {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut allocator = alloc::G15SharedBank1Allocator::new_range7_timestamp_queue(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let block = allocator.array_empty_shared_data::<u8>(
            fw::workqueue::G15_TIMESTAMP_QUEUE_BACKING_BYTES,
        )?;
        let base: u64 = block.weak_pointer().into();
        if block.len() != fw::workqueue::G15_TIMESTAMP_QUEUE_BACKING_BYTES
            || base == 0
            || base & (mmu::UAT_PGSZ as u64 - 1) != 0
            || fw::workqueue::G15_TIMESTAMP_QUEUE_STATES_PER_BACKING != 0x2aa
        {
            return Err(EIO);
        }
        Ok(Self { block })
    }

    const fn slot_offset(index: usize) -> Option<usize> {
        if index < fw::workqueue::G15_TIMESTAMP_QUEUE_STATES_PER_BACKING {
            Some(index * fw::workqueue::G15_TIMESTAMP_QUEUE_STATE_BYTES)
        } else {
            None
        }
    }

    fn put_u32(slot: &mut [u8], offset: usize, value: u32) -> Result {
        let end = offset.checked_add(4).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn put_u64(slot: &mut [u8], offset: usize, value: u64) -> Result {
        let end = offset.checked_add(8).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Exact `AGXTimeStampQueue::resetTimeStampQueueState()` image for one
    /// caller-selected local state. The routine clears all 0x18 bytes, writes
    /// its own translated GPUVA/FWVA at +0x08, then writes +0x10 as the
    /// `(mode == 2)` flag while +0x14 remains zero. ChinookV9 conversion is
    /// identity for this standard FW range. No global stack index is inferred.
    fn reset_selected(
        &mut self,
        slot_index: usize,
        update_mode_2: bool,
    ) -> Result<G15PreparedTimestampQueueState> {
        let offset = Self::slot_offset(slot_index).ok_or(EINVAL)?;
        let end = offset
            .checked_add(fw::workqueue::G15_TIMESTAMP_QUEUE_STATE_BYTES)
            .ok_or(EOVERFLOW)?;
        let base: u64 = self.block.weak_pointer().into();
        let fwva = base.checked_add(offset as u64).ok_or(EOVERFLOW)?;
        if fwva == 0 {
            return Err(EIO);
        }
        let slot = self.block.as_mut_slice().get_mut(offset..end).ok_or(EIO)?;
        for byte in slot.iter_mut() {
            *byte = 0;
        }
        Self::put_u64(slot, 0x08, fwva)?;
        Self::put_u32(slot, 0x10, update_mode_2 as u32)?;

        Ok(G15PreparedTimestampQueueState { slot_index, fwva })
    }
}

const _: [(); 0x0000] = [(); G15TimestampQueueBackingBlock::slot_offset(0).unwrap()];
const _: [(); 0x0018] = [(); G15TimestampQueueBackingBlock::slot_offset(1).unwrap()];
const _: [(); 0x3fd8] = [(); G15TimestampQueueBackingBlock::slot_offset(0x2a9).unwrap()];

/// One exact 23J220 `_AGFISchedulerState` range-8 backing block.
///
/// E126 proves one page-rounded 0x4000 block contains exactly 0x100 complete
/// 0x40-byte states. The global firmware resource stack owns selection/release;
/// this compile-only block has no externally reachable selected-state accessor.
#[allow(dead_code)]
struct G15SchedulerStateBackingBlock {
    block: alloc::G15SharedGpuArray<u8>,
}

/// Private proof token for one caller-selected local scheduler state. The FWVA
/// is consumed only by definition-only channel-state reconstruction; the local
/// index is deliberately not claimed to equal Apple's global stack index.
#[derive(Debug)]
#[allow(dead_code)]
struct G15PreparedSchedulerState {
    slot_index: usize,
    fwva: u64,
}

#[allow(dead_code)]
impl G15SchedulerStateBackingBlock {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut allocator = alloc::G15SharedBank1Allocator::new_range8_scheduler_state(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let block = allocator.array_empty_shared_data::<u8>(
            fw::workqueue::G15_SCHEDULER_STATE_BACKING_BYTES,
        )?;
        let base: u64 = block.weak_pointer().into();
        if block.len() != fw::workqueue::G15_SCHEDULER_STATE_BACKING_BYTES
            || base == 0
            || base & (mmu::UAT_PGSZ as u64 - 1) != 0
            || fw::workqueue::G15_SCHEDULER_STATES_PER_BACKING != 0x100
        {
            return Err(EIO);
        }
        Ok(Self { block })
    }

    const fn slot_offset(index: usize) -> Option<usize> {
        if index < fw::workqueue::G15_SCHEDULER_STATES_PER_BACKING {
            Some(index * fw::workqueue::G15_SCHEDULER_STATE_BYTES)
        } else {
            None
        }
    }

    fn put_u16(slot: &mut [u8], offset: usize, value: u16) -> Result {
        let end = offset.checked_add(2).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn put_u32(slot: &mut [u8], offset: usize, value: u32) -> Result {
        let end = offset.checked_add(4).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Exact normal-J615 `AGXCommandQueue::init()` host reset of one already
    /// selected scheduler state. Apple clears exactly the first 0x38 bytes,
    /// then writes +0x00/+0x01=0xff, +0x05=1, +0x22=0xff, zero at +0x23..26,
    /// and AGXShared+0x100 (=2) at +0x27. Bytes +0x38..0x3f are deliberately
    /// preserved because the exact host constructor does not clear them.
    fn reset_selected_j615(
        &mut self,
        slot_index: usize,
    ) -> Result<G15PreparedSchedulerState> {
        let offset = Self::slot_offset(slot_index).ok_or(EINVAL)?;
        let end = offset
            .checked_add(fw::workqueue::G15_SCHEDULER_STATE_BYTES)
            .ok_or(EOVERFLOW)?;
        let base: u64 = self.block.weak_pointer().into();
        let fwva = base.checked_add(offset as u64).ok_or(EOVERFLOW)?;
        if fwva == 0 {
            return Err(EIO);
        }
        let slot = self.block.as_mut_slice().get_mut(offset..end).ok_or(EIO)?;
        let reset = slot
            .get_mut(..fw::workqueue::G15_SCHEDULER_STATE_HOST_RESET_BYTES)
            .ok_or(EIO)?;
        for byte in reset.iter_mut() {
            *byte = 0;
        }
        Self::put_u16(slot, 0x00, u16::MAX)?;
        slot[0x05] = 1;
        slot[0x22] = 0xff;
        Self::put_u32(slot, 0x23, 0)?;
        slot[0x27] = fw::workqueue::G15_J615_SCHEDULER_SHARED_BYTE_27;

        Ok(G15PreparedSchedulerState { slot_index, fwva })
    }
}

const _: [(); 0x0000] = [(); G15SchedulerStateBackingBlock::slot_offset(0).unwrap()];
const _: [(); 0x0040] = [(); G15SchedulerStateBackingBlock::slot_offset(1).unwrap()];
const _: [(); 0x3fc0] = [(); G15SchedulerStateBackingBlock::slot_offset(0xff).unwrap()];

/// One exact 23J220 firmware `_AGFIChannelState` backing block.
///
/// E116 proves the global firmware resource stack allocates page-base 0x8000
/// blocks in the already-proven special range-8 class and slices each block
/// into three 0x24c0-byte channel states. This compile-only owner deliberately
/// does not select a slot, expose a slot FWVA, or initialize QueueInfo; those
/// are separate channel-lifetime gates.
#[allow(dead_code)]
struct G15ChannelStateBackingBlock {
    block: alloc::G15SharedGpuArray<u8>,
}

/// Exact host inputs for rebuilding the first 0xb0 bytes of one selected G15
/// `_AGFIChannelState` after Apple's full 0x24c0 reset. These are deliberately
/// FW addresses/runtime values, not borrowed Linux QueueInfo objects: E118/E119
/// close the byte image while the eventual live owner bridges remain gated.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct G15ChannelStateResetInputs {
    uncached_channel_fwva: u64,
    cached_channel_fwva: u64,
    timestamp_queue_state_fwva: u64,
    channel_4c_value: u32,
    scheduler_state_fwva: u64,
    effective_priority: u32,
    queue_qos: u32,
}

/// Private proof token for one reset/prioritized channel-state slot. It has no
/// conversion to a SKU input or RunCompute field and no external call site.
#[derive(Debug)]
#[allow(dead_code)]
struct G15PreparedChannelState {
    slot_index: usize,
    fwva: u64,
}

#[allow(dead_code)]
impl G15ChannelStateBackingBlock {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut state_alloc = alloc::G15SharedBank1Allocator::new_range8(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let block = state_alloc.array_empty_shared_data::<u8>(
            fw::workqueue::G15_CHANNEL_STATE_BACKING_BYTES,
        )?;
        let base: u64 = block.weak_pointer().into();
        if block.len() != fw::workqueue::G15_CHANNEL_STATE_BACKING_BYTES
            || base == 0
            || base & (mmu::UAT_PGSZ as u64 - 1) != 0
            || fw::workqueue::G15_CHANNEL_STATE_SLOTS_PER_BACKING != 3
        {
            return Err(EIO);
        }
        Ok(Self { block })
    }

    /// Pure geometry only: no FWVA is returned from the owner.
    const fn slot_offset(index: usize) -> Option<usize> {
        if index < fw::workqueue::G15_CHANNEL_STATE_SLOTS_PER_BACKING {
            Some(index * fw::workqueue::G15_CHANNEL_STATE_BYTES)
        } else {
            None
        }
    }

    fn put_u32(slot: &mut [u8], offset: usize, value: u32) -> Result {
        let end = offset.checked_add(4).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn put_u64(slot: &mut [u8], offset: usize, value: u64) -> Result {
        let end = offset.checked_add(8).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Rebuild one already-selected channel state at the exact E118/E119 host
    /// boundary. This models neither resource-stack allocation/release nor a
    /// live channel. It clears exactly one 0x24c0 slot, writes Apple's reset
    /// QueueInfo image, then applies the exact normal-J615 priority mutation.
    fn reset_selected_j615_cl(
        &mut self,
        slot_index: usize,
        input: G15ChannelStateResetInputs,
    ) -> Result<G15PreparedChannelState> {
        if input.uncached_channel_fwva == 0
            || input.cached_channel_fwva == 0
            || input.timestamp_queue_state_fwva == 0
            || input.scheduler_state_fwva == 0
        {
            return Err(EINVAL);
        }
        let priority = fw::workqueue::g15_j615_cl_priority_image(
            input.effective_priority,
            input.queue_qos,
        )
        .ok_or(EINVAL)?;
        let offset = Self::slot_offset(slot_index).ok_or(EINVAL)?;
        let base: u64 = self.block.weak_pointer().into();
        let fwva = base.checked_add(offset as u64).ok_or(EOVERFLOW)?;
        let gpu_buf_fwva = fwva
            .checked_add(fw::workqueue::G15_CHANNEL_STATE_GPU_BUF_OFFSET as u64)
            .ok_or(EOVERFLOW)?;
        if fwva == 0 || gpu_buf_fwva == 0 {
            return Err(EIO);
        }
        let end = offset
            .checked_add(fw::workqueue::G15_CHANNEL_STATE_BYTES)
            .ok_or(EOVERFLOW)?;
        let slot = self.block.as_mut_slice().get_mut(offset..end).ok_or(EIO)?;

        // Exact resetChannelState(): bulk zero selected 0x24c0 state first.
        for byte in slot.iter_mut() {
            *byte = 0;
        }
        // E122/E127: QueueInfo +0x00/+0x08 are the selected uncached/cached
        // channel-memory FWVAs, not independent external QueueInfo objects.
        Self::put_u64(slot, 0x00, input.uncached_channel_fwva)?;
        Self::put_u64(slot, 0x08, input.cached_channel_fwva)?;
        // E125: exact G15 QueueInfo +0x10 is the selected timestamp-queue
        // state FWVA; the inherited queue-wide notifier semantic is wrong here.
        Self::put_u64(slot, 0x10, input.timestamp_queue_state_fwva)?;
        // E121: AGXChannel::init() derives channel +0x88 from selected-state
        // GPUVA +0xb0; resetChannelState() then publishes that value at +0x18.
        Self::put_u64(slot, 0x18, gpu_buf_fwva)?;
        Self::put_u32(slot, 0x2c, u32::MAX)?;
        Self::put_u32(slot, 0x30, 4)?;
        Self::put_u32(slot, 0x4c, u32::MAX)?;
        Self::put_u32(slot, 0x50, input.channel_4c_value)?;
        // E126: exact G15 QueueInfo +0xa4 is selected scheduler-state FWVA.
        Self::put_u64(slot, 0xa4, input.scheduler_state_fwva)?;
        slot[0xac] = fw::workqueue::G15_J615_CDM_BACKOFF_TIMEOUT;

        // Exact later setChannelPriority() mutation. +0x30/+0x34 carry the
        // same class; +0x44 is the E119 constant 2. All other reset bytes stay
        // untouched/zero unless explicitly written here.
        Self::put_u32(slot, 0x30, priority.class_30)?;
        Self::put_u32(slot, 0x34, priority.class_30)?;
        Self::put_u64(slot, 0x38, priority.mask_38)?;
        Self::put_u32(slot, 0x40, priority.control_40)?;
        Self::put_u32(slot, 0x44, priority.integer_arg_44)?;
        Self::put_u32(slot, 0x48, priority.qos_value_48)?;

        Ok(G15PreparedChannelState { slot_index, fwva })
    }
}

const _: [(); 0x0000] = [(); G15ChannelStateBackingBlock::slot_offset(0).unwrap()];
const _: [(); 0x24c0] = [(); G15ChannelStateBackingBlock::slot_offset(1).unwrap()];
const _: [(); 0x4980] = [(); G15ChannelStateBackingBlock::slot_offset(2).unwrap()];

/// One exact 23J220 `AGXUncachedFWChannelMem` resource-stack backing block.
/// E122 proves this is a normal range-7 0x8000-byte allocation containing
/// three 0x2860-byte elements. Selection/index lifetime is deliberately not
/// modeled here and no element FWVA accessor exists.
#[allow(dead_code)]
struct G15UncachedChannelMemoryBackingBlock {
    block: alloc::G15SharedGpuArray<u8>,
}

/// Private E127 token for one caller-selected uncached channel-memory element.
/// It carries the locally derived FWVA only inside the unreachable owner graph;
/// the local index is not claimed to equal Apple's global resource-stack index.
#[derive(Debug)]
#[allow(dead_code)]
struct G15PreparedUncachedChannelMemory {
    slot_index: usize,
    fwva: u64,
}

#[allow(dead_code)]
impl G15UncachedChannelMemoryBackingBlock {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut allocator = alloc::G15SharedBank1Allocator::new_range7_channel_memory(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let block = allocator.array_empty_shared_data::<u8>(
            fw::workqueue::G15_J615_CHANNEL_MEMORY_BACKING_BYTES,
        )?;
        let base: u64 = block.weak_pointer().into();
        if block.len() != fw::workqueue::G15_J615_CHANNEL_MEMORY_BACKING_BYTES
            || base == 0
            || base & (mmu::UAT_PGSZ as u64 - 1) != 0
            || fw::workqueue::G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING != 3
        {
            return Err(EIO);
        }
        Ok(Self { block })
    }

    fn put_u32(slot: &mut [u8], offset: usize, value: u32) -> Result {
        let end = offset.checked_add(4).ok_or(EOVERFLOW)?;
        let dst = slot.get_mut(offset..end).ok_or(EINVAL)?;
        dst.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Apply only the exact E124 `resetChannelState()` writes to one already
    /// selected local uncached-memory element. Apple does not bulk-clear this
    /// element in resetChannelState(), so this helper deliberately touches only
    /// the six proven u32 header locations. E127 additionally returns the
    /// locally derived FWVA only to the unreachable combined owner graph.
    fn reset_selected_j615_cl(
        &mut self,
        slot_index: usize,
    ) -> Result<G15PreparedUncachedChannelMemory> {
        let offset = g15_j615_channel_memory_slot_offset(slot_index).ok_or(EINVAL)?;
        let end = offset
            .checked_add(fw::workqueue::G15_J615_CHANNEL_MEMORY_BYTES)
            .ok_or(EOVERFLOW)?;
        let base: u64 = self.block.weak_pointer().into();
        let fwva = base.checked_add(offset as u64).ok_or(EOVERFLOW)?;
        if fwva == 0 {
            return Err(EIO);
        }
        let slot = self.block.as_mut_slice().get_mut(offset..end).ok_or(EIO)?;

        Self::put_u32(slot, 0x00, 0)?;
        Self::put_u32(slot, 0x10, 0)?;
        Self::put_u32(slot, 0x20, 0)?;
        Self::put_u32(slot, 0x30, 0)?;
        Self::put_u32(slot, 0x40, 0)?;
        Self::put_u32(
            slot,
            0x50,
            fw::workqueue::G15_J615_CL_UNCACHED_CHANNEL_VALUE_50,
        )?;

        Ok(G15PreparedUncachedChannelMemory { slot_index, fwva })
    }
}

/// One exact 23J220 `AGXCachedFWChannelMem` resource-stack backing block.
/// It has the same element/block geometry as the uncached stack but uses the
/// independently proven special range-8 class. E127 permits only a private
/// local selection token; live channel publication remains absent.
#[allow(dead_code)]
struct G15CachedChannelMemoryBackingBlock {
    block: alloc::G15SharedGpuArray<u8>,
}

/// Private E127 token for one caller-selected cached channel-memory element.
/// resetChannelState() does not modify this object; only its exact selected
/// FWVA is needed for QueueInfo +0x08 inside the dormant channel image.
#[derive(Debug)]
#[allow(dead_code)]
struct G15PreparedCachedChannelMemory {
    slot_index: usize,
    fwva: u64,
}

#[allow(dead_code)]
impl G15CachedChannelMemoryBackingBlock {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        let mut allocator = alloc::G15SharedBank1Allocator::new_range8_channel_memory(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let block = allocator.array_empty_shared_data::<u8>(
            fw::workqueue::G15_J615_CHANNEL_MEMORY_BACKING_BYTES,
        )?;
        let base: u64 = block.weak_pointer().into();
        if block.len() != fw::workqueue::G15_J615_CHANNEL_MEMORY_BACKING_BYTES
            || base == 0
            || base & (mmu::UAT_PGSZ as u64 - 1) != 0
            || fw::workqueue::G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING != 3
        {
            return Err(EIO);
        }
        Ok(Self { block })
    }

    fn select_local(&self, slot_index: usize) -> Result<G15PreparedCachedChannelMemory> {
        let offset = g15_j615_channel_memory_slot_offset(slot_index).ok_or(EINVAL)?;
        let base: u64 = self.block.weak_pointer().into();
        let fwva = base.checked_add(offset as u64).ok_or(EOVERFLOW)?;
        if fwva == 0 {
            return Err(EIO);
        }
        Ok(G15PreparedCachedChannelMemory { slot_index, fwva })
    }
}

/// Pure E122 geometry helper used only by compile-time assertions. It does not
/// expose an address from either backing owner.
const fn g15_j615_channel_memory_slot_offset(index: usize) -> Option<usize> {
    if index < fw::workqueue::G15_J615_CHANNEL_MEMORY_SLOTS_PER_BACKING {
        Some(index * fw::workqueue::G15_J615_CHANNEL_MEMORY_BYTES)
    } else {
        None
    }
}

const _: [(); 0x0000] = [(); g15_j615_channel_memory_slot_offset(0).unwrap()];
const _: [(); 0x2860] = [(); g15_j615_channel_memory_slot_offset(1).unwrap()];
const _: [(); 0x50c0] = [(); g15_j615_channel_memory_slot_offset(2).unwrap()];

/// Persistent exact 23J220 CL-channel command-resource backing used by the
/// stock-empty Compute SKU stream. E114 proves J615 owns one normal option-3
/// eGartRange-5 resource of logical size 0x1f400. The already-proven range-5
/// option-3 class is Linux's dedicated uncached range-5 allocator.
///
/// This owner is reachable only through the definition-only stock-empty owner
/// graph below. It does not publish its FWVA to a live command.
#[derive(Debug)]
#[allow(dead_code)]
struct G15ClCommandResourceBacking {
    backing: GpuArray<u8>,
}

#[allow(dead_code)]
impl G15ClCommandResourceBacking {
    fn new(range5_uncached_alloc: &mut alloc::DefaultAllocator) -> Result<Self> {
        let backing = range5_uncached_alloc.array_empty_tagged::<u8>(
            fw::compute::G15_J615_CL_COMMAND_RESOURCE_BYTES,
            b"CLCR",
        )?;
        let base: u64 = backing.weak_pointer().into();
        if backing.len() != fw::compute::G15_J615_CL_COMMAND_RESOURCE_BYTES
            || base == 0
            || base & 0x3ff != 0
        {
            return Err(EIO);
        }
        Ok(Self { backing })
    }

    fn base_fwva(&self) -> Result<u64> {
        let base: u64 = self.backing.weak_pointer().into();
        if base == 0 || base & 0x3ff != 0 {
            return Err(EIO);
        }
        Ok(base)
    }
}

/// Definition-only E127 selection inputs for one coherent local CL-channel
/// image. Every firmware resource stack keeps its own index; callers must pass
/// them independently. Runtime queue values that are not yet bridged from the
/// live constructor remain explicit rather than being guessed.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct G15ChannelPrepareInputs {
    timestamp_queue_slot: usize,
    scheduler_state_slot: usize,
    channel_state_slot: usize,
    uncached_channel_slot: usize,
    cached_channel_slot: usize,
    timestamp_update_mode_2: bool,
    channel_4c_value: u32,
    effective_priority: u32,
    queue_qos: u32,
}

/// First phase of the E112 two-phase transaction. The rotating slots and
/// command-independent UMA/HWMetrics assets are known, but the SKU bytes have
/// not been serialized or written because the RunCompute FWVA is not known yet.
#[derive(Debug)]
#[allow(dead_code)]
struct G15UnpublishedStockEmptyPrepare {
    event_index: usize,
    event_control_fwva: u64,
    sku: fw::compute::G15ReservedSkuSlot,
    channel_state: G15PreparedChannelState,
    page_pool_state_fwva: u64,
    hwmetrics_fwva: u64,
    hardware_buffer_id: u32,
    state_sequence: u32,
}

/// E131 closes the remaining command-local stock-empty SKU sources at the
/// dormant finalize boundary. Exact 23J220 encodeCLCommandSKUStream() reads
/// these values back from the already-initialized RunCompute image: counter
/// +0x04, context ID +0x10, JobMeta fw-stamp/value/UUID/event-sequence at
/// +0x7f0/+0x7f8/+0x808/+0x80c, plus user-timestamp presence at +0x828/+0x830.
/// The finalizer therefore takes that typed command image instead of a parallel
/// bag of raw values that could drift from the command being serialized.

/// Fully materialized stock-empty command assets that are still deliberately
/// unpublished. This token is not a firmware structure and has no RunCompute
/// conversion/consumer. It exists only to prove the exact owner graph can
/// produce one coherent set of addresses after callers have separately proved
/// event-control and SKU slots retired.
#[derive(Debug)]
#[allow(dead_code)]
struct G15UnpublishedStockEmptyCommandAssets {
    event_control_fwva: u64,
    sku_fwva: u64,
    sku_size: u32,
    page_pool_state_fwva: u64,
    hwmetrics_fwva: u64,
    hardware_buffer_id: u32,
}

/// Host-only staging image for the exact stock-empty RunCompute pointer/UMA
/// fields closed by E071/E072/E088/E102. This is intentionally not a firmware
/// structure and there is no method that writes it into `fw::compute::RunCompute`.
/// Keeping the underlying non-Copy asset token inside preserves HardwareBuffer
/// completion ownership until this staging object is consumed.
#[derive(Debug)]
#[allow(dead_code)]
struct G15UnpublishedRunComputeFieldStage {
    assets: G15UnpublishedStockEmptyCommandAssets,
    event_control_fwva_14: u64,
    sku_fwva_760: u64,
    sku_size_768: u32,
    page_pool_state_fwva_83e: u64,
    uma_prepared_846: u8,
    uma_min_pool_size_847: u64,
    uma_ideal_pool_size_84f: u64,
    hwmetrics_fwva_857: u64,
}

#[allow(dead_code)]
impl G15UnpublishedRunComputeFieldStage {
    fn from_assets(assets: G15UnpublishedStockEmptyCommandAssets) -> Self {
        Self {
            event_control_fwva_14: assets.event_control_fwva,
            sku_fwva_760: assets.sku_fwva,
            sku_size_768: assets.sku_size,
            page_pool_state_fwva_83e: assets.page_pool_state_fwva,
            // Exact stock-empty prepared/min/ideal values from E071/E072.
            uma_prepared_846: 1,
            uma_min_pool_size_847: 0,
            uma_ideal_pool_size_84f: 0,
            hwmetrics_fwva_857: assets.hwmetrics_fwva,
            assets,
        }
    }
}

/// Unreachable channel/command-side ownership for exact stock-empty G15
/// Compute prerequisites.
///
/// E135 removes the FList/UMAPool from this lifetime: normal CL channels retain
/// a reusable shared Compute pool selected by priority class, while the 0x100
/// HardwareBuffer-ID namespace and pool-ID sequence are accelerator-global.
/// This owner therefore keeps only channel/command resources. Every phase that
/// needs Page-Pool State must receive a separate `G15SharedComputeUmaPoolOwner`.
/// There is still no Queue call site and no RunCompute writer.
#[allow(dead_code)]
struct G15StockEmptyComputeChannelOwners {
    // E130: exact Apple AGXFirmware +0x268 source is the 0xe10-byte Compute
    // statistics slice. Linux already owns the equivalent object under the
    // manager-global RuntimePointers lifetime; keep a typed weak FW pointer.
    _compute_stats: GpuWeakPointer<fw::initdata::G15StatsComp>,
    _event_control: G15EventControlBacking,
    _hwmetrics: G15HWMetricsBacking,
    _timestamp_queue: G15TimestampQueueBackingBlock,
    _scheduler_state: G15SchedulerStateBackingBlock,
    _channel_state: G15ChannelStateBackingBlock,
    _uncached_channel_memory: G15UncachedChannelMemoryBackingBlock,
    _cached_channel_memory: G15CachedChannelMemoryBackingBlock,
    _cl_command_resource: G15ClCommandResourceBacking,
    _sku: fw::compute::G15SkuBacking,
}

#[allow(dead_code)]
impl G15StockEmptyComputeChannelOwners {
    #[allow(clippy::too_many_arguments)]
    fn new_unpublished(
        dev: &AsahiDevice,
        compute_stats: GpuWeakPointer<fw::initdata::G15StatsComp>,
        range5_uncached_alloc: &mut alloc::DefaultAllocator,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        // E130 keeps this boundary typed instead of accepting an arbitrary raw
        // FWVA. The proven future producer is the manager-owned
        // InitData/RuntimePointers g15_stats_comp weak pointer.
        // Keep one shared bank-1/q22 lifetime for every exact shared resource.
        let event_control = G15EventControlBacking::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let hwmetrics = G15HWMetricsBacking::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let timestamp_queue = G15TimestampQueueBackingBlock::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let scheduler_state = G15SchedulerStateBackingBlock::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let channel_state = G15ChannelStateBackingBlock::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let uncached_channel_memory = G15UncachedChannelMemoryBackingBlock::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let cached_channel_memory = G15CachedChannelMemoryBackingBlock::new(
            dev,
            bank1.clone(),
            mapping_notifier.clone(),
        )?;
        let cl_command_resource = G15ClCommandResourceBacking::new(range5_uncached_alloc)?;
        let sku = fw::compute::G15SkuBacking::new(dev, bank1, mapping_notifier)?;

        Ok(Self {
            _compute_stats: compute_stats,
            _event_control: event_control,
            _hwmetrics: hwmetrics,
            _timestamp_queue: timestamp_queue,
            _scheduler_state: scheduler_state,
            _channel_state: channel_state,
            _uncached_channel_memory: uncached_channel_memory,
            _cached_channel_memory: cached_channel_memory,
            _cl_command_resource: cl_command_resource,
            _sku: sku,
        })
    }

    /// Phase 1: after the caller has bound both retirement guards, seed the
    /// selected event-control state, reserve (but do not write) the SKU slot,
    /// activate the exact stock-empty FList epoch and reserve the HWMetrics
    /// record. No RunCompute address is required or published here.
    fn prepare_unpublished_phase1(
        &mut self,
        uma_pool: &mut buffer::G15SharedComputeUmaPoolOwner,
        event_index: usize,
        state_sequence: u32,
        sku_index: usize,
        priority: u32,
        channel: G15ChannelPrepareInputs,
    ) -> Result<G15UnpublishedStockEmptyPrepare> {
        let timestamp_queue = self._timestamp_queue.reset_selected(
            channel.timestamp_queue_slot,
            channel.timestamp_update_mode_2,
        )?;
        let scheduler_state = self
            ._scheduler_state
            .reset_selected_j615(channel.scheduler_state_slot)?;
        let uncached_channel = self
            ._uncached_channel_memory
            .reset_selected_j615_cl(channel.uncached_channel_slot)?;
        let cached_channel = self
            ._cached_channel_memory
            .select_local(channel.cached_channel_slot)?;
        let channel_state = self._channel_state.reset_selected_j615_cl(
            channel.channel_state_slot,
            G15ChannelStateResetInputs {
                uncached_channel_fwva: uncached_channel.fwva,
                cached_channel_fwva: cached_channel.fwva,
                timestamp_queue_state_fwva: timestamp_queue.fwva,
                channel_4c_value: channel.channel_4c_value,
                scheduler_state_fwva: scheduler_state.fwva,
                effective_priority: channel.effective_priority,
                queue_qos: channel.queue_qos,
            },
        )?;

        self._event_control
            .seed_selected_after_event_finish(event_index, state_sequence)?;
        let event_control_fwva = self._event_control.control_fwva(event_index)?;
        let sku = self._sku.reserve_retired_slot(sku_index)?;
        if event_control_fwva == 0 || sku.fwva() == 0 {
            return Err(EIO);
        }

        let lease = uma_pool.prepare_stock_empty_reference(priority)?;
        let page_pool_state_fwva = match uma_pool.initialized_page_pool_state_fwva() {
            Ok(fwva) => fwva,
            Err(err) => {
                let _ = uma_pool.complete_reference(lease.hardware_buffer_id);
                return Err(err);
            }
        };
        let hwmetrics_fwva = match self._hwmetrics.take_record_fwva() {
            Ok(fwva) => fwva,
            Err(err) => {
                let _ = uma_pool.complete_reference(lease.hardware_buffer_id);
                return Err(err);
            }
        };
        if page_pool_state_fwva == 0 || hwmetrics_fwva == 0 {
            let _ = uma_pool.complete_reference(lease.hardware_buffer_id);
            return Err(EIO);
        }

        Ok(G15UnpublishedStockEmptyPrepare {
            event_index,
            event_control_fwva,
            sku,
            channel_state,
            page_pool_state_fwva,
            hwmetrics_fwva,
            hardware_buffer_id: lease.hardware_buffer_id,
            state_sequence,
        })
    }

    /// Phase 2: after a future caller has initialized the RunCompute backing
    /// and therefore knows both its typed image and FWVA, serialize E102 from
    /// that exact command plus the already-owned persistent resources, then
    /// write only the already-retired SKU slot. The result remains host-only.
    fn finalize_unpublished(
        &mut self,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        prepared: G15UnpublishedStockEmptyPrepare,
        command_fwva: u64,
        command: &fw::compute::raw::RunComputeG15V14_7<'_>,
    ) -> Result<G15UnpublishedRunComputeFieldStage> {
        if command_fwva == 0 {
            let _ = uma_pool.complete_reference(prepared.hardware_buffer_id);
            return Err(EINVAL);
        }

        let channel_command_region_base_fwva = match self._cl_command_resource.base_fwva() {
            Ok(fwva) => fwva,
            Err(err) => {
                let _ = uma_pool.complete_reference(prepared.hardware_buffer_id);
                return Err(err);
            }
        };
        let sku_input = fw::compute::G15StockEmptySkuInputs {
            command_fwva,
            stream_fwva: prepared.sku.fwva(),
            // E130: manager-owned G15StatsComp is the exact Linux counterpart
            // of Apple AGXFirmware +0x268 used by the stock-empty SKU stream.
            firmware_state_fwva: self._compute_stats.into(),
            channel_state_fwva: prepared.channel_state.fwva,
            channel_command_region_base_fwva,
            event_control_fwva: prepared.event_control_fwva,
            page_pool_state_fwva: prepared.page_pool_state_fwva,
            hwmetrics_fwva: prepared.hwmetrics_fwva,
            // E131 exact command-local sources. Apple reads these from the
            // same already-initialized RunCompute image immediately before
            // serializing the SKU stream, so do not accept duplicate values.
            fw_stamp_fwva: command.meta.fw_stamp.into(),
            user_timestamps_present: command.user_timestamp_pointers.start_addr.is_some()
                || command.user_timestamp_pointers.end_addr.is_some(),
            command_counter: command.counter.0,
            context_id: command.g15_context_id_10,
            state_sequence: prepared.state_sequence,
            queue_event_sequence: command.meta.event_seq,
            // E119/E128: this dormant graph models the first normal J615 CL
            // channel, whose AGXChannel +0x38 / G15JobMeta evctl_index is 0.
            evctl_index: fw::workqueue::G15_J615_FIRST_CL_EVCTL_INDEX,
            uuid: command.meta.uuid,
            stamp_value: command.meta.stamp_value.raw(),
            // E129 exact ordinary IOGPU device chain: userspace passes options=0,
            // so IOServiceOpen type 5 propagates zero through IOGPUDeviceUserClient,
            // IOGPU::createDevice(), AGXShared and AGXSecureGart. Together with the
            // normal J615 accelerator soft-fault feature byte staying zero,
            // AGXGart::isHWSoftFaultEnabled() is false for this target.
            gart_soft_fault_enabled: false,
            // E128 exact 23J220 configure chain: the base configure mask
            // 0xf4840fffffff7f clears packed feature bit 39 (halfword +0x654
            // bit 7); G15 and G15G preserve that bit. Stock J615 is false.
            accelerator_654_bit7: false,
        };
        let stream = match fw::compute::G15StockEmptySkuStream::new(sku_input) {
            Ok(stream) => stream,
            Err(err) => {
                let _ = uma_pool.complete_reference(prepared.hardware_buffer_id);
                return Err(err);
            }
        };
        let sku = match self
            ._sku
            .write_reserved_stock_empty_slot(prepared.sku, &stream)
        {
            Ok(sku) => sku,
            Err(err) => {
                let _ = uma_pool.complete_reference(prepared.hardware_buffer_id);
                return Err(err);
            }
        };
        if sku.fwva() == 0
            || sku.size() as usize != fw::compute::G15_STOCK_EMPTY_SKU_STREAM_SIZE
        {
            let _ = uma_pool.complete_reference(prepared.hardware_buffer_id);
            return Err(EIO);
        }

        Ok(G15UnpublishedRunComputeFieldStage::from_assets(
            G15UnpublishedStockEmptyCommandAssets {
                event_control_fwva: prepared.event_control_fwva,
                sku_fwva: sku.fwva(),
                sku_size: sku.size(),
                page_pool_state_fwva: prepared.page_pool_state_fwva,
                hwmetrics_fwva: prepared.hwmetrics_fwva,
                hardware_buffer_id: prepared.hardware_buffer_id,
            },
        ))
    }

    fn abort_unpublished(
        &self,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        prepared: G15UnpublishedStockEmptyPrepare,
    ) -> Result<bool> {
        uma_pool.complete_reference(prepared.hardware_buffer_id)
    }

    fn complete_unpublished(
        &self,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        assets: G15UnpublishedStockEmptyCommandAssets,
    ) -> Result<bool> {
        uma_pool.complete_reference(assets.hardware_buffer_id)
    }

    fn complete_staged(
        &self,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        stage: G15UnpublishedRunComputeFieldStage,
    ) -> Result<bool> {
        self.complete_unpublished(uma_pool, stage.assets)
    }
}

/// Compile-only owner for the exact 23J220 G15 command-buffer stamp and
/// event-control shared-data backings.
///
/// E092 proves two page-based mappings with logical data at page offset zero:
/// 36 four-byte stamps in normal range 7 and 36 0xc0-byte controls in range 8.
/// This owner remains unreachable and exposes no selected event-control FWVA.
/// E094 can seed the exact selected-state image only after its prior event has
/// been retired; the event-machine reuse/selection lifecycle remains gated.
#[allow(dead_code)]
struct G15EventControlBacking {
    _stamps: alloc::G15SharedGpuArray<Stamp>,
    _controls: alloc::G15SharedGpuArray<fw::event::raw::G15EventControlBlock>,
    _selector: fw::event::G15EventControlSelector,
}

#[allow(dead_code)]
impl G15EventControlBacking {
    fn new(
        dev: &AsahiDevice,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<Self> {
        // Apple shared-data logical addresses are page-base addresses. A page
        // minimum alignment forces G15SharedBank1Allocator's sub-page offset to
        // zero instead of packing these small objects at the end of a page.
        let mut stamp_alloc = alloc::G15SharedBank1Allocator::new_range7_event(
            dev,
            bank1.clone(),
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier.clone()),
        );
        let stamps = stamp_alloc.array_empty_shared_data::<Stamp>(
            fw::event::G15_EVENT_CONTROL_STATE_COUNT,
        )?;

        let mut control_alloc = alloc::G15SharedBank1Allocator::new_range8(
            dev,
            bank1,
            mmu::UAT_PGSZ,
            true,
            Some(mapping_notifier),
        );
        let mut controls = control_alloc
            .array_empty_shared_data::<fw::event::raw::G15EventControlBlock>(
                fw::event::G15_EVENT_CONTROL_STATE_COUNT,
            )?;

        // Exact AGXCommandBuffer::init() construction boundary: all controls
        // are zero except +0x00, which points at the corresponding stamp FWVA.
        // G15 GPUVA->FWVA conversion is identity on the exact target.
        for index in 0..fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            controls[index].stamp_fwva = U64(stamps.weak_item_pointer(index).into());
        }

        Ok(Self {
            _stamps: stamps,
            _controls: controls,
            _selector: Default::default(),
        })
    }

    /// Seed one already-selected state after the previous event occupying that
    /// slot has been retired. This is deliberately not a complete
    /// `nextCommandBufferState()` implementation: the exact IOGPU finish-event
    /// operation must happen before this reset and is not modeled here yet.
    fn seed_selected_after_event_finish(
        &mut self,
        index: usize,
        state_sequence: u32,
    ) -> Result {
        if index >= fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            return Err(EINVAL);
        }

        // Exact 23J220 rotation boundary: clear the selected 0xc0 control and
        // matching four-byte stamp, then restore the per-state fields.
        self._stamps[index].0.store(0, Ordering::Relaxed);
        let stamp_fwva: u64 = self._stamps.weak_item_pointer(index).into();
        self._controls[index] = Default::default();
        self._controls[index].stamp_fwva = U64(stamp_fwva);
        self._controls[index].state_sequence_08 = state_sequence;
        self._controls[index].effective_record_count_10 =
            fw::event::G15_EVENT_CONTROL_J615_EFFECTIVE_RECORD_COUNT;
        self._controls[index].sentinel_a8 = U64(u64::MAX);

        Ok(())
    }

    fn control_fwva(&self, index: usize) -> Result<u64> {
        if index >= fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            return Err(EINVAL);
        }
        Ok(self._controls.weak_item_pointer(index).into())
    }

    fn next_index(&self) -> usize {
        (self._selector.current() + 1) % fw::event::G15_EVENT_CONTROL_STATE_COUNT
    }

    fn advance_index(&mut self) -> usize {
        self._selector.advance()
    }
}

#[versions(AGX)]
pub(crate) struct Queue {
    dev: AsahiDevRef,
    _sched: sched::Scheduler<QueueJob::ver>,
    entity: sched::Entity<QueueJob::ver>,
    vm: mmu::Vm,
    q_vtx: Option<SubQueue::ver>,
    q_frag: Option<SubQueue::ver>,
    q_comp: Option<SubQueue::ver>,
    fence_ctx: FenceContexts,
    inner: QueueInner::ver,
}

#[versions(AGX)]
pub(crate) struct QueueInner {
    dev: AsahiDevRef,
    ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
    buffer: buffer::Buffer::ver,
    gpu_context: Arc<workqueue::GpuContext>,
    notifier_list: Arc<GpuObject<fw::event::NotifierList>>,
    notifier: Arc<GpuObject<fw::event::Notifier::ver>>,
    usc_exec_base: u64,
    id: u64,
    // Apple G15 stores this sequence per AGXCommandQueue (+0x5c8) and every
    // submitted channel command consumes one value from it across engines.
    #[ver(V >= V13_0B4)]
    counter: AtomicU64,
    // G15 AGXParameterManagement +0x2c equivalent. Apple scopes the PM to
    // AGX3DWorkQueue and advances this 80-record ring before each applicable
    // render setup. The selector is independent from the range-5 backing below.
    #[ver(G == G15)]
    g15_pm_record_index: AtomicU32,
    // Exact 0x6f0 Parameter Scene Allocations backing. Apple scopes this to
    // AGXShared/eGartRange 5 and PM/work-queue lifetime. Register emission is
    // still disabled; this only establishes the correct hidden-VA resource.
    #[ver(G == G15)]
    g15_pm_scene_alloc: GpuArray<u8>,
    // Exact G15 TA-channel object-payload backing. Apple allocates one normal
    // option-0x3 eGartRange-5 resource at channel initialization. J615's
    // topology fixes its size at 0x80000 bytes. Register emission remains
    // disabled; retaining the allocation here only reconstructs channel state.
    #[ver(G == G15)]
    g15_ta_object_payload: GpuArray<u8>,
    // Exact E139-E142 AGXShared/client address-space bridge. This retains only
    // the per-VM shared-pool selection container so future G15 Compute channel
    // creation can stay lazy. E143 performs no pool selection/construction and
    // consumes no device-global pool ID.
    #[ver(G == G15)]
    _g15_uma_shared_pools: Option<Arc<Mutex<buffer::G15ClientUmaPoolContainerState>>>,
    // Exact 0x2800 GPU-facing PM record backing. Apple uses range 5 with
    // compact PTE class 0x300; the separate 0x40 tail is intentionally absent.
    #[ver(G == G15)]
    g15_pm_records: GpuArray<buffer::G15PmRecord>,
    // Exact range-7 resources referenced by every PM record. Apple obtains the
    // 0x140 metrics element from a firmware-owned resource stack and allocates
    // the 0x80 page-list/statistics object separately; both map through shared
    // UAT bank 1 with option word 0x700000007.
    #[ver(G == G15)]
    _g15_pm_page_metrics: alloc::G15SharedGpuArray<u8>,
    #[ver(G == G15)]
    _g15_pm_scene_stats: alloc::G15SharedGpuArray<u8>,
}

#[versions(AGX)]
#[derive(Default)]
pub(crate) struct JobFence {
    id: u64,
    pending: AtomicU64,
}

#[versions(AGX)]
impl JobFence::ver {
    fn add_command(self: &FenceObject<Self>) {
        self.pending.fetch_add(1, Ordering::Relaxed);
    }

    fn command_complete(self: &FenceObject<Self>) {
        let remain = self.pending.fetch_sub(1, Ordering::Relaxed) - 1;
        mod_pr_debug!(
            "JobFence[{}]: Command complete (remain: {})\n",
            self.id,
            remain
        );
        if remain == 0 {
            mod_pr_debug!("JobFence[{}]: Signaling\n", self.id);
            self.signal();
        }
    }
}

/// Definition-only RAII command reference for the dormant G15 two-phase
/// transaction. E112 requires the submission fence to be armed before rotating
/// event/SKU state is reserved, but every constructor/finalizer failure must
/// release that reference. Keeping the decrement in Drop makes those rollback
/// paths structural rather than caller-convention dependent.
#[versions(AGX)]
#[allow(dead_code)]
struct G15CommandFenceArm {
    fence: UserFence<JobFence::ver>,
}

#[versions(AGX)]
#[allow(dead_code)]
impl G15CommandFenceArm::ver {
    fn new(fence: &UserFence<JobFence::ver>) -> Self {
        let fence = fence.clone();
        fence.add_command();
        Self { fence }
    }

    fn fence(&self) -> &UserFence<JobFence::ver> {
        &self.fence
    }
}

#[versions(AGX)]
impl Drop for G15CommandFenceArm::ver {
    fn drop(&mut self) {
        self.fence.command_complete();
    }
}

/// E134 pairs phase-1 ownership with the command fence reference that made the
/// rotating-slot bindings legitimately in-flight. The arm survives every
/// successful intermediate stage and rolls itself back automatically if any
/// later construction path returns an error.
#[versions(AGX)]
#[allow(dead_code)]
struct G15ArmedUnpublishedStockEmptyPrepare {
    arm: G15CommandFenceArm::ver,
    prepared: G15UnpublishedStockEmptyPrepare,
}

/// Final host-only E134 transaction token. The command reference remains armed
/// until this token is consumed at the future completion boundary; dropping it
/// on any unpublished/error path decrements the pending count automatically.
#[versions(AGX)]
#[allow(dead_code)]
struct G15ArmedUnpublishedRunComputeFieldStage {
    arm: G15CommandFenceArm::ver,
    stage: G15UnpublishedRunComputeFieldStage,
}

#[versions(AGX)]
#[vtable]
impl dma_fence::FenceOps for JobFence::ver {
    fn get_driver_name<'a>(self: &'a FenceObject<Self>) -> &'a CStr {
        c_str!("asahi")
    }
    fn get_timeline_name<'a>(self: &'a FenceObject<Self>) -> &'a CStr {
        c_str!("queue")
    }
}

/// Host-only reuse guards for the 36 exact G15 command-buffer states.
///
/// E095 proves Apple finishes the selected host event before resetting the
/// corresponding GPU-visible stamp/control state. Linux does not need to copy
/// that IOGPUEvent layout: retaining the submission fence expresses the same
/// lifetime. Reuse is fail-closed while any command covered by the fence is
/// still pending.
#[versions(AGX)]
#[allow(dead_code)]
struct G15EventControlRetirementGuards {
    slots: KVec<Option<UserFence<JobFence::ver>>>,
}

#[versions(AGX)]
#[allow(dead_code)]
impl G15EventControlRetirementGuards::ver {
    fn new() -> Result<Self> {
        let mut slots = KVec::with_capacity(
            fw::event::G15_EVENT_CONTROL_STATE_COUNT,
            GFP_KERNEL,
        )?;
        for _ in 0..fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            slots.push(None, GFP_KERNEL)?;
        }
        Ok(Self { slots })
    }

    /// Bind a selected state only after the submission has acquired at least
    /// one command reference. This prevents a not-yet-armed fence from being
    /// mistaken for an already-retired slot.
    fn bind_inflight(
        &mut self,
        index: usize,
        fence: &UserFence<JobFence::ver>,
    ) -> Result {
        if index >= fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            return Err(EINVAL);
        }
        if self.slots[index].is_some() {
            return Err(EBUSY);
        }
        if fence.pending.load(Ordering::Acquire) == 0 {
            return Err(EINVAL);
        }
        self.slots[index] = Some(fence.clone());
        Ok(())
    }

    /// Nonblocking Linux equivalent of the E095 reuse barrier. A caller may
    /// clear/reseed the GPU-visible state only after this returns true.
    fn try_finish(&mut self, index: usize) -> Result<bool> {
        if index >= fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            return Err(EINVAL);
        }
        let done = match self.slots[index].as_ref() {
            Some(fence) => fence.pending.load(Ordering::Acquire) == 0,
            None => true,
        };
        if done {
            self.slots[index] = None;
        }
        Ok(done)
    }

    fn rollback_bound(&mut self, index: usize) {
        if index < fw::event::G15_EVENT_CONTROL_STATE_COUNT {
            self.slots[index] = None;
        }
    }
}

/// Conservative Linux retirement guards for the exact 0xf0 G15 Compute SKU
/// slots. E105 proves Apple scans from the next slot, tests the per-slot host
/// event, and copies the new command event into the selected guard before
/// writing the slot. A whole-submission JobFence is stronger than Apple's
/// per-command event but cannot permit premature slot overwrite.
#[versions(AGX)]
#[allow(dead_code)]
struct G15SkuRetirementGuards {
    slots: KVec<Option<UserFence<JobFence::ver>>>,
    current: u32,
}

#[versions(AGX)]
#[allow(dead_code)]
impl G15SkuRetirementGuards::ver {
    fn new() -> Result<Self> {
        let mut slots = KVec::with_capacity(fw::compute::G15_SKU_SLOT_COUNT, GFP_KERNEL)?;
        for _ in 0..fw::compute::G15_SKU_SLOT_COUNT {
            slots.push(None, GFP_KERNEL)?;
        }
        Ok(Self {
            slots,
            // Exact encoder initialization uses -1 so the first candidate is 0.
            current: u32::MAX,
        })
    }

    /// Exact selection order with a conservative Linux completion predicate:
    /// `(current + 1) % 0xf0`, then scan at most 0xf0 candidates. The selected
    /// slot is bound to the new fence before being returned to a future writer.
    fn select_and_bind(&mut self, fence: &UserFence<JobFence::ver>) -> Result<usize> {
        if fence.pending.load(Ordering::Acquire) == 0 {
            return Err(EINVAL);
        }

        let count = fw::compute::G15_SKU_SLOT_COUNT as u32;
        let mut candidate = self.current.wrapping_add(1) % count;
        for _ in 0..fw::compute::G15_SKU_SLOT_COUNT {
            let index = candidate as usize;
            let reusable = match self.slots[index].as_ref() {
                Some(prior) => prior.pending.load(Ordering::Acquire) == 0,
                None => true,
            };
            if reusable {
                self.slots[index] = Some(fence.clone());
                self.current = candidate;
                return Ok(index);
            }
            candidate = (candidate + 1) % count;
        }
        Err(ENOSPC)
    }

    /// Host-only analogue of AGXSKUEncoder::scrubEvents(): discard completed
    /// retirement guards without changing the next-slot rotation point.
    fn scrub_completed(&mut self) {
        for slot in self.slots.iter_mut() {
            let complete = slot
                .as_ref()
                .map(|fence| fence.pending.load(Ordering::Acquire) == 0)
                .unwrap_or(false);
            if complete {
                *slot = None;
            }
        }
    }

    fn rollback_bound(&mut self, index: usize) {
        if index < fw::compute::G15_SKU_SLOT_COUNT {
            self.slots[index] = None;
        }
    }
}

/// Definition-only guard integration for one stock-empty Compute asset set.
/// The versioned wrapper is required only because the existing submission fence
/// and E096/E106 guard types are firmware-versioned. It has no Queue call site.
#[versions(AGX)]
#[allow(dead_code)]
struct G15StockEmptyAssetGuards {
    event: G15EventControlRetirementGuards::ver,
    sku: G15SkuRetirementGuards::ver,
}

#[versions(AGX)]
#[allow(dead_code)]
impl G15StockEmptyAssetGuards::ver {
    fn new() -> Result<Self> {
        Ok(Self {
            event: G15EventControlRetirementGuards::ver::new()?,
            sku: G15SkuRetirementGuards::ver::new()?,
        })
    }

    /// Phase 1: acquire the command fence reference first, then bind both exact
    /// rotating lifetimes and reserve command-independent assets without
    /// serializing or writing SKU bytes. The returned token owns the fence arm;
    /// every error path drops it after fresh slot/shared-pool rollback.
    fn prepare_unpublished_phase1(
        &mut self,
        owners: &mut G15StockEmptyComputeChannelOwners,
        uma_pool: &mut buffer::G15SharedComputeUmaPoolOwner,
        fence: &UserFence<JobFence::ver>,
        state_sequence: u32,
        priority: u32,
        channel: G15ChannelPrepareInputs,
    ) -> Result<G15ArmedUnpublishedStockEmptyPrepare::ver> {
        let arm = G15CommandFenceArm::ver::new(fence);

        // Apple selects exactly the next command-buffer state and waits for
        // that state to finish; it does not skip a busy event-control slot.
        let event_index = owners._event_control.next_index();
        if !self.event.try_finish(event_index)? {
            return Err(EBUSY);
        }
        if owners._event_control.advance_index() != event_index {
            return Err(EIO);
        }
        self.event.bind_inflight(event_index, arm.fence())?;

        let sku_index = match self.sku.select_and_bind(arm.fence()) {
            Ok(index) => index,
            Err(err) => {
                self.event.rollback_bound(event_index);
                return Err(err);
            }
        };

        match owners.prepare_unpublished_phase1(
            uma_pool,
            event_index,
            state_sequence,
            sku_index,
            priority,
            channel,
        ) {
            Ok(prepared) => Ok(G15ArmedUnpublishedStockEmptyPrepare::ver { arm, prepared }),
            Err(err) => {
                self.event.rollback_bound(event_index);
                self.sku.rollback_bound(sku_index);
                Err(err)
            }
        }
    }

    /// Phase 2: once a future caller has an initialized RunCompute image and
    /// its FWVA, finalize the reserved slot from that same command while keeping
    /// the command reference armed. This still has no firmware-command writer.
    fn finalize_unpublished(
        &mut self,
        owners: &mut G15StockEmptyComputeChannelOwners,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        armed: G15ArmedUnpublishedStockEmptyPrepare::ver,
        command_fwva: u64,
        command: &fw::compute::raw::RunComputeG15V14_7<'_>,
    ) -> Result<G15ArmedUnpublishedRunComputeFieldStage::ver> {
        let G15ArmedUnpublishedStockEmptyPrepare::ver { arm, prepared } = armed;
        let event_index = prepared.event_index;
        let sku_index = prepared.sku.index();
        match owners.finalize_unpublished(uma_pool, prepared, command_fwva, command) {
            Ok(stage) => Ok(G15ArmedUnpublishedRunComputeFieldStage::ver { arm, stage }),
            Err(err) => {
                self.event.rollback_bound(event_index);
                self.sku.rollback_bound(sku_index);
                Err(err)
            }
        }
    }

    fn abort_unpublished(
        &mut self,
        owners: &G15StockEmptyComputeChannelOwners,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        armed: G15ArmedUnpublishedStockEmptyPrepare::ver,
    ) -> Result<bool> {
        let G15ArmedUnpublishedStockEmptyPrepare::ver { arm, prepared } = armed;
        let event_index = prepared.event_index;
        let sku_index = prepared.sku.index();
        self.event.rollback_bound(event_index);
        self.sku.rollback_bound(sku_index);
        let result = owners.abort_unpublished(uma_pool, prepared);
        core::mem::drop(arm);
        result
    }

    /// Future completion boundary: release the FList HardwareBuffer epoch and
    /// then let the armed token's Drop decrement the submission-fence command
    /// reference. There is still no live caller in E136.
    fn complete_staged(
        &mut self,
        owners: &G15StockEmptyComputeChannelOwners,
        uma_pool: &buffer::G15SharedComputeUmaPoolOwner,
        armed: G15ArmedUnpublishedRunComputeFieldStage::ver,
    ) -> Result<bool> {
        let G15ArmedUnpublishedRunComputeFieldStage::ver { arm, stage } = armed;
        let result = owners.complete_staged(uma_pool, stage);
        core::mem::drop(arm);
        result
    }

    fn scrub_completed(&mut self) {
        self.sku.scrub_completed();
    }
}

#[versions(AGX)]
pub(crate) struct QueueJob {
    dev: AsahiDevRef,
    vm_bind: mmu::VmBind,
    op_guard: Option<gpu::OpGuard>,
    sj_vtx: Option<SubQueueJob::ver>,
    sj_frag: Option<SubQueueJob::ver>,
    sj_comp: Option<SubQueueJob::ver>,
    fence: UserFence<JobFence::ver>,
    notifier: Arc<GpuObject<fw::event::Notifier::ver>>,
    notification_count: u32,
    did_run: bool,
    id: u64,
}

#[versions(AGX)]
impl QueueJob::ver {
    fn get_vtx(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_vtx
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No vertex queue\n");
                EINVAL
            })?
            .get()
    }
    fn get_frag(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_frag
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No fragment queue\n");
                EINVAL
            })?
            .get()
    }
    fn get_comp(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_comp
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No compute queue\n");
                EINVAL
            })?
            .get()
    }

    fn commit(&mut self) -> Result {
        mod_dev_dbg!(self.dev, "QueueJob {}: Committing\n", self.id);

        self.sj_vtx.as_mut().map(|a| a.commit()).unwrap_or(Ok(()))?;
        self.sj_frag
            .as_mut()
            .map(|a| a.commit())
            .unwrap_or(Ok(()))?;
        self.sj_comp.as_mut().map(|a| a.commit()).unwrap_or(Ok(()))
    }
}

#[versions(AGX)]
impl sched::JobImpl for QueueJob::ver {
    fn prepare(job: &mut sched::Job<Self>) -> Option<Fence> {
        mod_dev_dbg!(job.dev, "QueueJob {}: Checking runnability\n", job.id);

        if let Some(sj) = job.sj_vtx.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to vertex queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        if let Some(sj) = job.sj_frag.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to fragment queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        if let Some(sj) = job.sj_comp.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to compute queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        None
    }

    #[allow(unused_assignments)]
    fn run(job: &mut sched::Job<Self>) -> Result<Option<dma_fence::Fence>> {
        mod_dev_dbg!(job.dev, "QueueJob {}: Running Job\n", job.id);

        // We can only increase the notifier threshold here, now that we are
        // actually running the job. We cannot increase it while queueing the
        // job without introducing subtle race conditions. Suppose we did, as
        // early versions of drm/asahi did:
        //
        // 1. When processing the ioctl submit, a job is queued to drm_sched.
        //    Incorrectly, the notifier threshold is increased, gating firmware
        //    events.
        // 2. When DRM schedules an event, the hardware is kicked.
        // 3. When the number of processed jobs equals the threshold, the
        //    firmware signals the complete event to the kernel
        // 4. When the kernel gets a complete event, we signal the out-syncs.
        //
        // Does that work? There are a few scenarios.
        //
        // 1. There is nothing else ioctl submitted before the job completes.
        //    The job is scheduled, completes, and signals immediately.
        //    Everything works.
        // 2. There is nontrivial sync across different queues. Since each queue
        //    has a separate own notifier threshold, submitting one does not
        //    block scheduling of the other. Everything works the way you'd
        //    expect. drm/sched handles the wait/signal ordering.
        // 3. Two ioctls are submitted back-to-back. The first signals a fence
        //    that the second waits on. Due to the notifier threshold increment,
        //    the first job's completion event is deferred. But in good
        //    conditions, drm/sched will schedule the second submit anyway
        //    because it kills the pointless intra-queue sync. Then both
        //    commands execute and are signalled together.
        // 4. Two ioctls are submitted back-to-back as above, but conditions are
        //    bad. Reporting completion of the first job is still masked by the
        //    notifier threshold, but the intra-queue fences are not optimized
        //    out in drm/sched... drm/sched doesn't schedule the second job
        //    until the first is signalled, but the first isn't signalled until
        //    the second is completed, but the second can't complete until it's
        //    scheduled. We hang!
        //
        // In good conditions, everything works properly and/or we win the race
        // to mask the issue. So the issue here is challenging to hit.
        // Nevertheless, we do need to get it right.
        //
        // The intention with drm/sched is that jobs that are not yet scheduled
        // are "invisible" to the firmware. Incrementing the notifier threshold
        // earlier than this violates that which leads to circles like the
        // above. Deferring the increment to submit solves the race.
        job.notifier.threshold.with(|raw, _inner| {
            raw.increase(job.notification_count);
        });

        let gpu = match (*job.dev)
            .gpu
            .clone()
            .arc_as_any()
            .downcast::<gpu::GpuManager::ver>()
        {
            Ok(gpu) => gpu,
            Err(_) => {
                dev_crit!(job.dev.as_ref(), "GpuManager mismatched with QueueJob!\n");
                return Err(EIO);
            }
        };

        if job.op_guard.is_none() {
            job.op_guard = Some(gpu.start_op()?);
        }

        // First submit all the commands for each queue. This can fail.

        let mut frag_job = None;
        let mut frag_sub = None;
        if let Some(sj) = job.sj_frag.as_mut() {
            frag_job = sj.job.take();
            if let Some(wqjob) = frag_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit fragment\n", job.id);
                frag_sub = Some(wqjob.submit()?);
            }
        }

        let mut vtx_job = None;
        let mut vtx_sub = None;
        if let Some(sj) = job.sj_vtx.as_mut() {
            vtx_job = sj.job.take();
            if let Some(wqjob) = vtx_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit vertex\n", job.id);
                vtx_sub = Some(wqjob.submit()?);
            }
        }

        let mut comp_job = None;
        let mut comp_sub = None;
        if let Some(sj) = job.sj_comp.as_mut() {
            comp_job = sj.job.take();
            if let Some(wqjob) = comp_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit compute\n", job.id);
                comp_sub = Some(wqjob.submit()?);
            }
        }

        // Now we fully commit to running the job
        mod_dev_dbg!(job.dev, "QueueJob {}: Run fragment\n", job.id);
        frag_sub.map(|a| gpu.run_job(a)).transpose()?;

        mod_dev_dbg!(job.dev, "QueueJob {}: Run vertex\n", job.id);
        vtx_sub.map(|a| gpu.run_job(a)).transpose()?;

        mod_dev_dbg!(job.dev, "QueueJob {}: Run compute\n", job.id);
        comp_sub.map(|a| gpu.run_job(a)).transpose()?;

        mod_dev_dbg!(job.dev, "QueueJob {}: Drop compute job\n", job.id);
        core::mem::drop(comp_job);
        mod_dev_dbg!(job.dev, "QueueJob {}: Drop vertex job\n", job.id);
        core::mem::drop(vtx_job);
        mod_dev_dbg!(job.dev, "QueueJob {}: Drop fragment job\n", job.id);
        core::mem::drop(frag_job);

        job.did_run = true;

        Ok(Some(Fence::from_fence(&job.fence)))
    }

    fn timed_out(job: &mut sched::Job<Self>) -> sched::Status {
        // FIXME: Handle timeouts properly
        dev_err!(
            job.dev.as_ref(),
            "QueueJob {}: Job timed out on the DRM scheduler, things will probably break (ran: {})\n",
            job.id, job.did_run
        );
        sched::Status::NoDevice
    }

    fn cancel(job: &mut sched::Job<Self>) {
        dev_info!(
            job.dev.as_ref(),
            "QueueJob {}: Job canceled on DRM scheduler teardown\n",
            job.id
        );
    }
}

#[versions(AGX)]
impl Drop for QueueJob::ver {
    fn drop(&mut self) {
        mod_dev_dbg!(self.dev, "QueueJob {}: Dropping\n", self.id);
    }
}

static QUEUE_NAME: &CStr = c_str!("asahi_fence");
static QUEUE_CLASS_KEY: Pin<&LockClassKey> = kernel::static_lock_class!();

#[versions(AGX)]
impl Queue::ver {
    /// Create a new user queue.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: &AsahiDevice,
        vm: mmu::Vm,
        alloc: &mut gpu::KernelAllocators,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        _g15_ualloc_range5_uncached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        _g15_ualloc_range5_cached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        _g15_uma_shared_pools: Option<Arc<Mutex<buffer::G15ClientUmaPoolContainerState>>>,
        _g15_shared_bank1: Option<mmu::G15SharedBank1>,
        _g15_mapping_notifier: Option<Arc<Mutex<mmu::G15MappingNotifier>>>,
        event_manager: Arc<event::EventManager>,
        mgr: &buffer::BufferManager::ver,
        id: u64,
        priority: u32,
        usc_exec_base: u64,
    ) -> Result<Queue::ver> {
        mod_dev_dbg!(dev, "[Queue {}] Creating queue\n", id);

        // Must be shared, no cache management on this one!
        let mut notifier_list = alloc.shared.new_default::<fw::event::NotifierList>()?;

        let self_ptr = notifier_list.weak_pointer();
        notifier_list.with_mut(|raw, _inner| {
            raw.list_head.next = Some(inner_weak_ptr!(self_ptr, list_head));
        });

        let threshold = alloc.shared.new_default::<fw::event::Threshold>()?;

        let notifier: Arc<GpuObject<fw::event::Notifier::ver>> = Arc::new(
            alloc.private.new_init(
                /*try_*/ init!(fw::event::Notifier::ver { threshold }),
                |inner, _p| {
                    try_init!(fw::event::raw::Notifier::ver {
                        threshold: inner.threshold.gpu_pointer(),
                        generation: AtomicU32::new(id as u32),
                        cur_count: AtomicU32::new(0),
                        unk_10: AtomicU32::new(0x50),
                        state: Default::default()
                    })
                },
            )?,
            GFP_KERNEL,
        )?;

        // Priorities are handled by the AGX scheduler, there is no meaning within a
        // per-queue scheduler. Use a single run queue wth Kernel priority.
        let sched =
            sched::Scheduler::new(dev.as_ref(), 1, WQ_SIZE, 0, 100000, c_str!("asahi_sched"))?;
        let entity = sched::Entity::new(&sched, sched::Priority::Kernel)?;

        let buffer =
            buffer::Buffer::ver::new(&*(*dev).gpu, alloc, ualloc.clone(), ualloc_priv, mgr)?;

        #[ver(G == G15)]
        let g15_pm_scene_alloc = _g15_ualloc_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged(buffer::G15_J615_PM_SCENE_ALLOC_BYTES, b"PMSC")?;
        #[ver(G == G15)]
        let g15_ta_object_payload = _g15_ualloc_range5_uncached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged(buffer::G15_J615_TA_OBJECT_PAYLOAD_BYTES, b"TAOP")?;
        #[ver(G == G15)]
        let mut g15_bank1_alloc = alloc::G15SharedBank1Allocator::new(
            dev,
            _g15_shared_bank1.ok_or(EINVAL)?,
            buffer::PAGE_SIZE,
            mmu::PROT_G15_RANGE7_FW,
            true,
            _g15_mapping_notifier,
        );
        #[ver(G == G15)]
        let g15_pm_page_metrics = g15_bank1_alloc.array_empty_tagged(
            buffer::G15_J615_PM_PAGE_METRICS_BYTES,
            b"PMET",
        )?;
        #[ver(G == G15)]
        let g15_pm_scene_stats = g15_bank1_alloc.array_empty_tagged(
            buffer::G15_PM_PAGE_LIST_STATS_BYTES,
            b"PSTS",
        )?;
        #[ver(G == G15)]
        let mut g15_pm_records = _g15_ualloc_range5_cached
            .as_ref()
            .ok_or(EINVAL)?
            .lock()
            .array_empty_tagged(buffer::G15_J615_PM_RECORD_COUNT, b"PMRC")?;
        #[ver(G == G15)]
        {
            let scene_base: u64 = g15_pm_scene_alloc.gpu_pointer().into();
            let shared_scene =
                scene_base + buffer::g15_j615_pm_shared_scene_slice_offset() as u64;
            let metrics_base: u64 = g15_pm_page_metrics.gpu_pointer().into();
            let scene_stats_fwva: u64 = g15_pm_scene_stats.gpu_offset_pointer(0x40).into();
            for (i, record) in g15_pm_records.as_mut_slice().iter_mut().enumerate() {
                let scene = scene_base + buffer::g15_j615_pm_scene_slice_offset(i as u32) as u64;
                let metrics_slot = metrics_base + (i * 4) as u64;
                *record = buffer::G15PmRecord::new_with_resources(
                    scene,
                    shared_scene,
                    metrics_slot,
                    scene_stats_fwva,
                );
            }
        }

        let mut ret = Queue::ver {
            dev: dev.into(),
            _sched: sched,
            entity,
            vm,
            q_vtx: None,
            q_frag: None,
            q_comp: None,
            fence_ctx: FenceContexts::new(1, QUEUE_NAME, QUEUE_CLASS_KEY)?,
            inner: QueueInner::ver {
                dev: dev.into(),
                ualloc,
                gpu_context: Arc::new(
                    workqueue::GpuContext::new(dev, alloc, buffer.any_ref())?,
                    GFP_KERNEL,
                )?,

                buffer,
                notifier_list: Arc::new(notifier_list, GFP_KERNEL)?,
                notifier,
                usc_exec_base,
                id,
                #[ver(V >= V13_0B4)]
                counter: AtomicU64::new(0),
                #[ver(G == G15)]
                g15_pm_record_index: AtomicU32::new(0),
                #[ver(G == G15)]
                g15_pm_scene_alloc,
                #[ver(G == G15)]
                g15_ta_object_payload,
                #[ver(G == G15)]
                _g15_uma_shared_pools,
                #[ver(G == G15)]
                g15_pm_records,
                #[ver(G == G15)]
                _g15_pm_page_metrics: g15_pm_page_metrics,
                #[ver(G == G15)]
                _g15_pm_scene_stats: g15_pm_scene_stats,
            },
        };

        // Rendering structures
        let tvb_blocks = *module_parameters::initial_tvb_size.value();

        ret.inner.buffer.ensure_blocks(tvb_blocks)?;

        ret.q_vtx = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager.clone(),
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Vertex,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        ret.q_frag = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager.clone(),
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Fragment,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        // Compute structures
        ret.q_comp = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager,
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Compute,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        mod_dev_dbg!(dev, "[Queue {}] Queue created\n", id);
        Ok(ret)
    }
}

const SQ_RENDER: usize = 0;
const SQ_COMPUTE: usize = 1;
const SQ_COUNT: usize = 2;

// SAFETY: All bit patterns are valid by construction.
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_header {}
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_render {}
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_compute {}
unsafe impl AnyBitPattern for uapi::drm_asahi_attachment {}

fn build_attachments(reader: &mut Reader<'_>, size: usize) -> Result<microseq::Attachments> {
    const STRIDE: usize = core::mem::size_of::<uapi::drm_asahi_attachment>();
    let count = size / STRIDE;

    if count > microseq::MAX_ATTACHMENTS {
        return Err(EINVAL);
    }

    let mut attachments: microseq::Attachments = Default::default();
    attachments.count = count as u32;

    for i in 0..count {
        let att: uapi::drm_asahi_attachment = reader.read()?;

        if att.flags != 0 || att.pad != 0 {
            return Err(EINVAL);
        }

        // Some kind of power-of-2 exponent related to attachment size, in
        // bounds [1, 6]? We don't know what this is exactly yet.
        let unk_e = 1;

        let cache_lines = (att.size + 127) >> 7;
        attachments.list[i as usize] = microseq::Attachment {
            address: U64(att.pointer),
            size: cache_lines.try_into()?,
            unk_c: 0x17,
            unk_e: unk_e as u16,
        };
    }

    Ok(attachments)
}

#[versions(AGX)]
impl Queue for Queue::ver {
    fn preflight_vm_bind(&mut self) -> Result<u32> {
        let bind = (*self.dev).gpu.bind_vm(&self.vm)?;
        let slot = bind.slot();

        #[ver(G == G15)]
        {
            let gpu = match (*self.dev)
                .gpu
                .clone()
                .arc_as_any()
                .downcast::<gpu::GpuManager::ver>()
            {
                Ok(gpu) => gpu,
                Err(_) => return Err(EIO),
            };

            // Use the compute subqueue for the bounded publication probe. With
            // wptr=0 the common scheduler path returns before any command entry
            // is dereferenced, and this avoids render/TVB-specific work.
            gpu.g15_set_command_submission_enabled(true)?;
            let publish_result = self
                .q_comp
                .as_ref()
                .ok_or(EIO)?
                .wq
                .g15_publish_empty(&gpu);

            // Restore the host runtime gate only after both the accelerator TX
            // entry and native ReleaseResource completed successfully. On any
            // ambiguous failure leave it enabled and fail-stop until reboot.
            if publish_result.is_ok() {
                gpu.g15_set_command_submission_enabled(false)?;
            }
            publish_result?;
        }

        core::mem::drop(bind);
        Ok(slot)
    }

    fn submit(
        &mut self,
        id: u64,
        mut syncs: KVec<file::SyncItem>,
        in_sync_count: usize,
        cmdbuf_raw: &[u8],
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
    ) -> Result {
        let gpu = match (*self.dev)
            .gpu
            .clone()
            .arc_as_any()
            .downcast::<gpu::GpuManager::ver>()
        {
            Ok(gpu) => gpu,
            Err(_) => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with JobImpl!\n");
                return Err(EIO);
            }
        };

        mod_dev_dbg!(self.dev, "[Submission {}] Submit job\n", id);

        if gpu.is_crashed() {
            dev_err!(
                self.dev.as_ref(),
                "[Submission {}] GPU is crashed, cannot submit\n",
                id
            );
            return Err(ENODEV);
        }

        let op_guard = if in_sync_count > 0 {
            Some(gpu.start_op()?)
        } else {
            None
        };

        let mut events: [KVec<Option<workqueue::QueueEventInfo::ver>>; SQ_COUNT] =
            Default::default();

        events[SQ_RENDER].push(
            self.q_frag.as_ref().and_then(|a| a.wq.event_info()),
            GFP_KERNEL,
        )?;
        events[SQ_COMPUTE].push(
            self.q_comp.as_ref().and_then(|a| a.wq.event_info()),
            GFP_KERNEL,
        )?;

        let vm_bind = gpu.bind_vm(&self.vm)?;
        let vm_slot = vm_bind.slot();

        mod_dev_dbg!(self.dev, "[Submission {}] Creating job\n", id);

        // FIXME: I think this can violate the fence seqno ordering contract.
        // If we have e.g. a render submission with no barriers and then a compute submission
        // with no barriers, it's possible for the compute submission to complete first, and
        // therefore its fence. Maybe we should have separate fence contexts for render
        // and compute, and then do a ? (Vert+frag should be fine since there is no vert
        // without frag, and frag always serializes.)
        let fence: UserFence<JobFence::ver> = self
            .fence_ctx
            .new_fence::<JobFence::ver>(
                0,
                JobFence::ver {
                    id,
                    pending: Default::default(),
                },
            )?
            .into();

        let mut cmdbuf = Reader::new(cmdbuf_raw);

        // First, parse the headers to determine the number of compute/render
        // commands. This will be used to determine when to flush stamps.
        //
        // We also use it to determine how many notifications the job will
        // generate. We could calculate that in the second pass since we don't
        // need until much later, but it's convenient to gather everything at
        // the same time.
        let mut nr_commands = 0;
        let mut last_compute = 0;
        let mut last_render = 0;
        let mut nr_render = 0;
        let mut nr_compute = 0;

        while !cmdbuf.is_empty() {
            let header: uapi::drm_asahi_cmd_header = cmdbuf.read()?;
            cmdbuf.skip(header.size as usize);
            nr_commands += 1;

            match header.cmd_type as u32 {
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => {
                    last_compute = nr_commands;
                    nr_render += 1;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => {
                    last_render = nr_commands;
                    nr_compute += 1;
                }
                _ => {}
            }
        }

        let mut job = self.entity.new_job(
            1,
            QueueJob::ver {
                dev: self.dev.clone(),
                vm_bind,
                op_guard,
                sj_vtx: self
                    .q_vtx
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                sj_frag: self
                    .q_frag
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                sj_comp: self
                    .q_comp
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                fence,
                notifier: self.inner.notifier.clone(),

                // Each render command generates 2 notifications: 1 for the
                // vertex part, 1 for the fragment part. Each compute command
                // generates 1 notification. Sum up to calculate the total
                // notification count for the job.
                notification_count: (2 * nr_render) + nr_compute,

                did_run: false,
                id,
            },
        )?;

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Adding {} in_syncs\n",
            id,
            in_sync_count
        );
        for sync in syncs.drain(0..in_sync_count) {
            if let Some(fence) = sync.fence {
                job.add_dependency(fence)?;
            }
        }

        // Validate the number of hardware commands, ignoring software commands
        let nr_hw_commands = nr_render + nr_compute;
        if nr_hw_commands == 0 || nr_hw_commands > MAX_COMMANDS_PER_SUBMISSION {
            cls_pr_debug!(
                Errors,
                "submit: Command count {} out of valid range [1, {}]\n",
                nr_hw_commands,
                MAX_COMMANDS_PER_SUBMISSION - 1
            );
            return Err(EINVAL);
        }

        cmdbuf.rewind();

        let mut command_index = 0;
        let mut vertex_attachments: microseq::Attachments = Default::default();
        let mut fragment_attachments: microseq::Attachments = Default::default();
        let mut compute_attachments: microseq::Attachments = Default::default();

        // Parse the full command buffer submitting as we go
        while !cmdbuf.is_empty() {
            let header: uapi::drm_asahi_cmd_header = cmdbuf.read()?;
            let header_size = header.size as usize;

            // Pre-increment command index to match last_compute/last_render
            command_index += 1;

            for (queue_idx, index) in [header.vdm_barrier, header.cdm_barrier].iter().enumerate() {
                if *index == uapi::DRM_ASAHI_BARRIER_NONE as u16 {
                    continue;
                }
                if let Some(event) = events[queue_idx].get(*index as usize).ok_or_else(|| {
                    cls_pr_debug!(Errors, "Invalid barrier #{}: {}\n", queue_idx, index);
                    EINVAL
                })? {
                    let mut alloc = gpu.alloc();
                    let queue_job = match header.cmd_type as u32 {
                        uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => job.get_vtx()?,
                        uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => job.get_comp()?,
                        _ => return Err(EINVAL),
                    };
                    mod_dev_dbg!(self.dev, "[Submission {}] Create Explicit Barrier\n", id);
                    let barrier = alloc.private.new_init(
                        pin_init::zeroed::<fw::workqueue::Barrier::ver>(),
                        |_inner, _p| {
                            let queue_job = &queue_job;
                            try_init!(fw::workqueue::raw::Barrier::ver {
                                tag: fw::workqueue::CommandType::Barrier,
                                wait_stamp: event.fw_stamp_pointer,
                                #[ver(G == G15)]
                                wait_stamp_2: event.fw_stamp_pointer,
                                wait_value: event.value,
                                wait_slot: event.slot,
                                stamp_self: queue_job.event_info().value.next(),
                                uuid: 0xffffbbbb,
                                external_barrier: 0,
                                internal_barrier_type: 1,
                                padding: Default::default(),
                            })
                        },
                    )?;
                    mod_dev_dbg!(self.dev, "[Submission {}] Add Explicit Barrier\n", id);
                    queue_job.add(barrier, vm_slot)?;
                } else {
                    assert!(*index == 0);
                }
            }

            match header.cmd_type as u32 {
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => {
                    let render: uapi::drm_asahi_cmd_render = cmdbuf.read_up_to(header_size)?;

                    self.inner.submit_render(
                        &mut job,
                        &render,
                        &vertex_attachments,
                        &fragment_attachments,
                        objects,
                        id,
                        command_index == last_render,
                    )?;
                    events[SQ_RENDER].push(
                        Some(
                            job.sj_frag
                                .as_ref()
                                .expect("No frag queue?")
                                .job
                                .as_ref()
                                .expect("No frag job?")
                                .event_info(),
                        ),
                        GFP_KERNEL,
                    )?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => {
                    let compute: uapi::drm_asahi_cmd_compute = cmdbuf.read_up_to(header_size)?;

                    self.inner.submit_compute(
                        &mut job,
                        &compute,
                        &compute_attachments,
                        objects,
                        id,
                        command_index == last_compute,
                    )?;
                    events[SQ_COMPUTE].push(
                        Some(
                            job.sj_comp
                                .as_ref()
                                .expect("No comp queue?")
                                .job
                                .as_ref()
                                .expect("No comp job?")
                                .event_info(),
                        ),
                        GFP_KERNEL,
                    )?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_VERTEX_ATTACHMENTS => {
                    vertex_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS => {
                    fragment_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_COMPUTE_ATTACHMENTS => {
                    compute_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                _ => {
                    cls_pr_debug!(Errors, "Unknown command type {}\n", header.cmd_type);
                    return Err(EINVAL);
                }
            }
        }

        mod_dev_dbg!(
            self.dev,
            "Queue {}: Committing job {}\n",
            self.inner.id,
            job.id
        );
        job.commit()?;

        mod_dev_dbg!(self.dev, "Queue {}: Arming job {}\n", self.inner.id, job.id);
        let mut job = job.arm();
        let out_fence = job.fences().finished();
        mod_dev_dbg!(
            self.dev,
            "Queue {}: Pushing job {}\n",
            self.inner.id,
            job.id
        );
        job.push();

        mod_dev_dbg!(
            self.dev,
            "Queue {}: Adding {} out_syncs\n",
            self.inner.id,
            syncs.len()
        );
        for mut sync in syncs {
            if let Some(chain) = sync.chain_fence.take() {
                sync.syncobj
                    .add_point(chain, &out_fence, sync.timeline_value);
            } else {
                sync.syncobj.replace_fence(Some(&out_fence));
            }
        }

        Ok(())
    }
}

#[versions(AGX)]
impl Drop for Queue::ver {
    fn drop(&mut self) {
        mod_dev_dbg!(self.dev, "[Queue {}] Dropping queue\n", self.inner.id);

        #[ver(G == G15)]
        if self.inner.gpu_context.is_published_to_firmware() {
            // Publication/release did not complete with a known-good outcome.
            // The compute WorkQueue owns the exact QueueInfo/ring/state backing
            // firmware may still reference. Retain it until reboot rather than
            // risking a firmware use-after-free from later scheduler activity.
            dev_err!(
                self.dev.as_ref(),
                "G15 queue {} has uncertain firmware publication; retaining QueueInfo backing\n",
                self.inner.id
            );
            core::mem::forget(self.q_comp.take());
        }
    }
}
