// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU command execution queues
//!
//! The AGX GPU firmware schedules GPU work commands out of work queues, which are ring buffers of
//! pointers to work commands. There can be an arbitrary number of work queues. Work queues have an
//! associated type (vertex, fragment, or compute) and may only contain generic commands or commands
//! specific to that type.
//!
//! This module manages queueing work commands into a work queue and submitting them for execution
//! by the firmware. An active work queue needs an event to signal completion of its work, which is
//! owned by what we call a batch. This event then notifies the work queue when work is completed,
//! and that triggers freeing of all resources associated with that work. An idle work queue gives
//! up its associated event.

use crate::debug::*;
use crate::fw::channels::{
    ChannelErrorType,
    PipeType, //
};
use crate::fw::types::*;
use crate::fw::workqueue::*;
use crate::gpu::GpuManager as _;
use crate::no_debug;
use crate::object::OpaqueGpuObject;
use crate::{
    channel,
    driver,
    event,
    fw,
    gpu,
    hw,
    regs, //
};
use core::any::Any;
use core::num::NonZeroU64;
use core::sync::atomic::{AtomicBool, Ordering};
use kernel::{
    dma_fence,
    error::code::*,
    new_mutex,
    prelude::*,
    sync::{
        lock::{
            mutex::MutexBackend,
            Guard, //
        },
        Arc,
        Mutex, //
    },
    workqueue::{
        self,
        impl_has_work,
        new_work,
        Work,
        WorkItem, //
    }, //
};

pub(crate) trait OpaqueCommandObject: OpaqueGpuObject {}

impl<T: GpuStruct + Sync + Send> OpaqueCommandObject for GpuObject<T> where T: Command {}

const DEBUG_CLASS: DebugFlags = DebugFlags::WorkQueue;

const MAX_JOB_SLOTS: u32 = 127;

/// Apple writes mach_absolute_time() into G15 accelerator-ring entries. On
/// Apple Silicon that is the architectural counter clock domain; the physical
/// counter used by the existing DRM_ASAHI_GET_TIME path has the same frequency
/// and differs only by a constant offset from CNTVCT on bare metal.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct G15WorkQueueTransportState {
    pub(crate) doneptr: u32,
    pub(crate) wptr: u32,
    pub(crate) ring_size: u32,
}

/// Typed E185 bridge from generic WorkQueue ownership to the selected G15 CL
/// channel transport. Methods remain unused by commit/submit/run in E185.
pub(crate) trait G15WorkQueueTransport: Send + Sync {
    fn queue_info_fwva(&self) -> Result<NonZeroU64>;
    fn state(&self) -> Result<G15WorkQueueTransportState>;
    fn write_command(&self, command_fwva: NonZeroU64) -> Result<u32>;
    fn begin_submission(&self) -> Result<bool>;
    fn finish_submission(&self, first_submission: bool) -> Result;
    fn fail_submission(&self, first_submission: bool);
}

#[inline(always)]
pub(crate) fn g15_submission_timestamp() -> u64 {
    let raw: u64;

    // SAFETY: this only reads the architectural counter.
    unsafe {
        core::arch::asm!(
            "mrs {x}, CNTPCT_EL0",
            x = out(reg) raw
        );
    }

    raw
}

/// An enum of possible errors that might cause a piece of work to fail execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WorkError {
    /// GPU timeout (command execution took too long).
    Timeout,
    /// GPU MMU fault (invalid access).
    Fault(regs::FaultInfo),
    /// Work failed due to an error caused by other concurrent GPU work.
    Killed,
    /// Channel error
    ChannelError(ChannelErrorType),
    /// The GPU crashed.
    NoDevice,
    /// Unknown reason.
    Unknown,
}

impl From<WorkError> for kernel::error::Error {
    fn from(err: WorkError) -> Self {
        match err {
            WorkError::Timeout => ETIMEDOUT,
            // Not EFAULT because that's for userspace faults
            WorkError::Fault(_) => EIO,
            WorkError::Unknown => ENODATA,
            WorkError::Killed => ECANCELED,
            WorkError::NoDevice => ENODEV,
            WorkError::ChannelError(_) => EIO,
        }
    }
}

/// A GPU context tracking structure, which must be explicitly invalidated when dropped.
pub(crate) struct GpuContext {
    dev: driver::AsahiDevRef,
    data: Option<KBox<GpuObject<fw::workqueue::GpuContextData>>>,
    /// True from the first possible QueueInfo publication until the native
    /// firmware ReleaseResource handshake completes. This is intentionally
    /// conservative: an uncertain pipe-send outcome must never turn into a
    /// host-only free of a context firmware may have observed.
    published_to_firmware: AtomicBool,
}
no_debug!(GpuContext);

impl GpuContext {
    /// Allocate a new GPU context.
    pub(crate) fn new(
        dev: &driver::AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
        buffer: Arc<dyn core::any::Any + Send + Sync>,
    ) -> Result<GpuContext> {
        let is_g15 = dev.gpu.get_cfg().gpu_gen == hw::GpuGen::G15;
        Ok(GpuContext {
            dev: dev.into(),
            published_to_firmware: AtomicBool::new(false),
            data: Some(KBox::new(
                alloc.shared.new_object(
                    fw::workqueue::GpuContextData { _buffer: buffer },
                    move |_inner| {
                        if is_g15 {
                            fw::workqueue::raw::GpuContextData::g15()
                        } else {
                            Default::default()
                        }
                    },
                )?,
                GFP_KERNEL,
            )?),
        })
    }

    /// Returns the GPU pointer to the inner GPU context data structure.
    pub(crate) fn gpu_pointer(&self) -> GpuPointer<'_, fw::workqueue::GpuContextData> {
        self.data.as_ref().unwrap().gpu_pointer()
    }

    pub(crate) fn data(&self) -> &GpuObject<fw::workqueue::GpuContextData> {
        self.data.as_ref().unwrap()
    }

    pub(crate) fn mark_published_to_firmware(&self) {
        self.published_to_firmware.store(true, Ordering::Release);
    }

    pub(crate) fn mark_released_from_firmware(&self) {
        self.published_to_firmware.store(false, Ordering::Release);
    }

    pub(crate) fn is_published_to_firmware(&self) -> bool {
        self.published_to_firmware.load(Ordering::Acquire)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        mod_dev_dbg!(self.dev, "GpuContext: Freeing GPU context\n");
        let data = self.data.take().unwrap();
        (*self.dev)
            .gpu
            .free_context(data, self.is_published_to_firmware());
    }
}

struct SubmittedWork<O, C>
where
    O: OpaqueCommandObject,
    C: FnOnce(Option<WorkError>) + Send + Sync + 'static,
{
    object: O,
    value: EventValue,
    error: Option<WorkError>,
    wptr: u32,
    vm_slot: u32,
    callback: Option<C>,
    fence: dma_fence::Fence,
}

pub(crate) trait GenSubmittedWork: Send + Sync {
    fn gpu_va(&self) -> NonZeroU64;
    fn value(&self) -> event::EventValue;
    fn wptr(&self) -> u32;
    fn set_wptr(&mut self, wptr: u32);
    fn mark_error(&mut self, error: WorkError);
    fn complete(&mut self);
    fn get_fence(&self) -> dma_fence::Fence;
}

#[pin_data]
struct SubmittedWorkContainer {
    #[pin]
    work: Work<Self>,
    inner: KBox<dyn GenSubmittedWork>,
}

impl_has_work! {
    impl HasWork<Self> for SubmittedWorkContainer { self.work }
}

impl WorkItem for SubmittedWorkContainer {
    type Pointer = Pin<KBox<SubmittedWorkContainer>>;

    fn run(this: Pin<KBox<SubmittedWorkContainer>>) {
        mod_pr_debug!("WorkQueue: Freeing command @ {:?}\n", this.inner.gpu_va());
    }
}

impl SubmittedWorkContainer {
    fn inner_mut(self: Pin<&mut Self>) -> &mut KBox<dyn GenSubmittedWork> {
        // SAFETY: inner does not require structural pinning.
        unsafe { &mut self.get_unchecked_mut().inner }
    }
}

impl<O: OpaqueCommandObject, C: FnOnce(Option<WorkError>) + Send + Sync> GenSubmittedWork
    for SubmittedWork<O, C>
{
    fn gpu_va(&self) -> NonZeroU64 {
        self.object.gpu_va()
    }

    fn value(&self) -> event::EventValue {
        self.value
    }

    fn wptr(&self) -> u32 {
        self.wptr
    }

    fn set_wptr(&mut self, wptr: u32) {
        self.wptr = wptr;
    }

    fn complete(&mut self) {
        if let Some(cb) = self.callback.take() {
            cb(self.error);
        }
    }

    fn mark_error(&mut self, error: WorkError) {
        mod_pr_debug!("WorkQueue: Command at value {:#x?} failed\n", self.value);
        self.error = Some(match error {
            WorkError::Fault(info) if info.vm_slot != self.vm_slot => WorkError::Killed,
            err => err,
        });
    }

    fn get_fence(&self) -> dma_fence::Fence {
        self.fence.clone()
    }
}

/// Inner data for managing a single work queue.
#[versions(AGX)]
struct WorkQueueInner {
    dev: driver::AsahiDevRef,
    event_manager: Arc<event::EventManager>,
    info: GpuObject<QueueInfo::ver>,
    new: bool,
    pipe_type: PipeType,
    size: u32,
    wptr: u32,
    pending: KVec<Pin<KBox<SubmittedWorkContainer>>>,
    last_token: Option<event::Token>,
    pending_jobs: usize,
    last_submitted: Option<event::EventValue>,
    last_completed: Option<event::EventValue>,
    event: Option<(event::Event, event::EventValue)>,
    priority: u32,
    commit_seq: u64,
    submit_seq: u64,
    event_seq: u64,
}

/// An instance of a work queue.
#[versions(AGX)]
#[pin_data]
pub(crate) struct WorkQueue {
    // E155 exact CL WorkQueue -> channel strong ownership. The type is opaque
    // here to keep the generic WorkQueue layer independent of Queue's G15
    // channel implementation, but the anchor is intentionally the first field:
    // the final WorkQueue Arc therefore releases the channel lifetime before
    // any base WorkQueue state is torn down. Every Job/Event WorkQueue Arc clone
    // implicitly retains this same anchor.
    _g15_owned_channel_lifetime: Option<Arc<dyn G15WorkQueueTransport>>,
    info_pointer: GpuWeakPointer<QueueInfo::ver>,
    #[pin]
    inner: Mutex<WorkQueueInner::ver>,
}

#[versions(AGX)]
impl WorkQueueInner::ver {
    /// Return the GPU done pointer, representing how many work items have been completed by the
    /// GPU.
    fn doneptr(&self) -> u32 {
        self.info
            .state
            .with(|raw, _inner| raw.gpu_doneptr.load(Ordering::Acquire))
    }
}

#[versions(AGX)]
#[derive(Copy, Clone)]
pub(crate) struct QueueEventInfo {
    pub(crate) stamp_pointer: GpuWeakPointer<Stamp>,
    pub(crate) fw_stamp_pointer: GpuWeakPointer<FwStamp>,
    pub(crate) slot: u32,
    pub(crate) value: event::EventValue,
    pub(crate) cmd_seq: u64,
    pub(crate) event_seq: u64,
    pub(crate) info_ptr: GpuWeakPointer<QueueInfo::ver>,
}

#[versions(AGX)]
pub(crate) struct Job {
    wq: Arc<WorkQueue::ver>,
    event_info: QueueEventInfo::ver,
    start_value: EventValue,
    pending: KVec<Pin<KBox<SubmittedWorkContainer>>>,
    committed: bool,
    submitted: bool,
    event_count: usize,
    fence: dma_fence::Fence,
}

#[versions(AGX)]
pub(crate) struct JobSubmission<'a> {
    inner: Option<Guard<'a, WorkQueueInner::ver, MutexBackend>>,
    wptr: u32,
    event_count: usize,
    command_count: usize,
}

/// E187 rollback-safe selected-channel submission token for the exact stock-empty
/// G15 path. Merely creating this object does not touch firmware-visible channel
/// memory; selected command placement occurs only in `run()`.
#[versions(AGX)]
#[allow(dead_code)]
pub(crate) struct G15SelectedJobSubmission<'a> {
    inner: Option<Guard<'a, WorkQueueInner::ver, MutexBackend>>,
    transport: Arc<dyn G15WorkQueueTransport>,
    event_count: usize,
    command_count: usize,
}

/// E192 handoff between selected RunWorkQueue encoding and the future EP21
/// doorbell. Dropping an uncompleted first-submit ticket makes scheduler
/// publication Uncertain, so Queue teardown cannot recycle its selected state.
pub(crate) struct G15SelectedRunCommit {
    transport: Arc<dyn G15WorkQueueTransport>,
    first_submission: bool,
    completed: bool,
}

impl G15SelectedRunCommit {
    pub(crate) fn complete(mut self) -> Result {
        self.transport.finish_submission(self.first_submission)?;
        self.completed = true;
        Ok(())
    }
}

impl Drop for G15SelectedRunCommit {
    fn drop(&mut self) {
        if !self.completed {
            self.transport.fail_submission(self.first_submission);
        }
    }
}

#[versions(AGX)]
impl Job::ver {
    pub(crate) fn event_info(&self) -> QueueEventInfo::ver {
        let mut info = self.event_info;
        info.cmd_seq += self.pending.len() as u64;
        info.event_seq += self.event_count as u64;

        info
    }

    pub(crate) fn next_seq(&mut self) {
        self.event_count += 1;
        self.event_info.value.increment();
    }

    pub(crate) fn add<O: OpaqueCommandObject + 'static>(
        &mut self,
        command: O,
        vm_slot: u32,
    ) -> Result {
        self.add_cb(command, vm_slot, |_| {})
    }

    pub(crate) fn add_cb<O: OpaqueCommandObject + 'static>(
        &mut self,
        command: O,
        vm_slot: u32,
        callback: impl FnOnce(Option<WorkError>) + Sync + Send + 'static,
    ) -> Result {
        if self.committed {
            pr_err!("WorkQueue: Tried to mutate committed Job\n");
            return Err(EINVAL);
        }

        let fence = self.fence.clone();
        let value = self.event_info.value.next();

        self.pending.push(
            KBox::try_pin_init(
                try_pin_init!(SubmittedWorkContainer {
                    work <- new_work!("SubmittedWorkWrapper::work"),
                    inner: KBox::new(SubmittedWork::<_, _> {
                        object: command,
                        value,
                        error: None,
                        callback: Some(callback),
                        wptr: 0,
                        vm_slot,
                        fence,
                    }, GFP_KERNEL)?
                }),
                GFP_KERNEL,
            )?,
            GFP_KERNEL,
        )?;

        Ok(())
    }

    pub(crate) fn commit(&mut self) -> Result {
        if self.committed {
            pr_err!("WorkQueue: Tried to commit committed Job\n");
            return Err(EINVAL);
        }

        if self.pending.is_empty() {
            pr_err!("WorkQueue: Job::commit() with no commands\n");
            return Err(EINVAL);
        }

        let mut inner = self.wq.inner.lock();

        let ev = inner.event.as_mut().expect("WorkQueue: Job lost its event");

        if ev.1 != self.start_value {
            pr_err!(
                "WorkQueue: Job::commit() out of order (event slot {} {:?} != {:?}\n",
                ev.0.slot(),
                ev.1,
                self.start_value
            );
            return Err(EINVAL);
        }

        ev.1 = self.event_info.value;
        inner.commit_seq += self.pending.len() as u64;
        inner.event_seq += self.event_count as u64;
        self.committed = true;

        Ok(())
    }

    pub(crate) fn can_submit(&self) -> Option<dma_fence::Fence> {
        let inner = self.wq.inner.lock();
        if inner.free_slots() > self.event_count && inner.free_space() > self.pending.len() {
            None
        } else if let Some(work) = inner.pending.first() {
            Some(work.inner.get_fence())
        } else {
            pr_err!(
                "WorkQueue: Cannot submit, but queue is empty? {} > {}, {} > {} (pend={} ls={:#x?} lc={:#x?}) ev={:#x?} cur={:#x?} slot {:?}\n",
                inner.free_slots(),
                self.event_count,
                inner.free_space(),
                self.pending.len(),
                inner.pending.len(),
                inner.last_submitted,
                inner.last_completed,
                inner.event.as_ref().map(|a| a.1),
                inner.event.as_ref().map(|a| a.0.current()),
                inner.event.as_ref().map(|a| a.0.slot()),
            );
            None
        }
    }

    /// E187 stock-empty selected-channel readiness check. This remains separate
    /// from the generic WorkQueue path and has no caller in this checkpoint.
    #[allow(dead_code)]
    pub(crate) fn can_submit_g15_selected_stock_empty(
        &self,
    ) -> Result<Option<dma_fence::Fence>> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if self.pending.len() != 1 {
                return Err(EINVAL);
            }
            let transport = self
                .wq
                ._g15_owned_channel_lifetime
                .as_ref()
                .ok_or(EINVAL)?;
            let state = transport.state()?;
            let next = (state.wptr + 1) % state.ring_size;
            let inner = self.wq.inner.lock();
            if inner.free_slots() > self.event_count && next != state.doneptr {
                return Ok(None);
            }
            if let Some(work) = inner.pending.first() {
                return Ok(Some(work.inner.get_fence()));
            }
            Err(EBUSY)
        }
    }

    /// Move exactly one finalized stock-empty command into host pending ownership
    /// while leaving the selected cached/uncached channel memory untouched. If
    /// this token is dropped before `run()`, the ordinary event/pending rollback
    /// remains complete because no firmware-visible ring mutation occurred.
    #[allow(dead_code)]
    pub(crate) fn submit_g15_selected_stock_empty(
        &mut self,
    ) -> Result<G15SelectedJobSubmission::ver<'_>> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if !self.committed || self.submitted || self.pending.len() != 1 {
                return Err(EINVAL);
            }
            let transport = self
                .wq
                ._g15_owned_channel_lifetime
                .as_ref()
                .ok_or(EINVAL)?
                .clone();
            let mut inner = self.wq.inner.lock();

            if inner.submit_seq != self.event_info.cmd_seq
                || inner.commit_seq < self.event_info.cmd_seq + 1
            {
                return Err(EINVAL);
            }
            let state = transport.state()?;
            let next = (state.wptr + 1) % state.ring_size;
            if next == state.doneptr {
                return Err(EBUSY);
            }

            inner.pending.reserve(1, GFP_KERNEL)?;
            inner.last_submitted = Some(self.event_info.value);

            for mut command in self.pending.drain(..) {
                command.as_mut().inner_mut().set_wptr(state.wptr);
                inner
                    .pending
                    .push(command, GFP_KERNEL)
                    .expect("push() failed after reserve()");
            }

            self.submitted = true;
            Ok(G15SelectedJobSubmission::ver {
                inner: Some(inner),
                transport,
                event_count: self.event_count,
                command_count: 1,
            })
        }
    }

    pub(crate) fn submit(&mut self) -> Result<JobSubmission::ver<'_>> {
        if !self.committed {
            pr_err!("WorkQueue: Tried to submit uncommitted Job\n");
            return Err(EINVAL);
        }

        if self.submitted {
            pr_err!("WorkQueue: Tried to submit Job twice\n");
            return Err(EINVAL);
        }

        if self.pending.is_empty() {
            pr_err!("WorkQueue: Job::submit() with no commands\n");
            return Err(EINVAL);
        }

        let mut inner = self.wq.inner.lock();

        if inner.submit_seq != self.event_info.cmd_seq {
            pr_err!(
                "WorkQueue: Job::submit() out of order (submit_seq {} != {})\n",
                inner.submit_seq,
                self.event_info.cmd_seq
            );
            return Err(EINVAL);
        }

        if inner.commit_seq < (self.event_info.cmd_seq + self.pending.len() as u64) {
            pr_err!(
                "WorkQueue: Job::submit() out of order (commit_seq {} != {})\n",
                inner.commit_seq,
                (self.event_info.cmd_seq + self.pending.len() as u64)
            );
            return Err(EINVAL);
        }

        let mut wptr = inner.wptr;
        let command_count = self.pending.len();

        if inner.free_space() <= command_count {
            pr_err!("WorkQueue: Job does not fit in ring buffer\n");
            return Err(EBUSY);
        }

        inner.pending.reserve(command_count, GFP_KERNEL)?;

        inner.last_submitted = Some(self.event_info.value);
        mod_dev_dbg!(
            inner.dev,
            "WorkQueue: submitting {} cmds at {:#x?}, lc {:#x?}, cur {:#x?}, pending {}, events {}\n",
            self.pending.len(),
            inner.last_submitted,
            inner.last_completed,
            inner.event.as_ref().map(|a| a.0.current()),
            inner.pending.len(),
            self.event_count,
        );

        for mut command in self.pending.drain(..) {
            command.as_mut().inner_mut().set_wptr(wptr);

            let next_wptr = (wptr + 1) % inner.size;
            assert!(inner.doneptr() != next_wptr);
            inner.info.ring[wptr as usize] = command.inner.gpu_va().get();
            wptr = next_wptr;

            // Cannot fail, since we did a reserve(1) above
            inner
                .pending
                .push(command, GFP_KERNEL)
                .expect("push() failed after reserve()");
        }

        self.submitted = true;

        Ok(JobSubmission::ver {
            inner: Some(inner),
            wptr,
            command_count,
            event_count: self.event_count,
        })
    }
}

#[versions(AGX)]
impl<'a> JobSubmission::ver<'a> {
    pub(crate) fn run(mut self, channel: &mut channel::PipeChannel::ver) {
        let command_count = self.command_count;
        let mut inner = self.inner.take().expect("No inner?");
        let wptr = self.wptr;
        core::mem::forget(self);

        inner
            .info
            .state
            .with(|raw, _inner| raw.cpu_wptr.store(wptr, Ordering::Release));

        inner.wptr = wptr;

        let event = inner.event.as_mut().expect("JobSubmission lost its event");

        let event_slot = event.0.slot();

        let msg = fw::channels::RunWorkQueueMsg::ver {
            #[ver(G == G15)]
            g15_timestamp: U64(g15_submission_timestamp()),
            #[ver(G != G15)]
            pipe_type: inner.pipe_type,
            #[ver(G != G15)]
            work_queue: Some(inner.info.weak_pointer()),
            #[ver(G == G15)]
            g15_work_queue_fwva: U64(inner.info.weak_pointer().into()),
            #[ver(G != G15)]
            wptr: inner.wptr,
            #[ver(G != G15)]
            event_slot,
            #[ver(G != G15)]
            is_new: inner.new,
            #[ver(G == G15)]
            g15_pipe_type: inner.pipe_type,
            #[ver(G == G15)]
            g15_wptr: inner.wptr as u16,
            #[ver(G == G15)]
            g15_event_slot: event_slot as u8,
            #[ver(G == G15)]
            g15_is_new: inner.new,
            #[ver(G != G15)]
            __pad: Default::default(),
        };
        channel.send(&msg);
        inner.new = false;

        inner.submit_seq += command_count as u64;
    }

    pub(crate) fn pipe_type(&self) -> PipeType {
        self.inner.as_ref().expect("No inner?").pipe_type
    }

    pub(crate) fn priority(&self) -> u32 {
        self.inner.as_ref().expect("No inner?").priority
    }
}

#[versions(AGX)]
#[allow(dead_code)]
impl<'a> G15SelectedJobSubmission::ver<'a> {
    pub(crate) fn pipe_type(&self) -> Result<PipeType> {
        self.inner.as_ref().map(|inner| inner.pipe_type).ok_or(EIO)
    }

    pub(crate) fn priority(&self) -> Result<u32> {
        self.inner.as_ref().map(|inner| inner.priority).ok_or(EIO)
    }

    /// Perform the first firmware-visible mutation only after every message input
    /// has been validated. After `write_command()` succeeds, the remaining path
    /// is intentionally infallible: enqueue the already-formed RunWorkQueue and
    /// advance host submission bookkeeping. The EP21 doorbell is still a later
    /// caller boundary and E187 has no caller for this method.
    pub(crate) fn run(
        mut self,
        channel: &mut channel::PipeChannel::ver,
    ) -> Result<G15SelectedRunCommit> {
        #[ver(G != G15)]
        {
            let _ = channel;
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if self.command_count != 1 {
                return Err(EINVAL);
            }
            let queue_info_fwva = self.transport.queue_info_fwva()?;
            let inner = self.inner.as_mut().ok_or(EIO)?;
            let event = inner.event.as_ref().ok_or(EIO)?;
            let event_slot: u8 = event.0.slot().try_into()?;
            let command_index = inner.pending.len().checked_sub(1).ok_or(EIO)?;
            let command_fwva = inner.pending[command_index].inner.gpu_va();
            let pipe_type = inner.pipe_type;

            // E188 exact authority: first/new state belongs to the selected
            // channel, not generic WorkQueueInner.new. The first submission also
            // starts the E191 scheduler publication transaction before any
            // selected command-ring mutation can become visible.
            let first_submission = self.transport.begin_submission()?;

            let wptr = match self.transport.write_command(command_fwva) {
                Ok(wptr) => wptr,
                Err(err) => {
                    self.transport.fail_submission(first_submission);
                    return Err(err);
                }
            };

            let msg = fw::channels::RunWorkQueueMsg::ver {
                g15_timestamp: U64(g15_submission_timestamp()),
                g15_work_queue_fwva: U64(queue_info_fwva.get()),
                g15_pipe_type: pipe_type,
                g15_wptr: wptr as u16,
                g15_event_slot: event_slot,
                g15_is_new: first_submission,
            };
            channel.send(&msg);
            inner.submit_seq += self.command_count as u64;

            let completion = G15SelectedRunCommit {
                transport: self.transport.clone(),
                first_submission,
                completed: false,
            };
            let inner = self.inner.take().expect("selected submission lost inner");
            core::mem::drop(inner);
            Ok(completion)
        }
    }
}

#[versions(AGX)]
impl<'a> Drop for G15SelectedJobSubmission::ver<'a> {
    fn drop(&mut self) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let new_len = inner.pending.len() - self.command_count;
        inner.pending.truncate(new_len);

        let event = inner.event.as_mut().expect("selected submission lost event");
        event.1.sub(self.event_count as u32);
        let val = event.1;
        inner.commit_seq -= self.command_count as u64;
        inner.event_seq -= self.event_count as u64;
        inner.last_submitted = Some(val);
    }
}

#[versions(AGX)]
impl Drop for Job::ver {
    fn drop(&mut self) {
        mod_pr_debug!("WorkQueue: Dropping Job\n");
        let mut inner = self.wq.inner.lock();

        if !self.committed {
            pr_info!(
                "WorkQueue: Dropping uncommitted job with {} events\n",
                self.event_count
            );
        }

        if self.committed && !self.submitted {
            let pipe_type = inner.pipe_type;
            let event = inner.event.as_mut().expect("Job lost its event");
            pr_info!(
                "WorkQueue({:?}): Roll back {} events (slot {} val {:#x?}) and {} commands\n",
                pipe_type,
                self.event_count,
                event.0.slot(),
                event.1,
                self.pending.len()
            );
            event.1.sub(self.event_count as u32);
            inner.commit_seq -= self.pending.len() as u64;
            inner.event_seq -= self.event_count as u64;
        }

        inner.pending_jobs -= 1;

        if inner.pending.is_empty() && inner.pending_jobs == 0 {
            mod_pr_debug!("WorkQueue({:?}): Dropping event\n", inner.pipe_type);
            inner.event = None;
            inner.last_submitted = None;
            inner.last_completed = None;
        }
        mod_pr_debug!("WorkQueue({:?}): Dropped Job\n", inner.pipe_type);
    }
}

#[versions(AGX)]
impl<'a> Drop for JobSubmission::ver<'a> {
    fn drop(&mut self) {
        let inner = self.inner.as_mut().expect("No inner?");
        mod_pr_debug!("WorkQueue({:?}): Dropping JobSubmission\n", inner.pipe_type);

        let new_len = inner.pending.len() - self.command_count;
        inner.pending.truncate(new_len);

        let pipe_type = inner.pipe_type;
        let event = inner.event.as_mut().expect("JobSubmission lost its event");
        pr_info!(
            "WorkQueue({:?}): JobSubmission: Roll back {} events (slot {} val {:#x?}) and {} commands\n",
            pipe_type,
            self.event_count,
            event.0.slot(),
            event.1,
            self.command_count
        );
        event.1.sub(self.event_count as u32);
        let val = event.1;
        inner.commit_seq -= self.command_count as u64;
        inner.event_seq -= self.event_count as u64;
        inner.last_submitted = Some(val);
        mod_pr_debug!("WorkQueue({:?}): Dropped JobSubmission\n", inner.pipe_type);
    }
}

#[versions(AGX)]
impl WorkQueueInner::ver {
    /// Return the number of free entries in the workqueue
    pub(crate) fn free_space(&self) -> usize {
        self.size as usize - self.pending.len() - 1
    }

    pub(crate) fn free_slots(&self) -> usize {
        let busy_slots = if let Some(ls) = self.last_submitted {
            let lc = self
                .last_completed
                .expect("last_submitted but not completed?");
            ls.delta(&lc)
        } else {
            0
        };

        ((MAX_JOB_SLOTS as i32) - busy_slots).max(0) as usize
    }
}

#[versions(AGX)]
impl WorkQueue::ver {
    /// Create a new WorkQueue of a given type and priority.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: &driver::AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
        event_manager: Arc<event::EventManager>,
        gpu_context: Arc<GpuContext>,
        notifier_list: Arc<GpuObject<fw::event::NotifierList>>,
        g15_owned_channel_lifetime: Option<Arc<dyn G15WorkQueueTransport>>,
        pipe_type: PipeType,
        id: u64,
        priority: u32,
        size: u32,
    ) -> Result<Arc<WorkQueue::ver>> {
        let gpu_buf = alloc.private.array_empty_tagged(0x2c18, b"GPBF")?;
        let mut state = alloc.shared.new_default::<RingState>()?;
        let ring = alloc.shared.array_empty(size as usize)?;
        let mut prio = *raw::PRIORITY.get(priority as usize).ok_or(EINVAL)?;

        if pipe_type == PipeType::Compute && !debug_enabled(DebugFlags::Debug0) {
            // Hack to disable compute preemption until we fix it
            prio.0 = 0;
            prio.5 = 1;
        }

        let inner = WorkQueueInner::ver {
            dev: dev.into(),
            event_manager,
            // Use shared (coherent) state with verbose faults so we can dump state correctly
            info: if debug_enabled(DebugFlags::VerboseFaults) {
                &mut alloc.shared
            } else {
                &mut alloc.private
            }
            .new_init(
                try_init!(QueueInfo::ver {
                    state: {
                        state.with_mut(|raw, _inner| {
                            raw.rb_size = size;
                        });
                        state
                    },
                    ring,
                    gpu_buf,
                    notifier_list: notifier_list,
                    gpu_context: gpu_context,
                }),
                |inner, _p| {
                    try_init!(raw::QueueInfo::ver {
                        state: inner.state.gpu_pointer(),
                        ring: inner.ring.gpu_pointer(),
                        notifier_list: inner.notifier_list.gpu_pointer(),
                        gpu_buf: inner.gpu_buf.gpu_pointer(),
                        gpu_rptr1: Default::default(),
                        gpu_rptr2: Default::default(),
                        gpu_rptr3: Default::default(),
                        event_id: AtomicI32::new(-1),
                        priority: prio,
                        unk_4c: -1,
                        uuid: id as u32,
                        unk_54: -1,
                        unk_58: Default::default(),
                        busy: Default::default(),
                        __pad: Default::default(),
                        #[ver(V >= V13_2 && G < G14X)]
                        unk_84_0: 0,
                        unk_84_state: Default::default(),
                        error_count: Default::default(),
                        unk_8c: 0,
                        unk_90: 0,
                        unk_94: 0,
                        pending: Default::default(),
                        unk_9c: 0,
                        gpu_context: inner.gpu_context.gpu_pointer(),
                        #[ver(G != G15)]
                        unk_a8: Default::default(),
                        #[ver(G == G15)]
                        cdm_backoff_timeout_ac: 4,
                        #[ver(G == G15)]
                        pad_ad: Default::default(),
                        #[ver(V >= V13_2 && G < G14X && G != G15)]
                        unk_b0: 0,
                    })
                },
            )?,
            new: true,
            pipe_type,
            size,
            wptr: 0,
            pending: KVec::new(),
            last_token: None,
            event: None,
            priority,
            pending_jobs: 0,
            commit_seq: 0,
            submit_seq: 0,
            event_seq: 0,
            last_completed: None,
            last_submitted: None,
        };

        let info_pointer = inner.info.weak_pointer();

        Arc::pin_init(
            pin_init!(Self {
                _g15_owned_channel_lifetime: g15_owned_channel_lifetime,
                info_pointer,
                inner <- match pipe_type {
                    PipeType::Vertex => new_mutex!(inner, "WorkQueue::inner (Vertex)"),
                    PipeType::Fragment => new_mutex!(inner, "WorkQueue::inner (Fragment)"),
                    PipeType::Compute => new_mutex!(inner, "WorkQueue::inner (Compute)"),
                },
            }),
            GFP_KERNEL,
        )
    }

    pub(crate) fn event_info(&self) -> Option<QueueEventInfo::ver> {
        let inner = self.inner.lock();

        inner.event.as_ref().map(|ev| QueueEventInfo::ver {
            stamp_pointer: ev.0.stamp_pointer(),
            fw_stamp_pointer: ev.0.fw_stamp_pointer(),
            slot: ev.0.slot(),
            value: ev.1,
            cmd_seq: inner.commit_seq,
            event_seq: inner.event_seq,
            info_ptr: self.info_pointer,
        })
    }

    pub(crate) fn new_job(self: &Arc<Self>, fence: dma_fence::Fence) -> Result<Job::ver> {
        let mut inner = self.inner.lock();

        if inner.event.is_none() {
            mod_pr_debug!("WorkQueue({:?}): Grabbing event\n", inner.pipe_type);
            let event = inner.event_manager.get(inner.last_token, self.clone())?;
            let cur = event.current();
            inner.last_token = Some(event.token());
            mod_pr_debug!(
                "WorkQueue({:?}): Grabbed event slot {}: {:#x?}\n",
                inner.pipe_type,
                event.slot(),
                cur
            );
            inner.event = Some((event, cur));
            inner.last_submitted = Some(cur);
            inner.last_completed = Some(cur);
        }

        inner.pending_jobs += 1;

        let ev = &inner.event.as_ref().unwrap();

        mod_pr_debug!(
            "WorkQueue({:?}): New job at value {:#x?} slot {}\n",
            inner.pipe_type,
            ev.1,
            ev.0.slot()
        );
        Ok(Job::ver {
            wq: self.clone(),
            event_info: QueueEventInfo::ver {
                stamp_pointer: ev.0.stamp_pointer(),
                fw_stamp_pointer: ev.0.fw_stamp_pointer(),
                slot: ev.0.slot(),
                value: ev.1,
                cmd_seq: inner.commit_seq,
                event_seq: inner.event_seq,
                info_ptr: self.info_pointer,
            },
            start_value: ev.1,
            pending: KVec::new(),
            event_count: 0,
            committed: false,
            submitted: false,
            fence,
        })
    }

    pub(crate) fn pipe_type(&self) -> PipeType {
        self.inner.lock().pipe_type
    }

    pub(crate) fn dump_info(&self) {
        pr_info!("WorkQueue @ {:?}:", self.info_pointer);
        self.inner.lock().info.with(|raw, _inner| {
            pr_info!("  GPU rptr1: {:#x}", raw.gpu_rptr1.load(Ordering::Relaxed));
            pr_info!("  GPU rptr1: {:#x}", raw.gpu_rptr2.load(Ordering::Relaxed));
            pr_info!("  GPU rptr1: {:#x}", raw.gpu_rptr3.load(Ordering::Relaxed));
            pr_info!("  Event ID: {:#x}", raw.event_id.load(Ordering::Relaxed));
            pr_info!("  Busy: {:#x}", raw.busy.load(Ordering::Relaxed));
            pr_info!("  Unk 84: {:#x}", raw.unk_84_state.load(Ordering::Relaxed));
            pr_info!(
                "  Error count: {:#x}",
                raw.error_count.load(Ordering::Relaxed)
            );
            pr_info!("  Pending: {:#x}", raw.pending.load(Ordering::Relaxed));
        });
    }

    pub(crate) fn info_pointer(&self) -> GpuWeakPointer<QueueInfo::ver> {
        self.info_pointer
    }

    /// E185 compile-only selected-channel transport view. The normal generic
    /// QueueInfo remains allocated, but this is the only typed route to the
    /// exact G15 firmware-facing QueueInfo/ring state.
    #[allow(dead_code)]
    pub(crate) fn g15_selected_transport_state(
        &self,
    ) -> Result<(NonZeroU64, G15WorkQueueTransportState)> {
        let transport = self._g15_owned_channel_lifetime.as_ref().ok_or(EINVAL)?;
        Ok((transport.queue_info_fwva()?, transport.state()?))
    }

    /// E185 compile-only command-placement route. No submit/run caller uses it
    /// yet; E184 requires this bridge before any G15 command can be published.
    #[allow(dead_code)]
    pub(crate) fn g15_selected_transport_write_command(
        &self,
        command_fwva: NonZeroU64,
    ) -> Result<u32> {
        self._g15_owned_channel_lifetime
            .as_ref()
            .ok_or(EINVAL)?
            .write_command(command_fwva)
    }

    /// Bounded G15 scheduler-registration probe. Publish exactly one firmware
    /// Barrier/type-4 record which waits on a fresh allocation-zero Apple stamp
    /// counter, then synchronously retire the scheduler/context resource with
    /// native G15 ReleaseResource. No GPU engine command is present.
    pub(crate) fn g15_register_barrier(
        self: &Arc<Self>,
        gpu: &gpu::GpuManager::ver,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = gpu;
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            let (
                context,
                pipe_type,
                priority,
                info_pointer,
                event_slot,
                wait_stamp,
                wait_value,
                stamp_self,
            ) = {
                let mut inner = self.inner.lock();

                if !inner.new
                    || inner.wptr != 0
                    || !inner.pending.is_empty()
                    || inner.event.is_some()
                    || inner.pending_jobs != 0
                {
                    return Err(EBUSY);
                }

                let event = inner.event_manager.get(inner.last_token, self.clone())?;
                let cur = event.current();
                let event_slot: u8 = event.slot().try_into()?;

                // E194 moves the G15 EventManager itself into Apple's exact
                // zero-based counter domain. This bounded registration probe
                // still accepts only a never-submitted fresh stamp.
                if cur.raw() != 0 {
                    return Err(EBUSY);
                }
                let wait_value = cur;
                let wait_stamp = event.fw_stamp_pointer();
                let stamp_self = wait_value.next();
                inner.last_token = Some(event.token());
                inner.last_submitted = Some(cur);
                inner.last_completed = Some(cur);
                inner.event = Some((event, cur));

                let context = inner
                    .info
                    .with(|_raw, info_inner| info_inner.gpu_context.clone());
                (
                    context,
                    inner.pipe_type,
                    inner.priority,
                    self.info_pointer,
                    event_slot,
                    wait_stamp,
                    wait_value,
                    stamp_self,
                )
            };

            let barrier = {
                let mut alloc = gpu.alloc();
                // `try_init!` binds field names internally; keep the second
                // identical pointer under a distinct local as E059 did.
                let wait_stamp_2 = wait_stamp;
                alloc.private.new_init(
                    pin_init::zeroed::<fw::workqueue::Barrier::ver>(),
                    |_inner, _p| {
                        try_init!(fw::workqueue::raw::Barrier::ver {
                            tag: fw::workqueue::CommandType::Barrier,
                            wait_stamp,
                            wait_stamp_2,
                            wait_value,
                            wait_slot: event_slot as u32,
                            stamp_self,
                            uuid: 0xffffbbbb,
                            external_barrier: 0,
                            internal_barrier_type: 1,
                            padding: Default::default(),
                        })
                    },
                )?
            };
            let barrier_va = barrier.gpu_va().get();

            {
                let mut inner = self.inner.lock();
                if !inner.new
                    || inner.wptr != 0
                    || !inner.pending.is_empty()
                    || inner.pending_jobs != 0
                    || inner.event.as_ref().map(|e| e.0.slot()) != Some(event_slot as u32)
                {
                    return Err(EBUSY);
                }

                inner.info.ring[0] = barrier_va;
                inner.info.state.with(|raw, _inner| {
                    raw.cpu_wptr.store(1, Ordering::Release);
                });
                inner.wptr = 1;
                // One-shot even if transport becomes uncertain.
                inner.new = false;
            }

            dev_info!(
                context.dev.as_ref(),
                "T8122 G15 E171 barrier registration slot={} barrier={:#x} wptr=1 wait={:#x} stamp_self={:#x}\n",
                event_slot,
                barrier_va,
                wait_value.raw(),
                stamp_self.raw()
            );

            // Mark before the first transport side effect. Queue::drop() will
            // retain QueueInfo backing if release does not reach known-good.
            context.mark_published_to_firmware();

            let publish_result = gpu.g15_publish_barrier_queue(
                pipe_type,
                priority,
                info_pointer,
                event_slot,
                1,
            );

            // If pipe delivery/retirement is uncertain, retain the command
            // backing so delayed firmware cannot fetch a freed Barrier object.
            if publish_result.is_err() {
                core::mem::forget(barrier);
            }

            let fields = context
                .data()
                .with(|raw, _inner| raw.g15_release_resource_fields());
            dev_info!(
                context.dev.as_ref(),
                "T8122 G15 E171 QueueInfo registration result={:?}, context fields={:02x?}\n",
                publish_result,
                fields
            );

            let release_result = gpu.release_context_now(context.data());
            if release_result.is_ok() {
                context.mark_released_from_firmware();
            }
            dev_info!(
                context.dev.as_ref(),
                "T8122 G15 E171 ReleaseResource result={:?}\n",
                release_result
            );

            match (publish_result, release_result) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(e), Ok(())) => Err(e),
                (_, Err(e)) => Err(e),
            }
        }
    }
}

/// Trait used to erase the version-specific type of WorkQueues, to avoid leaking
/// version-specificity into the event module.
pub(crate) trait WorkQueue {
    /// Cast as an Any type.
    fn as_any(&self) -> &dyn Any;

    fn signal(&self) -> bool;
    fn mark_error(&self, value: event::EventValue, error: WorkError);
    fn fail_all(&self, error: WorkError);
}

#[versions(AGX)]
impl WorkQueue for WorkQueue::ver {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Signal a workqueue that some work was completed.
    ///
    /// This will check the event stamp value to find out exactly how many commands were processed.
    fn signal(&self) -> bool {
        let mut inner = self.inner.lock();
        let event = inner.event.as_ref();
        let value = match event {
            None => {
                mod_pr_debug!("WorkQueue: signal() called but no event?\n");

                if inner.pending_jobs > 0 || !inner.pending.is_empty() {
                    pr_crit!("WorkQueue: signal() called with no event and pending jobs.\n");
                }
                return true;
            }
            Some(event) => event.0.current(),
        };

        if let Some(lc) = inner.last_completed {
            if value < lc {
                pr_err!(
                    "WorkQueue: event rolled back? cur {:#x?}, lc {:#x?}, ls {:#x?}",
                    value,
                    inner.last_completed,
                    inner.last_submitted
                );
            }
        } else {
            pr_crit!("WorkQueue: signal() called with no last_completed.\n");
        }
        inner.last_completed = Some(value);

        mod_pr_debug!(
            "WorkQueue({:?}): Signaling event {:?} value {:#x?}\n",
            inner.pipe_type,
            inner.last_token,
            value
        );

        let mut completed_commands: usize = 0;

        for cmd in inner.pending.iter() {
            if cmd.inner.value() <= value {
                mod_pr_debug!(
                    "WorkQueue({:?}): Command at value {:#x?} complete\n",
                    inner.pipe_type,
                    cmd.inner.value()
                );
                completed_commands += 1;
            } else {
                break;
            }
        }

        if completed_commands == 0 {
            return inner.pending.is_empty();
        }

        let last_wptr = inner.pending[completed_commands - 1].inner.wptr();
        let pipe_type = inner.pipe_type;

        for mut cmd in inner.pending.drain(..completed_commands) {
            mod_pr_debug!(
                "WorkQueue({:?}): Queueing command @ {:?} for cleanup\n",
                pipe_type,
                cmd.inner.gpu_va()
            );
            cmd.as_mut().inner_mut().complete();
            workqueue::system().enqueue(cmd);
        }

        mod_pr_debug!(
            "WorkQueue({:?}): Completed {} commands, left pending {}, ls {:#x?}, lc {:#x?}\n",
            inner.pipe_type,
            completed_commands,
            inner.pending.len(),
            inner.last_submitted,
            inner.last_completed,
        );

        inner
            .info
            .state
            .with(|raw, _inner| raw.cpu_freeptr.store(last_wptr, Ordering::Release));

        let empty = inner.pending.is_empty();
        if empty && inner.pending_jobs == 0 {
            inner.event = None;
            inner.last_submitted = None;
            inner.last_completed = None;
        }

        empty
    }

    /// Mark this queue's work up to a certain stamp value as having failed.
    fn mark_error(&self, value: event::EventValue, error: WorkError) {
        // If anything is marked completed, we can consider it successful
        // at this point, even if we didn't get the signal event yet.
        self.signal();

        let mut inner = self.inner.lock();

        if inner.event.is_none() {
            mod_pr_debug!("WorkQueue: signal_fault() called but no event?\n");

            if inner.pending_jobs > 0 || !inner.pending.is_empty() {
                pr_crit!("WorkQueue: signal_fault() called with no event and pending jobs.\n");
            }
            return;
        }

        mod_pr_debug!(
            "WorkQueue({:?}): Signaling fault for event {:?} at value {:#x?}\n",
            inner.pipe_type,
            inner.last_token,
            value
        );

        for cmd in inner.pending.iter_mut() {
            if cmd.inner.value() <= value {
                cmd.as_mut().inner_mut().mark_error(error);
            } else {
                break;
            }
        }
    }

    /// Mark all of this queue's work as having failed, and complete it.
    fn fail_all(&self, error: WorkError) {
        // If anything is marked completed, we can consider it successful
        // at this point, even if we didn't get the signal event yet.
        self.signal();

        let mut inner = self.inner.lock();

        if inner.event.is_none() {
            mod_pr_debug!("WorkQueue: fail_all() called but no event?\n");

            if inner.pending_jobs > 0 || !inner.pending.is_empty() {
                pr_crit!("WorkQueue: fail_all() called with no event and pending jobs.\n");
            }
            return;
        }

        mod_pr_debug!(
            "WorkQueue({:?}): Failing all jobs {:?}\n",
            inner.pipe_type,
            error
        );

        let mut cmds = KVec::new();

        core::mem::swap(&mut inner.pending, &mut cmds);

        if inner.pending_jobs == 0 {
            inner.event = None;
        }

        core::mem::drop(inner);

        for mut cmd in cmds {
            cmd.as_mut().inner_mut().mark_error(error);
            cmd.as_mut().inner_mut().complete();
        }
    }
}
