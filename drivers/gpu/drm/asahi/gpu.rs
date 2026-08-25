// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Top-level GPU manager
//!
//! This module is the root of all GPU firmware management for a given driver instance. It is
//! responsible for initialization, owning the top-level managers (events, UAT, etc.), and
//! communicating with the raw RtKit endpoints to send and receive messages to/from the GPU
//! firmware.
//!
//! It is also the point where diverging driver firmware/GPU variants (using the versions macro)
//! are unified, so that the top level of the driver itself (in `driver`) does not have to concern
//! itself with version dependence.

use core::any::Any;
use core::ops::Range;
use core::slice;
use core::sync::atomic::{
    AtomicBool,
    AtomicU64,
    Ordering, //
};

use kernel::{
    c_str,
    drm::gem::shmem,
    error::code::*,
    io::mem::{
        Mem,
        MemFlag, //
    },
    iosys_map::IoSysMapRef,
    macros::versions,
    new_mutex,
    prelude::*,
    soc::apple::rtkit,
    sync::{
        lock::{
            mutex::MutexBackend,
            Guard, //
        },
        Arc,
        Mutex,
        UniqueArc, //
    },
    time::{
        Delta,
        Instant,
        Monotonic, //
    },
    types::ForeignOwnable, //
};
#[cfg(CONFIG_DEV_COREDUMP)]
use kernel::{
    devcoredump,
    time::msecs_to_jiffies, //
};

use crate::alloc::Allocator;
use crate::debug::*;
use crate::driver::{
    AsahiDevRef,
    AsahiDevice, //
};
use crate::fw::channels::{
    ChannelErrorType,
    PipeType, //
};
use crate::fw::types::{
    U32,
    U64, //
};
use crate::{
    alloc,
    buffer,
    channel,
    event,
    fw,
    gem,
    hw,
    initdata,
    mem,
    mmu,
    queue,
    regs,
    workqueue, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Gpu;

/// Firmware endpoint for init & incoming notifications.
const EP_FIRMWARE: u8 = 0x20;

/// Doorbell endpoint for work/message submissions.
const EP_DOORBELL: u8 = 0x21;

/// Initialize the GPU firmware.
const MSG_INIT: u64 = 0x81 << 48;
const INIT_DATA_MASK: u64 = (1 << 44) - 1;

/// TX channel doorbell.
const MSG_TX_DOORBELL: u64 = 0x83 << 48;
/// Firmware control channel doorbell.
const MSG_FWCTL: u64 = 0x84 << 48;
// /// Halt the firmware (?).
// const MSG_HALT: u64 = 0x85 << 48;

/// Receive channel doorbell notification.
const MSG_RX_DOORBELL: u64 = 0x42 << 48;

/// Doorbell number for firmware kicks/wakeups.
const DOORBELL_KICKFW: u64 = 0x10;
/// Doorbell number for device control channel kicks.
const DOORBELL_DEVCTRL: u64 = 0x11;

// Upper kernel half VA address ranges.
/// Private (cached) firmware structure VA range base.
const IOVA_KERN_PRIV_RANGE: Range<u64> = 0xffffffa000000000..0xffffffa600000000;
/// Private (cached) GPU-RO firmware structure VA range base.
const IOVA_KERN_GPU_RO_RANGE: Range<u64> = 0xffffffa600000000..0xffffffa800000000;
/// Shared (uncached) firmware structure VA range base.
const IOVA_KERN_SHARED_RANGE: Range<u64> = 0xffffffa800000000..0xffffffaa00000000;
/// Shared (uncached) read-only firmware structure VA range base.
const IOVA_KERN_SHARED_RO_RANGE: Range<u64> = 0xffffffaa00000000..0xffffffac00000000;
/// GPU/FW shared structure VA range base.
const IOVA_KERN_GPU_RANGE: Range<u64> = 0xffffffac00000000..0xffffffae00000000;
/// GPU/FW shared structure VA range base.
const IOVA_KERN_RTKIT_RANGE: Range<u64> = 0xffffffae00000000..0xffffffae10000000;
/// Shared (uncached) timestamp region.
pub(crate) const IOVA_KERN_TIMESTAMP_RANGE: Range<u64> = 0xffffffae10000000..0xffffffae14000000;
/// FW MMIO VA range base.
const IOVA_KERN_MMIO_RANGE: Range<u64> = 0xffffffaf00000000..0xffffffb000000000;

/// GPU/FW buffer manager control address (context 0 low)
pub(crate) const IOVA_KERN_GPU_BUFMGR_LOW: u64 = 0x20_0000_0000;
/// GPU/FW buffer manager control address (context 0 high)
pub(crate) const IOVA_KERN_GPU_BUFMGR_HIGH: u64 = 0xffffffaeffff0000;

/// Timeout for entering the halt state after a fault or request.
const HALT_ENTER_TIMEOUT: Delta = Delta::from_millis(100);

/// Maximum amount of firmware-private memory garbage allowed before collection.
/// Collection flushes the FW cache and is expensive, so this needs to be
/// reasonably high.
const MAX_FW_ALLOC_GARBAGE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum count of firmware-private memory garbage objects allowed before collection.
/// This works out to 16K of memory in the garbage list (8 bytes each), which keeps us
/// within the safe range for kmalloc (on 16K page systems).
const MAX_FW_ALLOC_GARBAGE_OBJECTS: usize = 2048;

/// Global allocators used for kernel-half structures.
pub(crate) struct KernelAllocators {
    pub(crate) private: alloc::DefaultAllocator,
    pub(crate) shared: alloc::DefaultAllocator,
    pub(crate) shared_ro: alloc::DefaultAllocator,
    #[allow(dead_code)]
    pub(crate) gpu: alloc::DefaultAllocator,
    pub(crate) gpu_ro: alloc::DefaultAllocator,
}

/// Receive (GPU->driver) ring buffer channels.
#[versions(AGX)]
#[pin_data]
struct RxChannels {
    event: channel::EventChannel::ver,
    fw_log: channel::FwLogChannel,
    ktrace: channel::KTraceChannel,
    stats: channel::StatsChannel::ver,
}

/// GPU work submission pipe channels (driver->GPU).
#[versions(AGX)]
struct PipeChannels {
    pub(crate) vtx: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
    pub(crate) frag: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
    pub(crate) comp: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
}

/// Misc command transmit (driver->GPU) channels.
#[versions(AGX)]
#[pin_data]
struct TxChannels {
    pub(crate) device_control: channel::DeviceControlChannel::ver,
}

/// Number of work submission pipes per type, one for each priority level.
const NUM_PIPES: usize = 4;

/// A generic monotonically incrementing ID used to uniquely identify object instances within the
/// driver.
pub(crate) struct ID(AtomicU64);

impl ID {
    /// Create a new ID counter with a given value.
    fn new(val: u64) -> ID {
        ID(AtomicU64::new(val))
    }

    /// Fetch the next unique ID.
    pub(crate) fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for ID {
    /// IDs default to starting at 2, as 0/1 are considered reserved for the system.
    fn default() -> Self {
        Self::new(2)
    }
}

/// A guard representing one active submission on the GPU. When dropped, decrements the active
/// submission count.
pub(crate) struct OpGuard(Arc<dyn GpuManagerPriv>);

impl Drop for OpGuard {
    fn drop(&mut self) {
        self.0.end_op();
    }
}

/// Set of global sequence IDs used in the driver.
#[derive(Default)]
pub(crate) struct SequenceIDs {
    /// `File` instance ID.
    pub(crate) file: ID,
    /// `Vm` instance ID.
    pub(crate) vm: ID,
    /// Submission instance ID.
    pub(crate) submission: ID,
    /// `Queue` instance ID.
    pub(crate) queue: ID,
}

/// Top-level GPU manager that owns all the global state relevant to the driver instance.
#[versions(AGX)]
#[pin_data]
pub(crate) struct GpuManager {
    dev: AsahiDevRef,
    cfg: &'static hw::HwConfig,
    dyncfg: hw::DynConfig,
    pub(crate) initdata: fw::types::GpuObject<fw::initdata::InitData::ver>,
    uat: mmu::Uat,
    crashed: AtomicBool,
    #[pin]
    alloc: Mutex<KernelAllocators>,
    io_mappings: KVec<mmu::KernelMapping>,
    next_mmio_iova: u64,
    #[pin]
    rtkit: Mutex<Option<rtkit::RtKit<GpuManager::ver>>>,
    #[pin]
    rx_channels: Mutex<RxChannels::ver>,
    #[pin]
    tx_channels: Mutex<TxChannels::ver>,
    // G15 replaces the legacy FwStatus-owned firmware-control channel with
    // q22 +0x4568/+0x4570 (0x20 state + 0x1800 ring). Keep the old channel
    // allocation entirely out of the G15 object until that successor is wired.
    #[ver(G != G15)]
    #[pin]
    fwctl_channel: Mutex<channel::FwCtlChannel>,
    pipes: PipeChannels::ver,
    event_manager: Arc<event::EventManager>,
    buffer_mgr: buffer::BufferManager::ver,
    ids: SequenceIDs,
    #[allow(clippy::vec_box)]
    #[pin]
    garbage_contexts: Mutex<KVec<KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>>>,
}

/// Trait used to abstract the firmware/GPU-dependent variants of the GpuManager.
pub(crate) trait GpuManager: Send + Sync {
    /// Cast as an Any type.
    fn as_any(&self) -> &dyn Any;
    /// Cast Arc<Self> as an Any type.
    fn arc_as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send>;
    /// Initialize the GPU.
    fn init(&self) -> Result;
    /// Update the GPU globals from global info
    ///
    /// TODO: Unclear what can and cannot be updated like this.
    fn update_globals(&self);
    /// Get a reference to the KernelAllocators.
    fn alloc(&self) -> Guard<'_, KernelAllocators, MutexBackend>;
    /// Create a new `Vm` given a unique `File` ID.
    fn new_vm(&self, kernel_range: Range<u64>) -> Result<mmu::Vm>;
    /// Bind a `Vm` to an available slot and return the `VmBind`.
    fn bind_vm(&self, vm: &mmu::Vm) -> Result<mmu::VmBind>;
    /// Create a new user command queue.
    fn new_queue(
        &self,
        vm: mmu::Vm,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        g15_ualloc_range5_uncached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_cached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        priority: u32,
        usc_exec_base: u64,
    ) -> Result<KBox<dyn queue::Queue>>;
    /// Return a reference to the global `SequenceIDs` instance.
    fn ids(&self) -> &SequenceIDs;
    /// Kick the firmware (wake it up if asleep).
    ///
    /// This should be useful to reduce latency on work submission, so we can ask the firmware to
    /// wake up while we do some preparatory work for the work submission.
    fn kick_firmware(&self) -> Result;
    /// Flush the entire firmware cache.
    ///
    /// TODO: Does this actually work?
    fn flush_fw_cache(&self) -> Result;
    /// Handle a GPU work timeout event.
    fn handle_timeout(&self, counter: u32, event_slot: i32, unk: u32);
    /// Handle a GPU fault event.
    fn handle_fault(&self);
    /// Handle a channel error event.
    fn handle_channel_error(
        &self,
        error_type: ChannelErrorType,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    );
    /// Acknowledge a Buffer grow op.
    fn ack_grow(&self, buffer_slot: u32, vm_slot: u32, counter: u32);
    /// Send a firmware control command (secure cache flush).
    fn fwctl(&self, msg: fw::channels::FwCtlMsg) -> Result;
    /// Get the static GPU configuration for this SoC.
    fn get_cfg(&self) -> &'static hw::HwConfig;
    /// Get the dynamic GPU configuration for this SoC.
    fn get_dyncfg(&self) -> &hw::DynConfig;
    /// Register an unused context as garbage
    fn free_context(&self, data: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>);
    /// Check whether the GPU is crashed
    fn is_crashed(&self) -> bool;
    /// Map a BO as a timestamp buffer
    fn map_timestamp_buffer(
        &self,
        bo: gem::ObjectRef,
        range: Range<usize>,
    ) -> Result<mmu::KernelMapping>;
}

/// Private generic trait for functions that don't need to escape this module.
trait GpuManagerPriv {
    /// Decrement the pending submission counter.
    fn end_op(&self);
}

pub(crate) struct RtkitObject {
    vmap: shmem::VMap<gem::AsahiObject, u8>,
    mapping: mmu::KernelMapping,
}

pub(crate) struct RtkitPrealloc {
    phys: usize,
    mem: Mem,
}

pub(crate) enum RtkitBuffer {
    Allocated(RtkitObject),
    Preallocated(RtkitPrealloc),
}

impl rtkit::Buffer for RtkitBuffer {
    fn iova(&self) -> Result<usize> {
        Ok(match self {
            Self::Allocated(obj) => obj.mapping.iova() as usize,
            Self::Preallocated(obj) => obj.phys,
        })
    }

    fn buf(&mut self) -> Result<IoSysMapRef<'_, u8>> {
        Ok(match self {
            Self::Allocated(obj) => obj.vmap.get(),
            Self::Preallocated(obj) => obj.mem.as_iosys_map(),
        })
    }
}

#[versions(AGX)]
#[vtable]
impl rtkit::Operations for GpuManager::ver {
    type Data = Arc<GpuManager::ver>;
    type Buffer = RtkitBuffer;

    fn recv_message(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, ep: u8, msg: u64) {
        let dev = &data.dev;
        //dev_info!(dev.as_ref(), "RtKit message: {:#x}:{:#x}\n", ep, msg);

        if ep != EP_FIRMWARE || msg != MSG_RX_DOORBELL {
            dev_err!(dev.as_ref(), "Unknown message: {:#x}:{:#x}\n", ep, msg);
            return;
        }

        let mut ch = data.rx_channels.lock();

        ch.fw_log.poll();
        ch.ktrace.poll();
        ch.stats.poll();
        ch.event.poll();
    }

    fn crashed(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, crashlog: Option<&[u8]>) {
        let dev = &data.dev;

        data.crashed.store(true, Ordering::Relaxed);

        #[cfg(CONFIG_DEV_COREDUMP)]
        if let Err(e) = data.generate_crashdump(crashlog) {
            dev_err!(dev.as_ref(), "Could not generate crashdump: {:?}\n", e);
        }
        #[cfg(not(CONFIG_DEV_COREDUMP))]
        let _ = crashlog;

        if debug_enabled(DebugFlags::OopsOnGpuCrash) {
            panic!("GPU firmware crashed");
        } else {
            dev_err!(dev.as_ref(), "GPU firmware crashed, failing all jobs\n");
            data.event_manager.fail_all(workqueue::WorkError::NoDevice);
        }
    }

    fn shmem_alloc(
        data: <Self::Data as ForeignOwnable>::Borrowed<'_>,
        size: usize,
    ) -> Result<Self::Buffer> {
        let dev = &data.dev;
        mod_dev_dbg!(dev, "shmem_alloc() {:#x} bytes\n", size);

        let mut obj = gem::new_kernel_object(dev, size)?;
        let vmap = obj.gem.owned_vmap()?;
        let mapping = obj.map_into_range(
            data.uat.kernel_vm(),
            IOVA_KERN_RTKIT_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_FW_SHARED_RW,
            true,
        )?;
        mod_dev_dbg!(dev, "shmem_alloc() -> VA {:#x}\n", mapping.iova());
        Ok(RtkitBuffer::Allocated(RtkitObject { vmap, mapping }))
    }

    fn shmem_map(
        data: <Self::Data as ForeignOwnable>::Borrowed<'_>,
        iova: usize,
        size: usize,
    ) -> Result<Self::Buffer> {
        #[ver(G != G15)]
        {
            let _ = (data, iova, size);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            // G15 RTKit supplies its crashlog as a firmware-owned physical
            // DRAM allocation. The exact J615 live ADT places the observed
            // buffer inside a carveout, and live probing proves it has no AGX
            // UAT PTE but is CPU-readable with a WB memremap.
            //
            // Keep this deliberately narrow during bring-up: only accept the
            // high G15 DRAM aperture, page-aligned buffers, and a bounded size.
            if (iova >> 40) != 1
                || (iova & (mmu::UAT_PGSZ - 1)) != 0
                || size == 0
                || size > 0x0100_0000
            {
                dev_err!(
                    data.dev.as_ref(),
                    "G15 RTKit rejected preallocated buffer PA {:#x}, size {:#x}\n",
                    iova,
                    size
                );
                return Err(EINVAL);
            }

            // SAFETY: G15 firmware supplied this address as an RTKit
            // preallocated ordinary-memory buffer. The aperture/alignment/size
            // checks above keep MMIO and malformed requests out, and live J615
            // evidence confirms the buffer resides in firmware-carved DRAM.
            let mem = unsafe { Mem::try_new_phys(iova as u64, size, (MemFlag::WB).into())? };
            mod_dev_dbg!(
                data.dev.as_ref(),
                "G15 RTKit mapped firmware-preallocated physical buffer {:#x}+{:#x}\n",
                iova,
                size
            );
            Ok(RtkitBuffer::Preallocated(RtkitPrealloc { phys: iova, mem }))
        }
    }
}

#[versions(AGX)]
impl GpuManager::ver {
    /// Create a new GpuManager of this version/GPU combination.
    #[inline(never)]
    pub(crate) fn new(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<Arc<GpuManager::ver>> {
        let mgr = Self::build_pre_rtkit(dev, res, cfg)?;
        let mgr = Arc::from(mgr);

        let rtkit = rtkit::RtKit::<GpuManager::ver>::new(dev.as_ref(), None, 0, mgr.clone())?;

        *mgr.rtkit.lock() = Some(rtkit);

        {
            let mut rxc = mgr.rx_channels.lock();
            rxc.event.set_manager(mgr.clone());
        }

        Ok(mgr)
    }

    /// Build and wire the complete manager object graph up to, but excluding,
    /// RTKit construction. This is also the safe G15 bring-up boundary used by
    /// the build-and-destroy preflight.
    #[inline(never)]
    fn build_pre_rtkit(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<Pin<UniqueArc<GpuManager::ver>>> {
        let uat = Self::make_uat(dev, cfg)?;
        let dyncfg = Self::make_dyncfg(dev, res, cfg, &uat)?;

        let mut alloc = KernelAllocators {
            private: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_PRIV_RANGE,
                0x80,
                mmu::PROT_FW_PRIV_RW,
                1024 * 1024,
                true,
                fmt!("Kernel Private"),
                true,
            )?,
            shared: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_SHARED_RANGE,
                0x80,
                mmu::PROT_FW_SHARED_RW,
                1024 * 1024,
                true,
                fmt!("Kernel Shared"),
                false,
            )?,
            shared_ro: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_SHARED_RO_RANGE,
                0x80,
                mmu::PROT_FW_SHARED_RO,
                64 * 1024,
                true,
                fmt!("Kernel RO Shared"),
                false,
            )?,
            gpu: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_GPU_RANGE,
                0x80,
                mmu::PROT_GPU_FW_SHARED_RW,
                64 * 1024,
                true,
                fmt!("Kernel GPU Shared"),
                false,
            )?,
            gpu_ro: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_GPU_RO_RANGE,
                0x80,
                mmu::PROT_GPU_RO_FW_PRIV_RW,
                1024 * 1024,
                true,
                fmt!("Kernel GPU RO Shared"),
                true,
            )?,
        };

        let event_manager = Self::make_event_manager(&mut alloc)?;
        let mut initdata = Self::make_initdata(dev, cfg, &dyncfg, &mut alloc)?;

        initdata.runtime_pointers.buffer_mgr_ctl_low_mapping =
            Some(initdata.runtime_pointers.buffer_mgr_ctl.map_at(
                uat.kernel_lower_vm(),
                IOVA_KERN_GPU_BUFMGR_LOW,
                mmu::PROT_GPU_SHARED_RW,
                false,
            )?);
        initdata.runtime_pointers.buffer_mgr_ctl_high_mapping =
            Some(initdata.runtime_pointers.buffer_mgr_ctl.map_at(
                uat.kernel_vm(),
                IOVA_KERN_GPU_BUFMGR_HIGH,
                mmu::PROT_FW_SHARED_RW,
                false,
            )?);

        let mut mgr = Self::make_mgr(dev, cfg, dyncfg, uat, alloc, event_manager, initdata)?;

        #[ver(G != G15)]
        {
            let fwctl = mgr.fwctl_channel.lock();
            let p_fwctl = fwctl.to_raw();
            core::mem::drop(fwctl);

            mgr.as_mut()
                .initdata_mut()
                .fw_status
                .with_mut(|raw, _inner| {
                    raw.fwctl_channel = p_fwctl;
                });
        }
        #[ver(G == G15)]
        {
            // G15 no longer exposes the legacy FwStatus object in InitData.
            // Its exact 0x20 state + 0x1800 mapping/control ring already live
            // in q22 +0x4568/+0x4570. Runtime notification production remains
            // fail-closed; there is no separate legacy FwCtlChannel to wire.
        }

        {
            let txc = mgr.tx_channels.lock();
            #[ver(G != G15)]
            let p_device_control = txc.device_control.to_raw();
            #[ver(G == G15)]
            let p_device_control = txc.device_control.to_raw_g15();
            core::mem::drop(txc);

            let rxc = mgr.rx_channels.lock();
            let p_event = rxc.event.to_raw();
            let p_fw_log = rxc.fw_log.to_raw();
            let p_ktrace = rxc.ktrace.to_raw();
            let p_stats = rxc.stats.to_raw();
            let p_fwlog_buf = rxc.fw_log.get_buf();
            core::mem::drop(rxc);

            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .with_mut(|raw, _inner| {
                    raw.device_control = p_device_control;
                    raw.event = p_event;
                    raw.fw_log = p_fw_log;
                    raw.ktrace = p_ktrace;
                    raw.stats = p_stats;
                    raw.fwlog_buf = Some(p_fwlog_buf);
                });
        }

        #[ver(G != G15)]
        let mut p_pipes: KVec<fw::initdata::raw::PipeChannels::ver> = KVec::new();
        #[ver(G == G15)]
        let mut p_pipes: KVec<fw::initdata::raw::G15PipeChannels> = KVec::new();

        for ((v, f), c) in mgr
            .pipes
            .vtx
            .iter()
            .zip(&mgr.pipes.frag)
            .zip(&mgr.pipes.comp)
        {
            #[ver(G != G15)]
            p_pipes.push(
                fw::initdata::raw::PipeChannels::ver {
                    vtx: v.lock().to_raw(),
                    frag: f.lock().to_raw(),
                    comp: c.lock().to_raw(),
                },
                GFP_KERNEL,
            )?;
            #[ver(G == G15)]
            p_pipes.push(
                fw::initdata::raw::G15PipeChannels {
                    vtx: v.lock().to_raw_g15(),
                    frag: f.lock().to_raw_g15(),
                    comp: c.lock().to_raw_g15(),
                },
                GFP_KERNEL,
            )?;
        }

        mgr.as_mut()
            .initdata_mut()
            .runtime_pointers
            .with_mut(|raw, _inner| {
                for (i, p) in p_pipes.into_iter().enumerate() {
                    raw.pipes[i].vtx = p.vtx;
                    raw.pipes[i].frag = p.frag;
                    raw.pipes[i].comp = p.comp;
                }
            });

        for (i, map) in cfg.io_mappings.iter().enumerate() {
            if let Some(map) = map.as_ref() {
                Self::iomap(&mut mgr, cfg, i, map)?;
            }
        }

        #[ver(V >= V13_0B4)]
        if let Some(base) = cfg.sram_base {
            let size = cfg.sram_size.unwrap();
            let iova = mgr.as_mut().alloc_mmio_iova(size);

            let mapping = mgr
                .uat
                .kernel_vm()
                .map_io(iova, base, size, mmu::PROT_FW_SHARED_RW)?;

            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .hwdata_b
                .with_mut(|raw, _| {
                    raw.sgx_sram_ptr = U64(mapping.iova());
                });

            mgr.as_mut().io_mappings_mut().push(mapping, GFP_KERNEL)?;
        }

        Ok(mgr)
    }

    /// G15 bring-up checkpoint: construct the full InitData allocation tree,
    /// validate the deterministic J615/C0 HwDataA pre-tail in CPU memory, and
    /// drop everything without creating RTKit or sending firmware messages.
    pub(crate) fn initdata_preflight(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<()> {
        #[ver(G != G15)]
        {
            let _ = (dev, res, cfg);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            let uat = Self::make_uat(dev, cfg)?;
            let dyncfg = Self::make_dyncfg(dev, res, cfg, &uat)?;

            let mut alloc = KernelAllocators {
                private: alloc::DefaultAllocator::new(
                    dev,
                    uat.kernel_vm(),
                    IOVA_KERN_PRIV_RANGE,
                    0x80,
                    mmu::PROT_FW_PRIV_RW,
                    1024 * 1024,
                    true,
                    fmt!("Kernel Private"),
                    true,
                )?,
                shared: alloc::DefaultAllocator::new(
                    dev,
                    uat.kernel_vm(),
                    IOVA_KERN_SHARED_RANGE,
                    0x80,
                    mmu::PROT_FW_SHARED_RW,
                    1024 * 1024,
                    true,
                    fmt!("Kernel Shared"),
                    false,
                )?,
                shared_ro: alloc::DefaultAllocator::new(
                    dev,
                    uat.kernel_vm(),
                    IOVA_KERN_SHARED_RO_RANGE,
                    0x80,
                    mmu::PROT_FW_SHARED_RO,
                    64 * 1024,
                    true,
                    fmt!("Kernel RO Shared"),
                    false,
                )?,
                gpu: alloc::DefaultAllocator::new(
                    dev,
                    uat.kernel_vm(),
                    IOVA_KERN_GPU_RANGE,
                    0x80,
                    mmu::PROT_GPU_FW_SHARED_RW,
                    64 * 1024,
                    true,
                    fmt!("Kernel GPU Shared"),
                    false,
                )?,
                gpu_ro: alloc::DefaultAllocator::new(
                    dev,
                    uat.kernel_vm(),
                    IOVA_KERN_GPU_RO_RANGE,
                    0x80,
                    mmu::PROT_GPU_RO_FW_PRIV_RW,
                    1024 * 1024,
                    true,
                    fmt!("Kernel GPU RO Shared"),
                    true,
                )?,
            };

            let initdata = Self::make_initdata(dev, cfg, &dyncfg, &mut alloc)?;
            let hwdata_a_va = initdata.runtime_pointers.hwdata_a.gpu_va().get();
            let initdata_va = initdata.gpu_va().get();

            let pre_ok = initdata.runtime_pointers.hwdata_a.with(|raw, _inner| {
                let pre = &raw.g15_pretail_3a9c;
                let dpe = &pre.dpe_00c;
                let u64_is = |v: &U64, x: u64| v.0 == x;

                pre.constant_008.to_bits() == 0x40a0_0000
                    && dpe.loop_count_000 == 0
                    && dpe.all_ones_0cc.iter().all(|v| u64_is(v, u64::MAX))
                    && u64_is(&dpe.bootstrap_0ec, 0x0000_0000_0008_0000)
                    && dpe
                        .q_3fffff_0f4
                        .iter()
                        .all(|v| u64_is(v, 0x003f_ffff_003f_ffff))
                    && u64_is(&dpe.literal_114, 0x003f_0000_0000_0000)
                    && u64_is(&dpe.literal_13c, 0x2000_0000_0000_0000)
                    && dpe.all_ones_144.iter().all(|v| u64_is(v, u64::MAX))
                    && dpe
                        .q_0f07_164
                        .iter()
                        .all(|v| u64_is(v, 0x0f07_0f07_0f07_0f07))
                    && dpe
                        .bank1_184
                        .iter()
                        .all(|v| u64_is(v, 0x03ff_ffff_03ff_ffff))
                    && u64_is(&dpe.special_384, 0xa000_0000_0000_0017)
                    && dpe.all_ones_38c.iter().all(|v| u64_is(v, u64::MAX))
                    && dpe
                        .q_0f07_3ac
                        .iter()
                        .all(|v| u64_is(v, 0x0f07_0f07_0f07_0f07))
                    && dpe
                        .bank2_3cc
                        .iter()
                        .all(|v| u64_is(v, 0x03ff_ffff_03ff_ffff))
                    && u64_is(&dpe.control_5d4, 0x0000_0000_00c0_0000)
                    && u64_is(&pre.sochot_6ec.sensor_mask_010, 0x4248)
                    && u64_is(&pre.sochot_6ec.constant_018, 125)
            });

            if !pre_ok {
                dev_err!(dev.as_ref(), "T8122 G15 InitData preflight HwDataA pre-tail mismatch\n");
                return Err(EIO);
            }

            dev_info!(
                dev.as_ref(),
                "T8122 G15 InitData construction PASS (InitData VA {:#x}, HwDataA VA {:#x}, pre-tail 0x3a9c..0x421b exact); dropping without RTKit/MSG_INIT\n",
                initdata_va,
                hwdata_a_va
            );

            core::mem::drop(initdata);
            core::mem::drop(alloc);
            core::mem::drop(dyncfg);
            core::mem::drop(uat);
            Ok(())
        }
    }

    /// G15 bring-up checkpoint: build and wire the complete manager graph up
    /// to the exact RTKit construction boundary, validate all startup-visible
    /// channel/backing pointers, then drop it without constructing RTKit or
    /// sending MSG_INIT.
    pub(crate) fn manager_preflight(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<()> {
        #[ver(G != G15)]
        {
            let _ = (dev, res, cfg);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            // Preserve the already-proven HwDataA deterministic pre-tail gate
            // before advancing to the larger manager/channel object graph.
            Self::initdata_preflight(dev, res, cfg)?;

            let mgr = Self::build_pre_rtkit(dev, res, cfg)?;

            let initdata_va = mgr.initdata.gpu_va().get();
            let init_sequence_va = mgr.initdata.g15_init_sequence.gpu_va().get();
            let wrapper_va = mgr.initdata.runtime_pointers.gpu_va().get();
            let globals_va = mgr.initdata.g15_globals.gpu_va().get();
            let q21_va = mgr.initdata.g15_q21.gpu_va().get();
            let q22_va = mgr.initdata.g15_q22.gpu_va().get();
            let q23_va = mgr.initdata.g15_q23.gpu_va().get();
            let q22_state_va = mgr.initdata.g15_cache_flush_state.gpu_va().get();
            let q22_ring_va = mgr.initdata.g15_cache_flush_ring.gpu_va().get();
            let hwdata_a_va = mgr.initdata.runtime_pointers.hwdata_a.gpu_va().get();
            let hwdata_b_va = mgr.initdata.runtime_pointers.hwdata_b.gpu_va().get();
            let persistent_time_va = mgr.initdata.runtime_pointers.g15_persistent_time.gpu_va().get();
            let hwds_va = mgr.initdata.runtime_pointers.g15_hwds_counters.gpu_va().get();
            let pb_desc_va = mgr.initdata.runtime_pointers.g15_pb_desc_table.gpu_va().get();
            let uma_desc_va = mgr.initdata.runtime_pointers.g15_uma_page_pool_desc_table.gpu_va().get();

            // Validate every populated qword in the exact 24-qword G15 root.
            // q2 is allocation-zero; q19/q20 are the trailing zero pad of the
            // compact UAT descriptor. The level geometry is exact for J615.
            let root_ok = mgr.initdata.with(|raw, _inner| {
                let uat = &raw.g15_q6_q20_uat;
                let uat_bytes = unsafe {
                    core::slice::from_raw_parts(
                        uat as *const _ as *const u8,
                        core::mem::size_of_val(uat),
                    )
                };
                let mut expected_uat = [0u8; 0x78];
                expected_uat[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
                expected_uat[2] = 14;
                expected_uat[3] = 3;
                let oas_mask = ((1u64 << cfg.uat_oas) - 1) & !(mmu::UAT_PGMSK as u64);
                for (i, (shift, entries)) in [(36u8, 64u16), (25, 2048), (14, 2048)]
                    .into_iter()
                    .enumerate()
                {
                    let off = 4 + i * 0x20;
                    expected_uat[off] = 8;
                    expected_uat[off + 1] = 14;
                    expected_uat[off + 2] = 14;
                    expected_uat[off + 3] = shift;
                    expected_uat[off + 4..off + 6]
                        .copy_from_slice(&entries.to_le_bytes());
                    expected_uat[off + 6..off + 8]
                        .copy_from_slice(&0x4000u16.to_le_bytes());
                    expected_uat[off + 8..off + 16]
                        .copy_from_slice(&1u64.to_le_bytes());
                    expected_uat[off + 16..off + 24]
                        .copy_from_slice(&oas_mask.to_le_bytes());
                    let index_mask = ((entries as u64 - 1) << shift) as u64;
                    expected_uat[off + 24..off + 32]
                        .copy_from_slice(&index_mask.to_le_bytes());
                }

                raw.g15_q0_signature.0 == 0x0c08_e21e_8380_0490
                    && raw.g15_q1_init_sequence.0 == init_sequence_va
                    && raw.g15_q2.0 == 0
                    && raw.g15_q3_runtime_pointers.0 == wrapper_va
                    && raw.g15_q4_globals.0 == globals_va
                    && raw.g15_q5_host_mapped.0 == 0x0000_0001_0000_0000
                    && uat_bytes == expected_uat
                    && raw.g15_q21.0 == q21_va
                    && raw.g15_q22.0 == q22_va
                    && raw.g15_q23.0 == q23_va
            });

            let q22_ok = mgr.initdata.g15_q22.with(|raw, _inner| {
                // Root importer FUN_fffffc00000066dc snapshots q22 +0x008
                // before firmware_ready. Apple clears both bootstrap words.
                raw.host_zero_0000.0 == 0
                    && raw.host_zero_0008.0 == 0
                    && raw.shared_ptr_4568.0 == q22_state_va
                    && raw.shared_ptr_4570.0 == q22_ring_va
                    && raw.host_flag_45c4.0 == 1
                    && raw.kick_channel_qos_valid_c3c8.0 == 0
                    && raw.feature_c3cc.0 == 0
            });

            // Exact fields consumed while q21 +0x14 (firmware_ready) is still 0.
            // G15/G15G's host init-sequence callback is a no-op, so q1 starts
            // with the type-0 terminator. q4 +0x964 is an explicitly zeroed
            // patch-record count, disabling the +0x7f4 register-patch loop.
            // q23 +0x1e8 is likewise explicitly host-zero and gates its only
            // direct pre-ready branch observed in the boot consumer sweep.
            let pre_ready_ok = mgr.initdata.g15_init_sequence.with(|raw, _inner| {
                raw.terminator.value.0 == 0
                    && raw.terminator.register_offset.0 == 0
                    && raw.terminator.shift.0 == 0
                    && raw.terminator.kind.0 == 0
                    && raw.terminator.reserved.0 == 0
            }) && mgr.initdata.g15_globals.with(|raw, _inner| {
                raw.flags_000.0 == 0
                    && raw.zero_004.0 == 0
                    && raw.zero_008.0 == 0
                    && raw.zero_00c.0 == 0
                    && raw.zero_010.0 == 0
                    && raw.zero_014.0 == 0
                    && raw.zero_018.0 == 0
                    && raw.unk_01c.0 == 0
                    && raw.relaxed_cl_kill_timeout_020.0 == 3000
                    && raw.frg_task_timeout_024.0 == 0
                    && raw.accelerator_config_660_028.0 == 0
                    && raw.accelerator_config_bit0_02c.0 == 1
                    && raw.smart_idle_off_enabled_030.0 == 1
                    && raw.zero_034.0 == 0
                    && raw.constant_038.0 == 0x78
                    && raw.cpms_window_size_03c.0 == 0
                    && raw.cpms_tfca_size_040.0 == 0
                    && raw.zero_044.0 == 0
                    && raw.kick_channel_qos_arg2_048.0 == 0
                    && raw.kick_channel_qos_arg1_04c.0 == 0
                    && raw.unk_050 == 0xffff
                    && raw.unk_052 == 0x28
                    && raw.unk_054 == 0xffff
                    && raw.zero_056.0 == 0
                    && raw.low2_clear_05a.0 == 1
                    && raw.cswitch_timer_multiplier_05e.0 == 1
                    && raw.cdm_cswitch_mode_change_062.0 == 0
                    && raw.command_submission_enabled_070.0 == 0
                    && raw.zero_074.0 == 0
                    && raw.gpu_max_power_078.0 == 4070
                    && raw.power_interface_1_target_07c.0 == 100
                    && raw.power_interface_2_target_080.0 == 100
                    && raw.perf_state_cap_x100_08c.0 == 100
                    && raw.performance_target_090.0 == 0
                    && raw.performance_transfer_output_094.0 == 0
                    && raw.performance_boost_min_util_098.0 == 0
                    && raw.performance_boost_ce_step_09c.0 == 0
                    && raw.performance_reset_iters_0a0.0 == 0
                    && raw.performance_boost_min_util_valid_0a4 == 0
                    && raw.conflict_scan_enable_26f == 0
                    && raw.smart_idle_standby_timer_us_7a8.0 == 700
                    && raw.smart_idle_prob_init_val_7ac.to_bits() == 1.0f32.to_bits()
                    && raw.smart_idle_fn_hit_7b0.to_bits() == 0.8f32.to_bits()
                    && raw.smart_idle_fi_hit_7b4.to_bits() == 0.2f32.to_bits()
                    && raw.smart_idle_fn_miss_7b8.to_bits() == 0.9f32.to_bits()
                    && raw.smart_idle_fi_miss_7bc.to_bits() == 0.1f32.to_bits()
                    && raw.smart_idle_nei_hit_7c0.to_bits() == 0.25f32.to_bits()
                    && raw.smart_idle_min_confidence_7c4.to_bits() == 0.7f32.to_bits()
                    && raw.smart_idle_high_confidence_7c8.to_bits() == 0.9f32.to_bits()
                    && raw.smart_idle_reset_iterations_7cc.0 == 6
                    && raw.ut_engagement_enabled_7d0.0 == 1
                    && raw.clvr_engagement_enabled_7d4.0 == 1
                    && raw.gpu_keepalive_perf_mode_threshold_7d8.0 == 100
                    && raw.gpu_keepalive_off_mode_threshold_7dc.0 == 100
                    && raw.boot_patch_record_count_964.0 == 0
                    && raw.boot_patch_count_zero_968.0 == 0
                    && raw.display_pm_timestamp_96c.0 == 0
                    && raw.display_pm_interval_974.0 == 0
                    && raw.progress_check_interval_3d_97c.0 == 40
                    && raw.progress_check_interval_ta_980.0 == 10
                    && raw.progress_check_interval_cl_984.0 == 250
                    && raw.unk_988.0 == 1
                    && raw.unk_98c.0 == 1
                    && raw.unk_990.0 == 100
                    && raw.unk_994.0 == 1
                    && raw.gpu_idle_off_delay_ms_9a0.0 == 2
                    && raw.fender_idle_off_delay_ms_9a4.0 == 40
                    && raw.fw_early_wake_timeout_ms_9a8.0 == 5
                    && raw.gvdm_timer_interval_9ac.0 == 0
                    && raw.cl_context_switch_timeout_9b0.0 == 0x28
                    && raw.cl_kill_timeout_9b4.0 == 0x32
                    && raw.zero_9b8.0 == 0
                    && raw.cdm_backoff_timeout_9bc == 4
                    && raw.table_selector_9dd.0 == 0
                    && raw.zero_de5.0 == 0
                    && raw.gpu_keepalive_override_de9.0 == 0
                    && raw.gfxc_keepalive_override_ded.0 == 0
                    && raw.gpu_keepalive_perf_mode_threshold_override_df1.0 == 0
                    && raw.gpu_keepalive_off_mode_threshold_override_df5.0 == 0
                    && raw.soft_fault_settings_df9.0 == 1
            }) && mgr.initdata.g15_q21.with(|raw, _inner| {
                raw.host_flags == 0
                    && raw.banner_guard.load(Ordering::Relaxed) == 1
                    && raw.busy.load(Ordering::Relaxed) == 0
                    && raw.firmware_ready.load(Ordering::Relaxed) == 0
                    && raw.power_state.load(Ordering::Relaxed) == 0
            }) && mgr.initdata.g15_q23.with(|raw, _inner| {
                raw.tuning_update_1d0 == 0
                    && raw.host_zero_1e8.0 == 0
            });

            let tx_ring_ok = |r: &fw::initdata::raw::G15TxChannelRing| {
                let read = r.read_ptr.0;
                read != 0
                    && r.write_ptr_shadow.0 == read + 0x10
                    && r.write_ptr.0 == read + 0x20
                    && r.ring.0 != 0
            };

            let persistent_time_ok = mgr
                .initdata
                .runtime_pointers
                .g15_persistent_time
                .with(|raw, _inner| {
                    raw.initialized.load(Ordering::Relaxed) == 0
                        && raw.values.iter().all(|v| v.0 == 0)
                        && raw.tail_84.0 == 0
                });

            let wrapper_ok = mgr.initdata.runtime_pointers.with(|raw, _inner| {
                let pipes_ok = raw.pipes.iter().all(|p| {
                    tx_ring_ok(&p.vtx) && tx_ring_ok(&p.frag) && tx_ring_ok(&p.comp)
                });

                // Consumer-driven pre-ready wrapper invariants. The root importer
                // binds q3 as this 0x490-byte object; nested boot helpers consume
                // these fields before q21 +0x14 becomes firmware_ready=1.
                // Read the packed GPU pointers exactly as firmware-visible qwords.
                let wrapper_base = raw as *const _ as *const u8;
                let ptr_000 = u64::from_le(unsafe {
                    core::ptr::read_unaligned(wrapper_base.add(0x000) as *const u64)
                });
                let ptr_010 = u64::from_le(unsafe {
                    core::ptr::read_unaligned(wrapper_base.add(0x010) as *const u64)
                });

                ptr_000 == hwdata_b_va
                    && raw.g15_fwbrn_table.0 == 0
                    && ptr_010 == persistent_time_va
                    && persistent_time_ok
                    && pipes_ok
                    && tx_ring_ok(&raw.device_control)
                    && raw.event.state.is_some()
                    && raw.event.ring.is_some()
                    && raw.fw_log.state.is_some()
                    && raw.fw_log.ring.is_some()
                    && raw.ktrace.state.is_some()
                    && raw.ktrace.ring.is_some()
                    && raw.stats.state.is_some()
                    && raw.stats.ring.is_some()
                    && raw.fwlog_buf.is_some()
                    // FUN_fffffc000003da5c exits before touching FWLog backing
                    // while this Apple-explicit bootstrap gate is zero.
                    && raw.fwlog_enabled_230 == 0
                    && raw.g15_hwds_counters.0 == hwds_va
                    // FUN_fffffc000000f34c directly consumes +0x2b0 and +0x2c0.
                    // The adjacent FW-VA aliases are exact host-image invariants;
                    // convertGPUVAToFWVA() is identity on this G15 host.
                    && raw.g15_pb_desc_addr.0 == pb_desc_va
                    && raw.g15_pb_desc_fw_addr.0 == pb_desc_va
                    && raw.g15_uma_page_pool_desc_addr.0 == uma_desc_va
                    && raw.g15_uma_page_pool_desc_fw_addr.0 == uma_desc_va
                    // FUN_fffffc000000f34c indexes wrapper[0x5a] == +0x2d0;
                    // FUN_fffffc0000018864 directly reads +0x2d4.
                    && raw.g15_usc_max_tgmem == 4
                    && raw.g15_zero_2d4 == 0
                    // FUN_fffffc0000006908 dereferences +0x441 as HwDataA.
                    && raw.g15_ptr_441.0 == hwdata_a_va
                    && raw.g15_zero_449.0 == 0
                    && raw.g15_zero_451.0 == 0
            });

            let bufmgr_ok = mgr
                .initdata
                .runtime_pointers
                .buffer_mgr_ctl_low_mapping
                .is_some()
                && mgr
                    .initdata
                    .runtime_pointers
                    .buffer_mgr_ctl_high_mapping
                    .is_some();

            // Validate the complete first-read HwDataB image that is known
            // mechanically for J615: chip identity, live ADT perf-state tables,
            // all 31 firmware MMIO descriptors, and the exact G15 startup suffix.
            let expected_io_mapping_count: usize = cfg
                .io_mappings
                .iter()
                .filter_map(|map| map.as_ref())
                .map(|map| {
                    let dies = if map.per_die { cfg.num_dies as usize } else { 1 };
                    map.count * dies
                })
                .sum();

            let hwdatab_ok = mgr.initdata.runtime_pointers.hwdata_b.with(|raw, _inner| {
                let pwr = &mgr.dyncfg.pwr;
                let mut perf_ok = raw.max_pstate as usize + 1 == pwr.perf_states.len();

                for i in 0..raw.frequencies.len() {
                    if i < pwr.perf_states.len() {
                        let ps = &pwr.perf_states[i];
                        perf_ok &= raw.frequencies[i] == ps.freq_hz / 1_000_000;
                        for j in 0..raw.voltages[i].len() {
                            let mv = if j < ps.volt_mv.len() {
                                ps.volt_mv[j]
                            } else {
                                ps.volt_mv[0]
                            };
                            let sram_mv = mv.max(pwr.min_sram_microvolt / 1000);
                            perf_ok &= raw.voltages[i][j] == mv;
                            perf_ok &= raw.voltages_sram[i][j] == sram_mv;
                        }
                    } else {
                        perf_ok &= raw.frequencies[i] == 0;
                        perf_ok &= raw.voltages[i].iter().all(|v| *v == 0);
                        perf_ok &= raw.voltages_sram[i].iter().all(|v| *v == 0);
                    }
                }

                let mut map_ok = raw.io_mappings.len() == cfg.io_mappings.len();
                for (i, rec) in raw.io_mappings.iter().enumerate() {
                    if let Some(map) = cfg.io_mappings[i].as_ref() {
                        let dies = if map.per_die { cfg.num_dies as usize } else { 1 };
                        let expected_total = map.size * map.count * dies;
                        let off = map.base & mmu::UAT_PGMSK;
                        map_ok &= rec.phys_addr.0 == map.base as u64;
                        map_ok &= rec.virt_addr.0 != 0;
                        map_ok &= (rec.virt_addr.0 & mmu::UAT_PGMSK as u64) == off as u64;
                        map_ok &= rec.total_size as usize == expected_total;
                        map_ok &= rec.element_size as usize == map.size;
                        map_ok &= rec.readwrite.0 == map.writable as u64;
                    } else {
                        map_ok &= rec.phys_addr.0 == 0;
                        map_ok &= rec.virt_addr.0 == 0;
                        map_ok &= rec.total_size == 0;
                        map_ok &= rec.element_size == 0;
                        map_ok &= rec.readwrite.0 == 0;
                    }
                }

                let tail = &raw.g15_startup_17ec;
                let tail_bytes = unsafe {
                    core::slice::from_raw_parts(
                        tail as *const _ as *const u8,
                        core::mem::size_of_val(tail),
                    )
                };
                let expected_tail_words: [u32; 29] = [
                    0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
                    0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff,
                    0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff,
                    0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff,
                    0, 0, 0, 0, 1, 0,
                ];
                let mut expected_tail = [0u8; 0x74];
                for (i, word) in expected_tail_words.into_iter().enumerate() {
                    expected_tail[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                let tail_ok = tail_bytes == expected_tail;

                raw.chip_id == 0x8122
                    && raw.unk_454 == 2
                    && raw.unk_458 == 0
                    && raw.unk_45c == 4
                    && raw.base_clock_khz == 24_000
                    && raw.power_sample_period == 1
                    // Exact G15 UAT/GPTBAT + core-config header consumed by
                    // RTKit-2419 firmware. The GPTBAT base is dynamic but must
                    // be the physical base of Linux's ttbs region and 16 KiB aligned.
                    && raw.g15_zero_b38 == 0
                    && raw.g15_zero_b3c == 0
                    && raw.g15_uat_mode_b40 == 1
                    && raw.uat_ttb_base.0 == mgr.dyncfg.uat_ttb_base
                    && raw.uat_ttb_base.0 != 0
                    && (raw.uat_ttb_base.0 & mmu::UAT_PGMSK as u64) == 0
                    && raw.gpu_core_id == cfg.gpu_core as u32
                    && raw.gpu_rev_id == hw::GpuRevisionID::C0 as u32
                    && raw.num_cores == mgr.dyncfg.id.num_cores * mgr.dyncfg.id.num_clusters
                    // Physical HwDataB +0xad0 is source field unk_4f8 on G15.
                    && raw.unk_4f8 == 1
                    // Firmware pre-ready qword at +0x17e4 remains allocation-zero.
                    && raw.unk_b1c == 0
                    && raw.unk_b20 == 0
                    && raw.num_frags == mgr.dyncfg.id.num_frags * mgr.dyncfg.id.num_clusters
                    && perf_ok
                    && map_ok
                    && tail_ok
            });

            let mappings_ok = cfg.io_mappings.len() == 0x1f
                && cfg.sram_base.is_none()
                && cfg.sram_size.is_none()
                && expected_io_mapping_count == 13
                && mgr.io_mappings.len() == expected_io_mapping_count;

            if !(root_ok && q22_ok && pre_ready_ok && wrapper_ok && bufmgr_ok && hwdatab_ok && mappings_ok) {
                dev_err!(
                    dev.as_ref(),
                    "T8122 G15 pre-RTKit manager validation failed: root={} q22={} pre_ready={} wrapper={} bufmgr={} hwdatab={} mappings={}\n",
                    root_ok,
                    q22_ok,
                    pre_ready_ok,
                    wrapper_ok,
                    bufmgr_ok,
                    hwdatab_ok,
                    mappings_ok
                );
                return Err(EIO);
            }

            dev_info!(
                dev.as_ref(),
                "T8122 G15 pre-RTKit manager PASS (InitData {:#x}, wrapper {:#x}, q22 {:#x}, q22 ctl {:#x}/{:#x}; root + pre-ready q1/q4/q21/q23 gates + HwDataB GPTBAT/core/scalars/perf + 31 MMIO records/13 maps + G15 startup exact; dropping before RtKit::new/MSG_INIT\n",
                initdata_va,
                wrapper_va,
                q22_va,
                q22_state_va,
                q22_ring_va
            );

            core::mem::drop(mgr);
            Ok(())
        }
    }

    /// G15 bring-up checkpoint: complete RTKit management, then start the
    /// firmware and doorbell application endpoints without sending InitData,
    /// device-control traffic, a doorbell, or MSG_INIT. RTKit is destroyed
    /// again before ASC is stopped by the caller.
    pub(crate) fn rtkit_endpoint_preflight(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<()> {
        #[ver(G != G15)]
        {
            let _ = (dev, res, cfg);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            // Keep every previous host-side validation as a prerequisite.
            Self::manager_preflight(dev, res, cfg)?;

            let mgr = Arc::from(Self::build_pre_rtkit(dev, res, cfg)?);
            let rtkit = rtkit::RtKit::<GpuManager::ver>::new(
                dev.as_ref(),
                None,
                0,
                mgr.clone(),
            )?;
            *mgr.rtkit.lock() = Some(rtkit);

            let (fw_ep, doorbell_ep) = {
                let mut guard = mgr.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().ok_or(EIO)?;

                // Complete RTKit management/system-endpoint boot first.
                rtk.as_mut().boot()?;
                let fw = rtk.as_mut().has_endpoint(EP_FIRMWARE);
                let doorbell = rtk.as_mut().has_endpoint(EP_DOORBELL);
                if !fw || !doorbell {
                    return Err(EIO);
                }

                // m1n1 starts these endpoints before it builds InitData. The
                // RTKit core sends only STARTEP management messages here; no
                // InitData address or application payload is sent.
                rtk.as_mut().start_endpoint(EP_FIRMWARE)?;
                rtk.as_mut().start_endpoint(EP_DOORBELL)?;
                (fw, doorbell)
            };

            let q21_untouched = mgr.initdata.g15_q21.with(|raw, _inner| {
                raw.busy.load(Ordering::Relaxed) == 0
                    && raw.firmware_ready.load(Ordering::Relaxed) == 0
                    && raw.power_state.load(Ordering::Relaxed) == 0
            });
            let crashed = mgr.crashed.load(Ordering::Relaxed);

            if !fw_ep || !doorbell_ep || !q21_untouched || crashed {
                dev_err!(
                    dev.as_ref(),
                    "T8122 G15 RTKit endpoint-start validation failed: fw_ep={} doorbell_ep={} q21_untouched={} crashed={}\n",
                    fw_ep,
                    doorbell_ep,
                    q21_untouched,
                    crashed
                );
                return Err(EIO);
            }

            dev_info!(
                dev.as_ref(),
                "T8122 G15 RTKit endpoint-start PASS (EP20 firmware + EP21 doorbell started; no InitData/device-control/doorbell/MSG_INIT; q21 untouched); destroying RTKit\n"
            );

            // Stop mailbox RX and release any RTKit system buffers before the
            // manager/UAT disappear. The caller stops ASC immediately after.
            let rtkit = mgr.rtkit.lock().take();
            core::mem::drop(rtkit);
            core::mem::drop(mgr);
            Ok(())
        }
    }

    /// Return a mutable reference to the initdata member
    fn initdata_mut(
        self: Pin<&mut Self>,
    ) -> &mut fw::types::GpuObject<fw::initdata::InitData::ver> {
        // SAFETY: initdata does not require structural pinning.
        unsafe { &mut self.get_unchecked_mut().initdata }
    }

    /// Return a mutable reference to the io_mappings member
    fn io_mappings_mut(self: Pin<&mut Self>) -> &mut KVec<mmu::KernelMapping> {
        // SAFETY: io_mappings does not require structural pinning.
        unsafe { &mut self.get_unchecked_mut().io_mappings }
    }

    /// Allocate an MMIO iova range
    fn alloc_mmio_iova(self: Pin<&mut Self>, size: usize) -> u64 {
        // SAFETY: next_mmio_iova does not require structural pinning.
        let next_ref = unsafe { &mut self.get_unchecked_mut().next_mmio_iova };

        let addr = *next_ref;
        let next = addr + (size + mmu::UAT_PGSZ) as u64;

        assert!(next <= IOVA_KERN_MMIO_RANGE.end);

        *next_ref = next;

        addr
    }

    /// Build the entire GPU InitData structure tree and return it as a boxed GpuObject.
    fn make_initdata(
        dev: &AsahiDevice,
        cfg: &'static hw::HwConfig,
        dyncfg: &hw::DynConfig,
        alloc: &mut KernelAllocators,
    ) -> Result<KBox<fw::types::GpuObject<fw::initdata::InitData::ver>>> {
        let mut builder = initdata::InitDataBuilder::ver::new(dev, alloc, cfg, dyncfg);
        builder.build()
    }

    /// Create a fresh boxed Uat instance.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_uat(dev: &AsahiDevice, cfg: &'static hw::HwConfig) -> Result<KBox<mmu::Uat>> {
        // G14X has a new thing in the Scene structure that unfortunately requires
        // write access from user contexts. Hopefully it's not security-sensitive.
        #[ver(G >= G14X)]
        let map_kernel_to_user = true;
        #[ver(G < G14X)]
        let map_kernel_to_user = false;

        Ok(KBox::new(
            mmu::Uat::new(dev, cfg, map_kernel_to_user)?,
            GFP_KERNEL,
        )?)
    }

    /// Actually create the final GpuManager instance, as a UniqueArc.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_mgr(
        dev: &AsahiDevice,
        cfg: &'static hw::HwConfig,
        dyncfg: KBox<hw::DynConfig>,
        uat: KBox<mmu::Uat>,
        mut alloc: KernelAllocators,
        event_manager: Arc<event::EventManager>,
        initdata: KBox<fw::types::GpuObject<fw::initdata::InitData::ver>>,
    ) -> Result<Pin<UniqueArc<GpuManager::ver>>> {
        let mut pipes = PipeChannels::ver {
            vtx: KVec::new(),
            frag: KVec::new(),
            comp: KVec::new(),
        };

        for _i in 0..=NUM_PIPES - 1 {
            pipes.vtx.push(
                KBox::pin_init(
                    new_mutex!(channel::PipeChannel::ver::new(dev, &mut alloc)?, "pipe_vtx",),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
            pipes.frag.push(
                KBox::pin_init(
                    new_mutex!(
                        channel::PipeChannel::ver::new(dev, &mut alloc)?,
                        "pipe_frag",
                    ),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
            pipes.comp.push(
                KBox::pin_init(
                    new_mutex!(
                        channel::PipeChannel::ver::new(dev, &mut alloc)?,
                        "pipe_comp",
                    ),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
        }

        #[ver(G != G15)]
        let fwctl_channel = channel::FwCtlChannel::new(dev, &mut alloc)?;

        let buffer_mgr = buffer::BufferManager::ver::new()?;
        let event_manager_clone = event_manager.clone();
        let buffer_mgr_clone = buffer_mgr.clone();
        let alloc_ref = &mut alloc;
        let rx_channels = KBox::init(
            try_init!(RxChannels::ver {
                event: channel::EventChannel::ver::new(
                    dev,
                    alloc_ref,
                    event_manager_clone,
                    buffer_mgr_clone,
                )?,
                fw_log: channel::FwLogChannel::new(dev, alloc_ref)?,
                ktrace: channel::KTraceChannel::new(dev, alloc_ref)?,
                stats: channel::StatsChannel::ver::new(dev, alloc_ref)?,
            }),
            GFP_KERNEL,
        )?;

        let alloc_ref = &mut alloc;
        let tx_channels = KBox::init(
            try_init!(TxChannels::ver {
                device_control: channel::DeviceControlChannel::ver::new(dev, alloc_ref)?,
            }),
            GFP_KERNEL,
        )?;

        let x = UniqueArc::pin_init(
            try_pin_init!(GpuManager::ver {
                dev: dev.into(),
                cfg,
                dyncfg: KBox::<hw::DynConfig>::into_inner(dyncfg),
                initdata: KBox::<fw::types::GpuObject<fw::initdata::InitData::ver>>::into_inner(initdata),
                uat: KBox::<mmu::Uat>::into_inner(uat),
                io_mappings: KVec::new(),
                next_mmio_iova: IOVA_KERN_MMIO_RANGE.start,
                rtkit <- new_mutex!(None, "rtkit"),
                crashed: AtomicBool::new(false),
                event_manager,
                alloc <- new_mutex!(alloc, "alloc"),
                #[ver(G != G15)]
                fwctl_channel <- new_mutex!(fwctl_channel, "fwctl_channel"),
                rx_channels <- new_mutex!(KBox::<RxChannels::ver>::into_inner(rx_channels), "rx_channels"),
                tx_channels <- new_mutex!(KBox::<TxChannels::ver>::into_inner(tx_channels), "tx_channels"),
                pipes,
                buffer_mgr,
                ids: Default::default(),
                garbage_contexts <- new_mutex!(KVec::new(), "garbage_contexts"),
            }),
            GFP_KERNEL,
        )?;

        Ok(x)
    }

    fn load_hwdata_blob(dev: &AsahiDevice, name: &CStr, size_name: &CStr) -> Result<KVVec<u8>> {
        let of_node = dev.as_ref().of_node().ok_or(EINVAL)?;
        let size: usize = dev
            .as_ref()
            .fwnode()
            .ok_or(ENOENT)?
            .property_read::<u32>(size_name)
            .or(0)
            .try_into()?;
        let res = of_node.reserved_mem_region_to_resource_byname(name)?;
        // SAFETY: No dma here, just loading init data.
        let mem = unsafe { Mem::try_new(res, (MemFlag::WB).into())? };
        if size > mem.size() {
            return Err(ENOENT);
        }
        // SAFETY: trusting the bootloader to fill it out correctly
        let blob_sl = unsafe { slice::from_raw_parts(mem.ptr(), size) };
        let mut blob = KVVec::new();
        blob.extend_from_slice(blob_sl, GFP_KERNEL)?;
        Ok(blob)
    }

    /// Fetch and validate the GPU dynamic configuration from the device tree and hardware.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_dyncfg(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
        uat: &mmu::Uat,
    ) -> Result<KBox<hw::DynConfig>> {
        let gpu_id = res.get_gpu_id()?;

        dev_info!(dev.as_ref(), "GPU Information:\n");
        dev_info!(
            dev.as_ref(),
            "  Type: {:?}{:?}\n",
            gpu_id.gpu_gen,
            gpu_id.gpu_variant
        );
        dev_info!(dev.as_ref(), "  Clusters: {}\n", gpu_id.num_clusters);
        dev_info!(
            dev.as_ref(),
            "  Cores: {} ({})\n",
            gpu_id.num_cores,
            gpu_id.num_cores * gpu_id.num_clusters
        );
        dev_info!(
            dev.as_ref(),
            "  Frags: {} ({})\n",
            gpu_id.num_frags,
            gpu_id.num_frags * gpu_id.num_clusters
        );
        dev_info!(
            dev.as_ref(),
            "  GPs: {} ({})\n",
            gpu_id.num_gps,
            gpu_id.num_gps * gpu_id.num_clusters
        );
        dev_info!(dev.as_ref(), "  Core masks: {:#x?}\n", gpu_id.core_masks);
        dev_info!(
            dev.as_ref(),
            "  Active cores: {}\n",
            gpu_id.total_active_cores
        );

        dev_info!(dev.as_ref(), "Getting configuration from device tree...\n");
        let pwr_cfg = hw::PwrConfig::load(dev, cfg)?;
        dev_info!(dev.as_ref(), "Dynamic configuration fetched\n");

        if gpu_id.gpu_gen != cfg.gpu_gen || gpu_id.gpu_variant != cfg.gpu_variant {
            dev_err!(
                dev.as_ref(),
                "GPU type mismatch (expected {:?}{:?}, found {:?}{:?})\n",
                cfg.gpu_gen,
                cfg.gpu_variant,
                gpu_id.gpu_gen,
                gpu_id.gpu_variant
            );
            return Err(EIO);
        }
        if gpu_id.num_clusters > cfg.max_num_clusters {
            dev_err!(
                dev.as_ref(),
                "Too many clusters ({} > {})\n",
                gpu_id.num_clusters,
                cfg.max_num_clusters
            );
            return Err(EIO);
        }
        if gpu_id.num_cores > cfg.max_num_cores {
            dev_err!(
                dev.as_ref(),
                "Too many cores ({} > {})\n",
                gpu_id.num_cores,
                cfg.max_num_cores
            );
            return Err(EIO);
        }
        if gpu_id.num_frags > cfg.max_num_frags {
            dev_err!(
                dev.as_ref(),
                "Too many frags ({} > {})\n",
                gpu_id.num_frags,
                cfg.max_num_frags
            );
            return Err(EIO);
        }
        if gpu_id.num_gps > cfg.max_num_gps {
            dev_err!(
                dev.as_ref(),
                "Too many GPs ({} > {})\n",
                gpu_id.num_gps,
                cfg.max_num_gps
            );
            return Err(EIO);
        }

        let fwnode = dev.as_ref().fwnode().ok_or(ENOENT)?;

        Ok(KBox::new(
            hw::DynConfig {
                pwr: pwr_cfg,
                uat_ttb_base: uat.ttb_base(),
                id: gpu_id,
                firmware_version: fwnode
                    .property_read_array_vec(c_str!("apple,firmware-version"), 3)?
                    .or(kernel::kvec![0; 3]?),

                hw_data_a: Self::load_hwdata_blob(
                    dev,
                    c_str!("hw-cal-a"),
                    c_str!("debug,hw-cal-a-size"),
                )
                .unwrap_or(KVVec::new()),
                hw_data_b: Self::load_hwdata_blob(
                    dev,
                    c_str!("hw-cal-b"),
                    c_str!("debug,hw-cal-b-size"),
                )
                .unwrap_or(KVVec::new()),
                hw_globals: Self::load_hwdata_blob(
                    dev,
                    c_str!("globals"),
                    c_str!("debug,globals-size"),
                )
                .unwrap_or(KVVec::new()),
            },
            GFP_KERNEL,
        )?)
    }

    /// Create the global GPU event manager, and return an `Arc<>` to it.
    fn make_event_manager(alloc: &mut KernelAllocators) -> Result<Arc<event::EventManager>> {
        Ok(Arc::new(event::EventManager::new(alloc)?, GFP_KERNEL)?)
    }

    /// Create a new MMIO mapping and add it to the mappings list in initdata at the specified
    /// index.
    fn iomap(
        this: &mut Pin<UniqueArc<GpuManager::ver>>,
        cfg: &'static hw::HwConfig,
        index: usize,
        map: &hw::IOMapping,
    ) -> Result {
        let dies = if map.per_die {
            cfg.num_dies as usize
        } else {
            1
        };

        let off = map.base & mmu::UAT_PGMSK;
        let base = map.base - off;
        let end = (map.base + map.size + mmu::UAT_PGMSK) & !mmu::UAT_PGMSK;
        let map_size = end - base;

        // Array mappings must be aligned
        assert!((off == 0 && map_size == map.size) || (map.count == 1 && !map.per_die));
        assert!(map.count > 0);

        let iova = this.as_mut().alloc_mmio_iova(map_size * map.count * dies);
        let mut cur_iova = iova;

        for die in 0..dies {
            for i in 0..map.count {
                let phys_off = die * 0x20_0000_0000 + i * map.stride;

                let mapping = this.uat.kernel_vm().map_io(
                    cur_iova,
                    base + phys_off,
                    map_size,
                    if map.writable {
                        mmu::PROT_FW_MMIO_RW
                    } else {
                        mmu::PROT_FW_MMIO_RO
                    },
                )?;

                this.as_mut().io_mappings_mut().push(mapping, GFP_KERNEL)?;
                cur_iova += map_size as u64;
            }
        }

        this.as_mut()
            .initdata_mut()
            .runtime_pointers
            .hwdata_b
            .with_mut(|raw, _| {
                raw.io_mappings[index] = fw::initdata::raw::IOMapping {
                    phys_addr: U64(map.base as u64),
                    virt_addr: U64(iova + off as u64),
                    total_size: (map.size * map.count * dies) as u32,
                    element_size: map.size as u32,
                    readwrite: U64(map.writable as u64),
                };
            });

        Ok(())
    }

    /// Mark work associated with currently in-progress event slots as failed, after a fault or
    /// timeout.
    fn mark_pending_events(&self, culprit_slot: Option<u32>, error: workqueue::WorkError) {
        dev_err!(self.dev.as_ref(), "  Pending events:\n");

        #[ver(G != G15)]
        self.initdata.globals.with(|raw, _inner| {
            for (index, i) in raw.pending_stamps.iter().enumerate() {
                let info = i.info.load(Ordering::Relaxed);
                let wait_value = i.wait_value.load(Ordering::Relaxed);

                if info & 1 != 0 {
                    #[ver(V >= V13_5)]
                    let slot = (info >> 4) & 0x7f;
                    #[ver(V < V13_5)]
                    let slot = (info >> 3) & 0x7f;
                    #[ver(V >= V13_5)]
                    let flags = info & 0xf;
                    #[ver(V < V13_5)]
                    let flags = info & 0x7;
                    dev_err!(
                        self.dev.as_ref(),
                        "    [{}:{}] flags={} value={:#x}\n",
                        index,
                        slot,
                        flags,
                        wait_value
                    );
                    let error = if culprit_slot.is_some() && culprit_slot != Some(slot) {
                        workqueue::WorkError::Killed
                    } else {
                        error
                    };
                    self.event_manager.mark_error(slot, wait_value, error);
                    i.info.store(0, Ordering::Relaxed);
                    i.wait_value.store(0, Ordering::Relaxed);
                }
            }
        });
        #[ver(G == G15)]
        dev_err!(
            self.dev.as_ref(),
            "G15 pending-stamp recovery is not wired yet; runtime enablement is blocked\n"
        );
    }

    /// Fetch the GPU MMU fault information from the hardware registers.
    fn get_fault_info(&self) -> Option<regs::FaultInfo> {
        let res = &(*self.dev).resources;

        let info = res.get_fault_info(self.cfg);
        if info.is_some() {
            dev_err!(
                self.dev.as_ref(),
                "  Fault info: {:#x?}\n",
                info.as_ref().unwrap()
            );
        }
        info
    }

    /// Resume the GPU firmware after it halts (due to a timeout, fault, or request).
    fn recover(&self) {
        #[ver(G == G15)]
        {
            // G15 replaces the legacy FwStatus halt/resume block. Do not guess
            // at q21/q22 offsets: an accidental runtime path must fail closed.
            dev_err!(
                self.dev.as_ref(),
                "G15 firmware recovery ABI is not mapped yet; refusing recovery\n"
            );
            return;
        }
        #[ver(G != G15)]
        self.initdata.fw_status.with(|raw, _inner| {
            let halt_count = raw.flags.halt_count.load(Ordering::Relaxed);
            let mut halted = raw.flags.halted.load(Ordering::Relaxed);
            dev_err!(self.dev.as_ref(), "  Halt count: {}\n", halt_count);
            dev_err!(self.dev.as_ref(), "  Halted: {}\n", halted);

            if halted == 0 {
                let start = Instant::<Monotonic>::now();
                while start.elapsed() < HALT_ENTER_TIMEOUT {
                    halted = raw.flags.halted.load(Ordering::Relaxed);
                    if halted != 0 {
                        break;
                    }
                    mem::sync();
                }
                halted = raw.flags.halted.load(Ordering::Relaxed);
            }

            if debug_enabled(DebugFlags::NoGpuRecovery) {
                dev_crit!(
                    self.dev.as_ref(),
                    "  GPU recovery is disabled, wedging forever!\n"
                );
            } else if halted != 0 {
                dev_err!(self.dev.as_ref(), "  Attempting recovery...\n");
                raw.flags.halted.store(0, Ordering::SeqCst);
                raw.flags.resume.store(1, Ordering::SeqCst);
            } else {
                dev_err!(self.dev.as_ref(), "  Cannot recover.\n");
            }
        });
    }

    /// Return the packed GPU enabled core masks.
    // Only used for some versions
    #[allow(dead_code)]
    pub(crate) fn core_masks_packed(&self) -> &[u32] {
        self.dyncfg.id.core_masks_packed.as_slice()
    }

    /// Kick a submission pipe for a submitted job to tell the firmware to start processing it.
    pub(crate) fn run_job(&self, job: workqueue::JobSubmission::ver<'_>) -> Result {
        mod_dev_dbg!(self.dev, "GPU: run_job\n");

        let pipe_type = job.pipe_type();
        mod_dev_dbg!(self.dev, "GPU: run_job: pipe_type={:?}\n", pipe_type);

        let pipes = match pipe_type {
            PipeType::Vertex => &self.pipes.vtx,
            PipeType::Fragment => &self.pipes.frag,
            PipeType::Compute => &self.pipes.comp,
        };

        let index: usize = job.priority() as usize;
        let mut pipe = pipes.get(index).ok_or(EIO)?.lock();

        mod_dev_dbg!(self.dev, "GPU: run_job: run()\n");
        job.run(&mut pipe);
        mod_dev_dbg!(self.dev, "GPU: run_job: ring doorbell\n");

        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().as_pin_mut().unwrap();
        rtk.send_message(
            EP_DOORBELL,
            MSG_TX_DOORBELL | pipe_type as u64 | ((index as u64) << 2),
        )?;
        mod_dev_dbg!(self.dev, "GPU: run_job: done\n");

        Ok(())
    }

    pub(crate) fn start_op(self: &Arc<GpuManager::ver>) -> Result<OpGuard> {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        #[ver(G == G15)]
        {
            // Submission accounting moved out of the legacy Globals object on
            // G15. Keep the compile-only variant incapable of queue execution
            // until the replacement shared field is proven.
            return Err(ENODEV);
        }
        #[ver(G != G15)]
        {
            let val = self
                .initdata
                .globals
                .with(|raw, _inner| raw.pending_submissions.fetch_add(1, Ordering::Acquire));

            mod_dev_dbg!(self.dev, "OP start (pending: {})\n", val + 1);
            self.kick_firmware()?;
            Ok(OpGuard(self.clone()))
        }
    }

    fn invalidate_context(
        &self,
        context: &fw::types::GpuObject<fw::workqueue::GpuContextData>,
    ) -> Result {
        mod_dev_dbg!(
            self.dev,
            "Invalidating GPU context @ {:?}\n",
            context.weak_pointer()
        );

        if self.is_crashed() {
            return Err(ENODEV);
        }

        let mut guard = self.alloc.lock();
        let (garbage_count, _) = guard.private.garbage();
        let (garbage_count_gpuro, _) = guard.gpu_ro.garbage();

        let dc = context.with(
            |raw, _inner| fw::channels::DeviceControlMsg::ver::DestroyContext {
                unk_4: 0,
                ctx_23: raw.unk_23,
                #[ver(V < V13_3)]
                __pad0: Default::default(),
                unk_c: U32(0),
                unk_10: U32(0),
                ctx_0: raw.unk_0,
                ctx_1: raw.unk_1,
                ctx_4: raw.unk_4,
                #[ver(V < V13_3)]
                __pad1: Default::default(),
                #[ver(V < V13_3)]
                unk_18: 0,
                gpu_context: Some(context.weak_pointer()),
                __pad2: Default::default(),
            },
        );

        mod_dev_dbg!(self.dev, "Context invalidation command: {:?}\n", &dc);

        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);

        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        }

        txch.device_control.wait_for(token)?;

        mod_dev_dbg!(
            self.dev,
            "GPU context invalidated: {:?}\n",
            context.weak_pointer()
        );

        // The invalidation does a cache flush, so it is okay to collect garbage
        guard.private.collect_garbage(garbage_count);
        guard.gpu_ro.collect_garbage(garbage_count_gpuro);

        Ok(())
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    fn generate_crashdump(&self, crashlog: Option<&[u8]>) -> Result {
        // Lock the allocators, to block kernel/FW memory mutations (mostly)
        let kalloc = self.alloc();
        let pages = self.uat.dump_kernel_pages()?;
        core::mem::drop(kalloc);

        let mut crashdump = crate::crashdump::CrashDumpBuilder::new(pages)?;
        let initdata_addr = self.initdata.gpu_va().get();
        crashdump.add_agx_info(self.cfg, &self.dyncfg, initdata_addr)?;
        if let Some(crashlog) = crashlog {
            crashdump.add_crashlog(crashlog)?;
        }
        let crashdump = KBox::new(crashdump.finalize()?, GFP_KERNEL)?;

        devcoredump::dev_coredump(
            self.dev.as_ref(),
            &crate::THIS_MODULE,
            crashdump,
            GFP_KERNEL,
            msecs_to_jiffies(60 * 60 * 1000),
        );

        Ok(())
    }
}

#[versions(AGX)]
impl GpuManager for GpuManager::ver {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn arc_as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send> {
        self as Arc<dyn Any + Sync + Send>
    }

    fn init(&self) -> Result {
        self.tx_channels.lock().device_control.send(
            &fw::channels::DeviceControlMsg::ver::Initialize(Default::default()),
        );

        let initdata = self.initdata.gpu_va().get();
        let mut guard = self.rtkit.lock();
        let mut rtk = guard.as_mut().as_pin_mut().unwrap();

        rtk.as_mut().boot()?;
        rtk.as_mut().start_endpoint(EP_FIRMWARE)?;
        rtk.as_mut().start_endpoint(EP_DOORBELL)?;
        rtk.as_mut()
            .send_message(EP_FIRMWARE, MSG_INIT | (initdata & INIT_DATA_MASK))?;
        rtk.as_mut()
            .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        core::mem::drop(guard);

        self.kick_firmware()?;
        Ok(())
    }

    fn update_globals(&self) {
        let mut timeout: u32 = 2;
        if debug_enabled(DebugFlags::WaitForPowerOff) {
            timeout = 0;
        } else if debug_enabled(DebugFlags::KeepGpuPowered) {
            timeout = 5000;
        }

        #[ver(G != G15)]
        self.initdata.globals.with(|raw, _inner| {
            raw.idle_off_delay_ms.store(timeout, Ordering::Relaxed);
        });
        #[ver(G == G15)]
        let _ = timeout;
    }

    fn alloc(&self) -> Guard<'_, KernelAllocators, MutexBackend> {
        /* Clean up idle contexts */
        let mut garbage_ctx = KVec::new();
        core::mem::swap(&mut *self.garbage_contexts.lock(), &mut garbage_ctx);

        for ctx in garbage_ctx {
            if self.invalidate_context(&ctx).is_err() {
                dev_err!(
                    self.dev.as_ref(),
                    "GpuContext: Failed to invalidate GPU context!\n"
                );
                if debug_enabled(DebugFlags::OopsOnGpuCrash) {
                    panic!("GPU firmware timed out");
                }
            }
        }

        let mut guard = self.alloc.lock();
        let (garbage_count, garbage_bytes) = guard.private.garbage();
        let (ro_garbage_count, ro_garbage_bytes) = guard.gpu_ro.garbage();

        if garbage_bytes > MAX_FW_ALLOC_GARBAGE_BYTES
            || ro_garbage_bytes > MAX_FW_ALLOC_GARBAGE_BYTES
            || garbage_count > MAX_FW_ALLOC_GARBAGE_OBJECTS
            || ro_garbage_count > MAX_FW_ALLOC_GARBAGE_OBJECTS
        {
            mod_dev_dbg!(
                self.dev,
                "Collecting kalloc garbage (private: {} objects, {} bytes, gpuro: {} objects, {} bytes)\n",
                garbage_count,
                garbage_bytes,
                ro_garbage_count,
                ro_garbage_bytes
            );
            if self.flush_fw_cache().is_err() {
                dev_err!(self.dev.as_ref(), "Failed to flush FW cache\n");
            } else {
                guard.private.collect_garbage(garbage_count);
                guard.gpu_ro.collect_garbage(ro_garbage_count);
            }
        }

        guard
    }

    fn new_vm(&self, kernel_range: Range<u64>) -> Result<mmu::Vm> {
        self.uat.new_vm(self.ids.vm.next(), kernel_range)
    }

    fn bind_vm(&self, vm: &mmu::Vm) -> Result<mmu::VmBind> {
        self.uat.bind(vm)
    }

    fn new_queue(
        &self,
        vm: mmu::Vm,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        g15_ualloc_range5_uncached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_cached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        priority: u32,
        usc_exec_base: u64,
    ) -> Result<KBox<dyn queue::Queue>> {
        let mut kalloc = self.alloc();
        let id = self.ids.queue.next();
        Ok(KBox::new(
            queue::Queue::ver::new(
                &self.dev,
                vm,
                &mut kalloc,
                ualloc,
                ualloc_priv,
                g15_ualloc_range5_uncached,
                g15_ualloc_range5_cached,
                self.uat.g15_shared_bank1(),
                self.event_manager.clone(),
                &self.buffer_mgr,
                id,
                priority,
                usc_exec_base,
            )?,
            GFP_KERNEL,
        )?)
    }

    fn kick_firmware(&self) -> Result {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().as_pin_mut().unwrap();
        rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_KICKFW)?;

        Ok(())
    }

    fn flush_fw_cache(&self) -> Result {
        mod_dev_dbg!(self.dev, "Flushing coprocessor data cache\n");

        if self.is_crashed() {
            return Err(ENODEV);
        }

        // ctx_0 == 0xff or ctx_1 == 0xff cause no effect on context,
        // but this command does a full cache flush too, so abuse it
        // for that.

        let dc = fw::channels::DeviceControlMsg::ver::DestroyContext {
            unk_4: 0,

            ctx_23: 0,
            #[ver(V < V13_3)]
            __pad0: Default::default(),
            unk_c: U32(0),
            unk_10: U32(0),
            ctx_0: 0xff,
            ctx_1: 0xff,
            ctx_4: 0,
            #[ver(V < V13_3)]
            __pad1: Default::default(),
            #[ver(V < V13_3)]
            unk_18: 0,
            gpu_context: None,
            __pad2: Default::default(),
        };

        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        }

        txch.device_control.wait_for(token)?;
        Ok(())
    }

    fn ids(&self) -> &SequenceIDs {
        &self.ids
    }

    fn handle_timeout(&self, counter: u32, event_slot: i32, unk: u32) {
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "** GPU timeout nya~!!!!! **\n");
        dev_err!(self.dev.as_ref(), "  Event slot: {}\n", event_slot);
        dev_err!(self.dev.as_ref(), "  Timeout count: {}\n", counter);
        dev_err!(self.dev.as_ref(), "  Unk: {}\n", unk);

        // If we have fault info, consider it a fault.
        let error = match self.get_fault_info() {
            Some(info) => workqueue::WorkError::Fault(info),
            None => workqueue::WorkError::Timeout,
        };
        self.mark_pending_events(event_slot.try_into().ok(), error);
        self.recover();
    }

    fn handle_fault(&self) {
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "GPU fault nya~!!!!!\n");
        let error = match self.get_fault_info() {
            Some(info) => workqueue::WorkError::Fault(info),
            None => workqueue::WorkError::Unknown,
        };
        self.mark_pending_events(None, error);
        self.recover();
    }

    fn handle_channel_error(
        &self,
        error_type: ChannelErrorType,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    ) {
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "GPU channel error nya~!!!!!\n");
        dev_err!(self.dev.as_ref(), "  Error type: {:?}\n", error_type);
        dev_err!(self.dev.as_ref(), "  Pipe type: {}\n", pipe_type);
        dev_err!(self.dev.as_ref(), "  Event slot: {}\n", event_slot);
        dev_err!(self.dev.as_ref(), "  Event value: {:#x?}\n", event_value);

        self.event_manager.mark_error(
            event_slot,
            event_value,
            workqueue::WorkError::ChannelError(error_type),
        );

        let wq = match self.event_manager.get_owner(event_slot) {
            Some(wq) => wq,
            None => {
                dev_err!(
                    self.dev.as_ref(),
                    "Workqueue not found for this event slot!\n"
                );
                return;
            }
        };

        let wq = match wq.as_any().downcast_ref::<workqueue::WorkQueue::ver>() {
            Some(wq) => wq,
            None => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with WorkQueue!\n");
                return;
            }
        };

        if debug_enabled(DebugFlags::VerboseFaults) {
            wq.dump_info();
        }

        let dc = fw::channels::DeviceControlMsg::ver::RecoverChannel {
            pipe_type,
            work_queue: wq.info_pointer(),
            event_value,
            __pad: Default::default(),
        };

        mod_dev_dbg!(self.dev, "Recover Channel command: {:?}\n", &dc);
        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            if rtk
                .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)
                .is_err()
            {
                dev_err!(
                    self.dev.as_ref(),
                    "Failed to send Recover Channel command\n"
                );
            }
        }

        if txch.device_control.wait_for(token).is_err() {
            dev_err!(
                self.dev.as_ref(),
                "Timed out waiting for Recover Channel command\n"
            );
        }

        if debug_enabled(DebugFlags::VerboseFaults) {
            wq.dump_info();
        }
    }

    fn ack_grow(&self, buffer_slot: u32, vm_slot: u32, counter: u32) {
        #[ver(G == G15)]
        {
            // The G15 halt-count source is not the legacy FwStatus block.
            // Refuse to emit a malformed grow acknowledgement.
            dev_err!(
                self.dev.as_ref(),
                "G15 TVB grow acknowledgement ABI is not mapped yet\n"
            );
            return;
        }
        #[ver(G != G15)]
        {
            let halt_count = self
                .initdata
                .fw_status
                .with(|raw, _inner| raw.flags.halt_count.load(Ordering::Relaxed));

            let dc = fw::channels::DeviceControlMsg::ver::GrowTVBAck {
                unk_4: 1,
                buffer_slot,
                vm_slot,
                counter,
                subpipe: 0, // TODO
                halt_count: U64(halt_count),
                __pad: Default::default(),
            };

            mod_dev_dbg!(self.dev, "TVB Grow Ack command: {:?}\n", &dc);

            let mut txch = self.tx_channels.lock();

            txch.device_control.send(&dc);
            {
                let mut guard = self.rtkit.lock();
                let rtk = guard.as_mut().as_pin_mut().unwrap();
                if rtk
                    .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)
                    .is_err()
                {
                    dev_err!(self.dev.as_ref(), "Failed to send TVB Grow Ack command\n");
                }
            }
        }
    }

    fn fwctl(&self, msg: fw::channels::FwCtlMsg) -> Result {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        #[ver(G == G15)]
        {
            // q22 +0x4568/+0x4570 is the exact G15 successor. Apple proves the
            // new +0x14 dword is zero, but notifyNewMapping() also constructs
            // the legacy-looking +0x08/+0x10/+0x12 fields differently. Keep
            // this fail-closed until the G15-native request encoder is ported.
            let _ = msg;
            return Err(ENODEV);
        }
        #[ver(G != G15)]
        {
            let mut fwctl = self.fwctl_channel.lock();
            let token = fwctl.send(&msg);
            {
                let mut guard = self.rtkit.lock();
                let rtk = guard.as_mut().as_pin_mut().unwrap();
                rtk.send_message(EP_DOORBELL, MSG_FWCTL)?;
            }
            fwctl.wait_for(token)?;
            Ok(())
        }
    }

    fn get_cfg(&self) -> &'static hw::HwConfig {
        self.cfg
    }

    fn get_dyncfg(&self) -> &hw::DynConfig {
        &self.dyncfg
    }

    fn free_context(&self, ctx: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>) {
        let mut garbage = self.garbage_contexts.lock();

        if garbage.push(ctx, GFP_KERNEL).is_err() {
            dev_err!(
                self.dev.as_ref(),
                "Failed to reserve space for freed context, deadlock possible.\n"
            );
        }
    }

    fn is_crashed(&self) -> bool {
        self.crashed.load(Ordering::Relaxed)
    }

    fn map_timestamp_buffer(
        &self,
        mut bo: gem::ObjectRef,
        range: Range<usize>,
    ) -> Result<mmu::KernelMapping> {
        bo.map_range_into_range(
            self.uat.kernel_vm(),
            range,
            IOVA_KERN_TIMESTAMP_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_FW_SHARED_RW,
            false,
        )
    }
}

#[versions(AGX)]
impl GpuManagerPriv for GpuManager::ver {
    fn end_op(&self) {
        #[ver(G != G15)]
        {
            let val = self
                .initdata
                .globals
                .with(|raw, _inner| raw.pending_submissions.fetch_sub(1, Ordering::Release));

            mod_dev_dbg!(self.dev, "OP end (pending: {})\n", val - 1);
        }
        #[ver(G == G15)]
        {
            // start_op() is hard-blocked for the compile-only G15 variant.
        }
    }
}
