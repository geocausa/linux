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
        delay::fsleep,
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

use crate::alloc::{Allocator, DmaSyncAllocation};
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

/// Zero-caller E739 owner for Apple's separate sampler selector-0 context.
/// Drop clears the published sampler context before returning the ASID slot.
pub(crate) struct G15RdeObserverRuntime {
    // Options let Drop fail closed if firmware is still active: retaining the
    // ASID and source mappings is safer than releasing live GPU-visible state.
    sampler_context: Option<mmu::G15SamplerVmBind>,
    perfctr_runtime: fw::initdata::G15PerfCtrRuntimeHandle,
    // E764: retain the selected Queue-owned split source until after the
    // global sampler image has been restored to its stable bootstrap backing.
    split_source: Option<G15RdeSplitSourceBacking>,
}


/// E790 bounded owner for one fully installed APS_USC source-5 image.
/// The selected-observer path may construct this owner, but E790 itself is not
/// loaded live. Explicit completion restores the dormant bootstrap image after
/// STOP; Drop remains the fail-closed fallback for every unwind path.
pub(crate) struct G15ApsObserverRuntime {
    sampler_context: Option<mmu::G15SamplerVmBind>,
    perfctr_runtime: fw::initdata::G15PerfCtrRuntimeHandle,
    split_source: Option<G15ApsSplitSourceBacking>,
    sampler_count: u32,
}


impl G15ApsObserverRuntime {
    /// E790 bounded lifecycle audit. This is observation-only: it reads the
    /// exact q22 fields that gate safe APS teardown and selector-0 release.
    pub(crate) fn lifecycle_state(&self) -> (u32, u8, u16, u64, u64) {
        self.perfctr_runtime.with_q22(|raw| {
            (
                raw.sampler_active_4030.0,
                raw.aps_usc_coord_4094,
                raw.sampler_source_context_id_4064,
                raw.sampler_mask_4068.0,
                raw.sampler_mask_4070.0,
            )
        })
    }
}

/// E767 bounded raw view of one inactive G15 ordinary-RDE ring.
/// The exact 23J220 userspace consumer treats trailer +0x30 as an absolute
/// producer pointer and consumes a 0x60-byte G15 raw sample header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G15RdeRawRingSnapshot {
    pub(crate) source_base: u64,
    pub(crate) producer: u64,
    pub(crate) producer_offset: u32,
    pub(crate) producer_in_ring: bool,
    pub(crate) raw_header: [u64; 12],
}

/// E770 bounded implementation of the exact userspace magic-to-magic framing
/// scan. Offsets are byte offsets within the 0x4000 hardware ring.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G15RdeMagicScanSnapshot {
    pub(crate) magic_count: u32,
    pub(crate) magic_offsets: [u16; 16],
    pub(crate) first_record_size: u32,
    pub(crate) record_qwords_valid: u32,
    pub(crate) first_record_qwords: [u64; 32],
}

/// E773 bounded statistics for the one selected 64-bit DCDM payload lane.
/// Exact userspace default flags place that lane at raw record +0x60.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G15RdePayloadStatsSnapshot {
    pub(crate) framed_records: u32,
    pub(crate) payload_records: u32,
    pub(crate) nonzero_records: u32,
    pub(crate) min_value: u64,
    pub(crate) max_value: u64,
    pub(crate) first_values_valid: u32,
    pub(crate) first_values: [u64; 16],
}

/// E776 exact four-input CSInvocation observation. Current Apple derived raw
/// requests use 32-bit DCDM lanes packed contiguously at raw +0x60..+0x6f.
/// The derived value is the u64 sum of the four lane values.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G15RdeCsInvocationStatsSnapshot {
    pub(crate) framed_records: u32,
    pub(crate) payload_records: u32,
    pub(crate) nonzero_records: u32,
    pub(crate) lane_nonzero: [u32; 4],
    pub(crate) min_value: u64,
    pub(crate) max_value: u64,
    pub(crate) first_values_valid: u32,
    pub(crate) first_values: [u64; 16],
    pub(crate) first_lanes: [[u32; 4]; 4],
    pub(crate) first_timestamp: u64,
    pub(crate) last_timestamp: u64,
    pub(crate) positive_valid: u32,
    pub(crate) positive_ordinals: [u16; 16],
    pub(crate) positive_offsets: [u16; 16],
    pub(crate) positive_timestamps: [u64; 16],
    pub(crate) positive_aux: [u64; 16],
    pub(crate) positive_values: [u64; 16],
    pub(crate) positive_lanes: [[u32; 4]; 16],
}

impl G15RdeObserverRuntime {
    /// Take one inactive-boundary snapshot from the exact selected split
    /// source backing while source/state/q22 metadata is synchronized through
    /// the retained global sampler runtime.
    pub(crate) fn state_snapshot_synced(
        &self,
    ) -> Result<fw::initdata::G15PerfCtrStateSnapshot> {
        let source = self.split_source.as_ref().ok_or(EIO)?;
        let (source_base, completion_cursor, first_header) =
            source.sample_fields_synced()?;
        self.perfctr_runtime
            .state_snapshot_synced_external(source_base, completion_cursor, first_header)
            .ok_or(EIO)
    }

    /// E767: snapshot exactly the raw bytes consumed by G15
    /// `ParseSampleHeader()` after STOP. This is observation-only and does not
    /// attempt to decode counter payloads.
    pub(crate) fn raw_ring_snapshot_synced(&self) -> Result<G15RdeRawRingSnapshot> {
        self.split_source.as_ref().ok_or(EIO)?.raw_ring_snapshot_synced()
    }


    /// E770: use the exact G15 sentinel to recover bounded record framing
    /// after STOP without decoding or mutating ring contents.
    pub(crate) fn magic_scan_snapshot_synced(&self) -> Result<G15RdeMagicScanSnapshot> {
        self.split_source
            .as_ref()
            .ok_or(EIO)?
            .magic_scan_snapshot_synced()
    }


    /// E773: evaluate the exact one-counter payload qword across the bounded
    /// magic-framed ring after STOP. No decode or GPU-visible mutation.
    pub(crate) fn payload_stats_snapshot_synced(&self) -> Result<G15RdePayloadStatsSnapshot> {
        self.split_source
            .as_ref()
            .ok_or(EIO)?
            .payload_stats_snapshot_synced()
    }


    /// E776: decode the four exact 32-bit CSInvocation raw inputs after STOP
    /// and report their Apple-derived sum. Observation-only.
    pub(crate) fn cs_invocation_stats_snapshot_synced(
        &self,
    ) -> Result<G15RdeCsInvocationStatsSnapshot> {
        self.split_source
            .as_ref()
            .ok_or(EIO)?
            .cs_invocation_stats_snapshot_synced()
    }
}

impl Drop for G15RdeObserverRuntime {
    fn drop(&mut self) {
        let Some(slot) = self.sampler_context.as_ref().map(|ctx| ctx.slot()) else {
            return;
        };
        // Replace every GPU-visible pointer that names the selected split
        // source before either alias or its selector-0 ASID can be released.
        if self.perfctr_runtime.restore_bootstrap_rde0(slot) {
            return;
        }

        // If firmware is still active (or an ownership transition failed),
        // fail closed by retaining both resources until reset/module lifetime.
        // Unmapping either one while dRDE may still dereference it is worse
        // than leaking this single bounded lab allocation.
        if let Some(source) = self.split_source.take() {
            core::mem::forget(source);
        }
        if let Some(context) = self.sampler_context.take() {
            core::mem::forget(context);
        }
    }
}

/// E763 compile-only collision-safe owner for Apple's ordinary-RDE command-3
/// split mapping.
///
/// The main source allocation comes from the Queue-owned range-5/PPL308 heap,
/// so its VA is reserved by the same allocator as every other Queue resource.
/// Only the final physical page of that exact allocation is additionally
/// aliased through accelerator-shared range 7. No live path constructs this
/// object in E763.
pub(crate) struct G15RdeSplitSourceBacking {
    main: fw::types::GpuArray<u8>,
    trailer_alias: mmu::G15SharedBank1Mapping,
}

impl G15RdeSplitSourceBacking {
    pub(crate) fn new_for_observer(
        range5: &mut alloc::DefaultAllocator,
        bank1: &mmu::G15SharedBank1,
        notifier: Option<Arc<Mutex<mmu::G15MappingNotifier>>>,
    ) -> Result<Self> {
        // Take CPU ownership before constructing the typed array. This keeps
        // the non-coherent J615 ownership transition ordered before the zero
        // initialization performed by GpuArray::empty().
        let main_alloc = range5.alloc_generic::<u8>(0x8000, mmu::UAT_PGSZ, None)?;
        main_alloc.dma_sync_for_cpu()?;
        let main = fw::types::GpuArray::<u8>::empty(main_alloc, 0x8000)?;

        let main_fwva = main.gpu_va().get();
        let trailer_fwva = main_fwva.checked_add(0x4000).ok_or(EOVERFLOW)?;
        if main_fwva & 0x3fff != 0
            || trailer_fwva & 0x3fff != 0
            || !mmu::G15_GART_RANGE5_PPL308.contains(&main_fwva)
        {
            return Err(EINVAL);
        }

        // Resolve the exact physical GEM page that backs bytes 0x4000..0x7fff
        // of this heap allocation. Heap/debug-allocation prefixes are already
        // accounted for by BackingObjectAllocation.
        let (trailer_object, trailer_object_offset) =
            main.backing_object_range(0x4000, 0x4000)?;
        let trailer_alias = bank1.map_subrange(
            mmu::G15SharedBank1Aperture::Range7,
            &trailer_object.gem,
            trailer_object_offset,
            0x4000,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_G15_RANGE7_FW,
            false,
            notifier,
            mmu::G15MappingNotifyMode::Immediate,
        )?;
        let alias_fwva = trailer_alias.iova();
        if alias_fwva & 0x3fff != 0 || !mmu::G15_GART_RANGE7.contains(&alias_fwva) {
            return Err(EINVAL);
        }

        // The range-5 mapping and range-7 alias refer to the same physical
        // backing. Publish the CPU-authored zeros once for both aliases.
        main.dma_sync_for_device()?;

        Ok(Self {
            main,
            trailer_alias,
        })
    }

    pub(crate) fn main_fwva(&self) -> u64 {
        self.main.gpu_va().get()
    }

    pub(crate) fn trailer_alias_fwva(&self) -> u64 {
        self.trailer_alias.iova()
    }

    pub(crate) fn state_and_trailer(
        &self,
    ) -> Option<(
        fw::initdata::raw::G15PerfCtrSamplerState,
        fw::initdata::raw::G15PerfCtrSamplerTrailer,
    )> {
        let main = self.main_fwva();
        fw::initdata::raw::G15PerfCtrSamplerState::ordinary_rde_one(
            1,
            0x4000,
            main,
            main.checked_add(0x4000)?,
            0,
        )
    }

    pub(crate) fn source_record(
        &self,
        period: u32,
    ) -> Option<fw::initdata::raw::G15PerfCtrSourceRecord> {
        fw::initdata::raw::G15PerfCtrSourceRecord::rde0_observer_split(
            self.main_fwva(),
            self.trailer_alias_fwva(),
            period,
        )
    }

    pub(crate) fn dma_sync_for_cpu(&self) -> Result {
        self.main.dma_sync_for_cpu()
    }

    pub(crate) fn dma_sync_for_device(&self) -> Result {
        self.main.dma_sync_for_device()
    }

    /// Seed the exact trailer bytes on the common backing and return the
    /// source/state records that must be installed atomically into q22's
    /// retained source/state arrays before START.
    pub(crate) fn prepare_runtime_image(
        &mut self,
        period: u32,
    ) -> Result<(
        fw::initdata::raw::G15PerfCtrSourceRecord,
        fw::initdata::raw::G15PerfCtrSamplerState,
    )> {
        self.main.dma_sync_for_cpu()?;
        let (state, trailer) = self.state_and_trailer().ok_or(EINVAL)?;
        let bytes = self.main.as_mut_slice();
        let off = 0x4000usize;
        if bytes.len() < off + core::mem::size_of::<fw::initdata::raw::G15PerfCtrSamplerTrailer>() {
            let _ = self.main.dma_sync_for_device();
            return Err(EINVAL);
        }
        // SAFETY: `main` is one CPU-mapped 0x8000 allocation and `off` is the
        // exact page-aligned trailer subrange proven above.
        unsafe {
            core::ptr::write_unaligned(
                bytes.as_mut_ptr().add(off).cast::<fw::initdata::raw::G15PerfCtrSamplerTrailer>(),
                trailer,
            );
        }
        self.main.dma_sync_for_device()?;
        let source = self.source_record(period).ok_or(EINVAL)?;
        Ok((source, state))
    }

    /// Read the same physical source/trailer backing used by dRDE. The
    /// ownership transition covers both GPU aliases because they name one GEM.
    pub(crate) fn sample_fields_synced(&self) -> Result<(u64, u64, u64)> {
        self.main.dma_sync_for_cpu()?;
        let bytes = self.main.as_slice();
        let read_u64 = |off: usize| -> Option<u64> {
            let raw: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
            Some(u64::from_le_bytes(raw))
        };
        let result = match (read_u64(0x4000 + 0x30), read_u64(0)) {
            (Some(cursor), Some(header)) => Ok((self.main_fwva(), cursor, header)),
            _ => Err(EINVAL),
        };
        let dev_result = self.main.dma_sync_for_device();
        match (result, dev_result) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }

    /// E767 exact-consumer diagnostic. Apple's G15 RingBufferImpl stores an
    /// absolute producer pointer in trailer +0x30; the raw sample header is
    /// 0x60 bytes and its first qword is not the normalized `0xfeed` token.
    pub(crate) fn raw_ring_snapshot_synced(&self) -> Result<G15RdeRawRingSnapshot> {
        self.main.dma_sync_for_cpu()?;
        let bytes = self.main.as_slice();
        let read_u64 = |off: usize| -> Option<u64> {
            let raw: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
            Some(u64::from_le_bytes(raw))
        };

        let result = (|| {
            let source_base = self.main_fwva();
            let producer = read_u64(0x4000 + 0x30).ok_or(EINVAL)?;
            let ring_end = source_base.checked_add(0x4000).ok_or(EOVERFLOW)?;
            let producer_in_ring = producer >= source_base && producer < ring_end;
            let producer_offset = if producer_in_ring {
                u32::try_from(producer - source_base).map_err(|_| EOVERFLOW)?
            } else {
                0
            };
            let mut raw_header = [0u64; 12];
            let mut i = 0usize;
            while i < raw_header.len() {
                raw_header[i] = read_u64(i * 8).ok_or(EINVAL)?;
                i += 1;
            }
            Ok(G15RdeRawRingSnapshot {
                source_base,
                producer,
                producer_offset,
                producer_in_ring,
                raw_header,
            })
        })();

        let dev_result = self.main.dma_sync_for_device();
        match (result, dev_result) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }

    /// E770 exact framing scan. The Apple consumer advances one qword at a
    /// time until `HasMagicToken()` succeeds, then defines sample size by the
    /// modular distance to the next token. Keep output bounded even if the
    /// ring contains many samples.
    pub(crate) fn magic_scan_snapshot_synced(&self) -> Result<G15RdeMagicScanSnapshot> {
        const RING_SIZE: usize = 0x4000;
        const MAGIC: u64 = 0xfeed_cba9_8765_4321;

        self.main.dma_sync_for_cpu()?;
        let bytes = self.main.as_slice();
        let read_u64 = |off: usize| -> Option<u64> {
            let raw: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
            Some(u64::from_le_bytes(raw))
        };

        let result = (|| {
            if bytes.len() < RING_SIZE {
                return Err(EINVAL);
            }
            let mut magic_offsets = [0u16; 16];
            let mut magic_count = 0u32;
            let mut off = 0usize;
            while off < RING_SIZE {
                if read_u64(off).ok_or(EINVAL)? == MAGIC {
                    if (magic_count as usize) < magic_offsets.len() {
                        magic_offsets[magic_count as usize] =
                            u16::try_from(off).map_err(|_| EOVERFLOW)?;
                    }
                    magic_count = magic_count.checked_add(1).ok_or(EOVERFLOW)?;
                }
                off += 8;
            }

            let mut first_record_size = 0u32;
            let mut record_qwords_valid = 0u32;
            let mut first_record_qwords = [0u64; 32];
            if magic_count >= 2 {
                let first = magic_offsets[0] as usize;
                let second = magic_offsets[1] as usize;
                if second <= first {
                    return Err(EINVAL);
                }
                let size = second - first;
                if size < 0x60 || size & 7 != 0 {
                    return Err(EINVAL);
                }
                first_record_size = u32::try_from(size).map_err(|_| EOVERFLOW)?;
                let qwords = core::cmp::min(size / 8, first_record_qwords.len());
                let mut i = 0usize;
                while i < qwords {
                    first_record_qwords[i] = read_u64(first + i * 8).ok_or(EINVAL)?;
                    i += 1;
                }
                record_qwords_valid = u32::try_from(qwords).map_err(|_| EOVERFLOW)?;
            }

            Ok(G15RdeMagicScanSnapshot {
                magic_count,
                magic_offsets,
                first_record_size,
                record_qwords_valid,
                first_record_qwords,
            })
        })();

        let dev_result = self.main.dma_sync_for_device();
        match (result, dev_result) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }

    /// E773 exact default-flags one-counter payload statistic. The hardware
    /// payload starts at raw +0x60; use exact magic-to-magic framing and wrap
    /// every read inside the 0x4000 source ring.
    pub(crate) fn payload_stats_snapshot_synced(&self) -> Result<G15RdePayloadStatsSnapshot> {
        const RING_SIZE: usize = 0x4000;
        const MAGIC: u64 = 0xfeed_cba9_8765_4321;
        const PAYLOAD_OFFSET: usize = 0x60;

        self.main.dma_sync_for_cpu()?;
        let bytes = self.main.as_slice();
        let read_u64 = |off: usize| -> Option<u64> {
            let off = off % RING_SIZE;
            let raw: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
            Some(u64::from_le_bytes(raw))
        };

        let result = (|| {
            if bytes.len() < RING_SIZE {
                return Err(EINVAL);
            }

            let mut first_magic = None;
            let mut previous_magic = None;
            let mut framed_records = 0u32;
            let mut payload_records = 0u32;
            let mut nonzero_records = 0u32;
            let mut min_value = u64::MAX;
            let mut max_value = 0u64;
            let mut first_values_valid = 0u32;
            let mut first_values = [0u64; 16];

            let mut consume_record = |record_start: usize, record_size: usize| -> Result {
                framed_records = framed_records.checked_add(1).ok_or(EOVERFLOW)?;
                if record_size < PAYLOAD_OFFSET + 8 || record_size & 7 != 0 {
                    return Ok(());
                }
                let value = read_u64(record_start + PAYLOAD_OFFSET).ok_or(EINVAL)?;
                payload_records = payload_records.checked_add(1).ok_or(EOVERFLOW)?;
                if value != 0 {
                    nonzero_records = nonzero_records.checked_add(1).ok_or(EOVERFLOW)?;
                }
                min_value = core::cmp::min(min_value, value);
                max_value = core::cmp::max(max_value, value);
                if (first_values_valid as usize) < first_values.len() {
                    first_values[first_values_valid as usize] = value;
                    first_values_valid += 1;
                }
                Ok(())
            };

            let mut off = 0usize;
            while off < RING_SIZE {
                if read_u64(off).ok_or(EINVAL)? == MAGIC {
                    if first_magic.is_none() {
                        first_magic = Some(off);
                    }
                    if let Some(previous) = previous_magic {
                        consume_record(previous, off - previous)?;
                    }
                    previous_magic = Some(off);
                }
                off += 8;
            }

            if let (Some(first), Some(last)) = (first_magic, previous_magic) {
                if first != last {
                    consume_record(last, RING_SIZE - last + first)?;
                }
            }

            if payload_records == 0 {
                min_value = 0;
            }

            Ok(G15RdePayloadStatsSnapshot {
                framed_records,
                payload_records,
                nonzero_records,
                min_value,
                max_value,
                first_values_valid,
                first_values,
            })
        })();

        let dev_result = self.main.dma_sync_for_device();
        match (result, dev_result) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }


    /// E776 exact four-lane CSInvocation statistic. The raw G15 header is
    /// 0x60 bytes. Exact postProcessData() packs each 32-bit selected counter
    /// consecutively, so lanes live at +0x60/+0x64/+0x68/+0x6c. Framing stays
    /// magic-to-magic and every source-ring read is explicitly wrap-safe.
    pub(crate) fn cs_invocation_stats_snapshot_synced(
        &self,
    ) -> Result<G15RdeCsInvocationStatsSnapshot> {
        const RING_SIZE: usize = 0x4000;
        const MAGIC: u64 = 0xfeed_cba9_8765_4321;
        const PAYLOAD_OFFSET: usize = 0x60;
        const PAYLOAD_BYTES: usize = 0x10;

        self.main.dma_sync_for_cpu()?;
        let bytes = self.main.as_slice();
        let read_u64 = |off: usize| -> Option<u64> {
            if bytes.len() < RING_SIZE {
                return None;
            }
            let mut raw = [0u8; 8];
            let mut i = 0;
            while i < raw.len() {
                raw[i] = *bytes.get((off + i) % RING_SIZE)?;
                i += 1;
            }
            Some(u64::from_le_bytes(raw))
        };
        let read_u32 = |off: usize| -> Option<u32> {
            if bytes.len() < RING_SIZE {
                return None;
            }
            let mut raw = [0u8; 4];
            let mut i = 0;
            while i < raw.len() {
                raw[i] = *bytes.get((off + i) % RING_SIZE)?;
                i += 1;
            }
            Some(u32::from_le_bytes(raw))
        };

        let result = (|| {
            if bytes.len() < RING_SIZE {
                return Err(EINVAL);
            }

            let mut first_magic = None;
            let mut previous_magic = None;
            let mut framed_records = 0u32;
            let mut payload_records = 0u32;
            let mut nonzero_records = 0u32;
            let mut lane_nonzero = [0u32; 4];
            let mut min_value = u64::MAX;
            let mut max_value = 0u64;
            let mut first_values_valid = 0u32;
            let mut first_values = [0u64; 16];
            let mut first_lanes = [[0u32; 4]; 4];
            let mut first_timestamp = 0u64;
            let mut last_timestamp = 0u64;
            let mut positive_valid = 0u32;
            let mut positive_ordinals = [0u16; 16];
            let mut positive_offsets = [0u16; 16];
            let mut positive_timestamps = [0u64; 16];
            let mut positive_aux = [0u64; 16];
            let mut positive_values = [0u64; 16];
            let mut positive_lanes = [[0u32; 4]; 16];

            let mut consume_record = |record_start: usize, record_size: usize| -> Result {
                let record_ordinal = framed_records;
                framed_records = framed_records.checked_add(1).ok_or(EOVERFLOW)?;
                if record_size < PAYLOAD_OFFSET + PAYLOAD_BYTES || record_size & 7 != 0 {
                    return Ok(());
                }
                let timestamp = read_u64(record_start + 0x10).ok_or(EINVAL)?;
                let aux = read_u64(record_start + 0x18).ok_or(EINVAL)?;
                if payload_records == 0 {
                    first_timestamp = timestamp;
                }
                last_timestamp = timestamp;
                let lanes = [
                    read_u32(record_start + PAYLOAD_OFFSET).ok_or(EINVAL)?,
                    read_u32(record_start + PAYLOAD_OFFSET + 4).ok_or(EINVAL)?,
                    read_u32(record_start + PAYLOAD_OFFSET + 8).ok_or(EINVAL)?,
                    read_u32(record_start + PAYLOAD_OFFSET + 12).ok_or(EINVAL)?,
                ];
                let value = lanes.iter().try_fold(0u64, |acc, lane| {
                    acc.checked_add(u64::from(*lane)).ok_or(EOVERFLOW)
                })?;
                payload_records = payload_records.checked_add(1).ok_or(EOVERFLOW)?;
                if value != 0 {
                    nonzero_records = nonzero_records.checked_add(1).ok_or(EOVERFLOW)?;
                    if (positive_valid as usize) < positive_values.len() {
                        let idx = positive_valid as usize;
                        positive_ordinals[idx] =
                            u16::try_from(record_ordinal).map_err(|_| EOVERFLOW)?;
                        positive_offsets[idx] =
                            u16::try_from(record_start).map_err(|_| EOVERFLOW)?;
                        positive_timestamps[idx] = timestamp;
                        positive_aux[idx] = aux;
                        positive_values[idx] = value;
                        positive_lanes[idx] = lanes;
                        positive_valid += 1;
                    }
                }
                let mut lane = 0;
                while lane < lane_nonzero.len() {
                    if lanes[lane] != 0 {
                        lane_nonzero[lane] = lane_nonzero[lane].checked_add(1).ok_or(EOVERFLOW)?;
                    }
                    lane += 1;
                }
                min_value = core::cmp::min(min_value, value);
                max_value = core::cmp::max(max_value, value);
                if (first_values_valid as usize) < first_values.len() {
                    first_values[first_values_valid as usize] = value;
                    if (first_values_valid as usize) < first_lanes.len() {
                        first_lanes[first_values_valid as usize] = lanes;
                    }
                    first_values_valid += 1;
                }
                Ok(())
            };

            let mut off = 0usize;
            while off < RING_SIZE {
                if read_u64(off).ok_or(EINVAL)? == MAGIC {
                    if first_magic.is_none() {
                        first_magic = Some(off);
                    }
                    if let Some(previous) = previous_magic {
                        consume_record(previous, off - previous)?;
                    }
                    previous_magic = Some(off);
                }
                off += 8;
            }

            if let (Some(first), Some(last)) = (first_magic, previous_magic) {
                if first != last {
                    consume_record(last, RING_SIZE - last + first)?;
                }
            }

            if payload_records == 0 {
                min_value = 0;
            }

            Ok(G15RdeCsInvocationStatsSnapshot {
                framed_records,
                payload_records,
                nonzero_records,
                lane_nonzero,
                min_value,
                max_value,
                first_values_valid,
                first_values,
                first_lanes,
                first_timestamp,
                last_timestamp,
                positive_valid,
                positive_ordinals,
                positive_offsets,
                positive_timestamps,
                positive_aux,
                positive_values,
                positive_lanes,
            })
        })();

        let dev_result = self.main.dma_sync_for_device();
        match (result, dev_result) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }
}


/// E783 compile-only owner for Apple's source-type-5 APS_USC source buffer.
///
/// Exact 23J220 userspace rounds the all-active source image to one 0x4000
/// data ring per active core plus one final 0x4000 control page. The main
/// allocation is client/range-5; the final physical page is additionally
/// aliased through shared range 7 and published as source +0x918. E790 wires
/// this owner only into the existing one-command selected-observer path; the
/// E790 module itself remains non-live.
pub(crate) struct G15ApsSplitSourceBacking {
    main: fw::types::GpuArray<u8>,
    trailer_alias: mmu::G15SharedBank1Mapping,
    sampler_count: u32,
    subbuffer_size: u64,
}

impl G15ApsSplitSourceBacking {
    pub(crate) fn new_for_observer(
        range5: &mut alloc::DefaultAllocator,
        bank1: &mmu::G15SharedBank1,
        notifier: Option<Arc<Mutex<mmu::G15MappingNotifier>>>,
        active_core_count: u32,
        total_core_count: u32,
    ) -> Result<Self> {
        // E782 closes only the exact all-active J615 geometry. A partial/FSTP
        // topology uses Apple's two-page tail and remains outside this slice.
        if active_core_count == 0 || active_core_count != total_core_count {
            return Err(EINVAL);
        }
        let subbuffer_size = 0x4000u64;
        let data_size = u64::from(active_core_count)
            .checked_mul(subbuffer_size)
            .ok_or(EOVERFLOW)?;
        let source_size = data_size.checked_add(0x4000).ok_or(EOVERFLOW)?;
        let source_size_usize = usize::try_from(source_size).map_err(|_| EOVERFLOW)?;
        let trailer_offset = usize::try_from(data_size).map_err(|_| EOVERFLOW)?;

        let main_alloc = range5.alloc_generic::<u8>(source_size_usize, mmu::UAT_PGSZ, None)?;
        main_alloc.dma_sync_for_cpu()?;
        let mut main = fw::types::GpuArray::<u8>::empty(main_alloc, source_size_usize)?;
        let main_fwva = main.gpu_va().get();
        let trailer_fwva = main_fwva.checked_add(data_size).ok_or(EOVERFLOW)?;
        if main_fwva & 0x3fff != 0
            || trailer_fwva & 0x3fff != 0
            || !mmu::G15_GART_RANGE5_PPL308.contains(&main_fwva)
        {
            return Err(EINVAL);
        }

        let (trailer_object, trailer_object_offset) =
            main.backing_object_range(trailer_offset, 0x4000)?;
        let trailer_alias = bank1.map_subrange(
            mmu::G15SharedBank1Aperture::Range7,
            &trailer_object.gem,
            trailer_object_offset,
            0x4000,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_G15_RANGE7_FW,
            false,
            notifier,
            mmu::G15MappingNotifyMode::Immediate,
        )?;
        if trailer_alias.iova() & 0x3fff != 0
            || !mmu::G15_GART_RANGE7.contains(&trailer_alias.iova())
        {
            return Err(EINVAL);
        }

        // Seed the exact 16-byte per-core control descriptors into the final
        // physical page before publishing either mapping to the firmware.
        let bytes = main.as_mut_slice();
        for core in 0..active_core_count {
            let subbuffer_fwva = main_fwva
                .checked_add(
                    u64::from(core)
                        .checked_mul(subbuffer_size)
                        .ok_or(EOVERFLOW)?,
                )
                .ok_or(EOVERFLOW)?;
            let entry = fw::initdata::raw::G15PerfCtrApsControlEntry::new(
                subbuffer_fwva,
                subbuffer_size,
            )
            .ok_or(EINVAL)?;
            let core_off = usize::try_from(core)
                .map_err(|_| EOVERFLOW)?
                .checked_mul(0x10)
                .ok_or(EOVERFLOW)?;
            let off = trailer_offset.checked_add(core_off).ok_or(EOVERFLOW)?;
            if bytes.len()
                < off + core::mem::size_of::<fw::initdata::raw::G15PerfCtrApsControlEntry>()
            {
                return Err(EINVAL);
            }
            // SAFETY: `main` is CPU-mapped, the final page is inside it, and
            // E782 proves each normal J615 APS core owns one 16-byte slot.
            unsafe {
                core::ptr::write_unaligned(
                    bytes
                        .as_mut_ptr()
                        .add(off)
                        .cast::<fw::initdata::raw::G15PerfCtrApsControlEntry>(),
                    entry,
                );
            }
        }
        main.dma_sync_for_device()?;

        Ok(Self {
            main,
            trailer_alias,
            sampler_count: active_core_count,
            subbuffer_size,
        })
    }

    pub(crate) fn source_size(&self) -> u64 {
        u64::from(self.sampler_count) * self.subbuffer_size + 0x4000
    }

    pub(crate) fn source_record(&self) -> Option<fw::initdata::raw::G15PerfCtrSourceRecord> {
        fw::initdata::raw::G15PerfCtrSourceRecord::aps_usc_buffer_attached(
            self.sampler_count,
            self.source_size(),
            self.main.gpu_va().get(),
            self.trailer_alias.iova(),
        )
    }


    /// E789 exact pre-START source-5 image: E783's split buffer ownership plus
    /// E787's one Compute Shader Launch selector and production trigger/control
    /// tail. `runtime_active_d8` remains zero for firmware to own at START.
    pub(crate) fn committed_source_record(
        &self,
    ) -> Option<fw::initdata::raw::G15PerfCtrSourceRecord> {
        self.source_record()?.aps_usc_counter_control(
            fw::initdata::raw::G15_PERFCTR_APS_COMPUTE_SHADER_LAUNCH_SELECTOR,
        )
    }

    pub(crate) fn state_for_core(
        &self,
        core_index: u32,
    ) -> Option<fw::initdata::raw::G15PerfCtrSamplerState> {
        if core_index >= self.sampler_count {
            return None;
        }
        let fwva = self.main.gpu_va().get().checked_add(
            u64::from(core_index).checked_mul(self.subbuffer_size)?,
        )?;
        fw::initdata::raw::G15PerfCtrSamplerState::aps_usc_core(self.subbuffer_size, fwva)
    }
}

impl G15ApsObserverRuntime {
    fn fail_closed(&mut self) {
        // A pending +0x4094 idle-coordination state or any ownership failure
        // means firmware may still retain source-5 state. Preserve mappings and
        // selector-0 context rather than releasing memory it could dereference.
        if let Some(source) = self.split_source.take() {
            core::mem::forget(source);
        }
        if let Some(context) = self.sampler_context.take() {
            core::mem::forget(context);
        }
    }

    /// E790 explicit post-STOP rollback. Returning success proves that the
    /// bootstrap source/state image replaced every APS pointer before either
    /// retained GPU-visible owner is released. Failure keeps both owners mapped.
    pub(crate) fn finish(mut self) -> Result {
        let Some(slot) = self.sampler_context.as_ref().map(|ctx| ctx.slot()) else {
            self.fail_closed();
            return Err(EIO);
        };
        if !self
            .perfctr_runtime
            .restore_bootstrap_aps_usc(slot, self.sampler_count)
        {
            self.fail_closed();
            return Err(EBUSY);
        }

        // Disarm Drop and release only after the bootstrap image is visible.
        let source = self.split_source.take();
        let context = self.sampler_context.take();
        core::mem::drop(source);
        core::mem::drop(context);
        Ok(())
    }
}

impl Drop for G15ApsObserverRuntime {
    fn drop(&mut self) {
        let Some(slot) = self.sampler_context.as_ref().map(|ctx| ctx.slot()) else {
            if self.split_source.is_some() {
                self.fail_closed();
            }
            return;
        };
        if self
            .perfctr_runtime
            .restore_bootstrap_aps_usc(slot, self.sampler_count)
        {
            return;
        }
        self.fail_closed();
    }
}

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
    #[ver(G == G15)]
    g15_init_preflight: AtomicBool,
    /// Exact accelerator +0x622 logical command-submission enable state.
    /// E193 keeps this separate from firmware-visible q4 +0x070.
    #[ver(G == G15)]
    g15_host_command_submission_enabled: AtomicBool,
    #[ver(G == G15)]
    g15_preflight_rx_doorbells: AtomicU64,
    #[ver(G == G15)]
    g15_preflight_unknown_messages: AtomicU64,
    /// Exact J615 accelerator-global UMA host state. E138 places only the
    /// 0x100 HardwareBuffer-ID namespace plus zero-seeded pool-ID sequence;
    /// it allocates no UMAPool and has no live Queue/RunCompute consumer.
    #[ver(G == G15)]
    #[pin]
    g15_uma: Mutex<buffer::G15DeviceUmaOwnerState>,
    /// E147-E152 exact accelerator-global firmware resource stacks. Their
    /// initial mapped backings are created in the pre-RTKit manager graph with
    /// E151 bootstrap-silent q22 policy; Queue/channel leases remain the only
    /// path that can select an element/FWVA, and Compute still has no live writer.
    #[ver(G == G15)]
    g15_fw_resources: buffer::G15DeviceFirmwareResourceState,
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
        g15_ualloc_range5_ppl108: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_uncached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_ppl308: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_cached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_uma_shared_pools: Option<Arc<Mutex<buffer::G15ClientUmaPoolContainerState>>>,
        owner_pid: u32,
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
    /// Send the native G15 q22 mapping-ring pressure async note on EP21.
    fn g15_mapping_pressure_kick(&self) -> Result;
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
    fn free_context(
        &self,
        data: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>,
        published_to_firmware: bool,
    );
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

        #[ver(G == G15)]
        if data.g15_init_preflight.load(Ordering::Relaxed) {
            if ep == EP_FIRMWARE && msg == MSG_RX_DOORBELL {
                data.g15_preflight_rx_doorbells
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                data.g15_preflight_unknown_messages
                    .fetch_add(1, Ordering::Relaxed);
                dev_err!(
                    dev.as_ref(),
                    "G15 first-init preflight unknown message: {:#x}:{:#x}\n",
                    ep,
                    msg
                );
            }
            return;
        }

        #[ver(G == G15)]
        {
            if ep == EP_FIRMWARE && msg == MSG_RX_DOORBELL {
                data.g15_preflight_rx_doorbells
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                data.g15_preflight_unknown_messages
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

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
        #[ver(G == G15)]
        let mgr = Self::build_validated_pre_rtkit(dev, res, cfg)?;
        #[ver(G != G15)]
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
        let mut initdata = Self::make_initdata(dev, cfg, &dyncfg, &mut alloc, uat.g15_shared_bank1())?;

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

            let initdata = Self::make_initdata(dev, cfg, &dyncfg, &mut alloc, uat.g15_shared_bank1())?;
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
                    && pre.dpe_leakage_6ec.value_000 == 0
                    && pre.dpe_leakage_6ec.value_004 == 1
                    && u64_is(&pre.dpe_leakage_6ec.value_008, 0x0000_0005_0000_01f4)
                    && u64_is(&pre.dpe_leakage_6ec.value_010, 0x0000_30d4_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_018, 0x0000_30d4_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_020, 0x0000_0bb8_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_028, 0x0000_0bb8_0000_0bb8)
                    && u64_is(&pre.dpe_leakage_6ec.value_030, 0x0000_30d4_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_038, 0x0000_30d4_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_040, 0x0000_30d4_0000_30d4)
                    && u64_is(&pre.dpe_leakage_6ec.value_048, 0x0000_30d4_0000_30d4)
                    && pre.dpe_leakage_6ec.value_050 == 1
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
    fn build_validated_pre_rtkit(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<Pin<UniqueArc<GpuManager::ver>>> {
        #[ver(G != G15)]
        {
            let _ = (dev, res, cfg);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            // E031: the persistent manager must be the sole owner of the live
            // accelerator-shared bank-1 mappings. The old initdata_preflight()
            // constructed and discarded a second complete q22/range-7 graph;
            // with the correct shared root that deliberately fail-closed graph
            // cannot be followed by another UAT in the same boot. The full
            // manager/root/q21/q22/q23 validation below remains intact.
            let mgr = Self::build_pre_rtkit(dev, res, cfg)?;

            let initdata_va = mgr.initdata.gpu_va().get();
            let init_sequence_va = mgr.initdata.g15_init_sequence.gpu_va().get();
            let wrapper_va = mgr.initdata.runtime_pointers.gpu_va().get();
            let globals_va = mgr.initdata.g15_globals.gpu_va().get();
            let q21_va = mgr.initdata.g15_q21.gpu_va().get();
            let q22_va = mgr.initdata.g15_perfctr_runtime.q22_gpu_va();
            let q23_va = mgr.initdata.g15_q23.gpu_va().get();
            let (q22_state_va, q22_ring_va, q22_ring_empty) = {
                let notifier = mgr.initdata.g15_mapping_notifier.lock();
                (
                    notifier.state_gpu_va(),
                    notifier.ring_gpu_va(),
                    notifier.is_empty(),
                )
            };
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

            let q22_ok = mgr.initdata.g15_perfctr_runtime.with_q22(|raw| {
                // Root importer FUN_fffffc00000066dc snapshots q22 +0x008
                // before firmware_ready. Apple clears both bootstrap words.
                raw.host_zero_0000.0 == 0
                    && raw.host_zero_0008.0 == 0
                    && raw.shared_ptr_4568.0 == q22_state_va
                    && raw.shared_ptr_4570.0 == q22_ring_va
                    && mmu::G15_GART_RANGE7.contains(&q22_state_va)
                    && mmu::G15_GART_RANGE7.contains(&q22_ring_va)
                    && q22_ring_empty
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
                    && raw.command_submission_enabled_070.load(Ordering::Acquire) == 0
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
                raw.host_flags.load(Ordering::Relaxed) == 0
                    && raw.banner_guard.load(Ordering::Relaxed) == 1
                    && raw.busy.load(Ordering::Relaxed) == 0
                    && raw.firmware_ready.load(Ordering::Relaxed) == 0
                    && raw.power_state.load(Ordering::Relaxed) == 0
            }) && mgr.initdata.g15_q23.with(|raw, _inner| {
                raw.snapshot_update_158.0 == 0
                    && raw.snapshot_15c.0 == 1
                    && raw.snapshot_160.0 == 500
                    && raw.snapshot_164.0 == 5
                    && raw.snapshot_168.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_170.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_178.0 == 0x0000_0bb8_0000_30d4
                    && raw.snapshot_180.0 == 0x0000_0bb8_0000_0bb8
                    && raw.snapshot_188.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_190.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_198.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_1a0.0 == 0x0000_30d4_0000_30d4
                    && raw.snapshot_1a8.0 == 1
                    && raw.tuning_update_1d0 == 0
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
                    // E642 proved exact +0xa38=0 has an unresolved paired cold-start
                    // prerequisite; retain the E641 value in this compile-only cleanup.
                    && raw.unk_460 == 1
                    && raw.g15_extra_a64 == 0
                    && raw.base_clock_khz == 24_000
                    && raw.g15_extra_a6c == 1
                    && raw.power_sample_period == mgr.dyncfg.pwr.power_sample_period
                    // Exact G15 UAT/GPTBAT + core-config header consumed by
                    // RTKit-2419 firmware. The GPTBAT base is dynamic but must
                    // be the physical base of Linux's ttbs region and 16 KiB aligned.
                    // E645/E646 require direct q22 until SecureGart registration exists.
                    && raw.unk_524 == 0
                    && raw.g15_extra_b1c == 0
                    && raw.unk_554 == 1
                    && raw.g15_zero_b3c == 0
                    && raw.g15_uat_mode_b40 == 1
                    && raw.uat_ttb_base.0 == mgr.dyncfg.uat_ttb_base
                    && raw.uat_ttb_base.0 != 0
                    && (raw.uat_ttb_base.0 & mmu::UAT_PGMSK as u64) == 0
                    && raw.gpu_core_id == cfg.gpu_core as u32
                    && raw.gpu_rev_id == hw::GpuRevisionID::C0 as u32
                    && raw.num_cores == mgr.dyncfg.id.num_cores * mgr.dyncfg.id.num_clusters
                    // V14.8.3 moves exact physical +0xad0 to source field unk_4f4.
                    && raw.unk_4f4 == 1
                    && raw.unk_4f8 == 0
                    && raw.unk_504_0 == 0
                    // Exact 23J220 and successful V14.8.3 M3 both publish
                    // physical +0xae4 as 0x31; RTKit consumes it during startup.
                    && raw.unk_504 == 0x31
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

            Ok(mgr)
        }
    }

    /// G15 bring-up checkpoint: validate and discard the exact pre-RTKit manager.
    pub(crate) fn manager_preflight(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<()> {
        let mgr = Self::build_validated_pre_rtkit(dev, res, cfg)?;
        #[ver(G == G15)]
        {
            mgr.initdata
                .g15_mapping_notifier
                .lock()
                .preflight_roundtrip()?;
            dev_info!(
                dev.as_ref(),
                "T8122 G15 q22 native mapping encoder PASS (map/unmap bytes + DMB-before-write-index publication; ring restored empty)\n"
            );
        }
        core::mem::drop(mgr);
        Ok(())
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
            // Validate the exact manager object that is about to enter RTKit.
            let mgr = Arc::from(Self::build_validated_pre_rtkit(dev, res, cfg)?);
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

    /// G15 first-handoff checkpoint: send only the InitData MSG_INIT on EP20,
    /// observe the firmware-owned q21 ready transition, then destroy RTKit.
    /// No EP21 application payload, device-control doorbell, firmware kick, DRM
    /// registration, or GPU work submission is permitted in this probe.
    pub(crate) fn first_msg_init_preflight(
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
            // The exact manager object that will be published to firmware must
            // itself pass the complete byte-exact pre-ready validator.
            let mgr = Arc::from(Self::build_validated_pre_rtkit(dev, res, cfg)?);
            mgr.g15_init_preflight.store(true, Ordering::Relaxed);
            mgr.g15_preflight_rx_doorbells.store(0, Ordering::Relaxed);
            mgr.g15_preflight_unknown_messages.store(0, Ordering::Relaxed);

            let rtkit = rtkit::RtKit::<GpuManager::ver>::new(
                dev.as_ref(),
                None,
                0,
                mgr.clone(),
            )?;
            *mgr.rtkit.lock() = Some(rtkit);

            let probe_result = (|| -> Result<(u64, u64, u32, u32, u32)> {
                let initdata_va = mgr.initdata.gpu_va().get();

                {
                    let mut guard = mgr.rtkit.lock();
                    let mut rtk = guard.as_mut().as_pin_mut().ok_or(EIO)?;

                    rtk.as_mut().boot()?;
                    if !rtk.as_mut().has_endpoint(EP_FIRMWARE)
                        || !rtk.as_mut().has_endpoint(EP_DOORBELL)
                    {
                        return Err(EIO);
                    }
                    rtk.as_mut().start_endpoint(EP_FIRMWARE)?;
                    rtk.as_mut().start_endpoint(EP_DOORBELL)?;
                }

                let bootstrap_ok = mgr.initdata.g15_q21.with(|raw, _inner| {
                    raw.banner_guard.load(Ordering::Relaxed) == 1
                        && raw.busy.load(Ordering::Relaxed) == 0
                        && raw.firmware_ready.load(Ordering::Relaxed) == 0
                        && raw.power_state.load(Ordering::Relaxed) == 0
                });
                if !bootstrap_ok || mgr.crashed.load(Ordering::Relaxed) {
                    return Err(EIO);
                }

                // This is the sole application TX in E012. In particular, do
                // not ring EP21 device-control or KICKFW after publication.
                {
                    let mut guard = mgr.rtkit.lock();
                    let mut rtk = guard.as_mut().as_pin_mut().ok_or(EIO)?;
                    rtk.as_mut().send_message(
                        EP_FIRMWARE,
                        MSG_INIT | (initdata_va & INIT_DATA_MASK),
                    )?;
                }

                let start = Instant::<Monotonic>::now();
                const READY_TIMEOUT: Delta = Delta::from_millis(1000);
                loop {
                    if mgr.crashed.load(Ordering::Relaxed) {
                        return Err(EIO);
                    }
                    let ready = mgr.initdata.g15_q21.with(|raw, _inner| {
                        raw.firmware_ready.load(Ordering::Relaxed)
                    });
                    if ready == 1 {
                        break;
                    }
                    if start.elapsed() >= READY_TIMEOUT {
                        return Err(ETIMEDOUT);
                    }
                    fsleep(Delta::from_millis(1));
                }

                // Give the post-ready thread/EP21-registration tail a bounded
                // window to surface an immediate crash or unexpected RTKit app
                // message without sending any host follow-on traffic.
                fsleep(Delta::from_millis(20));

                let (busy, ready, power_state) = mgr.initdata.g15_q21.with(|raw, _inner| {
                    (
                        raw.busy.load(Ordering::Relaxed),
                        raw.firmware_ready.load(Ordering::Relaxed),
                        raw.power_state.load(Ordering::Relaxed),
                    )
                });
                let rx_doorbells = mgr
                    .g15_preflight_rx_doorbells
                    .load(Ordering::Relaxed);
                let unknown = mgr
                    .g15_preflight_unknown_messages
                    .load(Ordering::Relaxed);

                if ready != 1 || mgr.crashed.load(Ordering::Relaxed) || unknown != 0 {
                    return Err(EIO);
                }

                Ok((initdata_va, rx_doorbells, busy, ready, power_state))
            })();

            // Always stop mailbox RX/release RTKit buffers before dropping the
            // manager. The caller stops ASC immediately after this function.
            let rtkit = mgr.rtkit.lock().take();
            core::mem::drop(rtkit);
            mgr.g15_init_preflight.store(false, Ordering::Relaxed);

            match probe_result {
                Ok((initdata_va, rx_doorbells, busy, ready, power_state)) => {
                    dev_info!(
                        dev.as_ref(),
                        "T8122 G15 first MSG_INIT PASS (InitData {:#x}; q21 busy={} ready={} power={}; EP20 RX doorbells={}; no EP21 TX/device-control/kick); RTKit destroyed\n",
                        initdata_va,
                        busy,
                        ready,
                        power_state,
                        rx_doorbells
                    );
                    core::mem::drop(mgr);
                    Ok(())
                }
                Err(e) => {
                    let (busy, ready, power_state) = mgr.initdata.g15_q21.with(|raw, _inner| {
                        (
                            raw.busy.load(Ordering::Relaxed),
                            raw.firmware_ready.load(Ordering::Relaxed),
                            raw.power_state.load(Ordering::Relaxed),
                        )
                    });
                    let rx_doorbells = mgr
                        .g15_preflight_rx_doorbells
                        .load(Ordering::Relaxed);
                    let unknown = mgr
                        .g15_preflight_unknown_messages
                        .load(Ordering::Relaxed);
                    let crashed = mgr.crashed.load(Ordering::Relaxed);
                    dev_err!(
                        dev.as_ref(),
                        "T8122 G15 first MSG_INIT preflight failed: {:?} (q21 busy={} ready={} power={}; EP20 RX doorbells={}; unknown={}; crashed={})\n",
                        e,
                        busy,
                        ready,
                        power_state,
                        rx_doorbells,
                        unknown,
                        crashed
                    );
                    core::mem::drop(mgr);
                    Err(e)
                }
            }
        }
    }

    /// G15 production-bootstrap/RX checkpoint: exercise the normal GpuManager
    /// new() -> init() pair, then enable the real Event/FWLog/KTrace/Stats RX
    /// callback path for a bounded idle window. Destroy the manager before DRM
    /// registration or any GPU work is possible.
    pub(crate) fn production_init_preflight(
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
            let mgr = Self::new(dev, res, cfg)?;
            mgr.g15_init_preflight.store(true, Ordering::Relaxed);
            mgr.g15_preflight_rx_doorbells.store(0, Ordering::Relaxed);
            mgr.g15_preflight_unknown_messages.store(0, Ordering::Relaxed);

            let probe_result = (|| -> Result<(u64, u64, u32, u32, u32)> {
                let initdata_va = mgr.initdata.gpu_va().get();
                let bootstrap_ok = mgr.initdata.g15_q21.with(|raw, _inner| {
                    raw.banner_guard.load(Ordering::Relaxed) == 1
                        && raw.busy.load(Ordering::Relaxed) == 0
                        && raw.firmware_ready.load(Ordering::Relaxed) == 0
                        && raw.power_state.load(Ordering::Relaxed) == 0
                });
                if !bootstrap_ok || mgr.crashed.load(Ordering::Relaxed) {
                    return Err(EIO);
                }

                // Exercise the actual production initializer. Its G15 branch
                // sends only EP20 MSG_INIT and waits for q21 firmware_ready.
                GpuManager::init(mgr.as_ref())?;

                // Drain any bootstrap RX records once while asynchronous RTKit
                // callbacks are still telemetry-only. This exercises the exact
                // G15 Event/FWLog/KTrace/Stats parsers without racing a callback.
                {
                    let mut ch = mgr.rx_channels.lock();
                    ch.fw_log.poll();
                    ch.ktrace.poll();
                    ch.stats.poll();
                    ch.event.poll();
                }

                mgr.g15_preflight_rx_doorbells.store(0, Ordering::Relaxed);
                mgr.g15_preflight_unknown_messages.store(0, Ordering::Relaxed);

                // E014 bounded normal-RX window. Firmware is idle and this
                // checkpoint submits no EP21 traffic or GPU work; only the real
                // production EP20 receive callback is enabled for 500 ms.
                mgr.g15_init_preflight.store(false, Ordering::Release);
                fsleep(Delta::from_millis(500));

                // Quiesce new callbacks before the final snapshot/drain, then
                // leave telemetry-only mode set until RTKit is destroyed.
                mgr.g15_init_preflight.store(true, Ordering::Release);
                fsleep(Delta::from_millis(5));
                {
                    let mut ch = mgr.rx_channels.lock();
                    ch.fw_log.poll();
                    ch.ktrace.poll();
                    ch.stats.poll();
                    ch.event.poll();
                }

                let (busy, ready, power_state) = mgr.initdata.g15_q21.with(|raw, _inner| {
                    (
                        raw.busy.load(Ordering::Relaxed),
                        raw.firmware_ready.load(Ordering::Relaxed),
                        raw.power_state.load(Ordering::Relaxed),
                    )
                });
                let rx_doorbells = mgr
                    .g15_preflight_rx_doorbells
                    .load(Ordering::Relaxed);
                let unknown = mgr
                    .g15_preflight_unknown_messages
                    .load(Ordering::Relaxed);

                if ready != 1 || mgr.crashed.load(Ordering::Relaxed) || unknown != 0 {
                    return Err(EIO);
                }

                Ok((initdata_va, rx_doorbells, busy, ready, power_state))
            })();

            // Always stop mailbox RX/release RTKit buffers before the manager
            // and its UAT allocations disappear. The caller stops ASC next.
            let rtkit = mgr.rtkit.lock().take();
            core::mem::drop(rtkit);
            mgr.g15_init_preflight.store(false, Ordering::Relaxed);

            match probe_result {
                Ok((initdata_va, rx_doorbells, busy, ready, power_state)) => {
                    dev_info!(
                        dev.as_ref(),
                        "T8122 G15 production init + normal RX PASS (InitData {:#x}; q21 busy={} ready={} power={}; normal EP20 RX doorbells={}; no unknown messages/EP21 TX/device-control/kick); RTKit destroyed\n",
                        initdata_va,
                        busy,
                        ready,
                        power_state,
                        rx_doorbells
                    );
                    core::mem::drop(mgr);
                    Ok(())
                }
                Err(e) => {
                    let (busy, ready, power_state) = mgr.initdata.g15_q21.with(|raw, _inner| {
                        (
                            raw.busy.load(Ordering::Relaxed),
                            raw.firmware_ready.load(Ordering::Relaxed),
                            raw.power_state.load(Ordering::Relaxed),
                        )
                    });
                    let rx_doorbells = mgr
                        .g15_preflight_rx_doorbells
                        .load(Ordering::Relaxed);
                    let unknown = mgr
                        .g15_preflight_unknown_messages
                        .load(Ordering::Relaxed);
                    let crashed = mgr.crashed.load(Ordering::Relaxed);
                    dev_err!(
                        dev.as_ref(),
                        "T8122 G15 production init preflight failed: {:?} (q21 busy={} ready={} power={}; EP20 RX doorbells={}; unknown={}; crashed={})\n",
                        e,
                        busy,
                        ready,
                        power_state,
                        rx_doorbells,
                        unknown,
                        crashed
                    );
                    core::mem::drop(mgr);
                    Err(e)
                }
            }
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
        g15_shared_bank1: Option<mmu::G15SharedBank1>,
    ) -> Result<KBox<fw::types::GpuObject<fw::initdata::InitData::ver>>> {
        let mut builder = initdata::InitDataBuilder::ver::new(
            dev,
            alloc,
            cfg,
            dyncfg,
            g15_shared_bank1,
        );
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

        #[ver(G == G15)]
        let g15_uma = buffer::G15DeviceUmaOwnerState::new_device_global()?;
        #[ver(G == G15)]
        let g15_fw_resources = buffer::G15DeviceFirmwareResourceState::new_mapped_device_global(
            dev,
            uat.g15_shared_bank1().ok_or(EINVAL)?,
            initdata.g15_mapping_notifier.arc(),
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
                #[ver(G == G15)]
                g15_init_preflight: AtomicBool::new(false),
                #[ver(G == G15)]
                g15_host_command_submission_enabled: AtomicBool::new(false),
                #[ver(G == G15)]
                g15_preflight_rx_doorbells: AtomicU64::new(0),
                #[ver(G == G15)]
                g15_preflight_unknown_messages: AtomicU64::new(0),
                #[ver(G == G15)]
                g15_uma <- new_mutex!(g15_uma, "g15_uma"),
                #[ver(G == G15)]
                g15_fw_resources,
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
        #[ver(G == G15)]
        let g15_zero_based_counters = true;
        #[ver(G != G15)]
        let g15_zero_based_counters = false;
        Ok(Arc::new(
            event::EventManager::new(alloc, g15_zero_based_counters)?,
            GFP_KERNEL,
        )?)
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
    pub(crate) fn get_fault_info(&self) -> Option<regs::FaultInfo> {
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

    /// E153 command-queue lifetime selection from the two exact mapped
    /// device-global firmware resource stacks. Queue construction initializes
    /// the selected timestamp/scheduler states before retaining these leases.
    pub(crate) fn g15_select_command_queue_resources(
        &self,
    ) -> Result<buffer::G15FirmwareCommandQueueResourceLeases> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            self.g15_fw_resources.select_command_queue_resources()
        }
    }

    /// E153 dormant AGXChannel-lifetime selection from the three exact mapped
    /// global channel-resource stacks. The unpublished CL-channel constructor
    /// initializes these selected states directly and retains all three leases.
    #[allow(dead_code)]
    pub(crate) fn g15_select_channel_resources(
        &self,
    ) -> Result<buffer::G15FirmwareChannelResourceLeases> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            self.g15_fw_resources.select_channel_resources()
        }
    }

    /// Definition-only E145 bridge from one client-VM weak-pool container to
    /// the accelerator/device-global UMA namespace. The client-container mutex
    /// is held by `select_or_create_compute_channel()` across the create closure,
    /// matching exact AGXChannel::init() replacement serialization. Pool/FList
    /// construction then consumes the global pool identity and cached range-5
    /// allocator only on the absent/dead-slot path. There is no live caller.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn g15_select_or_create_compute_pool(
        &self,
        container: &Arc<Mutex<buffer::G15ClientUmaPoolContainerState>>,
        range5_cached: &Arc<Mutex<alloc::DefaultAllocator>>,
        bank1: mmu::G15SharedBank1,
        mapping_notifier: Arc<Mutex<mmu::G15MappingNotifier>>,
    ) -> Result<buffer::G15ClientUmaComputeChannelRef> {
        #[ver(G != G15)]
        {
            let _ = (
                container,
                range5_cached,
                bank1,
                mapping_notifier,
            );
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            buffer::G15ClientUmaPoolContainerState::select_or_create_compute_channel(
                container,
                buffer::G15_J615_NORMAL_COMPUTE_POOL_PRIORITY_CLASS,
                || {
                    // Exact host ordering holds the shared-container lock while
                    // replacement construction runs. The wider UMA namespace
                    // and this VM's cached range-5 allocator are nested only on
                    // that replacement path.
                    let mut device_uma = self.g15_uma.lock();
                    let mut range5 = range5_cached.lock();
                    buffer::G15SharedComputeUmaPoolOwner::new_j615_unprepared(
                        &self.dev,
                        &mut device_uma,
                        buffer::G15_J615_NORMAL_COMPUTE_POOL_PRIORITY_CLASS,
                        &mut range5,
                        bank1,
                        Some(mapping_notifier),
                    )
                },
            )
        }
    }

    /// Exact accelerator +0x622 host/runtime command-submission state. Apple
    /// keeps this byte separate from q4 +0x070: ArmFirmware checks +0x622
    /// immediately before deciding whether to ring a work-pipe EP21 doorbell.
    pub(crate) fn g15_host_command_submission_enabled(&self) -> bool {
        #[ver(G != G15)]
        {
            false
        }
        #[ver(G == G15)]
        {
            self.g15_host_command_submission_enabled.load(Ordering::Acquire)
        }
    }

    /// Model the two exact state writes performed by the settled J615
    /// `AGXAccelerator::setCommandSubmissionEnabled()` transition. Auxiliary
    /// Apple host timer/event bookkeeping is deliberately not synthesized here;
    /// only the independently proven +0x622 and q4 +0x070 gate states are owned.
    fn g15_set_command_submission_state(&self, enabled: bool) -> Result {
        #[ver(G != G15)]
        {
            let _ = enabled;
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            let prior = self
                .g15_host_command_submission_enabled
                .swap(enabled, Ordering::AcqRel);
            if let Err(err) = self.g15_set_firmware_command_gate(enabled) {
                self.g15_host_command_submission_enabled
                    .store(prior, Ordering::Release);
                return Err(err);
            }
            Ok(())
        }
    }

    /// Exact firmware-visible q4 +0x070 command gate. E188 separates this
    /// from AGXArmFirmware::isCommandSubmissionEnabled(), which reads the
    /// independent accelerator +0x622 host/runtime byte. Normal J615 start
    /// enables q4 once after firmware bootstrap and keeps it enabled across
    /// ordinary CL submissions; bounded registration must not toggle it.
    pub(crate) fn g15_set_firmware_command_gate(&self, enabled: bool) -> Result {
        #[ver(G != G15)]
        {
            let _ = enabled;
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            let value = u32::from(enabled);
            let before = self.initdata.g15_globals.with(|raw, _inner| {
                raw.command_submission_enabled_070.load(Ordering::Acquire)
            });
            if before > 1 {
                dev_err!(
                    self.dev.as_ref(),
                    "T8122 G15 invalid q4 command-submission state {}\n",
                    before
                );
                return Err(EIO);
            }
            self.initdata.g15_globals.with(|raw, _inner| {
                raw.command_submission_enabled_070
                    .store(value, Ordering::Release);
            });
            mem::sync();
            let after = self.initdata.g15_globals.with(|raw, _inner| {
                raw.command_submission_enabled_070.load(Ordering::Acquire)
            });
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 E033 q4 command-submission gate {} -> {}\n",
                before,
                after
            );
            if after != value {
                return Err(EIO);
            }
            Ok(())
        }
    }

    pub(crate) fn g15_firmware_command_gate_enabled(&self) -> bool {
        #[ver(G != G15)]
        {
            false
        }
        #[ver(G == G15)]
        {
            self.initdata.g15_globals.with(|raw, _inner| {
                raw.command_submission_enabled_070.load(Ordering::Acquire) != 0
            })
        }
    }

    /// Publish the bounded G15 QueueInfo registration record. The caller owns
    /// one already-satisfied firmware Barrier at ring[0], so `wptr=1` and the
    /// EventManager-owned slot are firmware-visible without exposing any
    /// RunVertex/RunFragment/RunCompute command.
    pub(crate) fn g15_publish_barrier_queue(
        &self,
        pipe_type: PipeType,
        priority: u32,
        work_queue: fw::types::GpuWeakPointer<fw::workqueue::QueueInfo::ver>,
        event_slot: u8,
        wptr: u16,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = (pipe_type, priority, work_queue, event_slot, wptr);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if self.is_crashed() {
                return Err(ENODEV);
            }

            let pipes = match pipe_type {
                PipeType::Vertex => &self.pipes.vtx,
                PipeType::Fragment => &self.pipes.frag,
                PipeType::Compute => &self.pipes.comp,
            };
            let index: usize = priority.try_into()?;
            let mut pipe = pipes.get(index).ok_or(EINVAL)?.lock();

            // Verify EP21 is present before committing a TX-ring entry. This
            // cannot eliminate every transport failure, but avoids creating an
            // unreachable stale pipe entry when the endpoint is absent.
            {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                if !rtk.as_mut().has_endpoint(EP_DOORBELL) {
                    return Err(ENODEV);
                }
            }

            if !self.g15_firmware_command_gate_enabled() {
                dev_err!(
                    self.dev.as_ref(),
                    "T8122 G15 E033 refusing EP21 pipe doorbell while q4 command submission is disabled\n"
                );
                return Err(EACCES);
            }

            let msg = fw::channels::RunWorkQueueMsg::ver {
                g15_timestamp: U64(workqueue::g15_submission_timestamp()),
                // E186 keeps the E170/E171 bounded registration surrogate
                // byte-identical while correcting the G15 ABI type at +0x08.
                g15_work_queue_fwva: U64(work_queue.into()),
                g15_pipe_type: pipe_type,
                g15_wptr: wptr,
                // E059 proved that the scheduler needs the EventManager-owned
                // slot when ring[0] contains the satisfied Barrier record.
                g15_event_slot: event_slot,
                g15_is_new: true,
            };

            let state_before = pipe.g15_state();
            let token = pipe.send(&msg);
            let state_after_put = pipe.g15_state();
            let doorbell = MSG_TX_DOORBELL | pipe_type as u64 | ((index as u64) << 2);
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 E033 pipe {:?}/{} state before={:?} after-put={:?} token={} doorbell={:#018x}\n",
                pipe_type,
                index,
                state_before,
                state_after_put,
                token,
                doorbell
            );
            {
                let mut guard = self.rtkit.lock();
                let rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                if let Err(e) = rtk.send_message(
                    EP_DOORBELL,
                    doorbell,
                ) {
                    // A TX entry exists but its delivery state is uncertain.
                    // Fail-stop so no later host doorbell can accidentally
                    // consume it after backing storage is retired.
                    self.crashed.store(true, Ordering::Release);
                    return Err(e);
                }
            }

            let wait_result = pipe.wait_for(token);
            let state_after_wait = pipe.g15_state();
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 E033 pipe {:?}/{} wait={:?} final-state={:?}\n",
                pipe_type,
                index,
                wait_result,
                state_after_wait
            );
            if let Err(e) = wait_result {
                self.crashed.store(true, Ordering::Release);
                return Err(e);
            }
            Ok(())
        }
    }

    pub(crate) fn release_context_now(
        &self,
        context: &fw::types::GpuObject<fw::workqueue::GpuContextData>,
    ) -> Result {
        self.invalidate_context(context)
    }

    /// Exact G15 native ReleaseResource transport for the real selected
    /// scheduler resource. E190 keeps this separate from the legacy typed
    /// GpuContext surrogate and leaves it zero-caller until Queue teardown owns
    /// a proven firmware-publication state.
    pub(crate) fn g15_release_resource_fwva_now(
        &self,
        resource_fwva: u64,
        ctx_27: u8,
        ctx_0: u8,
        ctx_1: u8,
        ctx_4: u8,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = (resource_fwva, ctx_27, ctx_0, ctx_1, ctx_4);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if resource_fwva == 0 || self.is_crashed() {
                return Err(ENODEV);
            }
            let dc = fw::channels::DeviceControlMsg::ver::ReleaseResource {
                unk_4: 0,
                ctx_27,
                ctx_0,
                ctx_1,
                ctx_4,
                resource_fwva: U64(resource_fwva),
                __pad: Default::default(),
            };
            let tag = unsafe {
                core::ptr::read_unaligned(core::ptr::addr_of!(dc).cast::<u32>())
            };
            if tag != 0x11 {
                return Err(EIO);
            }

            let mut txch = self.tx_channels.lock();
            let token = txch.device_control.send(&dc);
            {
                let mut guard = self.rtkit.lock();
                let rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
            }
            txch.device_control.wait_for(token)?;

            // Opcode 0x13 marks the firmware-side host configuration dirty;
            // the exact refresh itself is deferred to firmware scheduling.
            // Keep this finite and tiny: it is not evidence by itself (0x613
            // remains the proof that bit 6 became active before START), but it
            // avoids racing START immediately against the dirty-bit handoff.
            fsleep(Delta::from_millis(2));
            Ok(())
        }
    }

    /// E193 zero-caller selected G15 submit/doorbell transaction. All RTKit
    /// endpoint and logical/firmware gate checks happen before E192 mutates the
    /// selected command ring. Once the selected RunWorkQueue is enqueued, any
    /// EP21 delivery failure fail-stops the manager and leaves the E192 ticket
    /// uncompleted so scheduler resources are retained as Uncertain.
    #[allow(dead_code)]
    pub(crate) fn run_g15_selected_job(
        &self,
        job: workqueue::G15SelectedJobSubmission::ver<'_>,
    ) -> Result {
        #[ver(G != G15)]
        {
            let _ = job;
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if self.is_crashed()
                || !self.g15_host_command_submission_enabled()
                || !self.g15_firmware_command_gate_enabled()
            {
                return Err(ENODEV);
            }

            // EP21 must exist before the first selected command/ring mutation.
            {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                if !rtk.as_mut().has_endpoint(EP_DOORBELL) {
                    return Err(ENODEV);
                }
            }

            let pipe_type = job.pipe_type()?;
            let index: usize = job.priority()?.try_into()?;
            let pipes = match pipe_type {
                PipeType::Vertex => &self.pipes.vtx,
                PipeType::Fragment => &self.pipes.frag,
                PipeType::Compute => &self.pipes.comp,
            };
            let mut pipe = pipes.get(index).ok_or(EIO)?.lock();
            let state_before = pipe.g15_state();
            let (mut completion, pipe_token) = job.run(&mut pipe)?;
            let state_after_put = pipe.g15_state();
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 selected pipe {:?}/{} before={:?} after-put={:?} token={}\n",
                pipe_type,
                index,
                state_before,
                state_after_put,
                pipe_token
            );

            // ArmFirmware checks accelerator +0x622 after encoding the pipe
            // record. If the modeled state changed underneath this transaction,
            // do not leave a queued entry available to any later submission.
            if !self.g15_host_command_submission_enabled()
                || !self.g15_firmware_command_gate_enabled()
            {
                self.crashed.store(true, Ordering::Release);
                core::mem::drop(completion);
                return Err(ENODEV);
            }

            let doorbell = MSG_TX_DOORBELL | pipe_type as u64 | ((index as u64) << 2);
            let send_result = {
                let mut guard = self.rtkit.lock();
                match guard.as_mut().as_pin_mut() {
                    Some(rtk) => rtk.send_message(EP_DOORBELL, doorbell),
                    None => Err(ENODEV),
                }
            };
            if let Err(err) = send_result {
                self.crashed.store(true, Ordering::Release);
                core::mem::drop(completion);
                return Err(err);
            }

            // Prove the scheduler consumed and returned from this selected
            // RunWorkQueue record before declaring first publication complete.
            // Engine completion remains asynchronous and is observed separately
            // through the stock-empty WorkQueue completion callback.
            let wait_result = pipe.wait_for(pipe_token);
            let state_after_wait = pipe.g15_state();
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 selected pipe {:?}/{} wait={:?} final={:?}\n",
                pipe_type,
                index,
                wait_result,
                state_after_wait
            );
            if let Err(err) = wait_result {
                self.crashed.store(true, Ordering::Release);
                core::mem::drop(completion);
                return Err(err);
            }
            core::mem::drop(pipe);

            if let Err(err) = completion.complete() {
                // EP21 succeeded but host publication bookkeeping did not. The
                // completion ticket marks the scheduler state Uncertain on Drop;
                // fail-stop to prevent a second submission into ambiguous state.
                self.crashed.store(true, Ordering::Release);
                return Err(err);
            }
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 selected first RunWorkQueue accepted by scheduler; awaiting engine completion\n"
            );

            // E480: preserve E478's CPU-mapped selected channel-ring sample
            // and add only the retained RunCompute start/end timestamp targets.
            // Exact E479 order is type0b -> start timestamp -> WFI -> end
            // timestamp -> type0c. No GPU register access occurs here.
            let log_wfi_state = |phase: &str| {
                mem::sync();
                match completion.transport_state() {
                    Ok(state) => dev_info!(
                        self.dev.as_ref(),
                        "T8122 G15 E480 channel-ring {} doneptr={} wptr={} ring_size={}\n",
                        phase, state.doneptr, state.wptr, state.ring_size,
                    ),
                    Err(err) => dev_info!(
                        self.dev.as_ref(),
                        "T8122 G15 E480 channel-ring {} error={:?}\n",
                        phase, err,
                    ),
                }
                match completion.execution_snapshot() {
                    Ok(state) => {
                        dev_info!(
                            self.dev.as_ref(),
                            "T8122 G15 E581 progress {} start={:#x} end={:#x} ctx_req={:#x} ctx_done={:#x} result={:#010x} hwid={} lifecycle={} shared={} seq={:#x} regs={}/{} recovery={:#x}\n",
                            phase,
                            state.timestamp_start,
                            state.timestamp_end,
                            state.context_store_req,
                            state.context_store_compl,
                            state.result_word,
                            state.hardware_buffer_id,
                            state.lifecycle_state,
                            state.shared_compute,
                            state.dispatch_seq,
                            state.register_count,
                            state.register_length,
                            state.recovery_marker,
                        );
                        dev_info!(
                            self.dev.as_ref(),
                            "T8122 G15 E662 dag {} entry0_state={:#010x} entry1_state={:#010x} entry1_flags={:#010x}\n",
                            phase,
                            state.dag0_state,
                            state.dag1_state,
                            state.dag1_flags,
                        );
                        dev_info!(
                            self.dev.as_ref(),
                            "T8122 G15 E665 replay-marker {} jp2_76c={:#010x} jp2_770={:#010x}\n",
                            phase,
                            state.job_params2_word0,
                            state.job_params2_word1,
                        );
                    }
                    Err(err) => dev_info!(
                        self.dev.as_ref(),
                        "T8122 G15 E487 progress {} error={:?}\n",
                        phase, err,
                    ),
                }
                // E661: E660 is the first lineage to cross WFI and publish the
                // end timestamp while doneptr remains zero. Sample only the
                // already CPU-mapped firmware-owned Compute slot-0 record so we
                // can distinguish normal type-0xc cleanup from context-store/
                // replay. This performs no GPU MMIO and changes no GPU-visible
                // command, mapping, or timing point.
                self.initdata
                    .runtime_pointers
                    .g15_stats_comp
                    .with(|raw, _inner| {
                        let base = raw as *const _ as *const u8;
                        let u32_at = |offset: usize| unsafe {
                            core::ptr::read_volatile(base.wrapping_add(offset).cast::<u32>())
                        };
                        let u64_at = |offset: usize| unsafe {
                            core::ptr::read_volatile(base.wrapping_add(offset).cast::<u64>())
                        };
                        dev_info!(
                            self.dev.as_ref(),
                            "T8122 G15 E661 slot0 {} 00={:#010x} 04={:#010x} 08={:#010x} 0c={:#010x} 10={:#018x} 18={:#018x}\n",
                            phase,
                            u32_at(0x00),
                            u32_at(0x04),
                            u32_at(0x08),
                            u32_at(0x0c),
                            u64_at(0x10),
                            u64_at(0x18),
                        );
                    });
                self.initdata
                    .runtime_pointers
                    .g15_control_state
                    .with(|raw, _inner| {
                        let base = raw as *const _ as *const u8;
                        let u32_at = |offset: usize| unsafe {
                            core::ptr::read_volatile(base.wrapping_add(offset).cast::<u32>())
                        };
                        dev_info!(
                            self.dev.as_ref(),
                            "T8122 G15 E587 control {} 00={:#010x} 04={:#010x} 08={:#010x} 0c={:#010x} 10={:#010x} 14={:#010x} 18={:#010x} 1c={:#010x} 20={:#010x} 24={:#010x} 28={:#010x} 2c={:#010x} 30={:#010x} 34={:#010x}\n",
                            phase,
                            u32_at(0x00), u32_at(0x04), u32_at(0x08),
                            u32_at(0x0c), u32_at(0x10), u32_at(0x14),
                            u32_at(0x18), u32_at(0x1c), u32_at(0x20),
                            u32_at(0x24), u32_at(0x28), u32_at(0x2c),
                            u32_at(0x30), u32_at(0x34),
                        );
                    });
            };
            log_wfi_state("immediate");
            // E667: observation-only timing refinement.  The total evidence
            // window remains 5 seconds; only intermediate CPU-side snapshots
            // are added to determine when the exact E665 replay marker flips.
            fsleep(Delta::from_millis(1));
            log_wfi_state("1ms");
            fsleep(Delta::from_millis(9));
            log_wfi_state("10ms");
            fsleep(Delta::from_millis(40));
            log_wfi_state("50ms");
            fsleep(Delta::from_millis(50));
            log_wfi_state("100ms");
            fsleep(Delta::from_millis(150));
            log_wfi_state("250ms");
            fsleep(Delta::from_millis(4750));
            log_wfi_state("5s");
            Ok(())
        }
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

        let dc = context.with(|raw, _inner| {
            #[ver(G == G15)]
            {
                let (ctx_27, ctx_0, ctx_1, ctx_4) = raw.g15_release_resource_fields();
                fw::channels::DeviceControlMsg::ver::ReleaseResource {
                    unk_4: 0,
                    ctx_27,
                    ctx_0,
                    ctx_1,
                    ctx_4,
                    resource_fwva: U64(context.weak_pointer().into()),
                    __pad: Default::default(),
                }
            }
            #[ver(G != G15)]
            {
                fw::channels::DeviceControlMsg::ver::DestroyContext {
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
                }
            }
        });

        mod_dev_dbg!(self.dev, "Context invalidation command: {:?}\n", &dc);

        #[ver(G == G15)]
        {
            // E060 exact-target guard: legacy V13.3 Unk0d must not shift G15
            // ReleaseResource away from native opcode 0x11.
            let tag = unsafe {
                core::ptr::read_unaligned(core::ptr::addr_of!(dc).cast::<u32>())
            };
            if tag != 0x11 {
                dev_err!(
                    self.dev.as_ref(),
                    "T8122 G15 E170 refusing malformed ReleaseResource tag {:#x}\n",
                    tag
                );
                return Err(EIO);
            }
            dev_info!(
                self.dev.as_ref(),
                "T8122 G15 E170 ReleaseResource tag=0x11 ABI check PASS\n"
            );
        }

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

    /// Exact G15 sampler DeviceControl transport. E732 proves ordinary START
    /// and STOP are otherwise-zero 0x38-byte entries with discriminants 4/5.
    /// Keep this private and zero-caller in E741.
    fn g15_perfctr_device_control(&self, start: bool) -> Result {
        #[ver(G != G15)]
        {
            let _ = start;
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            if self.is_crashed() || !self.g15_firmware_command_gate_enabled() {
                return Err(ENODEV);
            }
            let dc = if start {
                fw::channels::DeviceControlMsg::ver::PerfCounterStart(Default::default())
            } else {
                fw::channels::DeviceControlMsg::ver::PerfCounterStop(Default::default())
            };
            let tag = unsafe {
                core::ptr::read_unaligned(core::ptr::addr_of!(dc).cast::<u32>())
            };
            let expected = if start { 4 } else { 5 };
            if tag != expected {
                return Err(EIO);
            }

            let mut txch = self.tx_channels.lock();
            let token = txch.device_control.send(&dc);
            {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                if !rtk.as_mut().has_endpoint(EP_DOORBELL) {
                    return Err(ENODEV);
                }
                rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
            }
            txch.device_control.wait_for(token)
        }
    }

    /// E756: update the exact G15 firmware trace mask using Apple's runtime
    /// ForceConfigSnapshot sequence. q21 stays zero through boot; callers may
    /// request only sampler/KTrace class 6 (bit 6), or zero to close the window.
    ///
    /// Exact 23J220 AGXArmFirmware::submitCmdFirmwareForceConfigSnapshot()
    /// zero-fills a 0x38 DeviceControl entry, writes opcode 0x13, submits it,
    /// and waits when requested. Firmware opcode 0x13 marks the host config
    /// dirty so the next config refresh copies q21 host_flags into e89c.
    pub(crate) fn g15_set_sampler_trace_mask(&self, mask: u32) -> Result {
        #[ver(G != G15)]
        {
            let _ = mask;
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            if mask & !(1 << 6) != 0 {
                return Err(EINVAL);
            }

            self.initdata.g15_q21.with(|raw, _inner| {
                raw.host_flags.store(mask, Ordering::Release);
            });
            // Publish q21 before the DeviceControl command can ask firmware to
            // snapshot it. q21 itself is shared/coherent; this is the ordering
            // edge matching Apple's host store -> command sequence.
            mem::sync();

            let dc = fw::channels::DeviceControlMsg::ver::Unk13(Default::default());
            let tag = unsafe {
                core::ptr::read_unaligned(core::ptr::addr_of!(dc).cast::<u32>())
            };
            if tag != 0x13 {
                return Err(EIO);
            }

            let mut txch = self.tx_channels.lock();
            let token = txch.device_control.send(&dc);
            {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
                if !rtk.as_mut().has_endpoint(EP_DOORBELL) {
                    return Err(ENODEV);
                }
                rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
            }
            txch.device_control.wait_for(token)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn g15_perfctr_start(&self) -> Result {
        self.g15_perfctr_device_control(true)
    }

    #[allow(dead_code)]
    pub(crate) fn g15_perfctr_stop(&self) -> Result {
        self.g15_perfctr_device_control(false)
    }

    #[allow(dead_code)]
    pub(crate) fn g15_perfctr_first_sample(
        &self,
    ) -> Result<fw::initdata::G15PerfCtrFirstSample> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            mem::sync();
            Ok(self.initdata.g15_perfctr_runtime.first_sample())
        }
    }

    #[allow(dead_code)]
    pub(crate) fn g15_perfctr_state_snapshot_synced(
        &self,
    ) -> Result<fw::initdata::G15PerfCtrStateSnapshot> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            mem::sync();
            self.initdata
                .g15_perfctr_runtime
                .state_snapshot_synced()
                .ok_or(EIO)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn g15_perfctr_q22_sampler_state(&self) -> Result<(u32, u32, u16, u64, u64)> {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            mem::sync();
            Ok(self.initdata.g15_perfctr_runtime.q22_sampler_state())
        }
    }

    /// Prepare the exact G15 RDE observer's runtime source context for the
    /// same VM that owns the selected Compute submission. This remains
    /// deliberately zero-caller in E739 and sends no DeviceControl message.
    pub(crate) fn g15_prepare_rde_observer_context(
        &self,
        vm: &mmu::Vm,
        mut split_source: G15RdeSplitSourceBacking,
    ) -> Result<G15RdeObserverRuntime> {
        #[ver(G != G15)]
        {
            let _ = (vm, split_source);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            let (source, state) = split_source.prepare_runtime_image(
                fw::initdata::raw::G15_PERFCTR_TIMER_NCLOCK_REFERENCE_PERIOD,
            )?;
            let sampler_context = self.uat.bind_g15_sampler_context(vm)?;
            let context_id = sampler_context.slot();
            let perfctr_runtime = self.initdata.g15_perfctr_runtime.clone();
            if !perfctr_runtime.install_rde0_split_source(source, state) {
                return Err(EINVAL);
            }
            if !perfctr_runtime.patch_source_context(context_id) {
                perfctr_runtime.restore_bootstrap_rde0(context_id);
                return Err(EINVAL);
            }
            Ok(G15RdeObserverRuntime {
                sampler_context: Some(sampler_context),
                perfctr_runtime,
                split_source: Some(split_source),
            })
        }
    }


    /// E790 selected-observer APS_USC runtime installer. This closes Apple's
    /// pre-START ownership/publication sequence but itself sends no DeviceControl
    /// command; the bounded Queue transaction owns START/STOP ordering.
    pub(crate) fn g15_prepare_aps_observer_context(
        &self,
        vm: &mmu::Vm,
        split_source: G15ApsSplitSourceBacking,
    ) -> Result<G15ApsObserverRuntime> {
        #[ver(G != G15)]
        {
            let _ = (vm, split_source);
            return Err(EINVAL);
        }

        #[ver(G == G15)]
        {
            let sampler_count = split_source.sampler_count;
            let source = split_source.committed_source_record().ok_or(EINVAL)?;
            let sampler_context = self.uat.bind_g15_sampler_context(vm)?;
            let context_id = sampler_context.slot();
            let perfctr_runtime = self.initdata.g15_perfctr_runtime.clone();
            if !perfctr_runtime.install_aps_usc_split_source(source, context_id) {
                // Install may have reached streaming-DMA publication before an
                // error was observed. No START is sent, but retaining both
                // owners is safer than unmapping a possibly published pointer.
                core::mem::forget(split_source);
                core::mem::forget(sampler_context);
                return Err(EIO);
            }
            Ok(G15ApsObserverRuntime {
                sampler_context: Some(sampler_context),
                perfctr_runtime,
                split_source: Some(split_source),
                sampler_count,
            })
        }
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
        // Pre-G15 firmware expects a legacy device-control Initialize record as
        // part of startup. Exact RTKit-2419 G15 host/firmware evidence proves
        // there is no corresponding boot-time device-control producer: the G15
        // 0x38-byte ring is used only for real runtime commands, and opcode 0x1a
        // carries a non-null object pointer. Never enqueue the inherited zeroed
        // Initialize record on G15.
        #[ver(G != G15)]
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
        #[ver(G != G15)]
        rtk.as_mut()
            .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        core::mem::drop(guard);

        #[ver(G != G15)]
        self.kick_firmware()?;

        #[ver(G == G15)]
        {
            // J615/RTKit-2419 reaches its first stable bootstrap boundary by
            // consuming MSG_INIT and publishing q21.firmware_ready. EP21 is
            // registered by firmware only after this transition, so neither a
            // DEVCTRL doorbell nor an unconditional KICKFW is part of G15 boot.
            let start = Instant::<Monotonic>::now();
            const READY_TIMEOUT: Delta = Delta::from_millis(1000);
            loop {
                if self.crashed.load(Ordering::Relaxed) {
                    return Err(EIO);
                }
                let ready = self.initdata.g15_q21.with(|raw, _inner| {
                    raw.firmware_ready.load(Ordering::Relaxed)
                });
                if ready == 1 {
                    break;
                }
                if start.elapsed() >= READY_TIMEOUT {
                    return Err(ETIMEDOUT);
                }
                fsleep(Delta::from_millis(1));
            }

            // E152: q21 firmware_ready is the first Linux boundary at which the
            // native q22 consumer is known operational. The exact Apple eager
            // firmware-resource backings were already mapped before its gate was
            // enabled, so activation does not replay them. From this point on,
            // dynamic backing growth and teardown must publish q22 normally.
            self.initdata.g15_mapping_notifier.lock().activate()?;

            // E188 exact normal-start lifetime: q4 +0x070 is enabled once for
            // the settled device runtime, not toggled around individual CL
            // submissions. Userspace submission remains independently gated.
            self.g15_set_command_submission_state(true)?;
        }

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
        g15_ualloc_range5_ppl108: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_uncached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_ppl308: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_ualloc_range5_cached: Option<Arc<Mutex<alloc::DefaultAllocator>>>,
        g15_uma_shared_pools: Option<Arc<Mutex<buffer::G15ClientUmaPoolContainerState>>>,
        owner_pid: u32,
        priority: u32,
        usc_exec_base: u64,
    ) -> Result<KBox<dyn queue::Queue>> {
        // Exact E147/E148 command-queue-lifetime resource indices are selected
        // before taking the general kernel allocator lock. Any later queue
        // construction failure drops the pair and returns both global indices.
        #[ver(G == G15)]
        let g15_fw_queue_resources = Some(self.g15_select_command_queue_resources()?);
        #[ver(G != G15)]
        let g15_fw_queue_resources = None;
        let mut kalloc = self.alloc();
        let id = self.ids.queue.next();
        #[ver(G == G15)]
        let g15_mapping_notifier = Some(self.initdata.g15_mapping_notifier.arc());
        #[ver(G != G15)]
        let g15_mapping_notifier = None;
        Ok(KBox::new(
            queue::Queue::ver::new(
                &self.dev,
                vm,
                &mut kalloc,
                ualloc,
                ualloc_priv,
                g15_ualloc_range5_ppl108,
                g15_ualloc_range5_uncached,
                g15_ualloc_range5_ppl308,
                g15_ualloc_range5_cached,
                g15_uma_shared_pools,
                owner_pid,
                self.uat.g15_shared_bank1(),
                g15_mapping_notifier,
                g15_fw_queue_resources,
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

    fn g15_mapping_pressure_kick(&self) -> Result {
        #[ver(G != G15)]
        {
            return Err(EINVAL);
        }
        #[ver(G == G15)]
        {
            if self.is_crashed() {
                return Err(ENODEV);
            }

            let mut guard = self.rtkit.lock();
            let mut rtk = guard.as_mut().as_pin_mut().ok_or(ENODEV)?;
            if !rtk.as_mut().has_endpoint(EP_DOORBELL) {
                return Err(ENODEV);
            }
            // Exact AGXFirmwareKextRTBuddy::sendAsyncNoteToFirmware()
            // message used by insertNewMappingEntry() once q22 occupancy has
            // both the 0x80 and 0x40 bits set. On G15 this is the same EP21
            // 0x84 message class represented by MSG_FWCTL.
            rtk.send_message(EP_DOORBELL, MSG_FWCTL)
        }
    }

    fn flush_fw_cache(&self) -> Result {
        mod_dev_dbg!(self.dev, "Flushing coprocessor data cache\n");

        if self.is_crashed() {
            return Err(ENODEV);
        }

        #[ver(G == G15)]
        {
            // The inherited cache-flush trick used DeviceControl opcode 0x17
            // with invalid context IDs. RTKit-2419 G15 has a different 0x38-byte
            // device-control ABI and native context retirement is opcode 0x11
            // ReleaseResource. Do not emit 0x17 on G15 until an exact standalone
            // cache-flush producer is mechanically identified.
            return Err(ENODEV);
        }

        #[ver(G != G15)]
        {
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

    fn free_context(
        &self,
        ctx: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>,
        published_to_firmware: bool,
    ) {
        #[ver(G != G15)]
        let _ = published_to_firmware;

        #[ver(G == G15)]
        {
            if !published_to_firmware {
                core::mem::drop(ctx);
                return;
            }
        }

        // Older generations always need the legacy deferred invalidation. G15
        // reaches this path only if a publication attempt was not followed by a
        // confirmed native ReleaseResource completion; retaining the object is
        // safer than freeing a firmware-visible pointer.
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
