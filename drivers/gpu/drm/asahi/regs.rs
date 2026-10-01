// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU MMIO register abstraction
//!
//! Since the vast majority of the interactions with the GPU are brokered through the firmware,
//! there is very little need to interact directly with GPU MMIO register. This module abstracts
//! the few operations that require that, mainly reading the MMU fault status, reading GPU ID
//! information, and starting the GPU firmware coprocessor.

use crate::hw;
use kernel::{
    c_str,
    device::Core,
    devres::Devres,
    io::{
        mem::IoMem, //
        Io,
    },
    platform,
    prelude::*,
    sync::aref::ARef, //
};

/// Size of the ASC control MMIO region.
pub(crate) const ASC_CTL_SIZE: usize = 0x4000;

/// Size of the SGX MMIO region.
pub(crate) const SGX_SIZE: usize = 0x1000000;

const CPU_CONTROL: usize = 0x44;
const CPU_RUN: u32 = 0x1 << 4; // BIT(4)

const FAULT_INFO: usize = 0x17030;

const ID_VERSION: usize = 0xd04000;
const ID_UNK08: usize = 0xd04008;
const ID_COUNTS_1: usize = 0xd04010;
const ID_COUNTS_2: usize = 0xd04014;
const ID_UNK18: usize = 0xd04018;
const ID_CLUSTERS: usize = 0xd0401c;

const CORE_MASK_0: usize = 0xd01500;
const CORE_MASK_1: usize = 0xd01514;

const CORE_MASKS_G14X: usize = 0xe01500;
const FAULT_INFO_G14X: usize = 0xd8c0;
const FAULT_ADDR_G14X: usize = 0xd8c8;

// Apple G15 readChipInfo() derives its normal topology count from the same
// ID_COUNTS_1 fields used for the G14X clusters-per-die * dies calculation.
// Apple exposes this G15 value as AGXGPUCoreConfig +0x30 and labels it MGPUs.
// Keep this compile-only until the rest of generation-7 GPU-ID/core-mask
// decoding is independently closed; the live get_gpu_id() match still rejects 7.
const fn g15_mgpu_count_from_id_counts_1(id_counts_1: u32) -> u32 {
    ((id_counts_1 >> 8) & 0xff) * ((id_counts_1 >> 16) & 0xf)
}

// Apple G15 readChipInfo() has an explicit generation-7 core-ID switch.
// Core IDs are independently named by Apple's kAGXGPUCoreName[] table.
// Variant 1 deliberately remains unresolved here: the G15 subclass leaves
// the pre-existing CoreConfig value untouched for that case. Variant 4 is
// assigned G15C only when ID_COUNTS_1[19:16] == 1, exactly as Apple gates it.
const fn g15_core_from_ids(id_version: u32, id_counts_1: u32) -> Option<hw::GpuCore> {
    if (id_version >> 24) != 7 {
        return None;
    }

    match (id_version >> 16) & 0xff {
        0 => Some(hw::GpuCore::G15P),
        2 => Some(hw::GpuCore::G15G),
        3 => Some(hw::GpuCore::G15S),
        4 if ((id_counts_1 >> 16) & 0xf) == 1 => Some(hw::GpuCore::G15C),
        _ => None,
    }
}

const fn g15_core_id_or_zero(id_version: u32, id_counts_1: u32) -> u32 {
    match g15_core_from_ids(id_version, id_counts_1) {
        Some(core) => core as u32,
        None => 0,
    }
}

// CoreConfig +0x44 is the width of one MGPU's entry in Apple's exported
// `core_mask_list`. G15 normally takes ID_COUNTS_1[7:0]; the G15S case
// explicitly forces that width to 10 before publishing the topology.
const fn g15_cores_per_mgpu_from_ids(id_version: u32, id_counts_1: u32) -> u32 {
    if (id_version >> 24) == 7 && ((id_version >> 16) & 0xff) == 3 {
        10
    } else {
        id_counts_1 & 0xff
    }
}

// Exact J615 identification registers captured by the guarded E009 read-only
// probe. The gfx domain was restored fully off before the probe returned.
const J615_G15_ID_VERSION: u32 = 0x0702_2000;
const J615_G15_ID_COUNTS_1: u32 = 0x0011_010a;
const J615_G15_ID_COUNTS_2: u32 = 0x0004_0404;
const J615_G15_CORE_MASK_0: u32 = 0x0000_03ff;

const _: [(); 7] = [(); (J615_G15_ID_VERSION >> 24) as usize];
const _: [(); 2] = [(); ((J615_G15_ID_VERSION >> 16) & 0xff) as usize];
const _: [(); 0x20] = [(); ((J615_G15_ID_VERSION >> 8) & 0xff) as usize];
const _: [(); 1] = [(); g15_mgpu_count_from_id_counts_1(J615_G15_ID_COUNTS_1) as usize];
const _: [(); 10] = [(); g15_cores_per_mgpu_from_ids(J615_G15_ID_VERSION, J615_G15_ID_COUNTS_1) as usize];
const _: [(); 4] = [(); ((J615_G15_ID_COUNTS_2 >> 16) & 0xff) as usize];
const _: [(); 10] = [(); J615_G15_CORE_MASK_0.count_ones() as usize];
const _: [(); 0x3ff] = [(); (J615_G15_CORE_MASK_0 & ((1u32 << 10) - 1)) as usize];
// Pin Apple's G15S override independently of the low byte supplied here.
const _: [(); 10] = [(); g15_cores_per_mgpu_from_ids(0x0703_0000, 0x0011_0114) as usize];
// Pin Apple's generation-7 switch and exact kAGXGPUCoreName[] IDs without
// making get_gpu_id() accept generation 7 yet. The low 16 revision bits are
// immaterial to this decoder, hence zero in these compile-time probes.
const _: [(); 21] = [(); g15_core_id_or_zero(0x0700_0000, 0x0011_010a) as usize];
const _: [(); 22] = [(); g15_core_id_or_zero(J615_G15_ID_VERSION, J615_G15_ID_COUNTS_1) as usize];
const _: [(); 23] = [(); g15_core_id_or_zero(0x0703_0000, 0x0011_010a) as usize];
const _: [(); 24] = [(); g15_core_id_or_zero(0x0704_0000, 0x0011_010a) as usize];
const _: [(); 0] = [(); g15_core_id_or_zero(0x0701_0000, 0x0011_010a) as usize];
const _: [(); 0] = [(); g15_core_id_or_zero(0x0704_0000, 0x0012_010a) as usize];

/// Enum representing the unit that caused an MMU fault.
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaultUnit {
    /// Decompress / pixel fetch
    DCMP(u8),
    /// USC L1 Cache (device loads/stores)
    UL1C(u8),
    /// Compress / pixel store
    CMP(u8),
    GSL1(u8),
    IAP(u8),
    VCE(u8),
    /// Tiling Engine
    TE(u8),
    RAS(u8),
    /// Vertex Data Master
    VDM(u8),
    PPP(u8),
    /// ISP Parameter Fetch
    IPF(u8),
    IPF_CPF(u8),
    VF(u8),
    VF_CPF(u8),
    /// Depth/Stencil load/store
    ZLS(u8),

    /// Parameter Management
    dPM,
    /// Compute Data Master
    dCDM_KS(u8),
    dIPP,
    dIPP_CS,
    // Vertex Data Master
    dVDM_CSD,
    dVDM_SSD,
    dVDM_ILF,
    dVDM_ILD,
    dRDE(u8),
    FC,
    GSL2,

    /// Graphics L2 Cache Control?
    GL2CC_META(u8),
    GL2CC_MB,

    /// Parameter Management
    gPM_SP(u8),
    /// Vertex Data Master - CSD
    gVDM_CSD_SP(u8),
    gVDM_SSD_SP(u8),
    gVDM_ILF_SP(u8),
    gVDM_TFP_SP(u8),
    gVDM_MMB_SP(u8),
    /// Compute Data Master
    gCDM_CS_KS0_SP(u8),
    gCDM_CS_KS1_SP(u8),
    gCDM_CS_KS2_SP(u8),
    gCDM_KS0_SP(u8),
    gCDM_KS1_SP(u8),
    gCDM_KS2_SP(u8),
    gIPP_SP(u8),
    gIPP_CS_SP(u8),
    gRDE0_SP(u8),
    gRDE1_SP(u8),

    gCDM_CS,
    gCDM_ID,
    gCDM_CSR,
    gCDM_CSW,
    gCDM_CTXR,
    gCDM_CTXW,
    gIPP,
    gIPP_CS,
    gKSM_RCE,

    Unknown(u8),
}

/// Reason for an MMU fault.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaultReason {
    Unmapped,
    AfFault,
    WriteOnly,
    ReadOnly,
    NoAccess,
    Unknown(u8),
}

/// Collection of information about an MMU fault.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct FaultInfo {
    pub(crate) address: u64,
    pub(crate) sideband: u8,
    pub(crate) vm_slot: u32,
    pub(crate) unit_code: u8,
    pub(crate) unit: FaultUnit,
    pub(crate) level: u8,
    pub(crate) unk_5: u8,
    pub(crate) read: bool,
    pub(crate) reason: FaultReason,
}

/// Device resources for this GPU instance.
pub(crate) struct Resources {
    dev: ARef<platform::Device>,
    sgx: Pin<KBox<Devres<IoMem<SGX_SIZE>>>>,
}

/// E370 read-only J615/G15 hardware gate state.
///
/// Exact RTKit 2419 uses HwDataB I/O mapping record 3 (RGXRegs), whose
/// virt_addr lives at HwDataB +0x6a8. That record maps physical
/// 0x290000000, the same base as this sgx resource. Therefore the RTKit
/// register offsets below are directly readable through this host mapping.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G15HardwareGateSnapshot {
    pub(crate) status_c020: u64,
    pub(crate) register_list_c048: u64,
    pub(crate) control_c050: u64,
    pub(crate) slot_mask_c058: u64,
    pub(crate) active_mask_c120: u64,
    pub(crate) request_c140: u64,
    pub(crate) ack_c148: u64,
    pub(crate) active_slot_10398: u32,
}

impl Resources {
    /// Map the required resources given our platform device.
    pub(crate) fn new(pdev: &platform::Device<Core>) -> Result<Resources> {
        let sgx_req = pdev.io_request_by_name(c_str!("sgx")).ok_or(EINVAL)?;
        let sgx_iomem = KBox::pin_init(sgx_req.iomap_sized::<SGX_SIZE>(), GFP_KERNEL)?;

        Ok(Resources {
            // SAFETY: This device does DMA via the UAT IOMMU.
            dev: pdev.into(),
            sgx: sgx_iomem,
        })
    }

    fn sgx_read32<const OFF: usize>(&self) -> u32 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().read32(OFF)
        } else {
            0
        }
    }

    /* Not yet used
    fn sgx_write32<OFF: usize>(&self, val: u32) {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.write32_relaxed(val, OFF)
        }
    }
    */

    fn sgx_read64<const OFF: usize>(&self) -> u64 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().read64(OFF)
        } else {
            0
        }
    }

    /// Snapshot the exact hardware registers consumed by the retained G15
    /// scheduler/resource-admission path. This is observation-only.
    pub(crate) fn g15_hardware_gate_snapshot(&self) -> G15HardwareGateSnapshot {
        const STATUS: usize = 0xc020;
        const REGISTER_LIST: usize = 0xc048;
        const CONTROL: usize = 0xc050;
        const SLOT_MASK: usize = 0xc058;
        const ACTIVE_MASK: usize = 0xc120;
        const REQUEST: usize = 0xc140;
        const ACK: usize = 0xc148;
        const ACTIVE_SLOT: usize = 0x10398;

        const _: [(); 1] = [(); (STATUS + 8 <= 0x20000) as usize];
        const _: [(); 1] = [(); (ACTIVE_SLOT + 4 <= 0x20000) as usize];

        G15HardwareGateSnapshot {
            status_c020: self.sgx_read64::<STATUS>(),
            register_list_c048: self.sgx_read64::<REGISTER_LIST>(),
            control_c050: self.sgx_read64::<CONTROL>(),
            slot_mask_c058: self.sgx_read64::<SLOT_MASK>(),
            active_mask_c120: self.sgx_read64::<ACTIVE_MASK>(),
            request_c140: self.sgx_read64::<REQUEST>(),
            ack_c148: self.sgx_read64::<ACK>(),
            active_slot_10398: self.sgx_read32::<ACTIVE_SLOT>(),
        }
    }

    /* Not yet used
    fn sgx_write64<OFF: usize>(&self, val: u64) {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.write64_relaxed(val, OFF)
        }
    }
    */

    /// Initialize the MMIO registers for the GPU.
    pub(crate) fn init_mmio(&self) -> Result {
        // Nothing to do for now...

        Ok(())
    }

    /// Start the ASC coprocessor CPU.
    pub(crate) fn start_cpu(pdev: &platform::Device<Core>) -> Result {
        let asc_req = pdev.io_request_by_name(c_str!("asc")).ok_or(EINVAL)?;
        let asc_iomem = KBox::pin_init(asc_req.iomap_sized::<ASC_CTL_SIZE>(), GFP_KERNEL)?;
        let res = asc_iomem.access(pdev.as_ref())?.relaxed();

        let val = res.read32(CPU_CONTROL);
        res.write32(val | CPU_RUN, CPU_CONTROL);
        Ok(())
    }

    /// Stop the ASC coprocessor CPU.
    ///
    /// Used by the fail-closed T8122 UAT-handoff preflight so the preloaded
    /// firmware is explicitly stopped before platform genpd detaches/powers
    /// the GPU domain off. This is the exact inverse of `start_cpu()`.
    pub(crate) fn stop_cpu(pdev: &platform::Device<Core>) -> Result {
        let asc_req = pdev.io_request_by_name(c_str!("asc")).ok_or(EINVAL)?;
        let asc_iomem = KBox::pin_init(asc_req.iomap_sized::<ASC_CTL_SIZE>(), GFP_KERNEL)?;
        let res = asc_iomem.access(pdev.as_ref())?.relaxed();

        let val = res.read32(CPU_CONTROL);
        res.write32(val & !CPU_RUN, CPU_CONTROL);
        Ok(())
    }

    /// Get the GPU identification info from registers.
    ///
    /// See [`hw::GpuIdConfig`] for the result.
    pub(crate) fn get_gpu_id(&self) -> Result<hw::GpuIdConfig> {
        let id_version = self.sgx_read32::<ID_VERSION>();
        let id_unk08 = self.sgx_read32::<ID_UNK08>();
        let id_counts_1 = self.sgx_read32::<ID_COUNTS_1>();
        let id_counts_2 = self.sgx_read32::<ID_COUNTS_2>();
        let id_unk18 = self.sgx_read32::<ID_UNK18>();
        let id_clusters = self.sgx_read32::<ID_CLUSTERS>();

        dev_info!(
            self.dev.as_ref(),
            "GPU ID registers: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}\n",
            id_version,
            id_unk08,
            id_counts_1,
            id_counts_2,
            id_unk18,
            id_clusters
        );

        let gpu_gen = (id_version >> 24) & 0xff;

        let mut core_mask_regs = KVec::new();

        let num_clusters = match gpu_gen {
            4 | 5 => {
                // G13 | G14G
                core_mask_regs.push(self.sgx_read32::<CORE_MASK_0>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<CORE_MASK_1>(), GFP_KERNEL)?;
                (id_clusters >> 12) & 0xff
            }
            6 => {
                // G14X
                core_mask_regs.push(self.sgx_read32::<CORE_MASKS_G14X>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<{ CORE_MASKS_G14X + 4 }>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<{ CORE_MASKS_G14X + 8 }>(), GFP_KERNEL)?;
                // Clusters per die * num dies
                ((id_counts_1 >> 8) & 0xff) * ((id_counts_1 >> 16) & 0xf)
            }
            7 => {
                // G15: E007 static reconstruction + guarded E009 powered probe
                // prove the packed core mask begins at SGX +0xe01500 and the
                // exported topology count is the MGPU count from ID_COUNTS_1.
                core_mask_regs.push(self.sgx_read32::<CORE_MASKS_G14X>(), GFP_KERNEL)?;
                g15_mgpu_count_from_id_counts_1(id_counts_1)
            }
            a => {
                dev_err!(self.dev.as_ref(), "Unknown GPU generation {}\n", a);
                return Err(ENODEV);
            }
        };

        let mut core_masks_packed = KVec::new();
        core_masks_packed.extend_from_slice(&core_mask_regs, GFP_KERNEL)?;

        dev_info!(self.dev.as_ref(), "Core masks: {:#x?}\n", core_masks_packed);

        let num_cores = id_counts_1 & 0xff;

        if num_cores > 32 {
            dev_err!(
                self.dev.as_ref(),
                "Too many cores per cluster ({} > 32)\n",
                num_cores
            );
            return Err(ENODEV);
        }

        if num_cores * num_clusters > (core_mask_regs.len() * 32) as u32 {
            dev_err!(
                self.dev.as_ref(),
                "Too many total cores ({} x {} > {})\n",
                num_clusters,
                num_cores,
                core_mask_regs.len() * 32
            );
            return Err(ENODEV);
        }

        let mut core_masks = KVec::new();
        let mut total_active_cores: u32 = 0;

        let max_core_mask = ((1u64 << num_cores) - 1) as u32;
        for _ in 0..num_clusters {
            let mask = core_mask_regs[0] & max_core_mask;
            core_masks.push(mask, GFP_KERNEL)?;
            for i in 0..core_mask_regs.len() {
                core_mask_regs[i] >>= num_cores;
                if i < (core_mask_regs.len() - 1) {
                    core_mask_regs[i] |= core_mask_regs[i + 1] << (32 - num_cores);
                }
            }
            total_active_cores += mask.count_ones();
        }

        if core_mask_regs.iter().any(|a| *a != 0) {
            dev_err!(
                self.dev.as_ref(),
                "Leftover core mask: {:#x?}\n",
                core_mask_regs
            );
            return Err(EIO);
        }

        let (gpu_rev, gpu_rev_id) = match (id_version >> 8) & 0xff {
            0x00 => (hw::GpuRevision::A0, hw::GpuRevisionID::A0),
            0x01 => (hw::GpuRevision::A1, hw::GpuRevisionID::A1),
            0x10 => (hw::GpuRevision::B0, hw::GpuRevisionID::B0),
            0x11 => (hw::GpuRevision::B1, hw::GpuRevisionID::B1),
            0x20 => (hw::GpuRevision::C0, hw::GpuRevisionID::C0),
            0x21 => (hw::GpuRevision::C1, hw::GpuRevisionID::C1),
            a => {
                dev_err!(self.dev.as_ref(), "Unknown GPU revision {}\n", a);
                return Err(ENODEV);
            }
        };

        Ok(hw::GpuIdConfig {
            gpu_gen: match (id_version >> 24) & 0xff {
                4 => hw::GpuGen::G13,
                5 => hw::GpuGen::G14,
                6 => hw::GpuGen::G14, // G14X has a separate ID
                7 => hw::GpuGen::G15,
                a => {
                    dev_err!(self.dev.as_ref(), "Unknown GPU generation {}\n", a);
                    return Err(ENODEV);
                }
            },
            gpu_variant: match (id_version >> 16) & 0xff {
                1 => hw::GpuVariant::P, // Guess
                2 => hw::GpuVariant::G,
                3 => hw::GpuVariant::S,
                4 => {
                    if num_clusters > 4 {
                        hw::GpuVariant::D
                    } else {
                        hw::GpuVariant::C
                    }
                }
                a => {
                    dev_err!(self.dev.as_ref(), "Unknown GPU variant {}\n", a);
                    return Err(ENODEV);
                }
            },
            gpu_rev,
            gpu_rev_id,
            num_clusters,
            num_cores,
            num_frags: num_cores, // Used to be id_counts_1[15:8] but does not work for G14X
            num_gps: (id_counts_2 >> 16) & 0xff,
            total_active_cores,
            core_masks,
            core_masks_packed,
        })
    }

    /// Get the fault information from the MMU status register, if one occurred.
    pub(crate) fn get_fault_info(&self, cfg: &'static hw::HwConfig) -> Option<FaultInfo> {
        let g14x = cfg.gpu_core as u32 >= hw::GpuCore::G14S as u32;

        let fault_info = if g14x {
            self.sgx_read64::<FAULT_INFO_G14X>()
        } else {
            self.sgx_read64::<FAULT_INFO>()
        };

        if fault_info & 1 == 0 {
            return None;
        }

        let fault_addr = if g14x {
            self.sgx_read64::<FAULT_ADDR_G14X>()
        } else {
            fault_info >> 30
        };

        let unit_code = ((fault_info >> 9) & 0xff) as u8;
        let unit = match unit_code {
            0x00..=0x9f => match unit_code & 0xf {
                0x0 => FaultUnit::DCMP(unit_code >> 4),
                0x1 => FaultUnit::UL1C(unit_code >> 4),
                0x2 => FaultUnit::CMP(unit_code >> 4),
                0x3 => FaultUnit::GSL1(unit_code >> 4),
                0x4 => FaultUnit::IAP(unit_code >> 4),
                0x5 => FaultUnit::VCE(unit_code >> 4),
                0x6 => FaultUnit::TE(unit_code >> 4),
                0x7 => FaultUnit::RAS(unit_code >> 4),
                0x8 => FaultUnit::VDM(unit_code >> 4),
                0x9 => FaultUnit::PPP(unit_code >> 4),
                0xa => FaultUnit::IPF(unit_code >> 4),
                0xb => FaultUnit::IPF_CPF(unit_code >> 4),
                0xc => FaultUnit::VF(unit_code >> 4),
                0xd => FaultUnit::VF_CPF(unit_code >> 4),
                0xe => FaultUnit::ZLS(unit_code >> 4),
                _ => FaultUnit::Unknown(unit_code),
            },
            0xa1 => FaultUnit::dPM,
            0xa2 => FaultUnit::dCDM_KS(0),
            0xa3 => FaultUnit::dCDM_KS(1),
            0xa4 => FaultUnit::dCDM_KS(2),
            0xa5 => FaultUnit::dIPP,
            0xa6 => FaultUnit::dIPP_CS,
            0xa7 => FaultUnit::dVDM_CSD,
            0xa8 => FaultUnit::dVDM_SSD,
            0xa9 => FaultUnit::dVDM_ILF,
            0xaa => FaultUnit::dVDM_ILD,
            0xab => FaultUnit::dRDE(0),
            0xac => FaultUnit::dRDE(1),
            0xad => FaultUnit::FC,
            0xae => FaultUnit::GSL2,
            0xb0..=0xb7 => FaultUnit::GL2CC_META(unit_code & 0xf),
            0xb8 => FaultUnit::GL2CC_MB,
            0xd0..=0xdf if g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gCDM_CS,
                0x1 => FaultUnit::gCDM_ID,
                0x2 => FaultUnit::gCDM_CSR,
                0x3 => FaultUnit::gCDM_CSW,
                0x4 => FaultUnit::gCDM_CTXR,
                0x5 => FaultUnit::gCDM_CTXW,
                0x6 => FaultUnit::gIPP,
                0x7 => FaultUnit::gIPP_CS,
                0x8 => FaultUnit::gKSM_RCE,
                _ => FaultUnit::Unknown(unit_code),
            },
            0xe0..=0xff if g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gPM_SP((unit_code >> 4) & 1),
                0x1 => FaultUnit::gVDM_CSD_SP((unit_code >> 4) & 1),
                0x2 => FaultUnit::gVDM_SSD_SP((unit_code >> 4) & 1),
                0x3 => FaultUnit::gVDM_ILF_SP((unit_code >> 4) & 1),
                0x4 => FaultUnit::gVDM_TFP_SP((unit_code >> 4) & 1),
                0x5 => FaultUnit::gVDM_MMB_SP((unit_code >> 4) & 1),
                0x6 => FaultUnit::gRDE0_SP((unit_code >> 4) & 1),
                _ => FaultUnit::Unknown(unit_code),
            },
            0xe0..=0xff if !g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gPM_SP((unit_code >> 4) & 1),
                0x1 => FaultUnit::gVDM_CSD_SP((unit_code >> 4) & 1),
                0x2 => FaultUnit::gVDM_SSD_SP((unit_code >> 4) & 1),
                0x3 => FaultUnit::gVDM_ILF_SP((unit_code >> 4) & 1),
                0x4 => FaultUnit::gVDM_TFP_SP((unit_code >> 4) & 1),
                0x5 => FaultUnit::gVDM_MMB_SP((unit_code >> 4) & 1),
                0x6 => FaultUnit::gCDM_CS_KS0_SP((unit_code >> 4) & 1),
                0x7 => FaultUnit::gCDM_CS_KS1_SP((unit_code >> 4) & 1),
                0x8 => FaultUnit::gCDM_CS_KS2_SP((unit_code >> 4) & 1),
                0x9 => FaultUnit::gCDM_KS0_SP((unit_code >> 4) & 1),
                0xa => FaultUnit::gCDM_KS1_SP((unit_code >> 4) & 1),
                0xb => FaultUnit::gCDM_KS2_SP((unit_code >> 4) & 1),
                0xc => FaultUnit::gIPP_SP((unit_code >> 4) & 1),
                0xd => FaultUnit::gIPP_CS_SP((unit_code >> 4) & 1),
                0xe => FaultUnit::gRDE0_SP((unit_code >> 4) & 1),
                0xf => FaultUnit::gRDE1_SP((unit_code >> 4) & 1),
                _ => FaultUnit::Unknown(unit_code),
            },
            _ => FaultUnit::Unknown(unit_code),
        };

        let reason = match (fault_info >> 1) & 0x7 {
            0 => FaultReason::Unmapped,
            1 => FaultReason::AfFault,
            2 => FaultReason::WriteOnly,
            3 => FaultReason::ReadOnly,
            4 => FaultReason::NoAccess,
            a => FaultReason::Unknown(a as u8),
        };

        Some(FaultInfo {
            address: fault_addr << 6,
            sideband: ((fault_info >> 23) & 0x7f) as u8,
            vm_slot: ((fault_info >> 17) & 0x3f) as u32,
            unit_code,
            unit,
            level: ((fault_info >> 7) & 3) as u8,
            unk_5: ((fault_info >> 5) & 3) as u8,
            read: (fault_info & (1 << 4)) != 0,
            reason,
        })
    }
}
