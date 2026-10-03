// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Preflight-only hardware identity for T8122 (M3 / G15G).
//!
//! This remains a staged bring-up HwConfig, not a production runtime enablement.
//! Identity, power data, InitData, RTKit boot, endpoint start, and the J615 G15
//! firmware MMIO descriptor set are independently reconstructed.  The driver
//! still fails closed before InitData MSG_INIT and DRM registration.

use crate::f32;
use super::*;

// Exact J615 / G15G firmware MMIO descriptor set reconstructed from
// AGXAccelerator::configureDevice(), AGXAcceleratorG15::configureDevice(),
// AGXAcceleratorG15G::configureDevice(), and the J615 ADT resource layout.
//
// G15 HwDataB has 31 I/O mapping records.  Apple gives names/RW defaults to
// additional placeholder records, but only the entries below have a nonzero
// physical source and element size on J615 and therefore reach
// createFWPIOMapping().  Keep all other slots None so the raw array stays zero.
const J615_G15_IO_MAPPINGS: [Option<IOMapping>; 0x1f] = [
    //  0 FenderRegs.  G15 overrides the legacy 0x5000 window to 0x104000.
    Some(IOMapping::new(0x290d00000, false, 1, 0x104000, 0, true)),
    //  1 AICTimerRegs.  Apple aligns hard-coded 0x20e101000 to 16 KiB.
    Some(IOMapping::new(0x20e100000, false, 1, 0x4000, 0, false)),
    //  2 AICSWIntRegs.  J615 meta-sw-interrupt q0=0x2d1014048 -> 16 KiB base.
    Some(IOMapping::new(0x2d1014000, false, 1, 0x4000, 0, true)),
    //  3 RGXRegs: exact J615 GPU physical base.
    Some(IOMapping::new(0x290000000, false, 1, 0x20000, 0, true)),
    //  4 UVDRegs/UVWarn is disabled: the third resource selector remains 0xff.
    None,
    None, //  5 Unused
    None, //  6 DisplayUnderrunWA
    None, //  7 TempSensorRegs
    None, //  8 PMPDoorbell
    //  9 MetrologySensorRegs
    Some(IOMapping::new(0x290e08000, false, 1, 0x8000, 0, true)),
    // 10 GMGIFAFRegs
    Some(IOMapping::new(0x290d0d000, false, 1, 0x1000, 0, true)),
    // 11 MCache registers.  Apple supplies two physical bases,
    //    0x220000000 and 0x222000000; each window is 0x58000 after 16 KiB
    //    alignment, giving a 0x02000000 stride and 0xb0000 total payload.
    Some(IOMapping::new(0x220000000, false, 2, 0x58000, 0x02000000, true)),
    None, // 12 AICBankedRegisters
    None, // 13 PMGRScratch
    None, // 14 NIA special-agent idle die 0
    None, // 15 NIA special-agent idle die 1
    None, // 16 CRE registers
    None, // 17 Streaming codec registers
    // 18 PushTelemetryDashboardRegs
    Some(IOMapping::new(0x2d03d0000, false, 1, 0x1000, 0, true)),
    // 19 PushTelemetryDashboardReadRegs
    Some(IOMapping::new(0x2d03c0000, false, 1, 0x2000, 0, false)),
    None, // 20
    None, // 21
    None, // 22
    None, // 23
    None, // 24
    // 25 ANE0Doorbell
    Some(IOMapping::new(0x31145c000, false, 1, 0x4000, 0, true)),
    // 26 PMSMetrologySensorRegs
    Some(IOMapping::new(0x2d0280000, false, 1, 0x8000, 0, false)),
    None, // 27
    None, // 28
    // 29 GFXCLKGEN_MGPU
    Some(IOMapping::new(0x290e1c000, false, 1, 0x4000, 0, false)),
    None, // 30
];

pub(crate) const HWCONFIG_PREFLIGHT: super::HwConfig = HwConfig {
    chip_id: 0x8122,
    gpu_gen: GpuGen::G15,
    gpu_variant: GpuVariant::G,
    gpu_core: GpuCore::G15G,

    // G15 UAT geometry is independently closed: 42-bit bank-local VA
    // roots and a 42-bit physical/root limit. Apple G15 host timing code and
    // the J615 clock-ref both independently establish the 24 MHz base clock.
    base_clock_hz: 24_000_000,
    uat_ias: 42,
    uat_oas: 42,
    num_dies: 1,

    // Exact E009 J615 powered-ID result: one MGPU, ten core bits.
    max_num_clusters: 1,
    max_num_cores: 10,

    // Modern get_gpu_id() defines fragment units from the core count; the
    // exact J615 powered topology therefore has ten frags. ID_COUNTS_2[23:16]
    // independently gives four GPs.
    max_num_frags: 10,
    max_num_gps: 4,
    preempt1_size: 0,
    preempt2_size: 0,
    preempt3_size: 0,
    // Exact 23J220 G15 Compute data-buffer primary size.
    // AGX::G15::Device::setupDataBufferParams() computes
    // 0xd80 + 0x700 * AGXGPUCoreConfig.MGPUs; J615 has one MGPU.
    // Four additional 8-byte command-local slots follow in Linux's
    // co-owned backing, matching the raw +0x70..+0x88 pointer set.
    compute_preempt1_size: 0x1480,
    clustering: None,
    render: HwRenderConfig { tiling_control: 0 },

    // Exact J615 / 25F84 host-driver InitData scalars. These are not consumed
    // by the current preflight-only path, but keeping them exact prevents a
    // later runtime enablement from silently inheriting older-SoC values.
    //
    // FastDie output +0x10 is zero; process-node=4 selects the G15G 110 C
    // default; HwDataA +0x1280 is allocation-zero and has no G15 host writer.
    da: HwConfigA {
        unk_87c: 0,
        unk_8cc: 11_000,
        unk_e24: 0,
    },
    db: HwConfigB {
        // Runtime /arm-io chip-revision is 0x20, and Apple stores >> 4 here.
        unk_454: 2,
        // Explicitly zeroed by AGXArmFirmware::initFirmwareData().
        unk_4e0: 0,
        // High dword of the explicit zero qword pair copied from +0x1dc8.
        unk_534: 0,
        unk_ab8: 0,
        unk_abc: 0,
        // Explicit G15 host bootstrap value.
        unk_b30: 1,
    },
    shared1_tab: &[],
    shared1_a4: 0,
    shared2_tab: &[],
    shared2_unk_508: 0,
    shared2_curves: None,
    shared3_unk: 0,
    shared3_tab: &[],
    // Apple configurePowerAndPerformanceController() seeds 700 us and the
    // exact J615 ADT has no gpu-idleoff-standby-timer override.
    idle_off_standby_timer_default: 700,
    unk_hws2_4: None,
    unk_hws2_24: 0,
    global_unk_54: 0,
    // AGXAcceleratorG15G::populateSRAMPowerScaleData() fills 1.02.
    sram_k: f32!(1.02),
    unk_coef_a: &[],
    unk_coef_b: &[],
    global_tab: None,
    has_csafr: false,
    // G15G populateFastDieDeviceConfigData() replaces the required FastDie0
    // sensor mask with 0x4248. The later mask copies are independently zero
    // on G15; initdata.rs handles that generation-specific duplication rule.
    fast_sensor_mask: [0x4248, 0],
    fast_sensor_mask_alt: [0, 0],
    fast_die0_sensor_present: 0,
    io_mappings: &J615_G15_IO_MAPPINGS,
    sram_base: None,
    sram_size: None,
};
