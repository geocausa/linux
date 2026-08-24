// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Preflight-only hardware identity for T8122 (M3 / G15G).
//!
//! This is deliberately *not* a runtime HwConfig. The platform driver consumes
//! only the identity/topology fields before returning ENODEV, prior to DMA/UAT,
//! ASC firmware start, initdata construction, or DRM registration. Every field
//! that belongs to those later lifecycles remains zero/empty rather than being
//! inherited from an older SoC by analogy.

use crate::f32;
use super::*;

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
    compute_preempt1_size: 0,
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
    io_mappings: &[],
    sram_base: None,
    sram_size: None,
};
