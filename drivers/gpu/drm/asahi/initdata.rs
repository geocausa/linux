// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! GPU initialization data builder.
//!
//! The root of all interaction between the GPU firmware and the host driver is a complex set of
//! nested structures that we call InitData. This includes both GPU hardware/firmware configuration
//! and the pointers to the ring buffers and global data fields that are used for communication at
//! runtime.
//!
//! Many of these structures are poorly understood, so there are lots of hardcoded unknown values
//! derived from observing the InitData structures that macOS generates.

use crate::f32;
use crate::fw::initdata::*;
use crate::fw::types::*;
use crate::module_parameters;
use crate::{
    alloc,
    driver::AsahiDevice,
    gem,
    gpu,
    hw,
    mmu, //
};
use kernel::error::{
    Error,
    Result, //
};
use kernel::macros::versions;
use kernel::prelude::*;
use kernel::try_init;
use kernel::{new_mutex, sync::Arc};

use ::pin_init;
use ::pin_init::Init;

/// Builder helper for the global GPU InitData.
#[versions(AGX)]
pub(crate) struct InitDataBuilder<'a> {
    dev: &'a AsahiDevice,
    alloc: &'a mut gpu::KernelAllocators,
    cfg: &'static hw::HwConfig,
    dyncfg: &'a hw::DynConfig,
    g15_shared_bank1: Option<mmu::G15SharedBank1>,
}

#[versions(AGX)]
impl<'a> InitDataBuilder::ver<'a> {
    /// Create a new InitData builder
    pub(crate) fn new(
        dev: &'a AsahiDevice,
        alloc: &'a mut gpu::KernelAllocators,
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
        g15_shared_bank1: Option<mmu::G15SharedBank1>,
    ) -> InitDataBuilder::ver<'a> {
        InitDataBuilder::ver {
            dev,
            alloc,
            cfg,
            dyncfg,
            g15_shared_bank1,
        }
    }

    /// Create the HwDataShared1 structure, which is used in two places in InitData.
    fn hw_shared1(cfg: &'static hw::HwConfig) -> impl Init<raw::HwDataShared1> {
        init!(raw::HwDataShared1 {
            unk_a4: cfg.shared1_a4,
            ..Zeroable::init_zeroed()
        })
        .chain(|ret| {
            for (i, val) in cfg.shared1_tab.iter().enumerate() {
                ret.table[i] = *val;
            }
            Ok(())
        })
    }

    fn init_curve(
        curve: &mut raw::HwDataShared2Curve,
        unk_0: u32,
        unk_4: u32,
        t1: &[u16],
        t2: &[i16],
        t3: &[KVec<i32>],
    ) {
        curve.unk_0 = unk_0;
        curve.unk_4 = unk_4;
        (*curve.t1)[..t1.len()].copy_from_slice(t1);
        (*curve.t1)[t1.len()..].fill(t1[0]);
        (*curve.t2)[..t2.len()].copy_from_slice(t2);
        (*curve.t2)[t2.len()..].fill(t2[0]);
        for (i, a) in curve.t3.iter_mut().enumerate() {
            a.fill(0x3ffffff);
            if i < t3.len() {
                let b = &t3[i];
                (**a)[..b.len()].copy_from_slice(b);
            }
        }
    }

    /// Create the HwDataShared2 structure, which is used in two places in InitData.
    fn hw_shared2(
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
    ) -> impl Init<raw::HwDataShared2, Error> + 'a {
        try_init!(raw::HwDataShared2 {
            unk_28: Array::new([0xff; 16]),
            g14: Default::default(),
            unk_508: cfg.shared2_unk_508,
            ..Zeroable::init_zeroed()
        })
        .chain(|ret| {
            for (i, val) in cfg.shared2_tab.iter().enumerate() {
                ret.table[i] = *val;
            }

            let curve_cfg = match cfg.shared2_curves.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            };

            let mut t1 = KVec::new();
            let mut t3 = KVec::new();

            for _ in 0..curve_cfg.t3_scales.len() {
                t3.push(KVec::new(), GFP_KERNEL)?;
            }

            for (i, ps) in dyncfg.pwr.perf_states.iter().enumerate() {
                let t3_coef = curve_cfg.t3_coefs[i];
                if t3_coef == 0 {
                    t1.push(0xffff, GFP_KERNEL)?;
                    for j in t3.iter_mut() {
                        j.push(0x3ffffff, GFP_KERNEL)?;
                    }
                    continue;
                }

                let f_khz = (ps.freq_hz / 1000) as u64;
                let v_max = ps.max_volt_mv() as u64;

                t1.push(
                    (1000000000 * (curve_cfg.t1_coef as u64) / (f_khz * v_max))
                        .try_into()
                        .unwrap(),
                    GFP_KERNEL,
                )?;

                for (j, scale) in curve_cfg.t3_scales.iter().enumerate() {
                    t3[j].push(
                        (t3_coef as u64 * 1000000100 * *scale as u64 / (f_khz * v_max * 6))
                            .try_into()
                            .unwrap(),
                        GFP_KERNEL,
                    )?;
                }
            }

            ret.g14.unk_14 = 0x6000000;
            Self::init_curve(
                &mut ret.g14.curve1,
                0,
                0x20000000,
                &[0xffff],
                &[0x0f07],
                &[],
            );
            Self::init_curve(&mut ret.g14.curve2, 7, 0x80000000, &t1, curve_cfg.t2, &t3);

            Ok(())
        })
    }

    /// Create the HwDataShared3 structure, which is used in two places in InitData.
    fn hw_shared3(cfg: &'static hw::HwConfig) -> impl Init<raw::HwDataShared3> {
        pin_init::init_zeroed::<raw::HwDataShared3>().chain(|ret| {
            if !cfg.shared3_tab.is_empty() {
                ret.unk_0 = 1;
                ret.unk_4 = 500;
                ret.unk_8 = cfg.shared3_unk;
                ret.table.copy_from_slice(cfg.shared3_tab);
                ret.unk_4c = 1;
            }
            Ok(())
        })
    }

    /// Create an unknown T81xx-specific data structure.
    fn t81xx_data(
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
    ) -> impl Init<raw::T81xxData> {
        let _perf_max_pstate = dyncfg.pwr.perf_max_pstate;

        pin_init::init_zeroed::<raw::T81xxData>().chain(move |_ret| {
            match cfg.chip_id {
                0x8103 | 0x8112 => {
                    #[ver(V < V13_3)]
                    {
                        _ret.unk_d8c = 0x80000000;
                        _ret.unk_d90 = 4;
                        _ret.unk_d9c = f32!(0.6);
                        _ret.unk_da4 = f32!(0.4);
                        _ret.unk_dac = f32!(0.38552);
                        _ret.unk_db8 = f32!(65536.0);
                        _ret.unk_dbc = f32!(13.56);
                        _ret.max_pstate_scaled = 100 * _perf_max_pstate;
                    }
                }
                _ => (),
            }
            Ok(())
        })
    }

    /// Construct the exact J615/C0 G15 HwDataA pre-tail at +0x3a9c.
    ///
    /// The DPE dynamic patch count is zero on this path, so the C0 payload is
    /// deterministic. SoCHot discovers MTR sensors 3,6,8,9,11,14 on J615
    /// (bitmap 0x4b48) and masks them with 0x4248, yielding 0x4248.
    fn g15_hwdata_a_pretail() -> impl Init<raw::G15HwDataAPreTail> {
        pin_init::init_zeroed::<raw::G15HwDataAPreTail>().chain(|ret| {
            const Q_BANK: u64 = 0x03ff_ffff_03ff_ffff;
            const Q_3FFFFF: u64 = 0x003f_ffff_003f_ffff;
            const Q_0F07: u64 = 0x0f07_0f07_0f07_0f07;

            ret.constant_008 = F32::from_bits(0x40a0_0000); // 5.0f

            let dpe = &mut ret.dpe_00c;
            for v in dpe.all_ones_0cc.iter_mut() {
                *v = U64(u64::MAX);
            }
            dpe.bootstrap_0ec = U64(0x0000_0000_0008_0000);
            for v in dpe.q_3fffff_0f4.iter_mut() {
                *v = U64(Q_3FFFFF);
            }
            dpe.literal_114 = U64(0x003f_0000_0000_0000);
            dpe.literal_13c = U64(0x2000_0000_0000_0000);
            for v in dpe.all_ones_144.iter_mut() {
                *v = U64(u64::MAX);
            }
            for v in dpe.q_0f07_164.iter_mut() {
                *v = U64(Q_0F07);
            }
            for v in dpe.bank1_184.iter_mut() {
                *v = U64(Q_BANK);
            }
            dpe.special_384 = U64(0xa000_0000_0000_0017);
            for v in dpe.all_ones_38c.iter_mut() {
                *v = U64(u64::MAX);
            }
            for v in dpe.q_0f07_3ac.iter_mut() {
                *v = U64(Q_0F07);
            }
            for v in dpe.bank2_3cc.iter_mut() {
                *v = U64(Q_BANK);
            }
            dpe.control_5d4 = U64(0x0000_0000_00c0_0000);

            ret.sochot_6ec.sensor_mask_010 = U64(0x4248);
            ret.sochot_6ec.constant_018 = U64(125);
            Ok(())
        })
    }

    /// Create the HwDataA structure. This mostly contains power-related configuration.
    fn hwdata_a(&mut self) -> Result<GpuObject<HwDataA::ver>> {
        let pwr = &self.dyncfg.pwr;
        let period_ms = pwr.power_sample_period;
        let period_s = F32::from(period_ms) / f32!(1000.0);
        let ppm_filter_tc_periods = pwr.ppm_filter_time_constant_ms / period_ms;
        #[ver(V >= V13_0B4)]
        let ppm_filter_tc_ms_rounded = ppm_filter_tc_periods * period_ms;
        let ppm_filter_a = f32!(1.0) / ppm_filter_tc_periods.into();
        let perf_filter_a = f32!(1.0) / pwr.perf_filter_time_constant.into();
        let perf_filter_a2 = f32!(1.0) / pwr.perf_filter_time_constant2.into();
        let avg_power_target_filter_a = f32!(1.0) / pwr.avg_power_target_filter_tc.into();
        let avg_power_filter_tc_periods = pwr.avg_power_filter_tc_ms / period_ms;
        #[ver(V >= V13_0B4)]
        let avg_power_filter_tc_ms_rounded = avg_power_filter_tc_periods * period_ms;
        let avg_power_filter_a = f32!(1.0) / avg_power_filter_tc_periods.into();
        let pwr_filter_a = f32!(1.0) / pwr.pwr_filter_time_constant.into();

        let base_ps = pwr.perf_base_pstate;
        let base_ps_scaled = 100 * base_ps;
        let max_ps = pwr.perf_max_pstate;
        let max_ps_scaled = 100 * max_ps;
        let boost_ps_count = max_ps - base_ps;

        #[allow(unused_variables)]
        let base_clock_khz = self.cfg.base_clock_hz / 1000;
        let v_clocks_per_period = pwr.pwr_sample_period_aic_clks;

        #[allow(unused_variables)]
        let clocks_per_period_coarse = self.cfg.base_clock_hz / 1000 * pwr.power_sample_period;

        self.alloc
            .private
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                try_init!(raw::HwDataA::ver {
                    clocks_per_period: v_clocks_per_period,
                    #[ver(V >= V13_0B4)]
                    clocks_per_period_2: v_clocks_per_period,
                    pwr_status: AtomicU32::new(4),
                    unk_10: f32!(1.0),
                    actual_pstate: 1,
                    tgt_pstate: 1,
                    base_pstate_scaled: base_ps_scaled,
                    unk_40: 1,
                    max_pstate_scaled: max_ps_scaled,
                    min_pstate_scaled: 100,
                    unk_64c: 625,
                    pwr_filter_a_neg: f32!(1.0) - pwr_filter_a,
                    pwr_filter_a: pwr_filter_a,
                    pwr_integral_gain: pwr.pwr_integral_gain,
                    pwr_integral_min_clamp: pwr.pwr_integral_min_clamp.into(),
                    max_power_1: pwr.max_power_mw.into(),
                    pwr_proportional_gain: pwr.pwr_proportional_gain,
                    pwr_pstate_related_k: -F32::from(max_ps_scaled) / pwr.max_power_mw.into(),
                    pwr_pstate_max_dc_offset: pwr.pwr_min_duty_cycle as i32 - max_ps_scaled as i32,
                    max_pstate_scaled_2: max_ps_scaled,
                    max_power_2: pwr.max_power_mw,
                    max_pstate_scaled_3: max_ps_scaled,
                    ppm_filter_tc_periods_x4: ppm_filter_tc_periods * 4,
                    ppm_filter_a_neg: f32!(1.0) - ppm_filter_a,
                    ppm_filter_a: ppm_filter_a,
                    ppm_ki_dt: pwr.ppm_ki * period_s,
                    unk_6fc: f32!(65536.0),
                    ppm_kp: pwr.ppm_kp,
                    pwr_min_duty_cycle: pwr.pwr_min_duty_cycle,
                    max_pstate_scaled_4: max_ps_scaled,
                    unk_71c: f32!(0.0),
                    max_power_3: pwr.max_power_mw,
                    cur_power_mw_2: 0x0,
                    ppm_filter_tc_ms: pwr.ppm_filter_time_constant_ms,
                    #[ver(V >= V13_0B4)]
                    ppm_filter_tc_clks: ppm_filter_tc_ms_rounded * base_clock_khz,
                    perf_tgt_utilization: pwr.perf_tgt_utilization,
                    perf_boost_min_util: pwr.perf_boost_min_util,
                    perf_boost_ce_step: pwr.perf_boost_ce_step,
                    perf_reset_iters: pwr.perf_reset_iters,
                    unk_774: 6,
                    unk_778: 1,
                    perf_filter_drop_threshold: pwr.perf_filter_drop_threshold,
                    perf_filter_a_neg: f32!(1.0) - perf_filter_a,
                    perf_filter_a2_neg: f32!(1.0) - perf_filter_a2,
                    perf_filter_a: perf_filter_a,
                    perf_filter_a2: perf_filter_a2,
                    perf_ki: pwr.perf_integral_gain,
                    perf_ki2: pwr.perf_integral_gain2,
                    perf_integral_min_clamp: pwr.perf_integral_min_clamp.into(),
                    unk_79c: f32!(95.0),
                    perf_kp: pwr.perf_proportional_gain,
                    perf_kp2: pwr.perf_proportional_gain2,
                    boost_state_unk_k: F32::from(boost_ps_count) / f32!(0.95),
                    base_pstate_scaled_2: base_ps_scaled,
                    max_pstate_scaled_5: max_ps_scaled,
                    base_pstate_scaled_3: base_ps_scaled,
                    perf_tgt_utilization_2: pwr.perf_tgt_utilization,
                    base_pstate_scaled_4: base_ps_scaled,
                    unk_7fc: f32!(65536.0),
                    pwr_min_duty_cycle_2: pwr.pwr_min_duty_cycle.into(),
                    max_pstate_scaled_6: max_ps_scaled.into(),
                    max_freq_mhz: pwr.max_freq_mhz,
                    pwr_min_duty_cycle_3: pwr.pwr_min_duty_cycle,
                    min_pstate_scaled_4: f32!(100.0),
                    max_pstate_scaled_7: max_ps_scaled,
                    unk_alpha_neg: f32!(0.8),
                    unk_alpha: f32!(0.2),
                    fast_die0_sensor_mask: U64(cfg.fast_sensor_mask[0]),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask: U64(cfg.fast_sensor_mask[1]),
                    fast_die0_release_temp_cc: 100 * pwr.fast_die0_release_temp,
                    unk_87c: cfg.da.unk_87c,
                    unk_880: 0x4,
                    unk_894: f32!(1.0),

                    fast_die0_ki_dt: pwr.fast_die0_integral_gain * period_s,
                    unk_8a8: f32!(65536.0),
                    fast_die0_kp: pwr.fast_die0_proportional_gain,
                    pwr_min_duty_cycle_4: pwr.pwr_min_duty_cycle,
                    max_pstate_scaled_8: max_ps_scaled,
                    max_pstate_scaled_9: max_ps_scaled,
                    fast_die0_prop_tgt_delta: 100 * pwr.fast_die0_prop_tgt_delta,
                    unk_8cc: cfg.da.unk_8cc,
                    max_pstate_scaled_10: max_ps_scaled,
                    max_pstate_scaled_11: max_ps_scaled,
                    unk_c2c: 1,
                    power_zone_count: pwr.power_zones.len() as u32,
                    max_power_4: pwr.max_power_mw,
                    max_power_5: pwr.max_power_mw,
                    max_power_6: pwr.max_power_mw,
                    avg_power_target_filter_a_neg: f32!(1.0) - avg_power_target_filter_a,
                    avg_power_target_filter_a: avg_power_target_filter_a,
                    avg_power_target_filter_tc_x4: 4 * pwr.avg_power_target_filter_tc,
                    avg_power_target_filter_tc_xperiod: period_ms * pwr.avg_power_target_filter_tc,
                    #[ver(V >= V13_0B4)]
                    avg_power_target_filter_tc_clks: period_ms
                        * pwr.avg_power_target_filter_tc
                        * base_clock_khz,
                    avg_power_filter_tc_periods_x4: 4 * avg_power_filter_tc_periods,
                    avg_power_filter_a_neg: f32!(1.0) - avg_power_filter_a,
                    avg_power_filter_a: avg_power_filter_a,
                    avg_power_ki_dt: pwr.avg_power_ki_only * period_s,
                    unk_d20: f32!(65536.0),
                    avg_power_kp: pwr.avg_power_kp,
                    avg_power_min_duty_cycle: pwr.avg_power_min_duty_cycle,
                    max_pstate_scaled_12: max_ps_scaled,
                    max_pstate_scaled_13: max_ps_scaled,
                    max_power_7: pwr.max_power_mw.into(),
                    max_power_8: pwr.max_power_mw,
                    avg_power_filter_tc_ms: pwr.avg_power_filter_tc_ms,
                    #[ver(V >= V13_0B4)]
                    avg_power_filter_tc_clks: avg_power_filter_tc_ms_rounded * base_clock_khz,
                    max_pstate_scaled_14: max_ps_scaled,
                    t81xx_data <- Self::t81xx_data(cfg, dyncfg),
                    #[ver(V >= V13_0B4)]
                    unk_e10_0 <- {
                        let filter_a = f32!(1.0) / pwr.se_filter_time_constant.into();
                        let filter_1_a = f32!(1.0) / pwr.se_filter_time_constant_1.into();
                        try_init!(raw::HwDataA130Extra {
                            unk_38: 4,
                            unk_3c: 8000,
                            gpu_se_inactive_threshold: pwr.se_inactive_threshold,
                            gpu_se_engagement_criteria: pwr.se_engagement_criteria,
                            gpu_se_reset_criteria: pwr.se_reset_criteria,
                            unk_54: 50,
                            unk_58: 0x1,
                            gpu_se_filter_a_neg: f32!(1.0) - filter_a,
                            gpu_se_filter_1_a_neg: f32!(1.0) - filter_1_a,
                            gpu_se_filter_a: filter_a,
                            gpu_se_filter_1_a: filter_1_a,
                            gpu_se_ki_dt: pwr.se_ki * period_s,
                            gpu_se_ki_1_dt: pwr.se_ki_1 * period_s,
                            unk_7c: f32!(65536.0),
                            gpu_se_kp: pwr.se_kp,
                            gpu_se_kp_1: pwr.se_kp_1,

                            #[ver(V >= V13_3)]
                            unk_8c: 100,
                            #[ver(V < V13_3)]
                            unk_8c: 40,

                            max_pstate_scaled_1: max_ps_scaled,
                            unk_9c: f32!(8000.0),
                            unk_a0: 1400,
                            gpu_se_filter_time_constant_ms: pwr.se_filter_time_constant * period_ms,
                            gpu_se_filter_time_constant_1_ms: pwr.se_filter_time_constant_1
                                * period_ms,
                            gpu_se_filter_time_constant_clks: U64((pwr.se_filter_time_constant
                                * clocks_per_period_coarse)
                                .into()),
                            gpu_se_filter_time_constant_1_clks: U64((pwr
                                .se_filter_time_constant_1
                                * clocks_per_period_coarse)
                                .into()),
                            unk_c4: f32!(65536.0),
                            unk_114: f32!(65536.0),
                            unk_124: 40,
                            max_pstate_scaled_2: max_ps_scaled,
                            ..Zeroable::init_zeroed()
                        })
                    },
                    // G15 only programs the primary FastDie mask at HwDataA
                    // +0x8ac. The generated second copy at +0x1278 remains
                    // zero from the allocation clear; older generations clone
                    // the configured mask here.
                    #[ver(G != G15)]
                    fast_die0_sensor_mask_2: U64(cfg.fast_sensor_mask[0]),
                    #[ver(G == G15)]
                    fast_die0_sensor_mask_2: U64(0),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask_2: U64(cfg.fast_sensor_mask[1]),
                    unk_e24: cfg.da.unk_e24,
                    unk_e28: 1,
                    fast_die0_sensor_mask_alt: U64(cfg.fast_sensor_mask_alt[0]),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask_alt: U64(cfg.fast_sensor_mask_alt[1]),
                    #[ver(V < V13_0B4)]
                    fast_die0_sensor_present: U64(cfg.fast_die0_sensor_present as u64),
                    unk_163c: 1,
                    unk_3644: 0,
                    #[ver(G != G15)]
                    hws1 <- Self::hw_shared1(cfg),
                    #[ver(G != G15)]
                    hws2 <- Self::hw_shared2(cfg, dyncfg),
                    #[ver(G != G15)]
                    hws3 <- Self::hw_shared3(cfg),
                    #[ver(G != G15)]
                    unk_3ce8: 1,
                    #[ver(G == G15)]
                    g15_pretail_3a9c <- Self::g15_hwdata_a_pretail(),
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    #[ver(G == G15)]
                    {
                        // J615/T8122 MTR sensor topology imported by
                        // AGXArmFirmware::initPowerAndPerformanceData(). Apple's
                        // MtrPolynomGFX records select sensors 3, 6, 8, 9, 11,
                        // and 14, so accelerator +0x1a08 and HwDataA +0x1a98
                        // contain the exact bitmap 0x4b48. Firmware's MTR alarm
                        // handler treats a zero bitmap as fatal before it can
                        // match/acknowledge the hardware alarm.
                        const G15_MTR_SENSOR_MASK: u64 = 0x4b48;
                        const G15_MTR_POLYNOMS: [(usize, [u32; 4]); 6] = [
                            (3, [0x0000_860c, 0x0000_eed1, 0x01ff_f024, 0x01ff_fc00]),
                            (6, [0x0000_850a, 0x0000_ef64, 0x01ff_f00e, 0x01ff_fbf8]),
                            (8, [0x0000_862f, 0x0000_efe1, 0x01ff_effe, 0x01ff_fbf2]),
                            (9, [0x0000_85c2, 0x0000_f00d, 0x01ff_eff8, 0x01ff_fbf0]),
                            (11, [0x0000_8448, 0x0000_f00d, 0x01ff_eff8, 0x01ff_fbf0]),
                            (14, [0x0000_84ee, 0x0000_efbc, 0x01ff_f008, 0x01ff_fbf4]),
                        ];

                        // In the generated G15 layout `unk_1640` begins at
                        // HwDataA +0x1a94. Apple lays out the MTR bitmap at
                        // +0x1a98, a second bitmap word at +0x1aa0
                        // (zero on J615), then one 0x78-byte sensor record from
                        // +0x1aa4. Each J615 property carries four u32 values;
                        // the optional word at record +0x58 stays allocation-zero.
                        for (i, byte) in G15_MTR_SENSOR_MASK.to_le_bytes().iter().enumerate() {
                            raw.unk_1640[0x04 + i] = *byte;
                        }
                        for (sensor, coeffs) in G15_MTR_POLYNOMS {
                            let base = 0x10 + sensor * 0x78;
                            for (i, byte) in 4u32.to_le_bytes().iter().enumerate() {
                                raw.unk_1640[base + i] = *byte;
                            }
                            for (word_index, word) in coeffs.iter().enumerate() {
                                let off = base + 4 + word_index * 4;
                                for (i, byte) in word.to_le_bytes().iter().enumerate() {
                                    raw.unk_1640[off + i] = *byte;
                                }
                            }
                        }
                    }

                    for i in 0..self.dyncfg.pwr.perf_states.len() {
                        raw.sram_k[i] = self.cfg.sram_k;
                    }

                    #[ver(G != G15)]
                    for (i, coef) in pwr.core_leak_coef.iter().enumerate() {
                        raw.core_leak_coef[i] = *coef;
                    }

                    #[ver(G != G15)]
                    for (i, coef) in pwr.sram_leak_coef.iter().enumerate() {
                        raw.sram_leak_coef[i] = *coef;
                    }

                    #[ver(V >= V13_0B4 && G != G15)]
                    if let Some(csafr) = pwr.csafr.as_ref() {
                        for (i, coef) in csafr.leak_coef_afr.iter().enumerate() {
                            raw.aux_leak_coef.cs_1[i] = *coef;
                            raw.aux_leak_coef.cs_2[i] = *coef;
                        }

                        for (i, coef) in csafr.leak_coef_cs.iter().enumerate() {
                            raw.aux_leak_coef.afr_1[i] = *coef;
                            raw.aux_leak_coef.afr_2[i] = *coef;
                        }
                    }

                    for i in 0..self.dyncfg.id.num_clusters as usize {
                        if let Some(coef_a) = self.cfg.unk_coef_a.get(i) {
                            (*raw.unk_coef_a1[i])[..coef_a.len()].copy_from_slice(coef_a);
                            (*raw.unk_coef_a2[i])[..coef_a.len()].copy_from_slice(coef_a);
                        }
                        if let Some(coef_b) = self.cfg.unk_coef_b.get(i) {
                            (*raw.unk_coef_b1[i])[..coef_b.len()].copy_from_slice(coef_b);
                            (*raw.unk_coef_b2[i])[..coef_b.len()].copy_from_slice(coef_b);
                        }
                    }

                    for (i, pz) in pwr.power_zones.iter().enumerate() {
                        raw.power_zones[i].target = pz.target;
                        raw.power_zones[i].target_off = pz.target - pz.target_offset;
                        raw.power_zones[i].filter_tc_x4 = 4 * pz.filter_tc;
                        raw.power_zones[i].filter_tc_xperiod = period_ms * pz.filter_tc;
                        let filter_a = f32!(1.0) / pz.filter_tc.into();
                        raw.power_zones[i].filter_a = filter_a;
                        raw.power_zones[i].filter_a_neg = f32!(1.0) - filter_a;
                        #[ver(V >= V13_0B4)]
                        raw.power_zones[i].unk_10 = 1320000000;
                    }

                    #[ver(V >= V13_0B4 && G >= G14X && G != G15)]
                    for (i, j) in raw.hws2.g14.curve2.t1.iter().enumerate() {
                        raw.unk_hws2[i] = if *j == 0xffff { 0 } else { j / 2 };
                    }

                    if !dyncfg.hw_data_b.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::HwDataA::ver as *const u8,
                                core::mem::size_of::<raw::HwDataA::ver>(),
                            );
                            if sla.len() != dyncfg.hw_data_a.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Hwdata A size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_data_a.len(),
                                );
                            }
                            for i in 0..core::cmp::min(sla.len(), dyncfg.hw_data_a.len()) {
                                if sla[i] != dyncfg.hw_data_a[i] {
                                    matches = false;
                                    dev_err!(self.dev.as_ref(), "!!! Hwdata A first mismatch: {i}");
                                    break;
                                }
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Hwdata A match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the HwDataB structure. This mostly contains GPU-related configuration.
    fn hwdata_b(&mut self) -> Result<GpuObject<HwDataB::ver>> {
        self.alloc
            .private
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                try_init!(raw::HwDataB::ver {
                    // Userspace VA map related
                    #[ver(V < V13_0B4)]
                    unk_0: U64(0x13_00000000),
                    unk_8: U64(0x14_00000000),
                    #[ver(V < V13_0B4)]
                    unk_10: U64(0x1_00000000),
                    unk_18: U64(0xffc00000),
                    // USC start
                    unk_20: U64(0), // U64(0x11_00000000),
                    unk_28: U64(0), // U64(0x11_00000000),
                    // Unknown page
                    //unk_30: U64(0x6f_ffff8000),
                    unk_30: U64(mmu::IOVA_UNK_PAGE),
                    #[ver(G != G15)]
                    timestamp_area_base: U64(gpu::IOVA_KERN_TIMESTAMP_RANGE.start),
                    #[ver(G == G15)]
                    // On G15 this inherited field lands at exact HwDataB +0x28.
                    // Apple writes convertGPUVAToFWVA(0xfffffc2011800000) here;
                    // ChinookV9's conversion is identity (eGartRange 11 base).
                    timestamp_area_base: U64(0xffff_fc20_1180_0000),
                    // TODO: yuv matrices
                    chip_id: cfg.chip_id,
                    unk_454: cfg.db.unk_454,
                    #[ver(G != G15)]
                    unk_458: 0x1,
                    #[ver(G == G15)]
                    // Exact J615/G15G chip-info revision-low word at HwDataB +0xa30.
                    unk_458: 0x0,
                    #[ver(G == G15)]
                    // Exact J615/G15G process-node word at HwDataB +0xa34.
                    unk_45c: 0x4,
                    unk_460: 0x1,
                    unk_464: 0x1,
                    unk_468: 0x1,
                    unk_47c: 0x1,
                    unk_484: 0x1,
                    unk_48c: 0x1,
                    base_clock_khz: cfg.base_clock_hz / 1000,
                    #[ver(G != G15)]
                    power_sample_period: dyncfg.pwr.power_sample_period,
                    #[ver(G == G15)]
                    // Exact J615 production path: `model-slow` is absent from
                    // Apple's ADT, so AGXFirmware writes 1 to HwDataB +0xa6c.
                    power_sample_period: 1,
                    unk_49c: 0x1,
                    unk_4a0: 0x1,
                    unk_4a4: 0x1,
                    unk_4c0: 0x1f,
                    unk_4e0: U64(cfg.db.unk_4e0),
                    unk_4f0: 0x1,
                    unk_4f4: 0x1,
                    unk_504: 0x31,
                    unk_524: 0x1, // use_secure_cache_flush
                    unk_534: cfg.db.unk_534,
                    num_frags: dyncfg.id.num_frags * dyncfg.id.num_clusters,
                    unk_554: 0x1,
                    #[ver(G == G15)]
                    g15_uat_mode_b40: 1,
                    uat_ttb_base: U64(dyncfg.uat_ttb_base),
                    gpu_core_id: cfg.gpu_core as u32,
                    gpu_rev_id: dyncfg.id.gpu_rev_id as u32,
                    num_cores: dyncfg.id.num_cores * dyncfg.id.num_clusters,
                    max_pstate: dyncfg.pwr.perf_states.len() as u32 - 1,
                    #[ver(V < V13_0B4)]
                    num_pstates: dyncfg.pwr.perf_states.len() as u32,
                    #[ver(V < V13_0B4)]
                    min_sram_volt: dyncfg.pwr.min_sram_microvolt / 1000,
                    #[ver(V < V13_0B4)]
                    unk_ab8: cfg.db.unk_ab8,
                    #[ver(V < V13_0B4)]
                    unk_abc: cfg.db.unk_abc,
                    #[ver(V < V13_0B4)]
                    unk_ac0: 0x1020,

                    #[ver(G == G15)]
                    unk_4f8: 1,
                    #[ver(V >= V13_0B4)]
                    unk_ae4: Array::new([0x0, 0x3, 0x7, 0x7]),
                    #[ver(V < V13_0B4)]
                    unk_ae4: Array::new([0x0, 0xf, 0x3f, 0x3f]),
                    unk_b10: 0x1,
                    timer_offset: U64(0),
                    #[ver(G != G15)]
                    unk_b24: 0x1,
                    #[ver(G != G15)]
                    unk_b28: 0x1,
                    #[ver(G != G15)]
                    unk_b2c: 0x1,
                    #[ver(G != G15)]
                    unk_b30: cfg.db.unk_b30,
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_b38_0: 1,
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_b38_4: 1,
                    #[ver(G != G15)]
                    unk_b38: Array::new([0xffffffff; 12]),
                    #[ver(G == G15)]
                    // Exact final J615/G15G image produced by Apple's
                    // initFirmwareData() immediately before bootFirmware().
                    g15_startup_17ec: raw::G15HwDataBStartup {
                        zero_17ec: Array::new([0; 2]),
                        flag_17f4: 1,
                        one_17f8: 1,
                        one_17fc: 1,
                        flag_1800: 1,
                        zero_1804: 0,
                        one_1808: 1,
                        one_180c: 1,
                        flag_1810: 1,
                        zero_1814: 0,
                        sentinels_1818: Array::new([0xffff_ffff; 12]),
                        zero_1848: 0,
                        zero_184c: Array::new([0; 2]),
                        zero_1854: 0,
                        flag_1858: 1,
                        zero_185c: 0,
                    },
                    #[ver(V >= V13_0B4 && V < V13_3 && G != G15)]
                    unk_c3c: 0x19,
                    #[ver(V >= V13_3 && G != G15)]
                    unk_c3c: 0x1a,
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    #[ver(V >= V13_3)]
                    for i in 0..16 {
                        raw.unk_arr_0[i] = i as u32;
                    }

                    let base_ps = self.dyncfg.pwr.perf_base_pstate as usize;
                    let max_ps = self.dyncfg.pwr.perf_max_pstate as usize;
                    let base_freq = self.dyncfg.pwr.perf_states[base_ps].freq_hz;
                    let max_freq = self.dyncfg.pwr.perf_states[max_ps].freq_hz;

                    for (i, ps) in self.dyncfg.pwr.perf_states.iter().enumerate() {
                        raw.frequencies[i] = ps.freq_hz / 1000000;
                        for (j, mv) in ps.volt_mv.iter().enumerate() {
                            let sram_mv = (*mv).max(self.dyncfg.pwr.min_sram_microvolt / 1000);
                            raw.voltages[i][j] = *mv;
                            raw.voltages_sram[i][j] = sram_mv;
                        }
                        for j in ps.volt_mv.len()..raw.voltages[i].len() {
                            raw.voltages[i][j] = raw.voltages[i][0];
                            raw.voltages_sram[i][j] = raw.voltages_sram[i][0];
                        }
                        raw.sram_k[i] = self.cfg.sram_k;
                        raw.rel_max_powers[i] = ps.pwr_mw * 100 / self.dyncfg.pwr.max_power_mw;
                        raw.rel_boost_freqs[i] = if i > base_ps {
                            (ps.freq_hz - base_freq) / ((max_freq - base_freq) / 100)
                        } else {
                            0
                        };
                    }

                    #[ver(V >= V13_0B4)]
                    if let Some(csafr) = self.dyncfg.pwr.csafr.as_ref() {
                        let aux = &mut raw.aux_ps;
                        aux.cs_max_pstate = (csafr.perf_states_cs.len() - 1).try_into()?;
                        aux.afr_max_pstate = (csafr.perf_states_afr.len() - 1).try_into()?;

                        for (i, ps) in csafr.perf_states_cs.iter().enumerate() {
                            aux.cs_frequencies[i] = ps.freq_hz / 1000000;
                            for (j, mv) in ps.volt_mv.iter().enumerate() {
                                let sram_mv = (*mv).max(csafr.min_sram_microvolt / 1000);
                                aux.cs_voltages[i][j] = *mv;
                                aux.cs_voltages_sram[i][j] = sram_mv;
                            }
                        }

                        for (i, ps) in csafr.perf_states_afr.iter().enumerate() {
                            aux.afr_frequencies[i] = ps.freq_hz / 1000000;
                            for (j, mv) in ps.volt_mv.iter().enumerate() {
                                let sram_mv = (*mv).max(csafr.min_sram_microvolt / 1000);
                                aux.afr_voltages[i][j] = *mv;
                                aux.afr_voltages_sram[i][j] = sram_mv;
                            }
                        }
                    }

                    // Special case override for T602x
                    #[ver(G == G14X)]
                    if dyncfg.id.gpu_rev_id == hw::GpuRevisionID::B1 {
                        raw.gpu_rev_id = hw::GpuRevisionID::B0 as u32;
                    }

                    if !dyncfg.hw_data_b.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::HwDataB::ver as *const u8,
                                core::mem::size_of::<raw::HwDataB::ver>(),
                            );
                            if sla.len() != dyncfg.hw_data_b.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Hwdata B size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_data_b.len(),
                                );
                            }
                            for i in 0..core::cmp::min(sla.len(), dyncfg.hw_data_b.len()) {
                                if sla[i] != dyncfg.hw_data_b[i] {
                                    matches = false;
                                    dev_err!(self.dev.as_ref(), "!!! Hwdata B first mismatch: {i}");
                                    break;
                                }
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Hwdata B match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the Globals structure, which contains global firmware config including more power
    /// configuration data and globals used to exchange state between the firmware and driver.
    fn globals(&mut self) -> Result<GpuObject<Globals::ver>> {
        self.alloc
            .private
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                let pwr = &dyncfg.pwr;
                let period_ms = pwr.power_sample_period;
                let period_s = F32::from(period_ms) / f32!(1000.0);
                let avg_power_filter_tc_periods = pwr.avg_power_filter_tc_ms / period_ms;

                let max_ps = pwr.perf_max_pstate;
                let max_ps_scaled = 100 * max_ps;

                try_init!(raw::Globals::ver {
                    //ktrace_enable: 0xffffffff,
                    ktrace_enable: 0,
                    #[ver(V >= V13_2)]
                    unk_24_0: 3000,
                    unk_24: 0,
                    #[ver(V >= V13_0B4)]
                    debug: 0,
                    unk_28: 1,
                    #[ver(G >= G14X)]
                    unk_2c_0: 1,
                    #[ver(V >= V13_0B4 && G < G14X)]
                    unk_2c_0: 0,
                    unk_2c: 1,
                    unk_30: 0,
                    unk_34: 120,
                    // sub <- try_init!(raw::GlobalsSub::ver {
                        unk_54: cfg.global_unk_54,
                        unk_56: 40,
                        unk_58: 0xffff,
                        unk_5e: U32(1),
                        unk_66: U32(1),
                    //     ..Zeroable::init_zeroed()
                    // }),
                    unk_8900: 1,
                    pending_submissions: AtomicU32::new(0),
                    max_power: pwr.max_power_mw,
                    max_pstate_scaled: max_ps_scaled,
                    max_pstate_scaled_2: max_ps_scaled,
                    max_pstate_scaled_3: max_ps_scaled,
                    power_zone_count: pwr.power_zones.len() as u32,
                    avg_power_filter_tc_periods: avg_power_filter_tc_periods,
                    avg_power_ki_dt: pwr.avg_power_ki_only * period_s,
                    avg_power_kp: pwr.avg_power_kp,
                    avg_power_min_duty_cycle: pwr.avg_power_min_duty_cycle,
                    avg_power_target_filter_tc: pwr.avg_power_target_filter_tc,
                    unk_89bc: cfg.da.unk_8cc,
                    fast_die0_release_temp: 100 * pwr.fast_die0_release_temp,
                    unk_89c4: cfg.da.unk_87c,
                    fast_die0_prop_tgt_delta: 100 * pwr.fast_die0_prop_tgt_delta,
                    fast_die0_kp: pwr.fast_die0_proportional_gain,
                    fast_die0_ki_dt: pwr.fast_die0_integral_gain * period_s,
                    unk_89e0: 1,
                    max_power_2: pwr.max_power_mw,
                    ppm_kp: pwr.ppm_kp,
                    ppm_ki_dt: pwr.ppm_ki * period_s,
                    #[ver(V >= V13_0B4)]
                    unk_89f4_8: 1,
                    unk_89f4: 0,
                    hws1 <- Self::hw_shared1(cfg),
                    hws2 <- Self::hw_shared2(cfg, dyncfg),
                    hws3 <- Self::hw_shared3(cfg),
                    #[ver(V >= V13_0B4)]
                    idle_off_standby_timer: pwr.idle_off_standby_timer,
                    #[ver(V >= V13_0B4)]
                    unk_hws2_4: cfg.unk_hws2_4.map(Array::new).unwrap_or_default(),
                    #[ver(V >= V13_0B4)]
                    unk_hws2_24: cfg.unk_hws2_24,
                    unk_900c: 1,
                    #[ver(V >= V13_0B4)]
                    unk_9010_0: 1,
                    #[ver(V >= V13_0B4)]
                    unk_903c: 1,
                    #[ver(V < V13_0B4)]
                    unk_903c: 0,
                    fault_control: *module_parameters::fault_control.value(),
                    do_init: 1,
                    progress_check_interval_3d: 40,
                    progress_check_interval_ta: 10,
                    progress_check_interval_cl: 250,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_0: 1,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_4: 1,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_8: 100,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_c: 1,
                    idle_off_delay_ms: AtomicU32::new(pwr.idle_off_delay_ms),
                    fender_idle_off_delay_ms: pwr.fender_idle_off_delay_ms,
                    fw_early_wake_timeout_ms: pwr.fw_early_wake_timeout_ms,
                    cl_context_switch_timeout_ms: 40,
                    #[ver(V >= V13_0B4)]
                    cl_kill_timeout_ms: 50,
                    #[ver(V >= V13_0B4)]
                    unk_11edc: 0,
                    #[ver(V >= V13_0B4)]
                    unk_11efc: 0,
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    for (i, pz) in self.dyncfg.pwr.power_zones.iter().enumerate() {
                        raw.power_zones[i].target = pz.target;
                        raw.power_zones[i].target_off = pz.target - pz.target_offset;
                        raw.power_zones[i].filter_tc = pz.filter_tc;
                    }

                    if let Some(tab) = self.cfg.global_tab.as_ref() {
                        for (i, x) in tab.iter().enumerate() {
                            raw.unk_118ec[i] = *x;
                        }
                        raw.unk_118e8 = 1;
                    }

                    if !dyncfg.hw_globals.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::Globals::ver as *const u8,
                                core::mem::size_of::<raw::Globals::ver>(),
                            );
                            if sla.len() != dyncfg.hw_globals.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Globals size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_globals.len(),
                                );
                            }
                            for i in 0..core::cmp::min(sla.len(), dyncfg.hw_globals.len()) {
                                if sla[i] != dyncfg.hw_globals[i] {
                                    matches = false;
                                    dev_err!(self.dev.as_ref(), "!!! Globals first mismatch: {i}");
                                    break;
                                }
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Globals match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the RuntimePointers structure, which contains pointers to most of the other
    /// structures including the ring buffer channels, statistics structures, and HwDataA/HwDataB.
    fn runtime_pointers(&mut self) -> Result<GpuObject<RuntimePointers::ver>> {
        let hwa = self.hwdata_a()?;
        let hwb = self.hwdata_b()?;

        let mut buffer_mgr_ctl = gem::new_kernel_object(self.dev, 0x4000)?;
        buffer_mgr_ctl.vmap()?.memset(0);

        GpuObject::new_init_prealloc(
            self.alloc.private.alloc_object()?,
            |_ptr| {
                let alloc = &mut *self.alloc;
                try_init!(RuntimePointers::ver {
                    stats <- {
                        let alloc = &mut *alloc;
                        try_init!(Stats::ver {
                            vtx: alloc.private.new_default::<GpuGlobalStatsVtx>()?,
                            frag: alloc.private.new_init(
                                pin_init::init_zeroed::<GpuGlobalStatsFrag::ver>(),
                                |_inner, _ptr| {
                                    try_init!(raw::GpuGlobalStatsFrag::ver {
                                        total_cmds: 0,
                                        unk_4: 0,
                                        stats: Default::default(),
                                    })
                                }
                            )?,
                            comp: alloc.private.new_default::<GpuStatsComp>()?,
                        })
                    },
                    #[ver(G == G15)]
                    // Exact 0x88-byte persistent firmware time/activity
                    // snapshot. Zero marker means first use; firmware then
                    // owns the save/restore contents.
                    g15_persistent_time: alloc.private.new_default::<G15FirmwareTimeState>()?,
                    #[ver(G == G15)]
                    // Exact Apple 0x60-byte firmware control/state block.
                    g15_control_state: alloc.private.new_default::<G15ControlState>()?,
                    #[ver(G == G15)]
                    g15_stats_vtx: alloc.private.new_default::<G15StatsVtx>()?,
                    #[ver(G == G15)]
                    g15_stats_frag: alloc.private.new_object(
                        Default::default(),
                        |_inner| raw::G15StatsFrag {
                            pad_000: Default::default(),
                            cur_stamp_id: -1,
                            pad_c1c: Default::default(),
                            unk_id: -1,
                            pad_c34: Default::default(),
                        },
                    )?,
                    #[ver(G == G15)]
                    g15_stats_comp: alloc.private.new_default::<G15StatsComp>()?,
                    #[ver(G == G15)]
                    // Apple exposes these as named GPU/FW hardware mappings.
                    // Firmware proves 256 x 0x10 PB records and 256 x 0x20 UMA
                    // records. The GPU-shared allocator gives both GPU and FW
                    // read/write access, matching the observed ownership.
                    g15_pb_desc_table: alloc
                        .gpu
                        .array_empty::<raw::G15PBDescriptor>(0x100)?,
                    #[ver(G == G15)]
                    g15_uma_page_pool_desc_table: alloc
                        .gpu
                        .array_empty::<raw::G15UMAPagePoolDescriptor>(0x100)?,
                    #[ver(G == G15)]
                    // G15 HWDS-ID firmware state is exactly 256 x 8 bytes.
                    // Apple zeroes the full 0x800-byte mapping before boot;
                    // firmware reads/writes both u32 words in each entry.
                    g15_hwds_counters: alloc
                        .shared
                        .array_empty::<raw::G15HWDSCounterEntry>(0x100)?,

                    hwdata_a: hwa,
                    unkptr_190: alloc.private.array_empty_tagged(0x80, b"I190")?,
                    unkptr_198: alloc.private.array_empty_tagged(0xc0, b"I198")?,
                    hwdata_b: hwb,

                    unkptr_1b8: alloc.private.array_empty_tagged(0x1000, b"I1B8")?,
                    unkptr_1c0: alloc.private.array_empty_tagged(0x300, b"I1C0")?,
                    unkptr_1c8: alloc.private.array_empty_tagged(0x1000, b"I1C8")?,

                    buffer_mgr_ctl,
                    buffer_mgr_ctl_low_mapping: None,
                    buffer_mgr_ctl_high_mapping: None,
                })
            },
            |inner, _ptr| {
                try_init!(raw::RuntimePointers::ver {
                    #[ver(G != G15)]
                    pipes: Default::default(),
                    #[ver(G != G15)]
                    device_control: Default::default(),
                    #[ver(G != G15)]
                    event: Default::default(),
                    #[ver(G != G15)]
                    fw_log: Default::default(),
                    #[ver(G != G15)]
                    ktrace: Default::default(),
                    #[ver(G != G15)]
                    stats: Default::default(),

                    #[ver(G != G15)]
                    stats_vtx: inner.stats.vtx.gpu_pointer(),
                    #[ver(G != G15)]
                    stats_frag: inner.stats.frag.gpu_pointer(),
                    #[ver(G != G15)]
                    stats_comp: inner.stats.comp.gpu_pointer(),

                    #[ver(G != G15)]
                    hwdata_a: inner.hwdata_a.gpu_pointer(),
                    #[ver(G != G15)]
                    unkptr_190: inner.unkptr_190.gpu_pointer(),
                    #[ver(G != G15)]
                    unkptr_198: inner.unkptr_198.gpu_pointer(),
                    #[ver(G != G15)]
                    hwdata_b: inner.hwdata_b.gpu_pointer(),
                    #[ver(G != G15)]
                    hwdata_b_2: inner.hwdata_b.gpu_pointer(),

                    #[ver(G != G15)]
                    fwlog_buf: None,

                    #[ver(G != G15)]
                    unkptr_1b8: inner.unkptr_1b8.gpu_pointer(),

                    #[ver(G < G14X && G != G15)]
                    unkptr_1c0: inner.unkptr_1c0.gpu_pointer(),
                    #[ver(G < G14X && G != G15)]
                    unkptr_1c8: inner.unkptr_1c8.gpu_pointer(),

                    #[ver(G != G15)]
                    buffer_mgr_ctl_gpu_addr: U64(gpu::IOVA_KERN_GPU_BUFMGR_LOW),
                    #[ver(G != G15)]
                    buffer_mgr_ctl_fw_addr: U64(gpu::IOVA_KERN_GPU_BUFMGR_HIGH),

                    #[ver(G != G15)]
                    __pad0: Default::default(),
                    #[ver(G != G15)]
                    unk_160: U64(0),
                    #[ver(G != G15)]
                    unk_168: U64(0),
                    #[ver(G != G15)]
                    unk_1d0: 0,
                    #[ver(G != G15)]
                    unk_1d4: 0,
                    #[ver(G != G15)]
                    unk_1d8: Default::default(),

                    #[ver(G != G15)]
                    __pad1: Default::default(),
                    #[ver(G != G15)]
                    gpu_scratch: raw::RuntimeScratch::ver {
                        unk_6b38: 0xff,
                        ..Default::default()
                    },

                    // G15 compact wrapper. Only fields with exact semantics are
                    // populated here; opaque backing pointers stay zero until the
                    // matching Apple allocations are reconstructed.
                    #[ver(G == G15)]
                    hwdata_b: inner.hwdata_b.gpu_pointer(),
                    #[ver(G == G15)]
                    g15_fwbrn_table: U64(0),
                    #[ver(G == G15)]
                    g15_persistent_time: inner.g15_persistent_time.gpu_pointer(),
                    #[ver(G == G15)]
                    pipes: Default::default(),
                    #[ver(G == G15)]
                    device_control: Default::default(),
                    #[ver(G == G15)]
                    event: Default::default(),
                    #[ver(G == G15)]
                    fw_log: Default::default(),
                    #[ver(G == G15)]
                    ktrace: Default::default(),
                    #[ver(G == G15)]
                    stats: Default::default(),
                    #[ver(G == G15)]
                    fwlog_buf: None,
                    #[ver(G == G15)]
                    // allocateSharedData zero-fills the wrapper. Firmware owns
                    // these six-per-FWLog-channel sequences; both start at 0.
                    #[ver(G == G15)]
                    fwlog_payload_slot_seq_200: Default::default(),
                    #[ver(G == G15)]
                    fwlog_message_seq_218: Default::default(),
                    #[ver(G == G15)]
                    // Explicitly cleared again by initFirmwareSharedData().
                    fwlog_enabled_230: 0,
                    #[ver(G == G15)]
                    g15_ptr_234: U64(inner.g15_stats_vtx.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_ptr_23c: U64(inner.g15_stats_frag.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_ptr_244: U64(inner.g15_stats_comp.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_control_state: U64(inner.g15_control_state.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_pad_254: Default::default(),
                    #[ver(G == G15)]
                    g15_hwds_counters: U64(inner.g15_hwds_counters.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_pb_desc_addr: U64(inner.g15_pb_desc_table.gpu_va().get()),
                    #[ver(G == G15)]
                    // AGXArmFirmware::convertGPUVAToFWVA() is an identity
                    // function on this exact G15 host driver.
                    g15_pb_desc_fw_addr: U64(inner.g15_pb_desc_table.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_uma_page_pool_desc_addr: U64(
                        inner.g15_uma_page_pool_desc_table.gpu_va().get(),
                    ),
                    #[ver(G == G15)]
                    g15_uma_page_pool_desc_fw_addr: U64(
                        inner.g15_uma_page_pool_desc_table.gpu_va().get(),
                    ),
                    #[ver(G == G15)]
                    // All G15/G15G vtables resolve this source to
                    // AGXAcceleratorG15::halGetDefaultUscMaxTgmem() => 4.
                    g15_usc_max_tgmem: 4,
                    #[ver(G == G15)]
                    g15_zero_2d4: 0,
                    #[ver(G == G15)]
                    // Firmware owns +0x2d8..+0x3af and performs its own
                    // first-boot initialization (including +0x2fc = 1 and
                    // +0x314 = 0xabcdabcd). Keep the compile-only host image
                    // zeroed until any additional host-side initialization is
                    // directly proven.
                    g15_runtime_state: Default::default(),
                    #[ver(G == G15)]
                    g15_marker_3b0: 0xff,
                    #[ver(G == G15)]
                    g15_zero_3b1: Default::default(),
                    #[ver(G == G15)]
                    // Exact unaligned firmware pointer to the 0x4360 HwDataA
                    // allocation. Firmware dereferences this wrapper field
                    // extensively, including G15-only tail offsets.
                    g15_ptr_441: U64(inner.hwdata_a.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_zero_449: U64(0),
                    #[ver(G == G15)]
                    g15_zero_451: U64(0),
                    #[ver(G == G15)]
                    g15_tail_459: Default::default(),
                })
            },
        )
    }

    /// Create the FwStatus structure, which is used to coordinate the firmware halt state between
    /// the firmware and the driver.
    fn fw_status(&mut self) -> Result<GpuObject<FwStatus>> {
        self.alloc
            .shared
            .new_object(Default::default(), |_inner| Default::default())
    }

    /// Create one UatLevelInfo structure, which describes one level of translation for the UAT MMU.
    fn uat_level_info(
        cfg: &'static hw::HwConfig,
        index_shift: usize,
        num_entries: usize,
    ) -> raw::UatLevelInfo {
        raw::UatLevelInfo {
            index_shift: index_shift as _,
            unk_1: 14,
            unk_2: 14,
            unk_3: 8,
            unk_4: 0x4000,
            num_entries: num_entries as _,
            unk_8: U64(1),
            unk_10: U64(((1u64 << cfg.uat_oas) - 1) & !(mmu::UAT_PGMSK as u64)),
            index_mask: U64(((num_entries - 1) << index_shift) as u64),
        }
    }

    /// Build the top-level InitData object.
    #[inline(never)]
    pub(crate) fn build(&mut self) -> Result<KBox<GpuObject<InitData::ver>>> {
        let runtime_pointers = self.runtime_pointers()?;
        #[ver(G != G15)]
        let globals = self.globals()?;
        #[ver(G != G15)]
        let fw_status = self.fw_status()?;
        #[ver(G != G15)]
        let unk_buf = self.alloc.shared_ro.array_empty_tagged(0x4000, b"IDTA")?;
        #[ver(G == G15)]
        let g15_init_sequence = self.alloc.shared_ro.new_object(
            Default::default(),
            |_inner| raw::G15InitSequencePage {
                // Exact G15/G15G empty AGFA sequence. The firmware parser
                // starts with this record and exits immediately on kind == 0.
                terminator: raw::G15InitSequenceEntry {
                    value: U64(0),
                    register_offset: U32(0),
                    shift: U32(0),
                    kind: U32(0),
                    reserved: U32(0),
                },
                unused: Default::default(),
            },
        )?;
        #[ver(G == G15)]
        let g15_globals = self.alloc.private.new_object(
            Default::default(),
            |_inner| raw::G15Q4Config {
                // Exact G15G/J615 host values proven from the paired Apple
                // AGXG15G driver. These are established before firmware sees
                // q4; unresolved accelerator-derived fields remain zero.
                // Accelerator::start zeroes +0x2970/+0x2974 and its only
                // population path is gated by config bit 34. Base configure
                // clears bit 34 and G15 never restores it, so both are exact 0.
                // q4 +0x000 is 7 only when accelerator config bit 28 is set.
                // Exact J615 keeps bit 28 clear throughout setup.
                flags_000: U32(0),
                // Base/Arm initFirmwareData explicitly clear these prefix
                // dwords. +0x034 has no non-zero host writer and remains
                // zero on either branch from the zero-filled allocation.
                zero_004: U32(0),
                zero_008: U32(0),
                zero_00c: U32(0),
                zero_010: U32(0),
                zero_014: U32(0),
                zero_018: U32(0),
                unk_01c: U32(0),
                // Apple pairs getDefaultRelaxedCLContextSwitchTimeout() with
                // setRelaxedCLKillTimeout(); ChinookV9's default is 3000.
                relaxed_cl_kill_timeout_020: U32(0x0bb8),
                frg_task_timeout_024: U32(0),
                // Base configure zeroes the complete accelerator
                // +0x65c..+0x663 qword. Apple later copies +0x660 directly
                // into q4 +0x028; G15/G15G configure only rewrites the packed
                // +0x650..+0x656 feature word, so exact J615 remains zero.
                accelerator_config_660_028: U32(0),
                // Base AGXAccelerator::configureDevice() forces config bit 0
                // on before firmware data construction. AGXFirmware copies
                // that bit directly to q4 +0x02c.
                accelerator_config_bit0_02c: U32(1),
                // q4 +0x030 is accelerator config bit 1 / Smart Idle Off.
                // The exact AGXG15G personality table has three T8122
                // matches (G15G, G15G_A0, G15G_B0), and every matching
                // subclass returns 1 from halIsSmartidleOffEnabled().
                smart_idle_off_enabled_030: U32(1),
                zero_034: U32(0),
                constant_038: U32(0x78),
                // Apple explicitly starts the CPMS/QoS control words at
                // zero in AGXArmFirmware::initFirmwareData. +0x044 has
                // no host population path and remains allocation-zeroed.
                cpms_window_size_03c: U32(0),
                cpms_tfca_size_040: U32(0),
                zero_044: U32(0),
                kick_channel_qos_arg2_048: U32(0),
                kick_channel_qos_arg1_04c: U32(0),
                unk_050: 0xffff,
                unk_052: 0x0028,
                unk_054: 0xffff,
                // Exact host clear immediately after the 0x50/0x52/0x54 tuple.
                zero_056: U32(0),
                // G15 constructor/start establishes accelerator+0x9d08 low two
                // bits as zero; later writers touch bits 3/5 only. Apple stores
                // ((byte & 3) == 0) here, so the exact J615 value is 1.
                // Apple seeds power interface 0 from accelerator +0x2288.
                // Base configure uses 0xfe6 (4070), and exact J615 has no
                // `gpu-max-power` override, so q4 +0x078 receives 4070.
                // q4 +0x074 is never populated by the G15 host and has only
                // firmware readers before bootstrap; preserve the zeroed allocation.
                zero_074: U32(0),
                gpu_max_power_078: U32(4070),
                // setupConfig enables interfaces 1/2 with a full-scale 0x10000
                // target. With J615's two perf states, max pstate index 1 and
                // gpu-pwr-min-duty-cycle=30, normalization yields 100 for both.
                power_interface_1_target_07c: U32(100),
                power_interface_2_target_080: U32(100),
                low2_clear_05a: U32(1),
                // On exact J615, the `model-slow` probe property is absent.
                // Apple therefore writes 1 to HwDataB +0xa6c, and the arm
                // firmware setup copies/defaults that value into q4 +0x05e.
                // The same field is the target of setCSwitchTimerMultiplier().
                cswitch_timer_multiplier_05e: U32(1),
                // q4 +0x062/+0x070 are runtime-mutable controls, but the
                // exact G15 bootstrap path never invokes their setters. The
                // zero-filled q4 allocation therefore reaches firmware as 0.
                cdm_cswitch_mode_change_062: U32(0),
                command_submission_enabled_070: AtomicU32::new(0),
                // AGXAccelerator::start() builds the GPU PerfStateInfo from
                // exact J615 `gpu-num-perf-states = 2`. G15 constructor byte
                // +0x4e1 is zero, so Apple takes the direct 0x448-byte copy
                // to accelerator +0xa1e0 rather than the MGPU combiner.
                // getPerfStateCap(domain 0) returns count - 1, and the Arm
                // initializer stores that cap multiplied by 100 at q4+0x8c.
                perf_state_cap_x100_08c: U32(100),
                // setupConfig explicitly clears the firmware-object backing
                // words at +0x17a0/+0x17a4/+0x17a8/+0x17ac/+0x17b0 and
                // the +0x17b4 validity byte. initPowerAndPerformanceData()
                // copies those bootstrap values to q4 +0x90..+0xa4 before
                // any performance-controller runtime update. The J615 ADT
                // target/boost values are configuration inputs, not these
                // initial q4 words.
                performance_target_090: U32(0),
                performance_transfer_output_094: U32(0),
                performance_boost_min_util_098: U32(0),
                performance_boost_ce_step_09c: U32(0),
                performance_reset_iters_0a0: U32(0),
                performance_boost_min_util_valid_0a4: 0,
                // Firmware caches q4 +0x26f and uses it to gate an extra
                // conflict/resource scan. The exact J615 host never writes
                // this byte, so it remains zero from the allocation.
                conflict_scan_enable_26f: 0,
                // AGXAccelerator::configurePowerAndPerformanceController()
                // establishes the Smart Idle defaults before configureDevice
                // probes its optional gpu-idleoff-* overrides. Exact J615
                // omits all ten properties, so initFirmwareData copies these
                // unchanged defaults into q4 +0x7a8..+0x7cc.
                smart_idle_standby_timer_us_7a8: U32(700),
                smart_idle_prob_init_val_7ac: f32!(1.0),
                smart_idle_fn_hit_7b0: f32!(0.8),
                smart_idle_fi_hit_7b4: f32!(0.2),
                smart_idle_fn_miss_7b8: f32!(0.9),
                smart_idle_fi_miss_7bc: f32!(0.1),
                smart_idle_nei_hit_7c0: f32!(0.25),
                smart_idle_min_confidence_7c4: f32!(0.7),
                smart_idle_high_confidence_7c8: f32!(0.9),
                smart_idle_reset_iterations_7cc: U32(6),
                // initPowerAndPerformanceData() enables UT and CLVR by
                // default. It also copies the two keepalive thresholds from
                // accelerator +0x1e90/+0x1e8c. configureDevice establishes
                // both as 100 before optional property overrides; exact J615
                // omits both gpu-keepalive-* threshold properties.
                ut_engagement_enabled_7d0: U32(1),
                clvr_engagement_enabled_7d4: U32(1),
                gpu_keepalive_perf_mode_threshold_7d8: U32(100),
                gpu_keepalive_off_mode_threshold_7dc: U32(100),
                // AGXArmFirmware::init() explicitly clears the entire q4
                // register-patch staging span through +0x96b. Firmware later
                // reads +0x964 as the record count for entries at +0x7f4;
                // exact J615 bootstrap therefore takes the zero-count path.
                boot_patch_record_count_964: U32(0),
                boot_patch_count_zero_968: U32(0),
                // These are runtime inputs from the IOGPU
                // set_display_params_for_gpu user-client method. Apple starts
                // the zeroed q4 block with no display override; firmware falls
                // back to its live timestamp/interval state while both are 0.
                // The interval, once supplied, is clamped by the host to
                // 100000..400000 ticks in the firmware 24 MHz timebase.
                display_pm_timestamp_96c: U64(0),
                display_pm_interval_974: U64(0),
                // AGXArmFirmware::setupConfig copies the four values below
                // from accelerator +0x1e48..+0x1e60. AGXAccelerator's static
                // initialization proves their exact J615 values.
                // configureDevice establishes the exact seven-word source
                // record at accelerator +0x1e48..+0x1e60 and setupConfig
                // copies it to q4 +0x97c..+0x994. Direct setters prove the
                // first three words are the 3D/TA/CL progress-check intervals.
                progress_check_interval_3d_97c: U32(40),
                progress_check_interval_ta_980: U32(10),
                progress_check_interval_cl_984: U32(250),
                unk_988: U32(1),
                unk_98c: U32(1),
                unk_990: U32(100),
                unk_994: U32(1),
                // AGXFirmware::init() invokes the active ChinookV9
                // setupConfig() before firmware data initialization. That
                // establishes 2/40/5 as the three delay defaults and clears
                // their override state. Exact J615 omits the corresponding
                // gpu-*-delay/early-wake properties, so
                // initPowerAndPerformanceData() selects those defaults and
                // writes them directly to q4 +0x9a0/+0x9a4/+0x9a8.
                gpu_idle_off_delay_ms_9a0: U32(2),
                fender_idle_off_delay_ms_9a4: U32(40),
                fw_early_wake_timeout_ms_9a8: U32(5),
                // Exact ChinookV9 defaults, applied by paired getter/setter
                // calls in AGXArmFirmware::initFirmwareData.
                cl_context_switch_timeout_9b0: U32(0x28),
                cl_kill_timeout_9b4: U32(0x32),
                // initFirmwareData explicitly clears the GVDM timer field;
                // setGVDMTimerInterval() targets the same q4 +0x9ac word.
                gvdm_timer_interval_9ac: U32(0),
                cdm_backoff_timeout_9bc: 0x04,
                // Exact host bootstrap clears/retains these as zero. The
                // table selector is explicitly zeroed before firmware can
                // take the optional +0x9e5 table-copy path.
                zero_9b8: U32(0),
                table_selector_9dd: U64(0),
                zero_de5: U32(0),
                // Keepalive override setters are runtime-only. Zero means
                // no override; threshold getters fall back to +0x7d8/+0x7dc.
                gpu_keepalive_override_de9: U32(0),
                gfxc_keepalive_override_ded: U32(0),
                gpu_keepalive_perf_mode_threshold_override_df1: U32(0),
                gpu_keepalive_off_mode_threshold_override_df5: U32(0),
                // initFirmwareData calls ChinookV9 initSoftFaultSettings(true).
                // Its updater encodes enabled as bit0=1, bit1=0, bit2=0 and
                // masks the word to three bits, yielding exact q4 +0xdf9 = 1.
                soft_fault_settings_df9: U32(1),
                ..Default::default()
            },
        )?;
        #[ver(G == G15)]
        let g15_q21 = self.alloc.shared.new_object(
            Default::default(),
            |_inner| raw::G15SharedStatus {
                host_flags: 0,
                banner_guard: AtomicU32::new(1),
                unk_08: 0,
                busy: AtomicU32::new(0),
                unk_10: 0,
                firmware_ready: AtomicU32::new(0),
                power_state: AtomicU32::new(0),
                unk_1c: 0,
            },
        )?;
        #[ver(G == G15)]
        let mut g15_q22_alloc = alloc::G15SharedBank1Allocator::new(
            self.dev,
            self.g15_shared_bank1.clone().ok_or(EINVAL)?,
            mmu::UAT_PGSZ,
            mmu::PROT_G15_RANGE7_FW,
            true,
            None,
        );
        #[ver(G == G15)]
        let g15_mapping_ring_backing =
            g15_q22_alloc.new_default::<G15MappingRingBacking>()?;
        #[ver(G == G15)]
        let g15_mapping_notifier = mmu::G15MappingNotifierHandle::new(Arc::pin_init(
            new_mutex!(
                mmu::G15MappingNotifier::new(
                    self.dev,
                    g15_mapping_ring_backing,
                ),
                "g15_mapping_notifier"
            ),
            GFP_KERNEL,
        )?);
        #[ver(G == G15)]
        let (g15_cache_flush_state_va, g15_cache_flush_ring_va) = {
            let notifier = g15_mapping_notifier.lock();
            (notifier.state_gpu_va(), notifier.ring_gpu_va())
        };
        #[ver(G == G15)]
        let g15_q22 = self.alloc.shared.new_object(
            Default::default(),
            |_inner| raw::G15Q22Shared {
                // Apple allocFirmwareData() calls allocateSharedData(..., true,
                // false) with eGartRange=7, zero extra options, and exact 0x20 /
                // 0x1800 suballocations. That is option word 0x700000007, so
                // these bootstrap objects use the same shared bank-1 PTE class
                // as later range-7 PM resources. The q22 producer is not yet
                // attached while these two self-hosting mappings are created.
                shared_ptr_4568: U64(g15_cache_flush_state_va),
                shared_ptr_4570: U64(g15_cache_flush_ring_va),
                // Apple sets +0x45c4 to one before firmware starts. G15's
                // accelerator configure path clears feature bit 28 before the
                // const smart-idle query, so +0xc3cc is exactly zero on J615.
                host_flag_45c4: U32(1),
                // initFirmwareData starts this at zero; setKickChannelQos()
                // flips it to one before updating q4 +0x04c/+0x048.
                kick_channel_qos_valid_c3c8: U32(0),
                feature_c3cc: U32(0),
                ..Default::default()
            },
        )?;
        #[ver(G == G15)]
        let g15_q23 = self.alloc.shared.new_object(
            Default::default(),
            |_inner| raw::G15Q23Shared {
                // Apple explicitly clears these host-visible update fields;
                // all unresolved/runtime-owned fields retain their zeroed base.
                tuning_update_1d0: 0,
                host_zero_1e8: U32(0),
                ..Default::default()
            },
        )?;

        #[ver(G == G15)]
        {
            // q4/q21/q22/q23 are now exact-size typed G15 root objects.
            // Fields whose accelerator/ADT source is not yet reconstructed
            // deliberately remain zero; runtime G15 dispatch is still off.
        }
        let cfg = self.cfg;

        // 16 KiB UAT pages use 11 index bits at the lower levels. G15 keeps
        // the same three-level layout but widens the root (shift 36) from
        // 8 entries (39-bit IAS) to 64 entries (42-bit IAS).
        let root_bits = self.cfg.uat_ias.checked_sub(36).ok_or(EINVAL)?;
        if root_bits > 11 {
            return Err(EINVAL);
        }
        let root_entries = 1usize << root_bits;

        let obj = self.alloc.private.new_init(
            try_init!(InitData::ver {
                #[ver(G != G15)]
                unk_buf,
                runtime_pointers,
                #[ver(G != G15)]
                globals,
                #[ver(G != G15)]
                fw_status,
                #[ver(G == G15)]
                g15_init_sequence,
                #[ver(G == G15)]
                g15_globals,
                #[ver(G == G15)]
                g15_q21,
                #[ver(G == G15)]
                g15_mapping_notifier,
                #[ver(G == G15)]
                g15_q22,
                #[ver(G == G15)]
                g15_q23,
            }),
            |inner, _ptr| {
                try_init!(raw::InitData::ver {
                    #[ver(V == V13_5 && G != G14X && G != G15)]
                    ver_info: Array::new([0x6ba0, 0x1f28, 0x601, 0xb0]),
                    #[ver(V == V13_5 && G == G14X)]
                    ver_info: Array::new([0xb390, 0x70f8, 0x601, 0xb0]),
                    #[ver(G != G15)]
                    unk_buf: inner.unk_buf.gpu_pointer(),
                    #[ver(G != G15)]
                    unk_8: 0,
                    #[ver(G != G15)]
                    unk_c: 0,
                    #[ver(G != G15)]
                    runtime_pointers: inner.runtime_pointers.gpu_pointer(),
                    #[ver(G != G15)]
                    globals: inner.globals.gpu_pointer(),
                    #[ver(G != G15)]
                    fw_status: inner.fw_status.gpu_pointer(),
                    #[ver(G != G15)]
                    uat_page_size: 0x4000,
                    #[ver(G != G15)]
                    uat_page_bits: 14,
                    #[ver(G != G15)]
                    uat_num_levels: 3,
                    #[ver(G != G15)]
                    uat_level_info: Array::new([
                        Self::uat_level_info(&cfg, 36, root_entries),
                        Self::uat_level_info(&cfg, 25, 2048),
                        Self::uat_level_info(&cfg, 14, 2048),
                    ]),
                    #[ver(G != G15)]
                    __pad0: Default::default(),
                    #[ver(G != G15)]
                    host_mapped_fw_allocations: 1,
                    #[ver(G != G15)]
                    unk_ac: 0,
                    #[ver(G != G15)]
                    unk_b0: 0,
                    #[ver(G != G15)]
                    unk_b4: 0,
                    #[ver(G != G15)]
                    unk_b8: 0,

                    // G15 top-level root reconstructed from
                    // AGXArmFirmware::initFirmwareData and RTKit-2419.140.12.
                    #[ver(G == G15)]
                    g15_q0_signature: U64(0x0c08_e21e_8380_0490),
                    #[ver(G == G15)]
                    g15_q1_init_sequence: U64(inner.g15_init_sequence.gpu_va().get()),
                    // Exact: Apple's root allocation is zeroed and the G15
                    // host constructor never writes q2.
                    #[ver(G == G15)]
                    g15_q2: U64(0),
                    #[ver(G == G15)]
                    g15_q3_runtime_pointers: U64(inner.runtime_pointers.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_q4_globals: U64(inner.g15_globals.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_q5_host_mapped: U64(0x0000_0001_0000_0000),
                    // q6..q20 are the compact UAT description. Apple writes
                    // q6..q18 only; q19/q20 are the final 0x10 bytes of this
                    // block's zero pad and stay zero from the zeroed root allocation.
                    // G15's Apple ADT advertises a 43-bit virtual address size including the
                    // TTBR selector, i.e. a 42-bit per-TTBR input width. That
                    // widens the shift-36 root from 8 to 64 entries; lower levels
                    // remain the established 25/14 shifts with 2048 entries.
                    #[ver(G == G15)]
                    g15_q6_q20_uat: raw::G15UatConfig {
                        page_size: 0x4000,
                        page_bits: 14,
                        num_levels: 3,
                        level_info: Array::new([
                            Self::uat_level_info(&cfg, 36, root_entries),
                            Self::uat_level_info(&cfg, 25, 2048),
                            Self::uat_level_info(&cfg, 14, 2048),
                        ]),
                        __pad0: Default::default(),
                    },
                    #[ver(G == G15)]
                    g15_q21: U64(inner.g15_q21.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_q22: U64(inner.g15_q22.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_q23: U64(inner.g15_q23.gpu_va().get()),
                    #[ver(G == G15)]
                    g15_phantom: PhantomData,
                })
            },
        )?;
        Ok(KBox::new(obj, GFP_KERNEL)?)
    }
}
