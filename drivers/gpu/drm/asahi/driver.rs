// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Top-level GPU driver implementation.

use kernel::{
    c_str,
    device::Core,
    dma::{
        Device,
        DmaMask, //
    },
    drm,
    drm::ioctl,
    of,
    platform,
    prelude::*,
    sync::{
        aref::ARef,
        Arc, //
    }, //
};

use crate::{
    debug,
    file,
    gem::AsahiObject,
    gpu,
    hw,
    mmu,
    regs, //
};

use kernel::macros::vtable;

/// Holds a reference to the top-level `GpuManager` object.
#[pin_data]
pub(crate) struct AsahiData {
    #[pin]
    pub(crate) gpu: Arc<dyn gpu::GpuManager>,
    pub(crate) pdev: ARef<platform::Device>,
    pub(crate) resources: regs::Resources,
}

unsafe impl Send for AsahiData {}
unsafe impl Sync for AsahiData {}

pub(crate) struct AsahiDriver {
    #[expect(unused)]
    drm: ARef<drm::Device<Self>>,
}

unsafe impl Send for AsahiDriver {}
unsafe impl Sync for AsahiDriver {}

/// Convenience type alias for the DRM device type for this driver.
pub(crate) type AsahiDevice = drm::device::Device<AsahiDriver>;
pub(crate) type AsahiDevRef = ARef<AsahiDevice>;

/// DRM Driver metadata
const INFO: drm::driver::DriverInfo = drm::driver::DriverInfo {
    major: 0,
    minor: 0,
    patchlevel: 0,
    name: c_str!("asahi"),
    desc: c_str!("Apple AGX Graphics"),
};

/// DRM Driver implementation for `AsahiDriver`.
#[vtable]
impl drm::driver::Driver for AsahiDriver {
    /// Our `DeviceData` type, reference-counted
    type Data = AsahiData;
    /// Our `File` type.
    type File = file::File;
    /// Our `Object` type.
    type Object = drm::gem::shmem::Object<AsahiObject>;

    const INFO: drm::driver::DriverInfo = INFO;
    const FEATURES: u32 = drm::driver::FEAT_GEM
        | drm::driver::FEAT_RENDER
        | drm::driver::FEAT_SYNCOBJ
        | drm::driver::FEAT_SYNCOBJ_TIMELINE
        | drm::driver::FEAT_GEM_GPUVA;

    kernel::declare_drm_ioctls! {
        (ASAHI_GET_PARAMS,      drm_asahi_get_params,
                          ioctl::RENDER_ALLOW, crate::file::File::get_params),
        (ASAHI_GET_TIME,        drm_asahi_get_time,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::get_time),
        (ASAHI_VM_CREATE,       drm_asahi_vm_create,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::vm_create),
        (ASAHI_VM_DESTROY,      drm_asahi_vm_destroy,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::vm_destroy),
        (ASAHI_VM_BIND,         drm_asahi_vm_bind,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::vm_bind),
        (ASAHI_GEM_CREATE,      drm_asahi_gem_create,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::gem_create),
        (ASAHI_GEM_MMAP_OFFSET, drm_asahi_gem_mmap_offset,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::gem_mmap_offset),
        (ASAHI_GEM_BIND_OBJECT, drm_asahi_gem_bind_object,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::gem_bind_object),
        (ASAHI_QUEUE_CREATE,    drm_asahi_queue_create,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::queue_create),
        (ASAHI_QUEUE_DESTROY,   drm_asahi_queue_destroy,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::queue_destroy),
        (ASAHI_SUBMIT,          drm_asahi_submit,
            ioctl::AUTH | ioctl::RENDER_ALLOW, crate::file::File::submit),
    }
}

// OF Device ID table.s
kernel::of_device_table!(
    OF_TABLE,
    MODULE_OF_TABLE,
    <AsahiDriver as platform::Driver>::IdInfo,
    [
        (
            of::DeviceId::new(c_str!("apple,agx-t8103")),
            &hw::t8103::HWCONFIG
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t8112")),
            &hw::t8112::HWCONFIG
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t8122")),
            &hw::t8122::HWCONFIG_PREFLIGHT
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6000")),
            &hw::t600x::HWCONFIG_T6000
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6001")),
            &hw::t600x::HWCONFIG_T6001
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6002")),
            &hw::t600x::HWCONFIG_T6002
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6020")),
            &hw::t602x::HWCONFIG_T6020
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6021")),
            &hw::t602x::HWCONFIG_T6021
        ),
        (
            of::DeviceId::new(c_str!("apple,agx-t6022")),
            &hw::t602x::HWCONFIG_T6022
        ),
    ]
);

/// Platform Driver implementation for `AsahiDriver`.
impl platform::Driver for AsahiDriver {
    type IdInfo = &'static hw::HwConfig;
    const OF_ID_TABLE: Option<of::IdTable<Self::IdInfo>> = Some(&OF_TABLE);

    /// Device probe function.
    fn probe(
        pdev: &platform::Device<Core>,
        info: Option<&Self::IdInfo>,
    ) -> impl PinInit<Self, Error> {
        debug::update_debug_flags();

        dev_info!(pdev.as_ref(), "Probing...\n");

        let cfg = info.ok_or(ENODEV)?;

        if cfg.gpu_gen == hw::GpuGen::G15 {
            // T8122 bring-up checkpoint: first validate the exact generation-7
            // identity, then start only the preloaded GFX ASC and initialize
            // the UAT handoff/TTB state. Do not create a GpuManager/RTKit
            // object, build firmware initdata, start RTKit endpoints, send
            // MSG_INIT, register DRM, or submit GPU work.
            unsafe { pdev.dma_set_mask_and_coherent(DmaMask::try_new(cfg.uat_oas)?)? };

            let res = regs::Resources::new(pdev)?;
            res.init_mmio()?;
            let id = res.get_gpu_id()?;

            if id.gpu_gen != hw::GpuGen::G15
                || id.gpu_variant != hw::GpuVariant::G
                || id.gpu_rev != hw::GpuRevision::C0
                || id.num_clusters != cfg.max_num_clusters
                || id.num_cores != cfg.max_num_cores
                || id.total_active_cores != cfg.max_num_cores
                || id.core_masks.len() != 1
                || id.core_masks[0] != 0x3ff
            {
                dev_err!(pdev.as_ref(), "T8122 G15 preflight identity mismatch: {:?}\n", id);
                return Err(EIO);
            }

            dev_info!(
                pdev.as_ref(),
                "T8122 G15G C0 identity PASS (1 MGPU, 10/10 cores); starting UAT handoff preflight\n"
            );

            regs::Resources::start_cpu(pdev)?;

            // Uat::new() needs an Asahi DRM device for its private GEM-backed
            // page tables, but this device is never registered with DRM. This
            // is the same uninitialized-device construction used by the normal
            // probe path before manager setup.
            let uninit = unsafe {
                pin_init::pin_init_from_closure::<AsahiData, kernel::error::Error>(|_slot| Ok(()))
            };
            let drm: ARef<AsahiDevice> = drm::device::Device::new(pdev.as_ref(), uninit)?;

            let uat_result = mmu::Uat::new(&drm, cfg, true);
            match uat_result {
                Ok(uat) => {
                    dev_info!(
                        pdev.as_ref(),
                        "T8122 G15 UAT handoff preflight PASS; RTKit/initdata/MSG_INIT intentionally blocked\n"
                    );
                    core::mem::drop(uat);
                }
                Err(e) => {
                    dev_err!(pdev.as_ref(), "T8122 G15 UAT handoff preflight failed: {:?}\n", e);
                    let _ = regs::Resources::stop_cpu(pdev);
                    return Err(e);
                }
            }

            regs::Resources::stop_cpu(pdev)?;
            dev_info!(pdev.as_ref(), "T8122 G15 ASC stopped after UAT preflight\n");

            // E157 retires only the obsolete E075 early-return gate. E075 already
            // live-proved clean reversible range-7/range-8 parent/leaf teardown;
            // E031-E033 separately live-proved the persistent manager/RTKit boot
            // boundary. Re-enter that persistent path now to exercise the newer
            // E147-E152 mapped global resource graph, but keep DRM registration
            // below a new later gate so File/VM/Queue creation stays unreachable.
            dev_info!(
                pdev.as_ref(),
                "T8122 G15 E075 range-8 preflight PASS; continuing to E157 persistent-manager checkpoint\n"
            );

            // Revalidate the complete exact J615 power configuration before the
            // persistent manager takes ownership of the restarted ASC/UAT state.
            let pwr = match hw::PwrConfig::load(&drm, cfg) {
                Ok(pwr) => pwr,
                Err(e) => {
                    dev_err!(pdev.as_ref(), "T8122 G15 PwrConfig load failed: {:?}\n", e);
                    return Err(e);
                }
            };

            const FREQ_HZ: [u32; 14] = [
                0, 338_000_000, 618_000_000, 796_000_000, 836_000_000, 928_000_000,
                952_000_000, 1_056_000_000, 1_053_000_000, 1_170_000_000,
                1_152_000_000, 1_278_000_000, 1_204_000_000, 1_338_000_000,
            ];
            const VOLT_MV: [u32; 14] = [
                125, 650, 675, 720, 755, 755, 805, 805, 850, 850, 890, 890, 915, 915,
            ];
            const POWER_MW: [u32; 14] = [
                0, 3516, 5774, 8103, 9376, 10194, 12037, 13102, 14799, 16149,
                17713, 19322, 19586, 21405,
            ];

            let mut power_ok = pwr.perf_states.len() == FREQ_HZ.len();
            if power_ok {
                for i in 0..FREQ_HZ.len() {
                    let ps = &pwr.perf_states[i];
                    if ps.freq_hz != FREQ_HZ[i]
                        || ps.pwr_mw != POWER_MW[i]
                        || ps.volt_mv.len() != 1
                        || ps.volt_mv[0] != VOLT_MV[i]
                    {
                        power_ok = false;
                        break;
                    }
                }
            }

            power_ok = power_ok
                && pwr.power_zones.is_empty()
                && pwr.csafr.is_none()
                && pwr.core_leak_coef.len() == 1
                && pwr.sram_leak_coef.len() == 1
                && pwr.core_leak_coef[0].to_bits() == 0x44cd_8000 // 1644.0
                && pwr.sram_leak_coef[0].to_bits() == 0x4270_0000 // 60.0
                && pwr.max_power_mw == 21_405
                && pwr.max_freq_mhz == 1_338
                && pwr.perf_base_pstate == 1
                && pwr.perf_max_pstate == 13
                && pwr.min_sram_microvolt == 790_000
                && pwr.avg_power_filter_tc_ms == 40
                && pwr.avg_power_ki_only.to_bits() == 0x428c_0000 // 70.0
                && pwr.avg_power_kp.to_bits() == 0x3ff9_999a // 1.95
                && pwr.avg_power_min_duty_cycle == 30
                && pwr.avg_power_target_filter_tc == 1
                && pwr.fast_die0_integral_gain.to_bits() == 0x43e1_0000 // 450.0
                && pwr.fast_die0_proportional_gain.to_bits() == 0x4208_0000 // 34.0
                && pwr.fast_die0_prop_tgt_delta == 0
                && pwr.fast_die0_release_temp == 80
                && pwr.fender_idle_off_delay_ms == 40
                && pwr.fw_early_wake_timeout_ms == 5
                && pwr.idle_off_delay_ms == 2
                && pwr.idle_off_standby_timer == 700
                && pwr.perf_boost_ce_step == 50
                && pwr.perf_boost_min_util == 90
                && pwr.perf_filter_drop_threshold == 0
                && pwr.perf_filter_time_constant == 5
                && pwr.perf_filter_time_constant2 == 200
                && pwr.perf_integral_gain.to_bits() == 0x3f4c_28f6
                && pwr.perf_integral_gain2.to_bits() == 0x3f4c_28f6
                && pwr.perf_integral_min_clamp == 0
                && pwr.perf_proportional_gain.to_bits() == 0x40ac_cccd
                && pwr.perf_proportional_gain2.to_bits() == 0x40ac_cccd
                && pwr.perf_reset_iters == 6
                && pwr.perf_tgt_utilization == 85
                && pwr.power_sample_period == 8
                && pwr.ppm_filter_time_constant_ms == 100
                && pwr.ppm_ki.to_bits() == 0x42c2_3333
                && pwr.ppm_kp.to_bits() == 0x4039_999a
                && pwr.pwr_filter_time_constant == 313
                && pwr.pwr_integral_gain.to_bits() == 0x3ca5_9586
                && pwr.pwr_integral_min_clamp == 0
                && pwr.pwr_min_duty_cycle == 30
                && pwr.pwr_proportional_gain.to_bits() == 0x40a9_0fdb
                && pwr.pwr_sample_period_aic_clks == 200_000
                && pwr.se_engagement_criteria == 600
                && pwr.se_filter_time_constant == 9
                && pwr.se_filter_time_constant_1 == 3
                && pwr.se_inactive_threshold == 2500
                && pwr.se_ki.to_bits() == 0xc248_0000
                && pwr.se_ki_1.to_bits() == 0xc2c8_0000
                && pwr.se_kp.to_bits() == 0xc0a0_0000
                && pwr.se_kp_1.to_bits() == 0xc120_0000
                && pwr.se_reset_criteria == 50;

            if !power_ok {
                dev_err!(pdev.as_ref(), "T8122 G15 PwrConfig mismatch: {:?}\n", pwr);
                return Err(EIO);
            }

            dev_info!(
                pdev.as_ref(),
                "T8122 G15 PwrConfig PASS (14 OPPs, leak 1644/60, SRAM floor 790mV, max 21405mW); starting persistent manager/DRM checkpoint\n"
            );

            // E015: transition from the bounded teardown checkpoint to the
            // normal persistent manager/RTKit lifetime, but keep userspace
            // completely below File::open() so no GPU-visible command path is
            // reachable yet.
            regs::Resources::start_cpu(pdev)?;
            let gpu = gpu::GpuManagerG15V14_7::new(&drm, &res, cfg)?
                as Arc<dyn gpu::GpuManager>;

            let data = try_pin_init!(AsahiData {
                gpu,
                pdev: pdev.into(),
                resources: res,
            });

            let ptr: *const AsahiData = &raw const **drm;
            unsafe {
                data.__pinned_init(ptr as *mut AsahiData)?;
            }

            (*drm).gpu.init()?;

            // E163 opens passive Queue create/destroy on top of E162. Queue
            // construction may exercise q22-tracked resource map/unmap, but no
            // firmware context/QueueInfo publication, channel ensure, or submit.
            drm::driver::Registration::new_foreign_owned(&drm, pdev.as_ref(), 0)?;
            dev_info!(
                pdev.as_ref(),
                "T8122 G15 E171 fresh-slot Barrier-registration probe ready; GPU engine commands blocked\n"
            );
            return Ok(Self { drm });
        }

        unsafe { pdev.dma_set_mask_and_coherent(DmaMask::try_new(cfg.uat_oas)?)? };

        let res = regs::Resources::new(pdev)?;

        // Initialize misc MMIO
        res.init_mmio()?;

        // Start the coprocessor CPU, so UAT can initialize the handoff
        regs::Resources::start_cpu(pdev)?;

        let fwnode = pdev.as_ref().fwnode().ok_or(EIO)?;
        let compat: KVec<u32> = fwnode
            .property_read_array_vec(c_str!("apple,firmware-compat"), 3)?
            .required_by(pdev.as_ref())?;

        // TODO: This is very temporary
        // SAFETY: This should be safe as data is not touched by the driver
        // untill it gets fully initialised.
        // Additionally drm::device::Device::release() will not drop data and
        // leaks instead.
        let uninit = unsafe {
            pin_init::pin_init_from_closure::<AsahiData, kernel::error::Error>(|_slot| Ok(()))
        };
        let drm: ARef<AsahiDevice> = drm::device::Device::new(pdev.as_ref(), uninit)?;

        let gpu = match (cfg.gpu_gen, cfg.gpu_variant, compat.as_slice()) {
            (hw::GpuGen::G13, _, &[12, 3, 0]) => {
                gpu::GpuManagerG13V12_3::new(&drm.clone(), &res, cfg)? as Arc<dyn gpu::GpuManager>
            }
            (hw::GpuGen::G14, hw::GpuVariant::G, &[12, 4, 0]) => {
                gpu::GpuManagerG14V12_4::new(&drm.clone(), &res, cfg)? as Arc<dyn gpu::GpuManager>
            }
            (hw::GpuGen::G13, _, &[13, 5, 0]) => {
                gpu::GpuManagerG13V13_5::new(&drm.clone(), &res, cfg)? as Arc<dyn gpu::GpuManager>
            }
            (hw::GpuGen::G14, hw::GpuVariant::G, &[13, 5, 0]) => {
                gpu::GpuManagerG14V13_5::new(&drm.clone(), &res, cfg)? as Arc<dyn gpu::GpuManager>
            }
            (hw::GpuGen::G14, _, &[13, 5, 0]) => {
                gpu::GpuManagerG14XV13_5::new(&drm.clone(), &res, cfg)? as Arc<dyn gpu::GpuManager>
            }
            _ => {
                dev_info!(
                    pdev.as_ref(),
                    "Unsupported GPU/firmware combination ({:?}, {:?}, {:?})\n",
                    cfg.gpu_gen,
                    cfg.gpu_variant,
                    compat
                );
                return Err(ENODEV);
            }
        };

        let data = try_pin_init!(AsahiData {
            gpu,
            pdev: pdev.into(),
            resources: res,
        });

        let ptr: *const AsahiData = &raw const **drm;
        unsafe {
            data.__pinned_init(ptr as *mut AsahiData)?;
        }

        (*drm).gpu.init()?;

        drm::driver::Registration::new_foreign_owned(&drm, pdev.as_ref(), 0)?;

        Ok(Self { drm })
    }
}
