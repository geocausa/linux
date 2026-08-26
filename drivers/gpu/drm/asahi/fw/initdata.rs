// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU initialization / global structures

use super::channels;
use super::types::*;
use crate::{
    default_zeroed,
    gem,
    mmu,
    no_debug,
    trivial_gpustruct, //
};

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct ChannelRing<T: GpuStruct + Debug + Default, U: Copy> {
        pub(crate) state: Option<GpuWeakPointer<T>>,
        pub(crate) ring: Option<GpuWeakPointer<[U]>>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct PipeChannels {
        pub(crate) vtx: ChannelRing<channels::ChannelState, channels::PipeMsg::ver>,
        pub(crate) frag: ChannelRing<channels::ChannelState, channels::PipeMsg::ver>,
        pub(crate) comp: ChannelRing<channels::ChannelState, channels::PipeMsg::ver>,
    }
    #[versions(AGX)]
    default_zeroed!(PipeChannels::ver);

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct FwStatusFlags {
        pub(crate) halt_count: AtomicU64,
        __pad0: Pad<0x8>,
        pub(crate) halted: AtomicU32,
        __pad1: Pad<0xc>,
        pub(crate) resume: AtomicU32,
        __pad2: Pad<0xc>,
        pub(crate) unk_40: u32,
        __pad3: Pad<0xc>,
        pub(crate) unk_ctr: u32,
        __pad4: Pad<0xc>,
        pub(crate) unk_60: u32,
        __pad5: Pad<0xc>,
        pub(crate) unk_70: u32,
        __pad6: Pad<0xc>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct FwStatus {
        pub(crate) fwctl_channel: ChannelRing<channels::FwCtlChannelState, channels::FwCtlMsg>,
        pub(crate) flags: FwStatusFlags,
    }

    /// G15 root q21: exact 0x20-byte host/FW shared status block.
    ///
    /// This is deliberately *not* the legacy FwStatus layout. Direct firmware
    /// users prove +0x04 as a one-time banner guard, +0x0c as a transient busy
    /// flag, +0x14 as the firmware boot-ready marker, and +0x18 as a power/state
    /// index. The host supplies +0x00 as a flags word.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15SharedStatus {
        pub(crate) host_flags: u32,                // +0x00
        pub(crate) banner_guard: AtomicU32,        // +0x04
        pub(crate) unk_08: u32,                    // +0x08
        pub(crate) busy: AtomicU32,                // +0x0c
        pub(crate) unk_10: u32,                    // +0x10
        pub(crate) firmware_ready: AtomicU32,      // +0x14
        pub(crate) power_state: AtomicU32,         // +0x18
        pub(crate) unk_1c: u32,                    // +0x1c
    }
    default_zeroed!(G15SharedStatus);
    const _: [(); 0x20] = [(); core::mem::size_of::<G15SharedStatus>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(G15SharedStatus, banner_guard)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15SharedStatus, busy)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15SharedStatus, firmware_ready)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(G15SharedStatus, power_state)];

    /// G15 root q4: exact 0xe00-byte compact configuration block.
    ///
    /// The Apple host uses several deliberately unaligned fields, so U32/U64
    /// wrappers preserve byte-exact offsets without packing the whole struct.
    /// Names stay offset-oriented where semantics are not yet proven.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15Q4Config {
        pub(crate) flags_000: U32,                 // +0x000: exact J615 zero; host writes 7 only if cfg bit28
        pub(crate) zero_004: U32,
        pub(crate) zero_008: U32,
        pub(crate) zero_00c: U32,
        pub(crate) zero_010: U32,
        pub(crate) zero_014: U32,
        pub(crate) zero_018: U32,
        pub(crate) unk_01c: U32,                   // exact G15 source +0x2970; zero
        pub(crate) relaxed_cl_kill_timeout_020: U32,
        pub(crate) frg_task_timeout_024: U32,       // setFRGTaskTimeout(); exact G15 zero
        pub(crate) accelerator_config_660_028: U32, // Apple copies accelerator +0x660; exact J615 zero
        pub(crate) accelerator_config_bit0_02c: U32,
        pub(crate) smart_idle_off_enabled_030: U32,
        pub(crate) zero_034: U32,
        pub(crate) constant_038: U32,              // exact host constant 0x78
        pub(crate) cpms_window_size_03c: U32,       // setCPMSWindowSize()
        pub(crate) cpms_tfca_size_040: U32,         // setCPMSTFCASize()
        pub(crate) zero_044: U32,
        pub(crate) kick_channel_qos_arg2_048: U32, // setKickChannelQos() arg2
        pub(crate) kick_channel_qos_arg1_04c: U32, // setKickChannelQos() arg1
        pub(crate) unk_050: u16,                   // accelerator +0x6c4
        pub(crate) unk_052: u16,                   // accelerator +0x6c6
        pub(crate) unk_054: u16,                   // accelerator +0x6c8
        pub(crate) zero_056: U32,                  // deliberately unaligned
        pub(crate) low2_clear_05a: U32,            // ((accelerator[0x9d08] & 3) == 0)
        pub(crate) cswitch_timer_multiplier_05e: U32, // deliberately unaligned
        pub(crate) cdm_cswitch_mode_change_062: U32, // deliberately unaligned
        pub(crate) pad_066: Pad<0x0a>,
        pub(crate) command_submission_enabled_070: U32,
        pub(crate) zero_074: U32,                    // exact G15 bootstrap zero; firmware reads only
        pub(crate) gpu_max_power_078: U32,
        pub(crate) power_interface_1_target_07c: U32,
        pub(crate) power_interface_2_target_080: U32,
        pub(crate) pad_084: Pad<0x08>,
        pub(crate) perf_state_cap_x100_08c: U32,
        pub(crate) performance_target_090: U32,
        pub(crate) performance_transfer_output_094: U32,
        pub(crate) performance_boost_min_util_098: U32,
        pub(crate) performance_boost_ce_step_09c: U32,
        pub(crate) performance_reset_iters_0a0: U32,
        pub(crate) performance_boost_min_util_valid_0a4: u8,
        pub(crate) pad_0a5: Pad<0x1ca>,
        pub(crate) conflict_scan_enable_26f: u8, // exact J615 zero; gates FW conflict/resource scan
        pub(crate) pad_270: Pad<0x538>,
        pub(crate) smart_idle_standby_timer_us_7a8: U32,
        pub(crate) smart_idle_prob_init_val_7ac: F32,
        pub(crate) smart_idle_fn_hit_7b0: F32,
        pub(crate) smart_idle_fi_hit_7b4: F32,
        pub(crate) smart_idle_fn_miss_7b8: F32,
        pub(crate) smart_idle_fi_miss_7bc: F32,
        pub(crate) smart_idle_nei_hit_7c0: F32,
        pub(crate) smart_idle_min_confidence_7c4: F32,
        pub(crate) smart_idle_high_confidence_7c8: F32,
        pub(crate) smart_idle_reset_iterations_7cc: U32,
        pub(crate) ut_engagement_enabled_7d0: U32,
        pub(crate) clvr_engagement_enabled_7d4: U32,
        pub(crate) gpu_keepalive_perf_mode_threshold_7d8: U32,
        pub(crate) gpu_keepalive_off_mode_threshold_7dc: U32,
        pub(crate) pad_7e0: Pad<0x184>,
        // AGXArmFirmware::init() explicitly zeroes q4 +0x7e4..+0x96b.
        // Firmware treats +0x964 as the count for 0x18-byte register patch
        // records beginning at +0x7f4; exact J615 boot count is therefore 0.
        pub(crate) boot_patch_record_count_964: U32,
        pub(crate) boot_patch_count_zero_968: U32,
        pub(crate) display_pm_timestamp_96c: U64,
        pub(crate) display_pm_interval_974: U64,
        pub(crate) progress_check_interval_3d_97c: U32,
        pub(crate) progress_check_interval_ta_980: U32,
        pub(crate) progress_check_interval_cl_984: U32,
        pub(crate) unk_988: U32,                   // accelerator +0x1e54; exact 1
        pub(crate) unk_98c: U32,                   // accelerator +0x1e58; exact 1
        pub(crate) unk_990: U32,                   // accelerator +0x1e5c; exact 100
        pub(crate) unk_994: U32,                   // accelerator +0x1e60; exact 1
        pub(crate) pad_998: Pad<0x08>,
        pub(crate) gpu_idle_off_delay_ms_9a0: U32,
        pub(crate) fender_idle_off_delay_ms_9a4: U32,
        pub(crate) fw_early_wake_timeout_ms_9a8: U32,
        pub(crate) gvdm_timer_interval_9ac: U32,
        pub(crate) cl_context_switch_timeout_9b0: U32,
        pub(crate) cl_kill_timeout_9b4: U32,
        pub(crate) zero_9b8: U32,
        pub(crate) cdm_backoff_timeout_9bc: u8,
        pub(crate) pad_9bd: Pad<0x20>,
        pub(crate) table_selector_9dd: U64,
        pub(crate) table_9e5: Array<0x200, u8>,
        pub(crate) table_be5: Array<0x200, u8>,
        pub(crate) zero_de5: U32,
        pub(crate) gpu_keepalive_override_de9: U32,
        pub(crate) gfxc_keepalive_override_ded: U32,
        pub(crate) gpu_keepalive_perf_mode_threshold_override_df1: U32,
        pub(crate) gpu_keepalive_off_mode_threshold_override_df5: U32,
        pub(crate) soft_fault_settings_df9: U32,
        pub(crate) tail_dfd: Pad<0x03>,
    }
    default_zeroed!(G15Q4Config);
    const _: [(); 0xe00] = [(); core::mem::size_of::<G15Q4Config>()];
    const _: [(); 0x01c] = [(); core::mem::offset_of!(G15Q4Config, unk_01c)];
    const _: [(); 0x020] = [(); core::mem::offset_of!(G15Q4Config, relaxed_cl_kill_timeout_020)];
    const _: [(); 0x024] = [(); core::mem::offset_of!(G15Q4Config, frg_task_timeout_024)];
    const _: [(); 0x028] = [(); core::mem::offset_of!(G15Q4Config, accelerator_config_660_028)];
    const _: [(); 0x02c] = [(); core::mem::offset_of!(G15Q4Config, accelerator_config_bit0_02c)];
    const _: [(); 0x030] = [(); core::mem::offset_of!(G15Q4Config, smart_idle_off_enabled_030)];
    const _: [(); 0x038] = [(); core::mem::offset_of!(G15Q4Config, constant_038)];
    const _: [(); 0x03c] = [(); core::mem::offset_of!(G15Q4Config, cpms_window_size_03c)];
    const _: [(); 0x040] = [(); core::mem::offset_of!(G15Q4Config, cpms_tfca_size_040)];
    const _: [(); 0x048] = [(); core::mem::offset_of!(G15Q4Config, kick_channel_qos_arg2_048)];
    const _: [(); 0x04c] = [(); core::mem::offset_of!(G15Q4Config, kick_channel_qos_arg1_04c)];
    const _: [(); 0x056] = [(); core::mem::offset_of!(G15Q4Config, zero_056)];
    const _: [(); 0x05a] = [(); core::mem::offset_of!(G15Q4Config, low2_clear_05a)];
    const _: [(); 0x05e] = [(); core::mem::offset_of!(G15Q4Config, cswitch_timer_multiplier_05e)];
    const _: [(); 0x062] = [(); core::mem::offset_of!(G15Q4Config, cdm_cswitch_mode_change_062)];
    const _: [(); 0x070] = [(); core::mem::offset_of!(G15Q4Config, command_submission_enabled_070)];
    const _: [(); 0x074] = [(); core::mem::offset_of!(G15Q4Config, zero_074)];
    const _: [(); 0x078] = [(); core::mem::offset_of!(G15Q4Config, gpu_max_power_078)];
    const _: [(); 0x07c] = [(); core::mem::offset_of!(G15Q4Config, power_interface_1_target_07c)];
    const _: [(); 0x080] = [(); core::mem::offset_of!(G15Q4Config, power_interface_2_target_080)];
    const _: [(); 0x08c] = [(); core::mem::offset_of!(G15Q4Config, perf_state_cap_x100_08c)];
    const _: [(); 0x090] = [(); core::mem::offset_of!(G15Q4Config, performance_target_090)];
    const _: [(); 0x094] = [(); core::mem::offset_of!(G15Q4Config, performance_transfer_output_094)];
    const _: [(); 0x098] = [(); core::mem::offset_of!(G15Q4Config, performance_boost_min_util_098)];
    const _: [(); 0x09c] = [(); core::mem::offset_of!(G15Q4Config, performance_boost_ce_step_09c)];
    const _: [(); 0x0a0] = [(); core::mem::offset_of!(G15Q4Config, performance_reset_iters_0a0)];
    const _: [(); 0x0a4] = [(); core::mem::offset_of!(G15Q4Config, performance_boost_min_util_valid_0a4)];
    const _: [(); 0x26f] = [(); core::mem::offset_of!(G15Q4Config, conflict_scan_enable_26f)];
    const _: [(); 0x7a8] = [(); core::mem::offset_of!(G15Q4Config, smart_idle_standby_timer_us_7a8)];
    const _: [(); 0x7ac] = [(); core::mem::offset_of!(G15Q4Config, smart_idle_prob_init_val_7ac)];
    const _: [(); 0x7cc] = [(); core::mem::offset_of!(G15Q4Config, smart_idle_reset_iterations_7cc)];
    const _: [(); 0x7d0] = [(); core::mem::offset_of!(G15Q4Config, ut_engagement_enabled_7d0)];
    const _: [(); 0x7d4] = [(); core::mem::offset_of!(G15Q4Config, clvr_engagement_enabled_7d4)];
    const _: [(); 0x7d8] = [(); core::mem::offset_of!(G15Q4Config, gpu_keepalive_perf_mode_threshold_7d8)];
    const _: [(); 0x7dc] = [(); core::mem::offset_of!(G15Q4Config, gpu_keepalive_off_mode_threshold_7dc)];
    const _: [(); 0x964] = [(); core::mem::offset_of!(G15Q4Config, boot_patch_record_count_964)];
    const _: [(); 0x968] = [(); core::mem::offset_of!(G15Q4Config, boot_patch_count_zero_968)];
    const _: [(); 0x96c] = [(); core::mem::offset_of!(G15Q4Config, display_pm_timestamp_96c)];
    const _: [(); 0x974] = [(); core::mem::offset_of!(G15Q4Config, display_pm_interval_974)];
    const _: [(); 0x97c] = [(); core::mem::offset_of!(G15Q4Config, progress_check_interval_3d_97c)];
    const _: [(); 0x980] = [(); core::mem::offset_of!(G15Q4Config, progress_check_interval_ta_980)];
    const _: [(); 0x984] = [(); core::mem::offset_of!(G15Q4Config, progress_check_interval_cl_984)];
    const _: [(); 0x988] = [(); core::mem::offset_of!(G15Q4Config, unk_988)];
    const _: [(); 0x98c] = [(); core::mem::offset_of!(G15Q4Config, unk_98c)];
    const _: [(); 0x990] = [(); core::mem::offset_of!(G15Q4Config, unk_990)];
    const _: [(); 0x994] = [(); core::mem::offset_of!(G15Q4Config, unk_994)];
    const _: [(); 0x9a0] = [(); core::mem::offset_of!(G15Q4Config, gpu_idle_off_delay_ms_9a0)];
    const _: [(); 0x9a4] = [(); core::mem::offset_of!(G15Q4Config, fender_idle_off_delay_ms_9a4)];
    const _: [(); 0x9a8] = [(); core::mem::offset_of!(G15Q4Config, fw_early_wake_timeout_ms_9a8)];
    const _: [(); 0x9ac] = [(); core::mem::offset_of!(G15Q4Config, gvdm_timer_interval_9ac)];
    const _: [(); 0x9b0] = [(); core::mem::offset_of!(G15Q4Config, cl_context_switch_timeout_9b0)];
    const _: [(); 0x9b4] = [(); core::mem::offset_of!(G15Q4Config, cl_kill_timeout_9b4)];
    const _: [(); 0x9bc] = [(); core::mem::offset_of!(G15Q4Config, cdm_backoff_timeout_9bc)];
    const _: [(); 0x9dd] = [(); core::mem::offset_of!(G15Q4Config, table_selector_9dd)];
    const _: [(); 0x9e5] = [(); core::mem::offset_of!(G15Q4Config, table_9e5)];
    const _: [(); 0xbe5] = [(); core::mem::offset_of!(G15Q4Config, table_be5)];
    const _: [(); 0xde5] = [(); core::mem::offset_of!(G15Q4Config, zero_de5)];
    const _: [(); 0xde9] = [(); core::mem::offset_of!(G15Q4Config, gpu_keepalive_override_de9)];
    const _: [(); 0xded] = [(); core::mem::offset_of!(G15Q4Config, gfxc_keepalive_override_ded)];
    const _: [(); 0xdf1] = [(); core::mem::offset_of!(G15Q4Config, gpu_keepalive_perf_mode_threshold_override_df1)];
    const _: [(); 0xdf5] = [(); core::mem::offset_of!(G15Q4Config, gpu_keepalive_off_mode_threshold_override_df5)];
    const _: [(); 0xdf9] = [(); core::mem::offset_of!(G15Q4Config, soft_fault_settings_df9)];

    /// G15 q22 firmware-control/cache-flush ring state, exact 0x20 bytes.
    /// This is the G15 successor to the legacy FwStatus `FwCtlChannelState`:
    /// both place read_idx at +0x00 and write_idx at +0x10. Firmware consumes
    /// entries from read_idx and compares against write_idx modulo 256.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15CacheFlushState {
        pub(crate) read_idx: AtomicU32,   // +0x00
        pub(crate) pad_004: Pad<0x0c>,
        pub(crate) write_idx: AtomicU32,  // +0x10
        pub(crate) pad_014: Pad<0x0c>,
    }
    default_zeroed!(G15CacheFlushState);
    const _: [(); 0x20] = [(); core::mem::size_of::<G15CacheFlushState>()];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15CacheFlushState, write_idx)];

    #[allow(dead_code)]
    pub(crate) const G15_MAP_FLAG_MAP: u16 = 1 << 0;
    #[allow(dead_code)]
    pub(crate) const G15_MAP_FLAG_SPECIAL_APERTURE: u16 = 1 << 1;
    #[allow(dead_code)]
    pub(crate) const G15_MAP_FLAG_PROPERTY: u16 = 1 << 2;

    /// One G15 firmware-control/mapping entry, exact 0x18 bytes. Apple builds
    /// these from AGXMemoryMap page-walker callbacks before inserting them into
    /// q22's 256-entry ring. G15's GPU-VA-to-FW-VA conversion is identity.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15CacheFlushEntry {
        pub(crate) addr: U64,                 // +0x00: firmware-visible GPU VA
        pub(crate) phys_page_4k: U32,          // +0x08: physical address >> 12
        // Firmware's secure-flush branch consumes this as a context ID. The
        // normal Apple page walker emits 0 for map and 0xffff_ffff for unmap.
        pub(crate) secure_context_id: U32,    // +0x0c
        // 1 << (GART page shift - FW page shift). J615 uses ChinookV9 with
        // FW shift 14 and the platform GART shift follows kernel page_shift.
        pub(crate) fw_page_count: u16,        // +0x10
        // bit0=map, bit1=special 64MiB aperture, bit2=map property
        pub(crate) mapping_flags: u16,        // +0x12
        pub(crate) reserved_14: U32,          // +0x14: exact zero, map and unmap
    }
    const _: [(); 0x18] = [(); core::mem::size_of::<G15CacheFlushEntry>()];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15CacheFlushEntry, phys_page_4k)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15CacheFlushEntry, secure_context_id)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15CacheFlushEntry, fw_page_count)];
    const _: [(); 0x12] = [(); core::mem::offset_of!(G15CacheFlushEntry, mapping_flags)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15CacheFlushEntry, reserved_14)];
    // SAFETY: every field has an all-zero valid representation.
    unsafe impl Zeroable for G15CacheFlushEntry {}

    /// Exact packed backing produced by AGXFirmware::allocateSharedData() for
    /// the q22 mapping-control state/ring subdescriptor pair. Apple places the
    /// 0x20 state at +0x00, aligns the second child to 0x40, then places the
    /// 0x1800 ring at +0x40, for 0x1840 bytes before page rounding.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15MappingRingBacking {
        pub(crate) state: G15CacheFlushState,
        pub(crate) pad_020: Pad<0x20>,
        pub(crate) ring: Array<0x100, G15CacheFlushEntry>,
    }
    default_zeroed!(G15MappingRingBacking);
    const _: [(); 0x1840] = [(); core::mem::size_of::<G15MappingRingBacking>()];
    const _: [(); 0x40] = [(); core::mem::offset_of!(G15MappingRingBacking, ring)];

    /// G15 root q22: exact 0xc3d0-byte host/FW shared object.
    ///
    /// Apple maps CPU/GPU pair +0x620/+0x630 into root q22. The fields below
    /// are all directly observed in host initialization or firmware accesses;
    /// unresolved regions stay zeroed and offset-named.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15Q22Shared {
        pub(crate) host_zero_0000: U32,
        pub(crate) pad_0004: Pad<0x04>,
        pub(crate) host_zero_0008: U32,
        pub(crate) pad_000c: Pad<0x4024>,
        pub(crate) trace_enable_4030: U32,
        pub(crate) pad_4034: Pad<0x18>,
        pub(crate) trace_desc_ptr_404c: U64,
        pub(crate) trace_desc_count_4054: U32,
        pub(crate) pad_4058: Pad<0x510>,
        pub(crate) shared_ptr_4568: U64,
        pub(crate) shared_ptr_4570: U64,
        pub(crate) pad_4578: Pad<0x08>,
        pub(crate) epoch_4580: U64,
        pub(crate) pad_4588: Pad<0x08>,
        pub(crate) state_4590: U32,
        pub(crate) pad_4594: Pad<0x0c>,
        pub(crate) state_45a0: U32,
        pub(crate) pad_45a4: Pad<0x0c>,
        pub(crate) state_45b0: U32,
        pub(crate) pad_45b4: Pad<0x0c>,
        pub(crate) counter_45c0: U32,
        pub(crate) host_flag_45c4: U32,
        pub(crate) pad_45c8: Pad<0x7e00>,
        pub(crate) kick_channel_qos_valid_c3c8: U32,
        pub(crate) feature_c3cc: U32,
    }
    default_zeroed!(G15Q22Shared);
    const _: [(); 0xc3d0] = [(); core::mem::size_of::<G15Q22Shared>()];
    const _: [(); 0x4030] = [(); core::mem::offset_of!(G15Q22Shared, trace_enable_4030)];
    const _: [(); 0x404c] = [(); core::mem::offset_of!(G15Q22Shared, trace_desc_ptr_404c)];
    const _: [(); 0x4054] = [(); core::mem::offset_of!(G15Q22Shared, trace_desc_count_4054)];
    const _: [(); 0x4568] = [(); core::mem::offset_of!(G15Q22Shared, shared_ptr_4568)];
    const _: [(); 0x4580] = [(); core::mem::offset_of!(G15Q22Shared, epoch_4580)];
    const _: [(); 0x45c4] = [(); core::mem::offset_of!(G15Q22Shared, host_flag_45c4)];
    const _: [(); 0xc3c8] = [(); core::mem::offset_of!(G15Q22Shared, kick_channel_qos_valid_c3c8)];
    const _: [(); 0xc3cc] = [(); core::mem::offset_of!(G15Q22Shared, feature_c3cc)];

    /// G15 root q23: exact 0x238-byte host/FW shared runtime/tuning object.
    /// Apple maps CPU/GPU pair +0x628/+0x638 into q23. Fields stay
    /// offset-oriented unless their behavior is mechanically established.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15Q23Shared {
        pub(crate) runtime_value_000: U32,       // +0x000, firmware output
        pub(crate) pad_004: Pad<0x04>,
        pub(crate) update_008: U32,              // +0x008, consumed/cleared
        pub(crate) table_count_00c: U32,         // +0x00c
        pub(crate) base_params_010: Array<5, U32>, // +0x010..+0x023
        // FUN_...29a48 walks entries from +0x24 with 0x0c stride. The next
        // independently used field is +0x60, bounding this table to 5 slots.
        pub(crate) table_024: Array<5, Array<3, U32>>, // +0x024..+0x05f

        pub(crate) update_060: U32,              // +0x060, consumed/cleared
        pub(crate) value_064: U32,
        pub(crate) value_068: U32,
        pub(crate) mask_06c: U32,
        pub(crate) pad_070: Pad<0x04>,
        pub(crate) mask_074: U32,
        pub(crate) values_078: Array<10, U32>,   // +0x078..+0x09f
        pub(crate) value_0a0: U32,

        pub(crate) update_0a4: U32,              // +0x0a4, consumed/cleared
        pub(crate) limit_0a8: U32,
        pub(crate) tuning_0ac: Array<7, U32>,    // +0x0ac..+0x0c7
        pub(crate) gate_0c8: U32,
        pub(crate) update_0cc: U32,              // +0x0cc, consumed/cleared
        pub(crate) tuning_0d0: Array<6, U32>,    // +0x0d0..+0x0e7

        pub(crate) update_0e8: U32,              // +0x0e8, consumed/cleared
        pub(crate) hold_0ec: U32,
        pub(crate) pad_0f0: Pad<0x04>,
        pub(crate) override_0f4: U32,
        pub(crate) value_0f8: U32,
        pub(crate) enable_0fc: U32,
        pub(crate) value_100: U32,
        pub(crate) value_104: U32,
        pub(crate) enable_108: U32,
        pub(crate) enable_10c: U32,
        pub(crate) value_110: U32,
        pub(crate) value_114: U32,
        pub(crate) enable_118: U32,
        pub(crate) enable_11c: U32,
        pub(crate) value_120: U32,
        pub(crate) value_124: U32,
        pub(crate) enable_128: U32,
        pub(crate) enable_12c: U32,
        pub(crate) value_130: U32,
        pub(crate) value_134: U32,
        pub(crate) value_138: U32,
        pub(crate) value_13c: U32,
        pub(crate) value_140: U32,
        pub(crate) value_144: U32,
        pub(crate) value_148: U32,
        pub(crate) pad_14c: Pad<0x0c>,

        // Firmware clears +0x158 after importing the following snapshot into
        // HwDataA +0x4190..+0x41d8.
        pub(crate) snapshot_update_158: U32,
        pub(crate) pad_15c: Pad<0x04>,
        pub(crate) snapshot_160: U32,
        pub(crate) snapshot_164: U32,
        pub(crate) snapshot_168: U64,
        pub(crate) snapshot_170: U64,
        pub(crate) snapshot_178: U64,
        pub(crate) snapshot_180: U64,
        pub(crate) snapshot_188: U64,
        pub(crate) snapshot_190: U64,
        pub(crate) snapshot_198: U64,
        pub(crate) snapshot_1a0: U64,
        pub(crate) snapshot_1a8: U32,
        pub(crate) pad_1ac: Pad<0x08>,

        // These counters are deliberately unaligned 64-bit firmware writes.
        pub(crate) idle_entry_count_1b4: U64,
        pub(crate) idle_ticks_1bc: U64,
        pub(crate) pad_1c4: Pad<0x08>,
        pub(crate) epoch_1cc: U32,
        pub(crate) tuning_update_1d0: u8,          // host initializes zero
        pub(crate) min_state_1d1: U32,            // deliberately unaligned
        pub(crate) pad_1d5: Pad<0x13>,
        pub(crate) host_zero_1e8: U32,             // explicit Apple host write
        pub(crate) pad_1ec: Pad<0x44>,
        pub(crate) idle_gate_230: U32,
        pub(crate) pad_234: Pad<0x04>,
    }
    default_zeroed!(G15Q23Shared);
    const _: [(); 0x238] = [(); core::mem::size_of::<G15Q23Shared>()];
    const _: [(); 0x008] = [(); core::mem::offset_of!(G15Q23Shared, update_008)];
    const _: [(); 0x00c] = [(); core::mem::offset_of!(G15Q23Shared, table_count_00c)];
    const _: [(); 0x024] = [(); core::mem::offset_of!(G15Q23Shared, table_024)];
    const _: [(); 0x060] = [(); core::mem::offset_of!(G15Q23Shared, update_060)];
    const _: [(); 0x06c] = [(); core::mem::offset_of!(G15Q23Shared, mask_06c)];
    const _: [(); 0x078] = [(); core::mem::offset_of!(G15Q23Shared, values_078)];
    const _: [(); 0x0a4] = [(); core::mem::offset_of!(G15Q23Shared, update_0a4)];
    const _: [(); 0x0c8] = [(); core::mem::offset_of!(G15Q23Shared, gate_0c8)];
    const _: [(); 0x0cc] = [(); core::mem::offset_of!(G15Q23Shared, update_0cc)];
    const _: [(); 0x0e8] = [(); core::mem::offset_of!(G15Q23Shared, update_0e8)];
    const _: [(); 0x0f4] = [(); core::mem::offset_of!(G15Q23Shared, override_0f4)];
    const _: [(); 0x130] = [(); core::mem::offset_of!(G15Q23Shared, value_130)];
    const _: [(); 0x158] = [(); core::mem::offset_of!(G15Q23Shared, snapshot_update_158)];
    const _: [(); 0x160] = [(); core::mem::offset_of!(G15Q23Shared, snapshot_160)];
    const _: [(); 0x1a8] = [(); core::mem::offset_of!(G15Q23Shared, snapshot_1a8)];
    const _: [(); 0x1b4] = [(); core::mem::offset_of!(G15Q23Shared, idle_entry_count_1b4)];
    const _: [(); 0x1bc] = [(); core::mem::offset_of!(G15Q23Shared, idle_ticks_1bc)];
    const _: [(); 0x1cc] = [(); core::mem::offset_of!(G15Q23Shared, epoch_1cc)];
    const _: [(); 0x1d0] = [(); core::mem::offset_of!(G15Q23Shared, tuning_update_1d0)];
    const _: [(); 0x1d1] = [(); core::mem::offset_of!(G15Q23Shared, min_state_1d1)];
    const _: [(); 0x1e8] = [(); core::mem::offset_of!(G15Q23Shared, host_zero_1e8)];
    const _: [(); 0x230] = [(); core::mem::offset_of!(G15Q23Shared, idle_gate_230)];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared1 {
        pub(crate) table: Array<16, i32>,
        pub(crate) unk_44: Array<0x60, u8>,
        pub(crate) unk_a4: u32,
        pub(crate) unk_a8: u32,
    }
    default_zeroed!(HwDataShared1);

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct HwDataShared2Curve {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) t1: Array<16, u16>,
        pub(crate) t2: Array<16, i16>,
        pub(crate) t3: Array<8, Array<16, i32>>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct HwDataShared2G14 {
        pub(crate) unk_0: Array<5, u32>,
        pub(crate) unk_14: u32,
        pub(crate) unk_18: Array<8, u32>,
        pub(crate) curve1: HwDataShared2Curve,
        pub(crate) curve2: HwDataShared2Curve,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared2 {
        pub(crate) table: Array<10, i32>,
        pub(crate) unk_28: Array<0x10, u8>,
        pub(crate) g14: HwDataShared2G14,
        pub(crate) unk_500: u32,
        pub(crate) unk_504: u32,
        pub(crate) unk_508: u32,
        pub(crate) unk_50c: u32,
    }
    default_zeroed!(HwDataShared2);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared3 {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) unk_8: u32,
        pub(crate) table: Array<16, u32>,
        pub(crate) unk_4c: u32,
    }
    default_zeroed!(HwDataShared3);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataA130Extra {
        pub(crate) unk_0: Array<0x38, u8>,
        pub(crate) unk_38: u32,
        pub(crate) unk_3c: u32,
        pub(crate) gpu_se_inactive_threshold: u32,
        pub(crate) unk_44: u32,
        pub(crate) gpu_se_engagement_criteria: i32,
        pub(crate) gpu_se_reset_criteria: u32,
        pub(crate) unk_50: u32,
        pub(crate) unk_54: u32,
        pub(crate) unk_58: u32,
        pub(crate) unk_5c: u32,
        pub(crate) gpu_se_filter_a_neg: F32,
        pub(crate) gpu_se_filter_1_a_neg: F32,
        pub(crate) gpu_se_filter_a: F32,
        pub(crate) gpu_se_filter_1_a: F32,
        pub(crate) gpu_se_ki_dt: F32,
        pub(crate) gpu_se_ki_1_dt: F32,
        pub(crate) unk_78: F32,
        pub(crate) unk_7c: F32,
        pub(crate) gpu_se_kp: F32,
        pub(crate) gpu_se_kp_1: F32,
        pub(crate) unk_88: u32,
        pub(crate) unk_8c: u32,
        pub(crate) max_pstate_scaled_1: u32,
        pub(crate) unk_94: u32,
        pub(crate) unk_98: u32,
        pub(crate) unk_9c: F32,
        pub(crate) unk_a0: u32,
        pub(crate) unk_a4: u32,
        pub(crate) gpu_se_filter_time_constant_ms: u32,
        pub(crate) gpu_se_filter_time_constant_1_ms: u32,
        pub(crate) gpu_se_filter_time_constant_clks: U64,
        pub(crate) gpu_se_filter_time_constant_1_clks: U64,
        pub(crate) unk_c0: u32,
        pub(crate) unk_c4: F32,
        pub(crate) unk_c8: Array<0x4c, u8>,
        pub(crate) unk_114: F32,
        pub(crate) unk_118: u32,
        pub(crate) unk_11c: u32,
        pub(crate) unk_120: u32,
        pub(crate) unk_124: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) unk_12c: Array<0x8c, u8>,
    }
    default_zeroed!(HwDataA130Extra);

    #[repr(C)]
    pub(crate) struct T81xxData {
        pub(crate) unk_d8c: u32,
        pub(crate) unk_d90: u32,
        pub(crate) unk_d94: u32,
        pub(crate) unk_d98: u32,
        pub(crate) unk_d9c: F32,
        pub(crate) unk_da0: u32,
        pub(crate) unk_da4: F32,
        pub(crate) unk_da8: u32,
        pub(crate) unk_dac: F32,
        pub(crate) unk_db0: u32,
        pub(crate) unk_db4: u32,
        pub(crate) unk_db8: F32,
        pub(crate) unk_dbc: F32,
        pub(crate) unk_dc0: u32,
        pub(crate) unk_dc4: u32,
        pub(crate) unk_dc8: u32,
        pub(crate) max_pstate_scaled: u32,
    }
    default_zeroed!(T81xxData);

    #[versions(AGX)]
    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub(crate) struct PowerZone {
        pub(crate) val: F32,
        pub(crate) target: u32,
        pub(crate) target_off: u32,
        pub(crate) filter_tc_x4: u32,
        pub(crate) filter_tc_xperiod: u32,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_10: u32,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_14: u32,
        pub(crate) filter_a_neg: F32,
        pub(crate) filter_a: F32,
        pub(crate) pad: u32,
    }

    #[versions(AGX)]
    const MAX_CORES_PER_CLUSTER: usize = {
        #[ver(G >= G14X)]
        {
            16
        }
        #[ver(G < G14X)]
        {
            8
        }
    };

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct AuxLeakCoef {
        pub(crate) afr_1: Array<2, F32>,
        pub(crate) cs_1: Array<2, F32>,
        pub(crate) afr_2: Array<2, F32>,
        pub(crate) cs_2: Array<2, F32>,
    }

    /// J615/C0 G15 DPE/PPT payload at HwDataA +0x3aa8.
    ///
    /// This is the exact zero-count bootstrap image produced by
    /// AGXAcceleratorG15G::populateDPEPPTConfigData() on 25F84. The two
    /// repeated 64-qword banks are represented structurally rather than as an
    /// opaque captured byte blob.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15DpePptConfig {
        pub(crate) loop_count_000: u32,
        pub(crate) pad_004: Pad<0xc8>,
        pub(crate) all_ones_0cc: Array<4, U64>,
        pub(crate) bootstrap_0ec: U64,
        pub(crate) q_3fffff_0f4: Array<2, U64>,
        pub(crate) pad_104: Pad<0x10>,
        pub(crate) literal_114: U64,
        pub(crate) pad_11c: Pad<0x20>,
        pub(crate) literal_13c: U64,
        pub(crate) all_ones_144: Array<4, U64>,
        pub(crate) q_0f07_164: Array<4, U64>,
        pub(crate) bank1_184: Array<64, U64>,
        pub(crate) special_384: U64,
        pub(crate) all_ones_38c: Array<4, U64>,
        pub(crate) q_0f07_3ac: Array<4, U64>,
        pub(crate) bank2_3cc: Array<64, U64>,
        pub(crate) pad_5cc: Pad<0x08>,
        pub(crate) control_5d4: U64,
    }
    default_zeroed!(G15DpePptConfig);
    const _: [(); 0x5dc] = [(); core::mem::size_of::<G15DpePptConfig>()];
    const _: [(); 0x0cc] = [(); core::mem::offset_of!(G15DpePptConfig, all_ones_0cc)];
    const _: [(); 0x184] = [(); core::mem::offset_of!(G15DpePptConfig, bank1_184)];
    const _: [(); 0x384] = [(); core::mem::offset_of!(G15DpePptConfig, special_384)];
    const _: [(); 0x3cc] = [(); core::mem::offset_of!(G15DpePptConfig, bank2_3cc)];
    const _: [(); 0x5d4] = [(); core::mem::offset_of!(G15DpePptConfig, control_5d4)];

    /// Sparse SoCHot payload copied to HwDataA +0x4188 on J615/C0.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15SoCHotConfig {
        pub(crate) pad_000: Pad<0x10>,
        pub(crate) sensor_mask_010: U64,
        pub(crate) constant_018: U64,
        pub(crate) pad_020: Pad<0x34>,
    }
    default_zeroed!(G15SoCHotConfig);
    const _: [(); 0x54] = [(); core::mem::size_of::<G15SoCHotConfig>()];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15SoCHotConfig, sensor_mask_010)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(G15SoCHotConfig, constant_018)];

    /// G15 replacement for the inherited Shared1/2/3 through leakage-coef
    /// region, spanning HwDataA +0x3a9c..+0x421b exactly.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15HwDataAPreTail {
        pub(crate) pad_000: Pad<0x08>,
        pub(crate) constant_008: F32,
        pub(crate) dpe_00c: G15DpePptConfig,
        pub(crate) pad_5e8: Pad<0x104>,
        pub(crate) sochot_6ec: G15SoCHotConfig,
        pub(crate) pad_740: Pad<0x40>,
    }
    default_zeroed!(G15HwDataAPreTail);
    const _: [(); 0x780] = [(); core::mem::size_of::<G15HwDataAPreTail>()];
    const _: [(); 0x008] = [(); core::mem::offset_of!(G15HwDataAPreTail, constant_008)];
    const _: [(); 0x00c] = [(); core::mem::offset_of!(G15HwDataAPreTail, dpe_00c)];
    const _: [(); 0x6ec] = [(); core::mem::offset_of!(G15HwDataAPreTail, sochot_6ec)];

    /// G15-only extension at HwDataA +0x421c.
    ///
    /// Apple's G15 allocation is 0x4360 bytes. Firmware directly accesses
    /// several fields in this 0x144-byte extension. Unresolved fields remain
    /// offset-named and zero-initialized until host semantics are proven.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15HwDataATail {
        pub(crate) pad_421c: Pad<0x7c>,
        pub(crate) unk_4298: u32,
        pub(crate) pad_429c: Pad<0x88>,
        pub(crate) unk_4324: u32,
        pub(crate) unk_4328: F32,
        pub(crate) unk_432c: u32,
        pub(crate) unk_4330: U64,
        pub(crate) unk_4338: u32,
        pub(crate) unk_433c: u32,
        pub(crate) unk_4340: U64,
        pub(crate) unk_4348: U64,
        pub(crate) pad_4350: Pad<0x08>,
        pub(crate) unk_4358: u32,
        pub(crate) pad_435c: Pad<0x04>,
    }
    default_zeroed!(G15HwDataATail);
    const _: [(); 0x144] = [(); core::mem::size_of::<G15HwDataATail>()];
    const _: [(); 0x07c] = [(); core::mem::offset_of!(G15HwDataATail, unk_4298)];
    const _: [(); 0x108] = [(); core::mem::offset_of!(G15HwDataATail, unk_4324)];
    const _: [(); 0x120] = [(); core::mem::offset_of!(G15HwDataATail, unk_433c)];
    const _: [(); 0x13c] = [(); core::mem::offset_of!(G15HwDataATail, unk_4358)];

    #[versions(AGX)]
    #[repr(C)]
    pub(crate) struct HwDataA {
        pub(crate) unk_0: u32,
        pub(crate) clocks_per_period: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) clocks_per_period_2: u32,

        pub(crate) unk_8: u32,
        pub(crate) pwr_status: AtomicU32,
        pub(crate) unk_10: F32,
        pub(crate) unk_14: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_1c: u32,
        pub(crate) unk_20: u32,
        pub(crate) unk_24: u32,
        pub(crate) actual_pstate: u32,
        pub(crate) tgt_pstate: u32,
        pub(crate) unk_30: u32,
        pub(crate) cur_pstate: u32,
        pub(crate) unk_38: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_3c_0: u32,

        pub(crate) base_pstate_scaled: u32,
        pub(crate) unk_40: u32,
        pub(crate) max_pstate_scaled: u32,
        pub(crate) unk_48: u32,
        pub(crate) min_pstate_scaled: u32,
        pub(crate) freq_mhz: F32,
        pub(crate) unk_54: Array<0x20, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_74_0: u32,

        pub(crate) sram_k: Array<0x10, F32>,
        pub(crate) unk_b4: Array<0x100, u8>,
        pub(crate) unk_1b4: u32,
        pub(crate) temp_c: u32,
        pub(crate) avg_power_mw: u32,
        pub(crate) update_ts: U64,
        pub(crate) unk_1c8: u32,
        pub(crate) unk_1cc: Array<0x478, u8>,
        pub(crate) pad_644: Pad<0x8>,
        pub(crate) unk_64c: u32,
        pub(crate) unk_650: u32,
        pub(crate) pad_654: u32,
        pub(crate) pwr_filter_a_neg: F32,
        pub(crate) pad_65c: u32,
        pub(crate) pwr_filter_a: F32,
        pub(crate) pad_664: u32,
        pub(crate) pwr_integral_gain: F32,
        pub(crate) pad_66c: u32,
        pub(crate) pwr_integral_min_clamp: F32,
        pub(crate) max_power_1: F32,
        pub(crate) pwr_proportional_gain: F32,
        pub(crate) pad_67c: u32,
        pub(crate) pwr_pstate_related_k: F32,
        pub(crate) pwr_pstate_max_dc_offset: i32,
        pub(crate) unk_688: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) pad_690: u32,
        pub(crate) unk_694: u32,
        pub(crate) max_power_2: u32,
        pub(crate) pad_69c: Pad<0x18>,
        pub(crate) unk_6b4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_6b8_0: Array<0x10, u8>,

        pub(crate) max_pstate_scaled_3: u32,
        pub(crate) unk_6bc: u32,
        pub(crate) pad_6c0: Pad<0x14>,
        pub(crate) ppm_filter_tc_periods_x4: u32,
        pub(crate) unk_6d8: u32,
        pub(crate) pad_6dc: u32,
        pub(crate) ppm_filter_a_neg: F32,
        pub(crate) pad_6e4: u32,
        pub(crate) ppm_filter_a: F32,
        pub(crate) pad_6ec: u32,
        pub(crate) ppm_ki_dt: F32,
        pub(crate) pad_6f4: u32,
        pub(crate) pwr_integral_min_clamp_2: u32,
        pub(crate) unk_6fc: F32,
        pub(crate) ppm_kp: F32,
        pub(crate) pad_704: u32,
        pub(crate) unk_708: u32,
        pub(crate) pwr_min_duty_cycle: u32,
        pub(crate) max_pstate_scaled_4: u32,
        pub(crate) unk_714: u32,
        pub(crate) pad_718: u32,
        pub(crate) unk_71c: F32,
        pub(crate) max_power_3: u32,
        pub(crate) cur_power_mw_2: u32,
        pub(crate) ppm_filter_tc_ms: u32,
        pub(crate) unk_72c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) ppm_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_c: u32,

        pub(crate) unk_730: F32,
        pub(crate) unk_734: u32,
        pub(crate) unk_738: u32,
        pub(crate) unk_73c: u32,
        pub(crate) unk_740: u32,
        pub(crate) unk_744: u32,
        pub(crate) unk_748: Array<0x4, F32>,
        pub(crate) unk_758: u32,
        pub(crate) perf_tgt_utilization: u32,
        pub(crate) pad_760: u32,
        pub(crate) perf_boost_min_util: u32,
        pub(crate) perf_boost_ce_step: u32,
        pub(crate) perf_reset_iters: u32,
        pub(crate) pad_770: u32,
        pub(crate) unk_774: u32,
        pub(crate) unk_778: u32,
        pub(crate) perf_filter_drop_threshold: u32,
        pub(crate) perf_filter_a_neg: F32,
        pub(crate) perf_filter_a2_neg: F32,
        pub(crate) perf_filter_a: F32,
        pub(crate) perf_filter_a2: F32,
        pub(crate) perf_ki: F32,
        pub(crate) perf_ki2: F32,
        pub(crate) perf_integral_min_clamp: F32,
        pub(crate) unk_79c: F32,
        pub(crate) perf_kp: F32,
        pub(crate) perf_kp2: F32,
        pub(crate) boost_state_unk_k: F32,
        pub(crate) base_pstate_scaled_2: u32,
        pub(crate) max_pstate_scaled_5: u32,
        pub(crate) base_pstate_scaled_3: u32,
        pub(crate) pad_7b8: u32,
        pub(crate) perf_cur_utilization: F32,
        pub(crate) perf_tgt_utilization_2: u32,
        pub(crate) pad_7c4: Pad<0x18>,
        pub(crate) unk_7dc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_7e0_0: Array<0x10, u8>,

        pub(crate) base_pstate_scaled_4: u32,
        pub(crate) pad_7e4: u32,
        pub(crate) unk_7e8: Array<0x14, u8>,
        pub(crate) unk_7fc: F32,
        pub(crate) pwr_min_duty_cycle_2: F32,
        pub(crate) max_pstate_scaled_6: F32,
        pub(crate) max_freq_mhz: u32,
        pub(crate) pad_80c: u32,
        pub(crate) unk_810: u32,
        pub(crate) pad_814: u32,
        pub(crate) pwr_min_duty_cycle_3: u32,
        pub(crate) unk_81c: u32,
        pub(crate) pad_820: u32,
        pub(crate) min_pstate_scaled_4: F32,
        pub(crate) max_pstate_scaled_7: u32,
        pub(crate) unk_82c: u32,
        pub(crate) unk_alpha_neg: F32,
        pub(crate) unk_alpha: F32,
        pub(crate) unk_838: u32,
        pub(crate) unk_83c: u32,
        pub(crate) pad_840: Pad<0x2c>,
        pub(crate) unk_86c: u32,
        pub(crate) fast_die0_sensor_mask: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask: U64,
        pub(crate) fast_die0_release_temp_cc: u32,
        pub(crate) unk_87c: i32,
        pub(crate) unk_880: u32,
        pub(crate) unk_884: u32,
        pub(crate) pad_888: u32,
        pub(crate) unk_88c: u32,
        pub(crate) pad_890: u32,
        pub(crate) unk_894: F32,
        pub(crate) pad_898: u32,
        pub(crate) fast_die0_ki_dt: F32,
        pub(crate) pad_8a0: u32,
        pub(crate) unk_8a4: u32,
        pub(crate) unk_8a8: F32,
        pub(crate) fast_die0_kp: F32,
        pub(crate) pad_8b0: u32,
        pub(crate) unk_8b4: u32,
        pub(crate) pwr_min_duty_cycle_4: u32,
        pub(crate) max_pstate_scaled_8: u32,
        pub(crate) max_pstate_scaled_9: u32,
        pub(crate) fast_die0_prop_tgt_delta: u32,
        pub(crate) unk_8c8: u32,
        pub(crate) unk_8cc: u32,
        pub(crate) pad_8d0: Pad<0x14>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_8e4_0: Array<0x10, u8>,

        pub(crate) unk_8e4: u32,
        pub(crate) unk_8e8: u32,
        pub(crate) max_pstate_scaled_10: u32,
        pub(crate) unk_8f0: u32,
        pub(crate) unk_8f4: u32,
        pub(crate) pad_8f8: u32,
        pub(crate) pad_8fc: u32,
        pub(crate) unk_900: Array<0x24, u8>,

        pub(crate) unk_coef_a1: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,
        pub(crate) unk_coef_a2: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,

        pub(crate) pad_b24: Pad<0x70>,
        pub(crate) max_pstate_scaled_11: u32,
        pub(crate) freq_with_off: u32,
        pub(crate) unk_b9c: u32,
        pub(crate) unk_ba0: U64,
        pub(crate) unk_ba8: U64,
        pub(crate) unk_bb0: u32,
        pub(crate) unk_bb4: u32,

        #[ver(V >= V13_3)]
        pub(crate) pad_bb8_0: Pad<0x200>,
        #[ver(V >= V13_5)]
        pub(crate) pad_bb8_200: Pad<0x8>,

        pub(crate) pad_bb8: Pad<0x74>,
        pub(crate) unk_c2c: u32,
        pub(crate) power_zone_count: u32,
        pub(crate) max_power_4: u32,
        pub(crate) max_power_5: u32,
        pub(crate) max_power_6: u32,
        pub(crate) unk_c40: u32,
        pub(crate) unk_c44: F32,
        pub(crate) avg_power_target_filter_a_neg: F32,
        pub(crate) avg_power_target_filter_a: F32,
        pub(crate) avg_power_target_filter_tc_x4: u32,
        pub(crate) avg_power_target_filter_tc_xperiod: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) avg_power_target_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_c58_4: u32,

        pub(crate) power_zones: Array<5, PowerZone::ver>,
        pub(crate) avg_power_filter_tc_periods_x4: u32,
        pub(crate) unk_cfc: u32,
        pub(crate) unk_d00: u32,
        pub(crate) avg_power_filter_a_neg: F32,
        pub(crate) unk_d08: u32,
        pub(crate) avg_power_filter_a: F32,
        pub(crate) unk_d10: u32,
        pub(crate) avg_power_ki_dt: F32,
        pub(crate) unk_d18: u32,
        pub(crate) unk_d1c: u32,
        pub(crate) unk_d20: F32,
        pub(crate) avg_power_kp: F32,
        pub(crate) unk_d28: u32,
        pub(crate) unk_d2c: u32,
        pub(crate) avg_power_min_duty_cycle: u32,
        pub(crate) max_pstate_scaled_12: u32,
        pub(crate) max_pstate_scaled_13: u32,
        pub(crate) unk_d3c: u32,
        pub(crate) max_power_7: F32,
        pub(crate) max_power_8: u32,
        pub(crate) unk_d48: u32,
        pub(crate) avg_power_filter_tc_ms: u32,
        pub(crate) unk_d50: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) avg_power_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_d54_4: Array<0xc, u8>,

        pub(crate) unk_d54: Array<0x10, u8>,
        pub(crate) max_pstate_scaled_14: u32,
        pub(crate) unk_d68: Array<0x24, u8>,

        pub(crate) t81xx_data: T81xxData,

        pub(crate) unk_dd0: Array<0x40, u8>,

        #[ver(V >= V13_2)]
        pub(crate) unk_e10_pad: Array<0x10, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_e10_0: HwDataA130Extra,

        pub(crate) unk_e10: Array<0xc, u8>,

        pub(crate) fast_die0_sensor_mask_2: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask_2: U64,

        pub(crate) unk_e24: u32,
        pub(crate) unk_e28: u32,
        pub(crate) unk_e2c: Pad<0x1c>,
        pub(crate) unk_coef_b1: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,
        pub(crate) unk_coef_b2: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,

        #[ver(G >= G14X)]
        pub(crate) pad_1048_0: Pad<0x600>,

        pub(crate) pad_1048: Pad<0x5e4>,

        pub(crate) fast_die0_sensor_mask_alt: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask_alt: U64,
        #[ver(V < V13_0B4)]
        pub(crate) fast_die0_sensor_present: U64,

        pub(crate) unk_163c: u32,

        pub(crate) unk_1640: Array<0x2000, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_3640_0: Array<0x2000, u8>,

        pub(crate) unk_3640: u32,
        pub(crate) unk_3644: u32,

        #[ver(G != G15)]
        pub(crate) hws1: HwDataShared1,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_hws2: Array<16, u16>,

        #[ver(G != G15)]
        pub(crate) hws2: HwDataShared2,
        #[ver(G != G15)]
        pub(crate) unk_3c00: u32,
        #[ver(G != G15)]
        pub(crate) unk_3c04: u32,
        #[ver(G != G15)]
        pub(crate) hws3: HwDataShared3,
        #[ver(G != G15)]
        pub(crate) unk_3c58: Array<0x3c, u8>,
        #[ver(G != G15)]
        pub(crate) unk_3c94: u32,
        #[ver(G != G15)]
        pub(crate) unk_3c98: U64,
        #[ver(G != G15)]
        pub(crate) unk_3ca0: U64,
        #[ver(G != G15)]
        pub(crate) unk_3ca8: U64,
        #[ver(G != G15)]
        pub(crate) unk_3cb0: U64,
        #[ver(G != G15)]
        pub(crate) ts_last_idle: U64,
        #[ver(G != G15)]
        pub(crate) ts_last_poweron: U64,
        #[ver(G != G15)]
        pub(crate) ts_last_poweroff: U64,
        #[ver(G != G15)]
        pub(crate) unk_3cd0: U64,
        #[ver(G != G15)]
        pub(crate) unk_3cd8: U64,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_3ce0_0: u32,

        #[ver(G != G15)]
        pub(crate) unk_3ce0: u32,
        #[ver(G != G15)]
        pub(crate) unk_3ce4: u32,
        #[ver(G != G15)]
        pub(crate) unk_3ce8: u32,
        #[ver(G != G15)]
        pub(crate) unk_3cec: u32,
        #[ver(G != G15)]
        pub(crate) unk_3cf0: u32,
        #[ver(G != G15)]
        pub(crate) core_leak_coef: Array<8, F32>,
        #[ver(G != G15)]
        pub(crate) sram_leak_coef: Array<8, F32>,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) aux_leak_coef: AuxLeakCoef,
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_3d34_0: Array<0x18, u8>,

        #[ver(G != G15)]
        pub(crate) unk_3d34: Array<0x38, u8>,

        #[ver(G == G15)]
        pub(crate) g15_pretail_3a9c: G15HwDataAPreTail,
        #[ver(G == G15)]
        pub(crate) g15_tail_421c: G15HwDataATail,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataA::ver);
    #[versions(AGX)]
    no_debug!(HwDataA::ver);

    const _: [(); 0x4360] = [(); core::mem::size_of::<HwDataAG15V14_7>()];
    // J615 G15 MTR state lives inside the opaque +0x1640 block. Keep the
    // absolute producer/consumer offsets mechanically tied to this layout.
    const _: [(); 0x1a94] = [(); core::mem::offset_of!(HwDataAG15V14_7, unk_1640)];
    const _: [(); 0x1a98] =
        [(); core::mem::offset_of!(HwDataAG15V14_7, unk_1640) + 0x04];
    const _: [(); 0x1aa4] =
        [(); core::mem::offset_of!(HwDataAG15V14_7, unk_1640) + 0x10];
    const _: [(); 0x3a94] = [(); core::mem::offset_of!(HwDataAG15V14_7, unk_3640)];
    const _: [(); 0x3a9c] = [(); core::mem::offset_of!(HwDataAG15V14_7, g15_pretail_3a9c)];
    const _: [(); 0x421c] = [(); core::mem::offset_of!(HwDataAG15V14_7, g15_tail_421c)];

    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct IOMapping {
        pub(crate) phys_addr: U64,
        pub(crate) virt_addr: U64,
        pub(crate) total_size: u32,
        pub(crate) element_size: u32,
        pub(crate) readwrite: U64,
    }

    #[versions(AGX)]
    const IO_MAPPING_COUNT: usize = {
        #[ver(V < V13_0B4)]
        {
            0x14
        }
        #[ver(V >= V13_0B4 && V < V13_3)]
        {
            0x17
        }
        #[ver(V >= V13_3 && V < V13_5)]
        {
            0x18
        }
        #[ver(V >= V13_5 && G != G15)]
        {
            0x19
        }
        #[ver(G == G15)]
        {
            0x1f
        }
    };

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataBAuxPStates {
        pub(crate) cs_max_pstate: u32,
        pub(crate) cs_frequencies: Array<0x10, u32>,
        pub(crate) cs_voltages: Array<0x10, Array<0x2, u32>>,
        pub(crate) cs_voltages_sram: Array<0x10, Array<0x2, u32>>,
        pub(crate) cs_unkpad: u32,
        pub(crate) afr_max_pstate: u32,
        pub(crate) afr_frequencies: Array<0x8, u32>,
        pub(crate) afr_voltages: Array<0x8, Array<0x2, u32>>,
        pub(crate) afr_voltages_sram: Array<0x8, Array<0x2, u32>>,
        pub(crate) afr_unkpad: u32,
    }

    /// Exact final J615/G15 HwDataB startup block written by Apple's
    /// `AGXArmFirmware::initFirmwareData()` immediately before `bootFirmware()`.
    ///
    /// This starts at +0x17ec because Apple performs an 8-byte zero store there,
    /// spanning the first firmware-imported dword at +0x17f0. Keeping the whole
    /// suffix typed prevents legacy V13.x defaults from leaking into G15.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15HwDataBStartup {
        pub(crate) zero_17ec: Array<2, u32>, // +0x17ec/+0x17f0
        pub(crate) flag_17f4: u32,           // +0x17f4 = feature bit 36
        pub(crate) one_17f8: u32,            // +0x17f8 = 1
        pub(crate) one_17fc: u32,            // +0x17fc = 1
        pub(crate) flag_1800: u32,           // +0x1800 = feature bit 37
        pub(crate) zero_1804: u32,           // +0x1804 = 0 on J615
        pub(crate) one_1808: u32,            // +0x1808 = low dword of 0x1_00000001
        pub(crate) one_180c: u32,            // +0x180c = high dword of 0x1_00000001
        pub(crate) flag_1810: u32,           // +0x1810 = feature bit 7
        pub(crate) zero_1814: u32,           // +0x1814 = 0 in final init pass
        pub(crate) sentinels_1818: Array<12, u32>, // +0x1818..+0x1847 = 0xffffffff
        pub(crate) zero_1848: u32,           // +0x1848 = 0
        pub(crate) zero_184c: Array<2, u32>, // +0x184c..+0x1853 = 0
        pub(crate) zero_1854: u32,           // +0x1854 = 0
        pub(crate) flag_1858: u32,           // +0x1858 = 1 on exact G15G path
        pub(crate) zero_185c: u32,           // +0x185c = allocation zero
    }
    default_zeroed!(G15HwDataBStartup);
    const _: [(); 0x74] = [(); core::mem::size_of::<G15HwDataBStartup>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(G15HwDataBStartup, zero_17ec)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15HwDataBStartup, flag_17f4)];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15HwDataBStartup, flag_1800)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(G15HwDataBStartup, one_1808)];
    const _: [(); 0x24] = [(); core::mem::offset_of!(G15HwDataBStartup, flag_1810)];
    const _: [(); 0x2c] = [(); core::mem::offset_of!(G15HwDataBStartup, sentinels_1818)];
    const _: [(); 0x5c] = [(); core::mem::offset_of!(G15HwDataBStartup, zero_1848)];
    const _: [(); 0x6c] = [(); core::mem::offset_of!(G15HwDataBStartup, flag_1858)];

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataB {
        #[ver(V < V13_0B4)]
        pub(crate) unk_0: U64,

        pub(crate) unk_8: U64,

        #[ver(V < V13_0B4)]
        pub(crate) unk_10: U64,

        pub(crate) unk_18: U64,
        pub(crate) unk_20: U64,
        pub(crate) unk_28: U64,
        pub(crate) unk_30: U64,
        pub(crate) timestamp_area_base: U64,
        pub(crate) pad_40: Pad<0x20>,

        #[ver(V < V13_0B4)]
        pub(crate) yuv_matrices: Array<0xf, Array<3, Array<4, i16>>>,

        #[ver(V >= V13_0B4)]
        pub(crate) yuv_matrices: Array<0x3f, Array<3, Array<4, i16>>>,

        pub(crate) pad_1c8: Pad<0x8>,
        pub(crate) io_mappings: Array<IO_MAPPING_COUNT::ver, IOMapping>,

        #[ver(V >= V13_0B4)]
        pub(crate) sgx_sram_ptr: U64,

        pub(crate) chip_id: u32,
        pub(crate) unk_454: u32,
        pub(crate) unk_458: u32,
        pub(crate) unk_45c: u32,
        pub(crate) unk_460: u32,
        pub(crate) unk_464: u32,
        pub(crate) unk_468: u32,
        pub(crate) unk_46c: u32,
        pub(crate) unk_470: u32,
        pub(crate) unk_474: u32,
        pub(crate) unk_478: u32,
        pub(crate) unk_47c: u32,
        pub(crate) unk_480: u32,
        pub(crate) unk_484: u32,
        pub(crate) unk_488: u32,
        pub(crate) unk_48c: u32,
        pub(crate) base_clock_khz: u32,
        pub(crate) power_sample_period: u32,
        pub(crate) pad_498: Pad<0x4>,
        pub(crate) unk_49c: u32,
        pub(crate) unk_4a0: u32,
        pub(crate) unk_4a4: u32,
        pub(crate) pad_4a8: Pad<0x4>,
        pub(crate) unk_4ac: u32,
        pub(crate) pad_4b0: Pad<0x8>,
        pub(crate) unk_4b8: u32,
        pub(crate) unk_4bc: Array<0x4, u8>,
        pub(crate) unk_4c0: u32,
        pub(crate) unk_4c4: u32,
        pub(crate) unk_4c8: u32,
        pub(crate) unk_4cc: u32,
        pub(crate) unk_4d0: u32,
        pub(crate) unk_4d4: u32,
        pub(crate) unk_4d8: Array<0x4, u8>,
        pub(crate) unk_4dc: u32,
        pub(crate) unk_4e0: U64,
        pub(crate) unk_4e8: u32,
        pub(crate) unk_4ec: u32,
        pub(crate) unk_4f0: u32,
        pub(crate) unk_4f4: u32,
        pub(crate) unk_4f8: u32,
        pub(crate) unk_4fc: u32,
        pub(crate) unk_500: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_504_0: u32,

        pub(crate) unk_504: u32,
        pub(crate) unk_508: u32,
        pub(crate) unk_50c: u32,
        pub(crate) unk_510: u32,
        pub(crate) unk_514: u32,
        pub(crate) unk_518: u32,
        pub(crate) unk_51c: u32,
        pub(crate) unk_520: u32,
        pub(crate) unk_524: u32,
        pub(crate) unk_528: u32,
        pub(crate) unk_52c: u32,
        pub(crate) unk_530: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_534_0: u32,

        pub(crate) unk_534: u32,
        pub(crate) unk_538: u32,

        pub(crate) num_frags: u32,
        pub(crate) unk_540: u32,
        pub(crate) unk_544: u32,
        pub(crate) unk_548: u32,
        pub(crate) unk_54c: u32,
        pub(crate) unk_550: u32,
        pub(crate) unk_554: u32,

        // G15 inserts a 12-byte UAT/GPTBAT header ahead of the legacy
        // uat_ttb_base/core tuple. Apple zeros +0xb38/+0xb3c, marks the
        // non-legacy G15 UAT mode at +0xb40, then publishes the GPTBAT
        // physical base at +0xb44. Keeping uat_ttb_base as the shared field
        // name lets the builder use the same physical TTB/GPTBAT allocation.
        #[ver(G == G15)]
        pub(crate) g15_zero_b38: u32,
        #[ver(G == G15)]
        pub(crate) g15_zero_b3c: u32,
        #[ver(G == G15)]
        pub(crate) g15_uat_mode_b40: u32,

        pub(crate) uat_ttb_base: U64,
        pub(crate) gpu_core_id: u32,
        pub(crate) gpu_rev_id: u32,
        pub(crate) num_cores: u32,
        pub(crate) max_pstate: u32,

        #[ver(V < V13_0B4)]
        pub(crate) num_pstates: u32,

        pub(crate) frequencies: Array<0x10, u32>,
        pub(crate) voltages: Array<0x10, [u32; 0x8]>,
        pub(crate) voltages_sram: Array<0x10, [u32; 0x8]>,

        #[ver(V >= V13_3)]
        pub(crate) unk_9f4_0: Pad<64>,

        pub(crate) sram_k: Array<0x10, F32>,
        pub(crate) unk_9f4: Array<0x10, u32>,
        pub(crate) rel_max_powers: Array<0x10, u32>,
        pub(crate) rel_boost_freqs: Array<0x10, u32>,

        #[ver(V >= V13_3)]
        pub(crate) unk_arr_0: Array<32, u32>,

        #[ver(V < V13_0B4)]
        pub(crate) min_sram_volt: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_ab8: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_abc: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_ac0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) aux_ps: HwDataBAuxPStates,

        #[ver(V >= V13_3 && G != G15)]
        pub(crate) pad_ac4_0: Array<0x44c, u8>,
        // The G15 perf-state copy starts at exact +0x134c and is 0x448 bytes.
        // The inherited opaque span therefore loses the same 0xc bytes that
        // G15 inserted before the perf arrays, keeping the proven suffix fixed.
        #[ver(V >= V13_3 && G == G15)]
        pub(crate) pad_ac4_0: Array<0x440, u8>,

        pub(crate) pad_ac4: Pad<0x8>,
        pub(crate) unk_acc: u32,
        pub(crate) unk_ad0: u32,
        pub(crate) pad_ad4: Pad<0x10>,
        pub(crate) unk_ae4: Array<0x4, u32>,
        pub(crate) pad_af4: Pad<0x4>,
        pub(crate) unk_af8: u32,
        pub(crate) pad_afc: Pad<0x8>,
        pub(crate) unk_b04: u32,
        pub(crate) unk_b08: u32,
        pub(crate) unk_b0c: u32,

        #[ver(G >= G14X)]
        pub(crate) pad_b10_0: Array<0x8, u8>,

        pub(crate) unk_b10: u32,
        pub(crate) timer_offset: U64,
        pub(crate) unk_b1c: u32,
        pub(crate) unk_b20: u32,
        #[ver(G != G15)]
        pub(crate) unk_b24: u32,
        #[ver(G != G15)]
        pub(crate) unk_b28: u32,
        #[ver(G != G15)]
        pub(crate) unk_b2c: u32,
        #[ver(G != G15)]
        pub(crate) unk_b30: u32,
        #[ver(G != G15)]
        pub(crate) unk_b34: u32,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_b38_0: u32,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_b38_4: u32,

        #[ver(V >= V13_3 && G != G15)]
        pub(crate) unk_b38_8: u32,

        #[ver(G != G15)]
        pub(crate) unk_b38: Array<0xc, u32>,
        #[ver(G != G15)]
        pub(crate) unk_b68: u32,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_b6c: Array<0xd0, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_c3c_0: Array<0x8, u8>,

        #[ver(G < G14X && G != G15 && V >= V13_5)]
        pub(crate) unk_c3c_8: Array<0x10, u8>,

        #[ver(V >= V13_5 && G != G15)]
        pub(crate) unk_c3c_18: Array<0x20, u8>,

        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) unk_c3c: u32,

        // G15 replaces the legacy suffix beginning at exact +0x17ec with the
        // final Apple startup image. The block runs to the exact 0x1860 end.
        #[ver(G == G15)]
        pub(crate) g15_startup_17ec: G15HwDataBStartup,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataB::ver);

    /// G15 PB descriptor-table entry. Firmware indexes this table with an
    /// 8-bit ID and uses 0x10-byte records; the Apple mapping is exactly
    /// 0x1000 bytes, i.e. 256 records. Field semantics remain incremental.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15PBDescriptor {
        pub(crate) words: Array<4, U32>,
    }
    const _: [(); 0x10] = [(); core::mem::size_of::<G15PBDescriptor>()];

    /// G15 UMA page-pool descriptor-table entry. Firmware indexes this table
    /// with an 8-bit ID and uses 0x20-byte records; the Apple mapping is exactly
    /// 0x2000 bytes, i.e. 256 records.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15UMAPagePoolDescriptor {
        // RTKit FUN_fffffc0000037b48 packs Page Pool State into this record:
        //   q0 = (state[+0x1c] << 41) | (state[+0x14] >> 7)
        //   q1 = (state[+0x24] << 33) | (state[+0x20] << 5)
        //        | (state[+0x28] << 61)
        //   q2 = state[+0x2c]
        // FUN_fffffc0000038884 performs the inverse extraction.
        pub(crate) page_pool_list: U64, // +0x00: FWVA>>7 plus capacity<<41
        pub(crate) dynamic_state: U64,  // +0x08: three packed state fields
        pub(crate) page_count: U64,      // +0x10: low 22 bits mirror Page Pool State page_count
        pub(crate) unk_18: U64,         // +0x18: not touched by recovered RTKit table users
    }
    const _: [(); 0x20] = [(); core::mem::size_of::<G15UMAPagePoolDescriptor>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(G15UMAPagePoolDescriptor, page_pool_list)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15UMAPagePoolDescriptor, dynamic_state)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15UMAPagePoolDescriptor, page_count)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(G15UMAPagePoolDescriptor, unk_18)];

    /// One AGFA firmware-init sequence record. The firmware parser advances
    /// in exact 0x18-byte steps and terminates when `kind == 0`.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15InitSequenceEntry {
        pub(crate) value: U64,           // +0x00
        pub(crate) register_offset: U32, // +0x08
        pub(crate) shift: U32,           // +0x0c
        pub(crate) kind: U32,            // +0x10: 0=end, 1=u32, 2=u64, 3=u64>>shift
        pub(crate) reserved: U32,        // +0x14
    }
    const _: [(); 0x18] = [(); core::mem::size_of::<G15InitSequenceEntry>()];
    const _: [(); 0x08] = [(); core::mem::offset_of!(G15InitSequenceEntry, register_offset)];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15InitSequenceEntry, shift)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15InitSequenceEntry, kind)];

    /// Exact one-page G15/G15G AGFA init-sequence backing. Apple's G15 and
    /// G15G populateInitSequenceFirmware() overrides are BTI+RET stubs, so the
    /// first record is the type-0 terminator and the rest of the page is zero.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15InitSequencePage {
        pub(crate) terminator: G15InitSequenceEntry, // +0x0000
        pub(crate) unused: Pad<0x3fe8>,              // +0x0018..+0x3fff
    }
    default_zeroed!(G15InitSequencePage);
    const _: [(); 0x4000] = [(); core::mem::size_of::<G15InitSequencePage>()];
    const _: [(); 0x0000] = [(); core::mem::offset_of!(G15InitSequencePage, terminator)];
    const _: [(); 0x0018] = [(); core::mem::offset_of!(G15InitSequencePage, unused)];

    /// G15 persistent firmware time/activity snapshot referenced by wrapper
    /// +0x010. The exact Apple allocation is 0x88 bytes. Firmware treats +0x00
    /// as a first-use marker and saves/restores sixteen deliberately unaligned
    /// qwords at +0x04 + 8*n across sleep/restart transitions.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15FirmwareTimeState {
        pub(crate) initialized: AtomicU32, // +0x00, FW changes 0 -> 1
        pub(crate) values: Array<16, U64>, // +0x04..+0x83, intentionally unaligned
        pub(crate) tail_84: U32,           // +0x84, semantic unresolved
    }
    default_zeroed!(G15FirmwareTimeState);
    const _: [(); 0x88] = [(); core::mem::size_of::<G15FirmwareTimeState>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(G15FirmwareTimeState, values)];
    const _: [(); 0x84] = [(); core::mem::offset_of!(G15FirmwareTimeState, tail_84)];

    /// Exact 0x60-byte G15 firmware control/state block referenced by wrapper
    /// +0x24c. Firmware directly mutates every named field below. The three
    /// counters at +0x0c are walked as a contiguous array; the qwords at
    /// +0x1c/+0x2c/+0x4c are deliberately unaligned in Apple's ABI.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15ControlState {
        pub(crate) state: u32,                    // +0x00: compared against state 2
        pub(crate) request_pending: u32,          // +0x04: set by FW control message
        pub(crate) request_latched: u32,          // +0x08: mirrors request_pending != 0
        pub(crate) counters: Array<3, u32>,       // +0x0c..+0x17
        pub(crate) aggregate_count: u32,          // +0x18
        pub(crate) active_timestamp: U64,         // +0x1c, unaligned
        pub(crate) active: u32,                   // +0x24
        pub(crate) active_id: u32,                // +0x28
        pub(crate) secondary_timestamp: U64,      // +0x2c, unaligned
        pub(crate) event_count: u32,              // +0x34
        pub(crate) pad_038: Pad<0x14>,
        pub(crate) timestamp_ns: U64,             // +0x4c, 24-MHz ticks * 125 / 3
        pub(crate) pad_054: Pad<0x0c>,
    }
    default_zeroed!(G15ControlState);
    const _: [(); 0x60] = [(); core::mem::size_of::<G15ControlState>()];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15ControlState, counters)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(G15ControlState, active_timestamp)];
    const _: [(); 0x2c] = [(); core::mem::offset_of!(G15ControlState, secondary_timestamp)];
    const _: [(); 0x4c] = [(); core::mem::offset_of!(G15ControlState, timestamp_ns)];

    /// G15 HWDS-ID firmware counter entry. The Apple host allocates exactly
    /// 0x800 bytes and firmware indexes it with an 8-bit ID using 8-byte
    /// records, proving 256 entries. Both words are firmware-mutated counters.
    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct G15HWDSCounterEntry {
        pub(crate) word_0: AtomicU32,
        pub(crate) word_4: AtomicU32,
    }
    const _: [(); 0x08] = [(); core::mem::size_of::<G15HWDSCounterEntry>()];

    /// Firmware-owned G15 runtime accounting state embedded in the compact
    /// wrapper at +0x2d8. The observed host init methods do not populate this
    /// region; early firmware binds it as a persistent state block, clears the
    /// ranges below, sets +0x2fc to 1 and writes the +0x314 magic marker.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15RuntimeState {
        pub(crate) state_2d8: AtomicU32,             // +0x00 / wrapper +0x2d8
        pub(crate) zero_2dc_2fb: Pad<0x20>,          // +0x04
        pub(crate) fw_initialized_2fc: AtomicU32,    // +0x24, FW sets to 1
        pub(crate) active_mask_300: U64,             // +0x28
        pub(crate) active_mask_308: U64,             // +0x30
        pub(crate) state_310: AtomicU32,             // +0x38
        pub(crate) magic_314: U32,                   // +0x3c, FW sets 0xabcdabcd
        pub(crate) active_318: AtomicU32,            // +0x40
        pub(crate) active_31c: AtomicU32,            // +0x44
        pub(crate) active_320: AtomicU32,            // +0x48
        pub(crate) counts_324: Array<36, u16>,       // +0x4c, exact 0x48 bytes
        pub(crate) active_36c: AtomicU32,            // +0x94
        pub(crate) opaque_370: Pad<0x08>,            // +0x98
        pub(crate) zero_378: U32,                    // +0xa0
        pub(crate) zero_37c: U64,                    // +0xa4 (unaligned)
        pub(crate) zero_384: U64,                    // +0xac
        pub(crate) zero_38c: U64,                    // +0xb4
        pub(crate) zero_394: U64,                    // +0xbc
        pub(crate) zero_39c: U64,                    // +0xc4
        pub(crate) zero_3a4: U64,                    // +0xcc
        pub(crate) opaque_3ac: Pad<0x04>,            // +0xd4
    }
    default_zeroed!(G15RuntimeState);
    const _: [(); 0xd8] = [(); core::mem::size_of::<G15RuntimeState>()];
    const _: [(); 0x24] = [(); core::mem::offset_of!(G15RuntimeState, fw_initialized_2fc)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(G15RuntimeState, active_mask_300)];
    const _: [(); 0x38] = [(); core::mem::offset_of!(G15RuntimeState, state_310)];
    const _: [(); 0x3c] = [(); core::mem::offset_of!(G15RuntimeState, magic_314)];
    const _: [(); 0x40] = [(); core::mem::offset_of!(G15RuntimeState, active_318)];
    const _: [(); 0x4c] = [(); core::mem::offset_of!(G15RuntimeState, counts_324)];
    const _: [(); 0x94] = [(); core::mem::offset_of!(G15RuntimeState, active_36c)];
    const _: [(); 0xa0] = [(); core::mem::offset_of!(G15RuntimeState, zero_378)];
    const _: [(); 0xd4] = [(); core::mem::offset_of!(G15RuntimeState, opaque_3ac)];

    /// G15 wrapper tail at +0x459. The whole wrapper allocation is explicitly
    /// zero-filled by AGXFirmware::allocateSharedData(); AGXArmFirmware::init()
    /// additionally re-clears the four deliberately unaligned qwords at
    /// +0x46d/+0x475/+0x47d/+0x485. Semantics remain unresolved, but every
    /// byte in this tail is exactly zero at bootstrap.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15WrapperTail {
        pub(crate) zero_459: Pad<0x14>,
        pub(crate) zero_46d: U64,
        pub(crate) zero_475: U64,
        pub(crate) zero_47d: U64,
        pub(crate) zero_485: U64,
        pub(crate) zero_48d: Pad<0x03>,
    }
    default_zeroed!(G15WrapperTail);
    const _: [(); 0x37] = [(); core::mem::size_of::<G15WrapperTail>()];
    const _: [(); 0x14] = [(); core::mem::offset_of!(G15WrapperTail, zero_46d)];
    const _: [(); 0x1c] = [(); core::mem::offset_of!(G15WrapperTail, zero_475)];
    const _: [(); 0x24] = [(); core::mem::offset_of!(G15WrapperTail, zero_47d)];
    const _: [(); 0x2c] = [(); core::mem::offset_of!(G15WrapperTail, zero_485)];
    const _: [(); 0x00] = [(); core::mem::offset_of!(G15WrapperTail, zero_459)];
    const _: [(); 0x34] = [(); core::mem::offset_of!(G15WrapperTail, zero_48d)];

    /// Exact G15 global statistics backing allocations referenced by wrapper
    /// +0x234/+0x23c/+0x244. They remain separate from the legacy Stats owner
    /// until the G15 render/compute command statistics ABI is reconstructed.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15StatsVtx {
        pub(crate) opaque: Array<0xc10, u8>,
    }
    default_zeroed!(G15StatsVtx);
    const _: [(); 0xc10] = [(); core::mem::size_of::<G15StatsVtx>()];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15StatsFrag {
        pub(crate) pad_000: Pad<0xc18>,
        pub(crate) cur_stamp_id: i32, // +0xc18, Apple initializes to -1
        pub(crate) pad_c1c: Pad<0x14>,
        pub(crate) unk_id: i32,       // +0xc30, Apple initializes to -1
        pub(crate) pad_c34: Pad<0x614>,
    }
    const _: [(); 0x1248] = [(); core::mem::size_of::<G15StatsFrag>()];
    const _: [(); 0xc18] = [(); core::mem::offset_of!(G15StatsFrag, cur_stamp_id)];
    const _: [(); 0xc30] = [(); core::mem::offset_of!(G15StatsFrag, unk_id)];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15StatsComp {
        pub(crate) opaque: Array<0xe10, u8>,
    }
    default_zeroed!(G15StatsComp);
    const _: [(); 0xe10] = [(); core::mem::size_of::<G15StatsComp>()];

    #[derive(Debug)]
    #[repr(C, packed)]
    pub(crate) struct GpuStatsVtx {
        // This changes all the time and we don't use it, let's just make it a big buffer
        pub(crate) opaque: Array<0x3000, u8>,
    }
    default_zeroed!(GpuStatsVtx);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuStatsFrag {
        // This changes all the time and we don't use it, let's just make it a big buffer
        // except for these two fields which may need init.
        #[ver(G >= G14X)]
        pub(crate) unk1_0: Array<0x910, u8>,
        pub(crate) unk1: Array<0x100, u8>,
        pub(crate) cur_stamp_id: i32,
        pub(crate) unk2: Array<0x14, u8>,
        pub(crate) unk_id: i32,
        pub(crate) unk3: Array<0x1000, u8>,
    }

    #[versions(AGX)]
    impl Default for GpuStatsFrag::ver {
        fn default() -> Self {
            Self {
                #[ver(G >= G14X)]
                unk1_0: Default::default(),
                unk1: Default::default(),
                cur_stamp_id: -1,
                unk2: Default::default(),
                unk_id: -1,
                unk3: Default::default(),
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuGlobalStatsVtx {
        pub(crate) total_cmds: u32,
        pub(crate) stats: GpuStatsVtx,
    }
    default_zeroed!(GpuGlobalStatsVtx);

    #[versions(AGX)]
    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct GpuGlobalStatsFrag {
        pub(crate) total_cmds: u32,
        pub(crate) unk_4: u32,
        pub(crate) stats: GpuStatsFrag::ver,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuStatsComp {
        // This changes all the time and we don't use it, let's just make it a big buffer
        pub(crate) opaque: Array<0x3000, u8>,
    }
    default_zeroed!(GpuStatsComp);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RuntimeScratch {
        pub(crate) unk_280: Array<0x6800, u8>,
        pub(crate) unk_6a80: u32,
        pub(crate) gpu_idle: u32,
        pub(crate) unkpad_6a88: Pad<0x14>,
        pub(crate) unk_6a9c: u32,
        pub(crate) unk_ctr0: u32,
        pub(crate) unk_ctr1: u32,
        pub(crate) unk_6aa8: u32,
        pub(crate) unk_6aac: u32,
        pub(crate) unk_ctr2: u32,
        pub(crate) unk_6ab4: u32,
        pub(crate) unk_6ab8: u32,
        pub(crate) unk_6abc: u32,
        pub(crate) unk_6ac0: u32,
        pub(crate) unk_6ac4: u32,
        pub(crate) unk_ctr3: u32,
        pub(crate) unk_6acc: u32,
        pub(crate) unk_6ad0: u32,
        pub(crate) unk_6ad4: u32,
        pub(crate) unk_6ad8: u32,
        pub(crate) unk_6adc: u32,
        pub(crate) unk_6ae0: u32,
        pub(crate) unk_6ae4: u32,
        pub(crate) unk_6ae8: u32,
        pub(crate) unk_6aec: u32,
        pub(crate) unk_6af0: u32,
        pub(crate) unk_ctr4: u32,
        pub(crate) unk_ctr5: u32,
        pub(crate) unk_6afc: u32,
        pub(crate) pad_6b00: Pad<0x38>,

        #[ver(G >= G14X)]
        pub(crate) pad_6b00_extra: Array<0x4800, u8>,

        pub(crate) unk_6b38: u32,
        pub(crate) pad_6b3c: Pad<0x84>,
    }
    #[versions(AGX)]
    default_zeroed!(RuntimeScratch::ver);

    /// G15 transmit channels split the producer/consumer state across three
    /// firmware pointers and keep the ring buffer as the fourth pointer.
    /// Firmware consumes these as four consecutive qwords.
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15TxChannelRing {
        pub(crate) read_ptr: U64,
        pub(crate) write_ptr_shadow: U64,
        pub(crate) write_ptr: U64,
        pub(crate) ring: U64,
    }
    default_zeroed!(G15TxChannelRing);

    /// One G15 pipe has vertex/fragment/compute TX channels, 0x20 bytes each.
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15PipeChannels {
        pub(crate) vtx: G15TxChannelRing,
        pub(crate) frag: G15TxChannelRing,
        pub(crate) comp: G15TxChannelRing,
    }
    default_zeroed!(G15PipeChannels);

    const _: [(); 0x20] = [(); core::mem::size_of::<G15TxChannelRing>()];
    const _: [(); 0x60] = [(); core::mem::size_of::<G15PipeChannels>()];

    #[versions(AGX)]
    #[repr(C)]
    pub(crate) struct RuntimePointers<'a> {
        // Legacy RegionB / RuntimePointers layout through G14X.
        #[ver(G != G15)]
        pub(crate) pipes: Array<4, PipeChannels::ver>,

        #[ver(G != G15)]
        pub(crate) device_control:
            ChannelRing<channels::ChannelState, channels::DeviceControlMsg::ver>,
        #[ver(G != G15)]
        pub(crate) event: ChannelRing<channels::ChannelState, channels::RawEventMsg>,
        #[ver(G != G15)]
        pub(crate) fw_log: ChannelRing<channels::FwLogChannelState, channels::RawFwLogMsg>,
        #[ver(G != G15)]
        pub(crate) ktrace: ChannelRing<channels::ChannelState, channels::RawKTraceMsg>,
        #[ver(G != G15)]
        pub(crate) stats: ChannelRing<channels::ChannelState, channels::RawStatsMsg::ver>,

        #[ver(G != G15)]
        pub(crate) __pad0: Pad<0x50>,
        #[ver(G != G15)]
        pub(crate) unk_160: U64,
        #[ver(G != G15)]
        pub(crate) unk_168: U64,
        #[ver(G != G15)]
        pub(crate) stats_vtx: GpuPointer<'a, super::GpuGlobalStatsVtx>,
        #[ver(G != G15)]
        pub(crate) stats_frag: GpuPointer<'a, super::GpuGlobalStatsFrag::ver>,
        #[ver(G != G15)]
        pub(crate) stats_comp: GpuPointer<'a, super::GpuStatsComp>,
        #[ver(G != G15)]
        pub(crate) hwdata_a: GpuPointer<'a, super::HwDataA::ver>,
        #[ver(G != G15)]
        pub(crate) unkptr_190: GpuPointer<'a, &'a [u8]>,
        #[ver(G != G15)]
        pub(crate) unkptr_198: GpuPointer<'a, &'a [u8]>,
        #[ver(G != G15)]
        pub(crate) hwdata_b: GpuPointer<'a, super::HwDataB::ver>,
        #[ver(G != G15)]
        pub(crate) hwdata_b_2: GpuPointer<'a, super::HwDataB::ver>,
        #[ver(G != G15)]
        pub(crate) fwlog_buf: Option<GpuWeakPointer<[channels::RawFwLogPayloadMsg]>>,
        #[ver(G != G15)]
        pub(crate) unkptr_1b8: GpuPointer<'a, &'a [u8]>,

        #[ver(G < G14X && G != G15)]
        pub(crate) unkptr_1c0: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14X && G != G15)]
        pub(crate) unkptr_1c8: GpuPointer<'a, &'a [u8]>,

        #[ver(G != G15)]
        pub(crate) unk_1d0: u32,
        #[ver(G != G15)]
        pub(crate) unk_1d4: u32,
        #[ver(G != G15)]
        pub(crate) unk_1d8: Array<0x3c, u8>,
        #[ver(G != G15)]
        pub(crate) buffer_mgr_ctl_gpu_addr: U64,
        #[ver(G != G15)]
        pub(crate) buffer_mgr_ctl_fw_addr: U64,
        #[ver(G != G15)]
        pub(crate) __pad1: Pad<0x5c>,
        #[ver(G != G15)]
        pub(crate) gpu_scratch: RuntimeScratch::ver,

        // G15 replaces the large inline RegionB object with a compact 0x490-byte
        // wrapper. Offsets below are reconstructed from the exact AGXG15G host
        // driver and RTKit-2419.140.12 firmware. Unknown backing objects remain
        // opaque until their host allocation semantics are fully reconstructed.
        #[ver(G == G15)]
        pub(crate) hwdata_b: GpuPointer<'a, super::HwDataB::ver>, // +0x000
        #[ver(G == G15)]
        pub(crate) g15_fwbrn_table: U64, // +0x008: null on G15 (FWBRN size getter returns 0)
        #[ver(G == G15)]
        pub(crate) g15_persistent_time: GpuPointer<'a, super::G15FirmwareTimeState>, // +0x010
        #[ver(G == G15)]
        pub(crate) pipes: Array<4, G15PipeChannels>, // +0x018..+0x197
        #[ver(G == G15)]
        pub(crate) device_control: G15TxChannelRing, // +0x198..+0x1b7
        #[ver(G == G15)]
        pub(crate) event: ChannelRing<channels::ChannelState, channels::RawEventMsg>, // +0x1b8
        #[ver(G == G15)]
        pub(crate) fw_log: ChannelRing<channels::FwLogChannelState, channels::RawFwLogMsg>, // +0x1c8
        #[ver(G == G15)]
        pub(crate) ktrace: ChannelRing<channels::ChannelState, channels::RawKTraceMsg>, // +0x1d8
        #[ver(G == G15)]
        pub(crate) stats: ChannelRing<channels::ChannelState, channels::RawStatsMsg::ver>, // +0x1e8
        #[ver(G == G15)]
        pub(crate) fwlog_buf: Option<GpuWeakPointer<[channels::RawFwLogPayloadMsg]>>, // +0x1f8
        #[ver(G == G15)]
        // Six firmware-log channels. The first counter's low byte selects the
        // 0..255 payload slot for successfully queued records. The second is
        // the monotonic message sequence and advances even when the ring is
        // full, allowing the receiver to observe dropped records as gaps.
        #[ver(G == G15)]
        pub(crate) fwlog_payload_slot_seq_200: Array<6, u32>, // +0x200..+0x217
        #[ver(G == G15)]
        pub(crate) fwlog_message_seq_218: Array<6, u32>, // +0x218..+0x22f
        #[ver(G == G15)]
        pub(crate) fwlog_enabled_230: u32,
        #[ver(G == G15)]
        pub(crate) g15_ptr_234: U64,
        #[ver(G == G15)]
        pub(crate) g15_ptr_23c: U64,
        #[ver(G == G15)]
        pub(crate) g15_ptr_244: U64,
        #[ver(G == G15)]
        pub(crate) g15_control_state: U64, // +0x24c: exact 0x60-byte FW control state
        #[ver(G == G15)]
        pub(crate) g15_pad_254: Array<0x54, u8>,
        #[ver(G == G15)]
        pub(crate) g15_hwds_counters: U64, // +0x2a8: 256 x 8-byte HWDS-ID counters
        #[ver(G == G15)]
        pub(crate) g15_pb_desc_addr: U64, // +0x2b0
        #[ver(G == G15)]
        pub(crate) g15_pb_desc_fw_addr: U64, // +0x2b8
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_desc_addr: U64, // +0x2c0
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_desc_fw_addr: U64, // +0x2c8
        #[ver(G == G15)]
        pub(crate) g15_usc_max_tgmem: u32, // +0x2d0: halGetDefaultUscMaxTgmem() == 4
        #[ver(G == G15)]
        pub(crate) g15_zero_2d4: u32, // +0x2d4: accelerator +0x1e0c, explicitly zeroed
        #[ver(G == G15)]
        pub(crate) g15_runtime_state: G15RuntimeState, // +0x2d8..+0x3af, FW-owned
        #[ver(G == G15)]
        pub(crate) g15_marker_3b0: u8,
        #[ver(G == G15)]
        pub(crate) g15_zero_3b1: Array<0x90, u8>,
        #[ver(G == G15)]
        pub(crate) g15_ptr_441: U64,
        #[ver(G == G15)]
        pub(crate) g15_zero_449: U64, // +0x449: accelerator +0x1d80, zeroed by base configureDevice
        #[ver(G == G15)]
        pub(crate) g15_zero_451: U64, // +0x451: accelerator +0x1d88, same zeroed 16-byte block
        #[ver(G == G15)]
        pub(crate) g15_tail_459: G15WrapperTail,
    }
    #[versions(AGX)]
    no_debug!(RuntimePointers::ver<'_>);

    // The exact G15 wrapper allocation in Apple's host driver is 0x490 bytes.
    // Keep this as a hard compile-time ABI invariant while the remaining fields
    // are named and populated incrementally.
    const _: [(); 0x490] = [(); core::mem::size_of::<RuntimePointersG15V14_7<'static>>()];

    const _: [(); 0x000] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, hwdata_b)];
    const _: [(); 0x008] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_fwbrn_table)];
    const _: [(); 0x010] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_persistent_time)];
    const _: [(); 0x018] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, pipes)];
    const _: [(); 0x198] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, device_control)];
    const _: [(); 0x1b8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, event)];
    const _: [(); 0x1c8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fw_log)];
    const _: [(); 0x1d8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, ktrace)];
    const _: [(); 0x1e8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, stats)];
    const _: [(); 0x1f8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fwlog_buf)];
    const _: [(); 0x200] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fwlog_payload_slot_seq_200)];
    const _: [(); 0x218] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fwlog_message_seq_218)];
    const _: [(); 0x230] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fwlog_enabled_230)];
    const _: [(); 0x234] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_ptr_234)];
    const _: [(); 0x24c] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_control_state)];
    const _: [(); 0x2a8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_hwds_counters)];
    const _: [(); 0x2d0] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_usc_max_tgmem)];
    const _: [(); 0x2d4] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_zero_2d4)];
    const _: [(); 0x2d8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_runtime_state)];
    const _: [(); 0x2b0] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_pb_desc_addr)];
    const _: [(); 0x2c8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_uma_page_pool_desc_fw_addr)];
    const _: [(); 0x3b0] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_marker_3b0)];
    const _: [(); 0x441] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_ptr_441)];
    const _: [(); 0x449] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_zero_449)];
    const _: [(); 0x451] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_zero_451)];
    const _: [(); 0x459] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_tail_459)];

    // Six extra 0x20-byte I/O descriptors move the inherited SRAM pointer
    // from V13.5 +0x960 to the exact G15 host/firmware offset +0xa20.
    const _: [(); 0x28] = [(); core::mem::offset_of!(HwDataBG15V14_7, timestamp_area_base)];
    const _: [(); 0xa20] = [(); core::mem::offset_of!(HwDataBG15V14_7, sgx_sram_ptr)];
    // J615/G15G firmware imports this four-word chip identity block during
    // first init; keep the exact host/firmware offsets mechanically pinned.
    const _: [(); 0xa28] = [(); core::mem::offset_of!(HwDataBG15V14_7, chip_id)];
    const _: [(); 0xa2c] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_454)];
    const _: [(); 0xa30] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_458)];
    const _: [(); 0xa34] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_45c)];
    const _: [(); 0xa6c] = [(); core::mem::offset_of!(HwDataBG15V14_7, power_sample_period)];
    // Exact J615/G15G UAT + core-config header and perf-table boundary.
    const _: [(); 0xad0] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_4f8)];
    const _: [(); 0xb38] = [(); core::mem::offset_of!(HwDataBG15V14_7, g15_zero_b38)];
    const _: [(); 0xb3c] = [(); core::mem::offset_of!(HwDataBG15V14_7, g15_zero_b3c)];
    const _: [(); 0xb40] = [(); core::mem::offset_of!(HwDataBG15V14_7, g15_uat_mode_b40)];
    const _: [(); 0xb44] = [(); core::mem::offset_of!(HwDataBG15V14_7, uat_ttb_base)];
    const _: [(); 0xb4c] = [(); core::mem::offset_of!(HwDataBG15V14_7, gpu_core_id)];
    const _: [(); 0xb50] = [(); core::mem::offset_of!(HwDataBG15V14_7, gpu_rev_id)];
    const _: [(); 0xb54] = [(); core::mem::offset_of!(HwDataBG15V14_7, num_cores)];
    const _: [(); 0xb58] = [(); core::mem::offset_of!(HwDataBG15V14_7, max_pstate)];
    const _: [(); 0xb5c] = [(); core::mem::offset_of!(HwDataBG15V14_7, frequencies)];
    const _: [(); 0x134c] = [(); core::mem::offset_of!(HwDataBG15V14_7, pad_ac4_0)];
    const _: [(); 0x17e4] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_b1c)];
    const _: [(); 0x17e8] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_b20)];
    const _: [(); 0x17ec] = [(); core::mem::offset_of!(HwDataBG15V14_7, g15_startup_17ec)];
    const _: [(); 0x1860] = [(); core::mem::size_of::<HwDataBG15V14_7>()];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct PendingStamp {
        pub(crate) info: AtomicU32,
        pub(crate) wait_value: AtomicU32,
    }
    default_zeroed!(PendingStamp);

    #[derive(Debug, Clone, Copy)]
    #[repr(C, packed)]
    pub(crate) struct FaultInfo {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) queue_uuid: u32,
        pub(crate) unk_c: u32,
        pub(crate) unk_10: u32,
        pub(crate) unk_14: u32,
    }
    default_zeroed!(FaultInfo);

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct PowerZoneGlobal {
        pub(crate) target: u32,
        pub(crate) target_off: u32,
        pub(crate) filter_tc: u32,
    }
    default_zeroed!(PowerZoneGlobal);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Globals {
        pub(crate) ktrace_enable: u32,
        pub(crate) unk_4: Array<0x20, u8>,

        #[ver(V >= V13_2)]
        pub(crate) unk_24_0: u32,

        pub(crate) unk_24: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) debug: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_28_4: u32,

        pub(crate) unk_28: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_2c_0: u32,

        pub(crate) unk_2c: u32,
        pub(crate) unk_30: u32,
        pub(crate) unk_34: u32,
        pub(crate) unk_38: Array<0x1c, u8>,

        // pub(crate) sub: GlobalsSub::ver,
        pub(crate) unk_54: u16,
        pub(crate) unk_56: u16,
        pub(crate) unk_58: u16,
        pub(crate) unk_5a: U32,
        pub(crate) unk_5e: U32,
        pub(crate) unk_62: U32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_66_0: Array<0xc, u8>,

        pub(crate) unk_66: U32,
        pub(crate) unk_6a: Array<0x16, u8>,
        // end GlobalsSub::ver

        pub(crate) unk_80: Array<0xf80, u8>,
        pub(crate) unk_1000: Array<0x7000, u8>,
        pub(crate) unk_8000: Array<0x900, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_8900_pad: Array<0x484c, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_8900_pad2: Array<0x54, u8>,

        pub(crate) unk_8900: u32,
        pub(crate) pending_submissions: AtomicU32,
        pub(crate) max_power: u32,
        pub(crate) max_pstate_scaled: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) unk_8914: u32,
        pub(crate) unk_8918: u32,
        pub(crate) max_pstate_scaled_3: u32,
        pub(crate) unk_8920: u32,
        pub(crate) power_zone_count: u32,
        pub(crate) avg_power_filter_tc_periods: u32,
        pub(crate) avg_power_ki_dt: F32,
        pub(crate) avg_power_kp: F32,
        pub(crate) avg_power_min_duty_cycle: u32,
        pub(crate) avg_power_target_filter_tc: u32,
        pub(crate) power_zones: Array<5, PowerZoneGlobal>,
        pub(crate) unk_8978: Array<0x44, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89bc_0: Array<0x3c, u8>,

        pub(crate) unk_89bc: u32,
        pub(crate) fast_die0_release_temp: u32,
        pub(crate) unk_89c4: i32,
        pub(crate) fast_die0_prop_tgt_delta: u32,
        pub(crate) fast_die0_kp: F32,
        pub(crate) fast_die0_ki_dt: F32,
        pub(crate) unk_89d4: Array<0xc, u8>,
        pub(crate) unk_89e0: u32,
        pub(crate) max_power_2: u32,
        pub(crate) ppm_kp: F32,
        pub(crate) ppm_ki_dt: F32,
        pub(crate) unk_89f0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_0: Array<0x8, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_c: Array<0x50, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_89f4_5c: Array<0xc, u8>,

        pub(crate) unk_89f4: u32,
        pub(crate) hws1: HwDataShared1,
        pub(crate) hws2: HwDataShared2,

        #[ver(V >= V13_0B4)]
        pub(crate) idle_off_standby_timer: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2_4: Array<0x8, F32>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2_24: u32,

        pub(crate) unk_hws2_28: u32,

        pub(crate) hws3: HwDataShared3,
        pub(crate) unk_9004: Array<8, u8>,
        pub(crate) unk_900c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_9010_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_9010_4: Array<0x14, u8>,

        pub(crate) unk_9010: Array<0x2c, u8>,
        pub(crate) unk_903c: u32,
        pub(crate) unk_9040: Array<0xc0, u8>,
        pub(crate) unk_9100: Array<0x6f00, u8>,
        pub(crate) unk_10000: Array<0xe50, u8>,
        pub(crate) unk_10e50: u32,
        pub(crate) unk_10e54: Array<0x2c, u8>,

        #[ver((G >= G14X && V < V13_3) || (G <= G14 && V >= V13_3))]
        pub(crate) unk_x_pad: Array<0x4, u8>,

        // bit 0: sets sgx_reg 0x17620
        // bit 1: sets sgx_reg 0x17630
        pub(crate) fault_control: u32,
        pub(crate) do_init: u32,
        pub(crate) unk_10e88: Array<0x188, u8>,
        pub(crate) idle_ts: U64,
        pub(crate) idle_unk: U64,
        pub(crate) progress_check_interval_3d: u32,
        pub(crate) progress_check_interval_ta: u32,
        pub(crate) progress_check_interval_cl: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_10: u32,

        pub(crate) unk_1102c: u32,
        pub(crate) idle_off_delay_ms: AtomicU32,
        pub(crate) fender_idle_off_delay_ms: u32,
        pub(crate) fw_early_wake_timeout_ms: u32,
        #[ver(V == V13_3)]
        pub(crate) ps_pad_0: Pad<0x8>,
        pub(crate) pending_stamps: Array<0x100, PendingStamp>,
        #[ver(V != V13_3)]
        pub(crate) ps_pad_0: Pad<0x8>,
        pub(crate) unkpad_ps: Pad<0x78>,
        pub(crate) unk_117bc: u32,
        pub(crate) fault_info: FaultInfo,
        pub(crate) counter: u32,
        pub(crate) unk_118dc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_118e0_0: Array<0x9c, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_118e0_9c: Array<0x580, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_118e0_9c_x: Array<0x8, u8>,

        pub(crate) cl_context_switch_timeout_ms: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) cl_kill_timeout_ms: u32,

        pub(crate) cdm_context_store_latency_threshold: u32,
        pub(crate) unk_118e8: u32,
        pub(crate) unk_118ec: Array<0x400, u8>,
        pub(crate) unk_11cec: Array<0x54, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11d40: Array<0x19c, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11edc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11ee0: Array<0x1c, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11efc: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_11f00: Array<0x280, u8>,
    }
    #[versions(AGX)]
    default_zeroed!(Globals::ver);

    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C, packed)]
    pub(crate) struct UatLevelInfo {
        pub(crate) unk_3: u8,
        pub(crate) unk_1: u8,
        pub(crate) unk_2: u8,
        pub(crate) index_shift: u8,
        pub(crate) num_entries: u16,
        pub(crate) unk_4: u16,
        pub(crate) unk_8: U64,
        pub(crate) unk_10: U64,
        pub(crate) index_mask: U64,
    }

    /// G15 q6..q20: exact 0x78-byte UAT description block.
    ///
    /// Apple's G15 host copies q6..q18 from accelerator firmware-mapper data
    /// and leaves q19/q20 untouched. The byte span is exactly the legacy UAT
    /// header + three 0x20-byte level descriptors + 0x14 bytes of zero pad.
    /// For G15 the per-TTBR input width is 42 bits (the Apple ADT's 43-bit
    /// `uat-vaddr-size` includes the TTBR0/TTBR1 selector bit), hence the
    /// shift-36 root has 64 entries.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15UatConfig {
        pub(crate) page_size: u16,
        pub(crate) page_bits: u8,
        pub(crate) num_levels: u8,
        pub(crate) level_info: Array<3, UatLevelInfo>,
        pub(crate) __pad0: Pad<0x14>,
    }
    default_zeroed!(G15UatConfig);
    const _: [(); 0x78] = [(); core::mem::size_of::<G15UatConfig>()];
    const _: [(); 0x04] = [(); core::mem::offset_of!(G15UatConfig, level_info)];
    const _: [(); 0x64] = [(); core::mem::offset_of!(G15UatConfig, __pad0)];

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct InitData<'a> {
        // Legacy top-level interface. G15 replaces this field layout wholesale
        // with the 24-qword RTKit-2419.140.12 host interface below.
        #[ver(V >= V13_0B4 && G != G15)]
        pub(crate) ver_info: Array<0x4, u16>,
        #[ver(G != G15)]
        pub(crate) unk_buf: GpuPointer<'a, &'a [u8]>,
        #[ver(G != G15)]
        pub(crate) unk_8: u32,
        #[ver(G != G15)]
        pub(crate) unk_c: u32,
        #[ver(G != G15)]
        pub(crate) runtime_pointers: GpuPointer<'a, super::RuntimePointers::ver>,
        #[ver(G != G15)]
        pub(crate) globals: GpuPointer<'a, super::Globals::ver>,
        #[ver(G != G15)]
        pub(crate) fw_status: GpuPointer<'a, super::FwStatus>,
        #[ver(G != G15)]
        pub(crate) uat_page_size: u16,
        #[ver(G != G15)]
        pub(crate) uat_page_bits: u8,
        #[ver(G != G15)]
        pub(crate) uat_num_levels: u8,
        #[ver(G != G15)]
        pub(crate) uat_level_info: Array<0x3, UatLevelInfo>,
        #[ver(G != G15)]
        pub(crate) __pad0: Pad<0x14>,
        #[ver(G != G15)]
        pub(crate) host_mapped_fw_allocations: u32,
        #[ver(G != G15)]
        pub(crate) unk_ac: u32,
        #[ver(G != G15)]
        pub(crate) unk_b0: u32,
        #[ver(G != G15)]
        pub(crate) unk_b4: u32,
        #[ver(G != G15)]
        pub(crate) unk_b8: u32,

        // Exact G15 / V14.7 top-level ABI: 24 consecutive qwords (0xc0).
        // q0 is the interface signature. q3/q4 are the compact runtime wrapper
        // and 0xe00 Globals allocation. q5's upper dword enables host-mapped
        // firmware allocations. q21..q23 point at exact 0x20/0xc3d0/0x238
        // backing objects. q1 and q6..q18 are kept raw until their builders are
        // reconstructed byte-for-byte from the Apple host implementation.
        #[ver(G == G15)]
        pub(crate) g15_q0_signature: U64,
        #[ver(G == G15)]
        pub(crate) g15_q1_init_sequence: U64,
        // Apple never writes q2 after allocating the root with the same
        // zeroing allocation flags as IOMallocZero(), so q2 is exactly zero.
        #[ver(G == G15)]
        pub(crate) g15_q2: U64,
        #[ver(G == G15)]
        pub(crate) g15_q3_runtime_pointers: U64,
        #[ver(G == G15)]
        pub(crate) g15_q4_globals: U64,
        #[ver(G == G15)]
        pub(crate) g15_q5_host_mapped: U64,
        #[ver(G == G15)]
        pub(crate) g15_q6_q20_uat: G15UatConfig,
        #[ver(G == G15)]
        pub(crate) g15_q21: U64,
        #[ver(G == G15)]
        pub(crate) g15_q22: U64,
        #[ver(G == G15)]
        pub(crate) g15_q23: U64,
        #[ver(G == G15)]
        pub(crate) g15_phantom: PhantomData<&'a ()>,
    }

    const _: [(); 0xc0] = [(); core::mem::size_of::<InitDataG15V14_7<'static>>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q0_signature)];
    const _: [(); 0x08] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q1_init_sequence)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q2)];
    const _: [(); 0x18] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q3_runtime_pointers)];
    const _: [(); 0x20] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q4_globals)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q5_host_mapped)];
    const _: [(); 0x30] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q6_q20_uat)];
    const _: [(); 0xa8] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q21)];
    const _: [(); 0xb0] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q22)];
    const _: [(); 0xb8] = [(); core::mem::offset_of!(InitDataG15V14_7<'static>, g15_q23)];
}

#[derive(Debug)]
pub(crate) struct ChannelRing<T: GpuStruct + Debug + Default, U: Copy>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug,
{
    pub(crate) state: GpuObject<T>,
    pub(crate) ring: GpuArray<U>,
}

impl<T: GpuStruct + Debug + Default, U: Copy> ChannelRing<T, U>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug,
{
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<T, U> {
        raw::ChannelRing {
            state: Some(self.state.weak_pointer()),
            ring: Some(self.ring.weak_pointer()),
        }
    }
}

impl<U: Copy> ChannelRing<channels::ChannelState, U> {
    pub(crate) fn to_raw_g15_tx(&self) -> raw::G15TxChannelRing {
        let state = u64::from(self.state.weak_pointer());
        raw::G15TxChannelRing {
            read_ptr: U64(state),
            write_ptr_shadow: U64(state + 0x10),
            write_ptr: U64(state + 0x20),
            ring: U64(u64::from(self.ring.weak_pointer())),
        }
    }
}

trivial_gpustruct!(FwStatus);
trivial_gpustruct!(G15SharedStatus);
trivial_gpustruct!(G15FirmwareTimeState);
trivial_gpustruct!(G15InitSequencePage);
trivial_gpustruct!(G15ControlState);
trivial_gpustruct!(G15Q4Config);
trivial_gpustruct!(G15MappingRingBacking);
trivial_gpustruct!(G15Q22Shared);
trivial_gpustruct!(G15Q23Shared);
trivial_gpustruct!(G15StatsVtx);
trivial_gpustruct!(G15StatsFrag);
trivial_gpustruct!(G15StatsComp);
trivial_gpustruct!(GpuGlobalStatsVtx);
#[versions(AGX)]
trivial_gpustruct!(GpuGlobalStatsFrag::ver);
trivial_gpustruct!(GpuStatsComp);

#[versions(AGX)]
trivial_gpustruct!(HwDataA::ver);

#[versions(AGX)]
trivial_gpustruct!(HwDataB::ver);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Stats {
    pub(crate) vtx: GpuObject<GpuGlobalStatsVtx>,
    pub(crate) frag: GpuObject<GpuGlobalStatsFrag::ver>,
    pub(crate) comp: GpuObject<GpuStatsComp>,
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RuntimePointers {
    // Keep the legacy Stats owner for generated command-layout scaffolding.
    // G15's wrapper-visible global stats are separate exact-size allocations
    // until the G15 queue statistics pointer ABI is reconstructed.
    pub(crate) stats: Stats::ver,
    #[ver(G == G15)]
    pub(crate) g15_persistent_time: GpuObject<G15FirmwareTimeState>, // exact Apple backing size 0x88
    #[ver(G == G15)]
    pub(crate) g15_control_state: GpuObject<G15ControlState>,
    #[ver(G == G15)]
    pub(crate) g15_stats_vtx: GpuObject<G15StatsVtx>,
    #[ver(G == G15)]
    pub(crate) g15_stats_frag: GpuObject<G15StatsFrag>,
    #[ver(G == G15)]
    pub(crate) g15_stats_comp: GpuObject<G15StatsComp>,
    #[ver(G == G15)]
    pub(crate) g15_pb_desc_table: GpuArray<raw::G15PBDescriptor>,
    #[ver(G == G15)]
    pub(crate) g15_uma_page_pool_desc_table: GpuArray<raw::G15UMAPagePoolDescriptor>,
    #[ver(G == G15)]
    pub(crate) g15_hwds_counters: GpuArray<raw::G15HWDSCounterEntry>,

    pub(crate) hwdata_a: GpuObject<HwDataA::ver>,
    pub(crate) unkptr_190: GpuArray<u8>,
    pub(crate) unkptr_198: GpuArray<u8>,
    pub(crate) hwdata_b: GpuObject<HwDataB::ver>,

    pub(crate) unkptr_1b8: GpuArray<u8>,
    pub(crate) unkptr_1c0: GpuArray<u8>,
    pub(crate) unkptr_1c8: GpuArray<u8>,

    pub(crate) buffer_mgr_ctl: gem::ObjectRef,
    pub(crate) buffer_mgr_ctl_low_mapping: Option<mmu::KernelMapping>,
    pub(crate) buffer_mgr_ctl_high_mapping: Option<mmu::KernelMapping>,
}

#[versions(AGX)]
impl GpuStruct for RuntimePointers::ver {
    type Raw<'a> = raw::RuntimePointers::ver<'a>;
}

#[versions(AGX)]
trivial_gpustruct!(Globals::ver);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct InitData {
    #[ver(G != G15)]
    pub(crate) unk_buf: GpuArray<u8>,
    pub(crate) runtime_pointers: GpuObject<RuntimePointers::ver>,
    #[ver(G != G15)]
    pub(crate) globals: GpuObject<Globals::ver>,
    #[ver(G != G15)]
    pub(crate) fw_status: GpuObject<FwStatus>,

    // G15 replaces the legacy top-level backing set. q1 is an exact one-page
    // AGFA init-sequence mapping. G15/G15G's population hook is a BTI+RET stub,
    // so the first 0x18-byte record is the explicit type-0 terminator.
    #[ver(G == G15)]
    pub(crate) g15_init_sequence: GpuObject<G15InitSequencePage>,
    // Exact-size G15 top-level backing allocations. These remain opaque while
    // their individual fields are reconstructed from the Apple host driver.
    #[ver(G == G15)]
    pub(crate) g15_globals: GpuObject<G15Q4Config>,
    #[ver(G == G15)]
    pub(crate) g15_q21: GpuObject<G15SharedStatus>,
    #[ver(G == G15)]
    pub(crate) g15_mapping_notifier: mmu::G15MappingNotifierHandle,
    #[ver(G == G15)]
    pub(crate) g15_q22: GpuObject<G15Q22Shared>,
    #[ver(G == G15)]
    pub(crate) g15_q23: GpuObject<G15Q23Shared>,
}

#[versions(AGX)]
impl GpuStruct for InitData::ver {
    type Raw<'a> = raw::InitData::ver<'a>;
}
