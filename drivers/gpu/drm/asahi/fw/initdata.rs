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
        pub(crate) flags_000: U32,                 // +0x000: host writes 0 or 7
        pub(crate) zero_004: U32,
        pub(crate) zero_008: U32,
        pub(crate) zero_00c: U32,
        pub(crate) zero_010: U32,
        pub(crate) zero_014: U32,
        pub(crate) zero_018: U32,
        pub(crate) unk_01c: U32,                   // accelerator +0x2970
        pub(crate) unk_020: U32,
        pub(crate) unk_024: U32,
        pub(crate) unk_028: U32,
        pub(crate) unk_02c: U32,
        pub(crate) unk_030: U32,
        pub(crate) zero_034: U32,
        pub(crate) constant_038: U32,              // exact host constant 0x78
        pub(crate) zero_03c: U32,
        pub(crate) zero_040: U32,
        pub(crate) zero_044: U32,
        pub(crate) zero_048: U32,
        pub(crate) zero_04c: U32,
        pub(crate) unk_050: u16,                   // accelerator +0x6c4
        pub(crate) unk_052: u16,                   // accelerator +0x6c6
        pub(crate) unk_054: u16,                   // accelerator +0x6c8
        pub(crate) zero_056: U32,                  // deliberately unaligned
        pub(crate) unk_05a: U32,                   // deliberately unaligned
        pub(crate) unk_05e: U32,                   // deliberately unaligned
        pub(crate) unk_062: U32,                   // deliberately unaligned
        pub(crate) pad_066: Pad<0x0a>,
        pub(crate) unk_070: U32,
        pub(crate) unk_074: U32,
        pub(crate) unk_078: U32,
        pub(crate) pad_07c: Pad<0x04>,
        pub(crate) unk_080: U32,
        pub(crate) pad_084: Pad<0x08>,
        pub(crate) unk_08c: U32,
        pub(crate) unk_090: U32,
        pub(crate) unk_094: U32,
        pub(crate) unk_098: U32,
        pub(crate) unk_09c: U32,
        pub(crate) pad_0a0: Pad<0x04>,
        pub(crate) unk_0a4: u8,
        pub(crate) pad_0a5: Pad<0x1ca>,
        pub(crate) unk_26f: u8,
        pub(crate) pad_270: Pad<0x538>,
        pub(crate) unk_7a8: U32,
        pub(crate) unk_7ac: U32,
        pub(crate) unk_7b0: U32,
        pub(crate) unk_7b4: U32,
        pub(crate) unk_7b8: U32,
        pub(crate) unk_7bc: U32,
        pub(crate) unk_7c0: U32,
        pub(crate) unk_7c4: U32,
        pub(crate) unk_7c8: U32,
        pub(crate) unk_7cc: U32,
        pub(crate) pad_7d0: Pad<0x04>,
        pub(crate) unk_7d4: U32,
        pub(crate) unk_7d8: U32,
        pub(crate) unk_7dc: U32,
        pub(crate) pad_7e0: Pad<0x18c>,
        pub(crate) unk_96c: U64,
        pub(crate) unk_974: U64,
        pub(crate) unk_97c: U64,                   // accelerator +0x1e48
        pub(crate) unk_984: U64,                   // accelerator +0x1e50
        pub(crate) unk_98c: U64,                   // accelerator +0x1e58
        pub(crate) unk_994: U32,                   // accelerator +0x1e60
        pub(crate) pad_998: Pad<0x08>,
        pub(crate) unk_9a0: U32,
        pub(crate) unk_9a4: U32,
        pub(crate) unk_9a8: U32,
        pub(crate) zero_9ac: U32,
        pub(crate) unk_9b0: U32,
        pub(crate) unk_9b4: U32,
        pub(crate) zero_9b8: U32,
        pub(crate) pad_9bc: Pad<0x21>,
        pub(crate) table_selector_9dd: U64,
        pub(crate) table_9e5: Array<0x200, u8>,
        pub(crate) table_be5: Array<0x200, u8>,
        pub(crate) zero_de5: U32,
        pub(crate) unk_de9: U32,
        pub(crate) unk_ded: U32,
        pub(crate) unk_df1: U32,
        pub(crate) unk_df5: U32,
        pub(crate) unk_df9: U32,
        pub(crate) tail_dfd: Pad<0x03>,
    }
    default_zeroed!(G15Q4Config);
    const _: [(); 0xe00] = [(); core::mem::size_of::<G15Q4Config>()];
    const _: [(); 0x038] = [(); core::mem::offset_of!(G15Q4Config, constant_038)];
    const _: [(); 0x056] = [(); core::mem::offset_of!(G15Q4Config, zero_056)];
    const _: [(); 0x05a] = [(); core::mem::offset_of!(G15Q4Config, unk_05a)];
    const _: [(); 0x7a8] = [(); core::mem::offset_of!(G15Q4Config, unk_7a8)];
    const _: [(); 0x96c] = [(); core::mem::offset_of!(G15Q4Config, unk_96c)];
    const _: [(); 0x97c] = [(); core::mem::offset_of!(G15Q4Config, unk_97c)];
    const _: [(); 0x9ac] = [(); core::mem::offset_of!(G15Q4Config, zero_9ac)];
    const _: [(); 0x9dd] = [(); core::mem::offset_of!(G15Q4Config, table_selector_9dd)];
    const _: [(); 0x9e5] = [(); core::mem::offset_of!(G15Q4Config, table_9e5)];
    const _: [(); 0xbe5] = [(); core::mem::offset_of!(G15Q4Config, table_be5)];
    const _: [(); 0xde5] = [(); core::mem::offset_of!(G15Q4Config, zero_de5)];

    /// G15 q22 cache-flush ring control block, exact 0x20 bytes.
    /// Firmware consumes entries from read_idx and compares against write_idx
    /// at +0x10 modulo 256.
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

    /// One G15 cache-flush ring entry, exact 0x18 bytes. Firmware walks 256
    /// entries, so the Apple ring allocation is exactly 0x1800 bytes.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15CacheFlushEntry {
        pub(crate) addr: U64,        // +0x00
        pub(crate) unk_08: U32,      // +0x08
        pub(crate) context_id: U32,  // +0x0c
        pub(crate) page_count: u16,  // +0x10
        pub(crate) flags: u16,       // +0x12
        pub(crate) unk_14: U32,      // +0x14
    }
    const _: [(); 0x18] = [(); core::mem::size_of::<G15CacheFlushEntry>()];
    const _: [(); 0x0c] = [(); core::mem::offset_of!(G15CacheFlushEntry, context_id)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15CacheFlushEntry, page_count)];
    const _: [(); 0x12] = [(); core::mem::offset_of!(G15CacheFlushEntry, flags)];

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
        pub(crate) host_zero_c3c8: U32,
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
    const _: [(); 0xc3c8] = [(); core::mem::offset_of!(G15Q22Shared, host_zero_c3c8)];
    const _: [(); 0xc3cc] = [(); core::mem::offset_of!(G15Q22Shared, feature_c3cc)];

    /// G15 root q23: exact 0x238-byte host/FW shared state object. Apple maps
    /// CPU/GPU pair +0x628/+0x638 into q23. Host initialization is zero for the
    /// observed fields (including +0x1e8); firmware owns most runtime updates.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15Q23Shared {
        pub(crate) opaque: Array<0x238, u8>,
    }
    default_zeroed!(G15Q23Shared);
    const _: [(); 0x238] = [(); core::mem::size_of::<G15Q23Shared>()];

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

    /// G15-only extension appended to the inherited HwDataA prefix.
    ///
    /// The generated pre-G15-style object ends exactly at +0x421c, while
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
        pub(crate) hws1: HwDataShared1,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2: Array<16, u16>,

        pub(crate) hws2: HwDataShared2,
        pub(crate) unk_3c00: u32,
        pub(crate) unk_3c04: u32,
        pub(crate) hws3: HwDataShared3,
        pub(crate) unk_3c58: Array<0x3c, u8>,
        pub(crate) unk_3c94: u32,
        pub(crate) unk_3c98: U64,
        pub(crate) unk_3ca0: U64,
        pub(crate) unk_3ca8: U64,
        pub(crate) unk_3cb0: U64,
        pub(crate) ts_last_idle: U64,
        pub(crate) ts_last_poweron: U64,
        pub(crate) ts_last_poweroff: U64,
        pub(crate) unk_3cd0: U64,
        pub(crate) unk_3cd8: U64,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_3ce0_0: u32,

        pub(crate) unk_3ce0: u32,
        pub(crate) unk_3ce4: u32,
        pub(crate) unk_3ce8: u32,
        pub(crate) unk_3cec: u32,
        pub(crate) unk_3cf0: u32,
        pub(crate) core_leak_coef: Array<8, F32>,
        pub(crate) sram_leak_coef: Array<8, F32>,

        #[ver(V >= V13_0B4)]
        pub(crate) aux_leak_coef: AuxLeakCoef,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_3d34_0: Array<0x18, u8>,

        pub(crate) unk_3d34: Array<0x38, u8>,

        #[ver(G == G15)]
        pub(crate) g15_tail_421c: G15HwDataATail,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataA::ver);
    #[versions(AGX)]
    no_debug!(HwDataA::ver);

    const _: [(); 0x4360] = [(); core::mem::size_of::<HwDataAG15V14_7>()];
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

    /// Final G15 HwDataB trailer, replacing the larger V13.5 legacy tail.
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct G15HwDataBTail {
        pub(crate) pad_1840: Pad<0x18>,
        // Apple host writes the inverse of accelerator-global flag bit 4 here;
        // firmware imports this exact dword during early init.
        pub(crate) flag_1858: u32,
        pub(crate) pad_185c: Pad<0x04>,
    }
    default_zeroed!(G15HwDataBTail);
    const _: [(); 0x20] = [(); core::mem::size_of::<G15HwDataBTail>()];
    const _: [(); 0x18] = [(); core::mem::offset_of!(G15HwDataBTail, flag_1858)];

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

        #[ver(V >= V13_3)]
        pub(crate) pad_ac4_0: Array<0x44c, u8>,

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
        pub(crate) unk_b24: u32,
        pub(crate) unk_b28: u32,
        pub(crate) unk_b2c: u32,
        pub(crate) unk_b30: u32,
        pub(crate) unk_b34: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_b38_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_b38_4: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_b38_8: u32,

        pub(crate) unk_b38: Array<0xc, u32>,
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

        // G15 keeps the inherited layout byte-exact through +0x183f
        // (`unk_b68` at +0x183c), then replaces the old 0x104-byte V13.5
        // trailer with an exact 0x20-byte tail. Apple's allocation is 0x1860
        // and firmware directly reads the final active word at +0x1858.
        #[ver(G == G15)]
        pub(crate) g15_tail_1840: G15HwDataBTail,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataB::ver);

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
        pub(crate) g15_unk_010: U64,
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
        pub(crate) g15_counters_200: Array<6, u32>, // +0x200..+0x217
        #[ver(G == G15)]
        pub(crate) g15_counters_218: Array<6, u32>, // +0x218..+0x22f
        #[ver(G == G15)]
        pub(crate) g15_enable_230: u32,
        #[ver(G == G15)]
        pub(crate) g15_ptr_234: U64,
        #[ver(G == G15)]
        pub(crate) g15_ptr_23c: U64,
        #[ver(G == G15)]
        pub(crate) g15_ptr_244: U64,
        #[ver(G == G15)]
        pub(crate) g15_ptr_24c: U64,
        #[ver(G == G15)]
        pub(crate) g15_pad_254: Array<0x54, u8>,
        #[ver(G == G15)]
        pub(crate) g15_ptr_2a8: U64,
        #[ver(G == G15)]
        pub(crate) g15_pb_desc_addr: U64, // +0x2b0
        #[ver(G == G15)]
        pub(crate) g15_pb_desc_fw_addr: U64, // +0x2b8
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_desc_addr: U64, // +0x2c0
        #[ver(G == G15)]
        pub(crate) g15_uma_page_pool_desc_fw_addr: U64, // +0x2c8
        #[ver(G == G15)]
        pub(crate) g15_unk_2d0: u32,
        #[ver(G == G15)]
        pub(crate) g15_unk_2d4: u32,
        #[ver(G == G15)]
        pub(crate) g15_opaque_2d8: Array<0xd8, u8>,
        #[ver(G == G15)]
        pub(crate) g15_marker_3b0: u8,
        #[ver(G == G15)]
        pub(crate) g15_zero_3b1: Array<0x90, u8>,
        #[ver(G == G15)]
        pub(crate) g15_ptr_441: U64,
        #[ver(G == G15)]
        pub(crate) g15_unk_449: U64,
        #[ver(G == G15)]
        pub(crate) g15_unk_451: U64,
        #[ver(G == G15)]
        pub(crate) g15_tail_459: Array<0x37, u8>,
    }
    #[versions(AGX)]
    no_debug!(RuntimePointers::ver<'_>);

    // The exact G15 wrapper allocation in Apple's host driver is 0x490 bytes.
    // Keep this as a hard compile-time ABI invariant while the remaining fields
    // are named and populated incrementally.
    const _: [(); 0x490] = [(); core::mem::size_of::<RuntimePointersG15V14_7<'static>>()];

    const _: [(); 0x000] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, hwdata_b)];
    const _: [(); 0x008] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_fwbrn_table)];
    const _: [(); 0x018] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, pipes)];
    const _: [(); 0x198] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, device_control)];
    const _: [(); 0x1b8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, event)];
    const _: [(); 0x1c8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fw_log)];
    const _: [(); 0x1d8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, ktrace)];
    const _: [(); 0x1e8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, stats)];
    const _: [(); 0x1f8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, fwlog_buf)];
    const _: [(); 0x230] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_enable_230)];
    const _: [(); 0x234] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_ptr_234)];
    const _: [(); 0x2b0] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_pb_desc_addr)];
    const _: [(); 0x2c8] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_uma_page_pool_desc_fw_addr)];
    const _: [(); 0x3b0] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_marker_3b0)];
    const _: [(); 0x441] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_ptr_441)];
    const _: [(); 0x459] = [(); core::mem::offset_of!(RuntimePointersG15V14_7<'static>, g15_tail_459)];

    // Six extra 0x20-byte I/O descriptors move the inherited SRAM pointer
    // from V13.5 +0x960 to the exact G15 host/firmware offset +0xa20.
    const _: [(); 0xa20] = [(); core::mem::offset_of!(HwDataBG15V14_7, sgx_sram_ptr)];
    const _: [(); 0x1860] = [(); core::mem::size_of::<HwDataBG15V14_7>()];
    const _: [(); 0x183c] = [(); core::mem::offset_of!(HwDataBG15V14_7, unk_b68)];
    const _: [(); 0x1840] = [(); core::mem::offset_of!(HwDataBG15V14_7, g15_tail_1840)];

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
trivial_gpustruct!(G15Q4Config);
trivial_gpustruct!(G15CacheFlushState);
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
    pub(crate) g15_aux_010: GpuArray<u8>, // exact Apple backing size 0x88
    #[ver(G == G15)]
    pub(crate) g15_aux_24c: GpuArray<u8>, // exact Apple backing size 0x60
    #[ver(G == G15)]
    pub(crate) g15_stats_vtx: GpuObject<G15StatsVtx>,
    #[ver(G == G15)]
    pub(crate) g15_stats_frag: GpuObject<G15StatsFrag>,
    #[ver(G == G15)]
    pub(crate) g15_stats_comp: GpuObject<G15StatsComp>,

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

    // G15 replaces the legacy top-level backing set. The init sequence is an
    // exact one-page (0x4000) host/FW mapping; Apple never writes its CPU view
    // after allocation and the G15 populateInitSequenceFirmware() override is
    // a BTI+RET stub, so a zero record at +0x08 is the firmware terminator.
    #[ver(G == G15)]
    pub(crate) g15_init_sequence: GpuArray<u8>,
    // Exact-size G15 top-level backing allocations. These remain opaque while
    // their individual fields are reconstructed from the Apple host driver.
    #[ver(G == G15)]
    pub(crate) g15_globals: GpuObject<G15Q4Config>,
    #[ver(G == G15)]
    pub(crate) g15_q21: GpuObject<G15SharedStatus>,
    #[ver(G == G15)]
    pub(crate) g15_cache_flush_state: GpuObject<G15CacheFlushState>,
    #[ver(G == G15)]
    pub(crate) g15_cache_flush_ring: GpuArray<raw::G15CacheFlushEntry>,
    #[ver(G == G15)]
    pub(crate) g15_q22: GpuObject<G15Q22Shared>,
    #[ver(G == G15)]
    pub(crate) g15_q23: GpuObject<G15Q23Shared>,
}

#[versions(AGX)]
impl GpuStruct for InitData::ver {
    type Raw<'a> = raw::InitData::ver<'a>;
}
