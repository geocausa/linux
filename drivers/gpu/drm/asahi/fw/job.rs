// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Common GPU job firmware structures

use super::types::*;
use crate::{
    default_zeroed,
    mmu,
    trivial_gpustruct, //
};
use kernel::prelude::Result;
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct JobMeta {
        // These legacy names describe the pre-G15 layout only. Apple G15
        // reuses the entire first dword as engine-specific state: Compute
        // repacks four raw command bytes, while TA and 3D synthesize different
        // feature-gated dwords. Keep G15 zero/fail-closed until that producer
        // contract is modeled per engine; do not infer no_preemption at +0x03.
        pub(crate) unk_0: u16,
        pub(crate) unk_2: u8,
        pub(crate) no_preemption: u8,
        pub(crate) stamp: GpuWeakPointer<Stamp>,
        pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
        pub(crate) stamp_value: EventValue,
        pub(crate) stamp_slot: u32,
        pub(crate) evctl_index: u32,
        pub(crate) flush_stamps: u32,
        pub(crate) uuid: u32,
        pub(crate) event_seq: u32,
    }

    // G15 host submission copies the low 32 bits of its shared atomic
    // command-descriptor ID to meta +0x24 and the independent queue-local
    // event sequence to meta +0x28.
    const _: [(); 0x2c] = [(); core::mem::size_of::<JobMeta>()];
    const _: [(); 0x24] = [(); core::mem::offset_of!(JobMeta, uuid)];
    const _: [(); 0x28] = [(); core::mem::offset_of!(JobMeta, event_seq)];

    /// G15 TA/3D SKU-local timing state. Firmware scheduling writes the first
    /// timestamp when work is dispatched. Completion paths require the first
    /// timestamp to precede the third qword and feed their difference to the
    /// common latency histogram helper. The other two qwords remain unnamed.
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct G15SkuTimingState {
        pub(crate) start_timestamp: U64,    // +0x00
        pub(crate) unk_08: U64,             // +0x08
        pub(crate) complete_timestamp: U64, // +0x10
        pub(crate) unk_18: U64,             // +0x18
    }
    const _: [(); 0x20] = [(); core::mem::size_of::<G15SkuTimingState>()];
    const _: [(); 0x00] = [(); core::mem::offset_of!(G15SkuTimingState, start_timestamp)];
    const _: [(); 0x10] = [(); core::mem::offset_of!(G15SkuTimingState, complete_timestamp)];

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct EncoderParams {
        pub(crate) unk_8: u32,
        pub(crate) sync_grow: u32,
        pub(crate) unk_10: u32,
        pub(crate) encoder_id: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_mask: u32,
        pub(crate) sampler_array: U64,
        pub(crate) sampler_count: u32,
        pub(crate) sampler_max: u32,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobTimestamps {
        pub(crate) start: AtomicU64,
        pub(crate) end: AtomicU64,
    }
    default_zeroed!(JobTimestamps);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RenderTimestamps {
        pub(crate) vtx: JobTimestamps,
        pub(crate) frag: JobTimestamps,
    }
    default_zeroed!(RenderTimestamps);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Register {
        pub(crate) number: u32,
        pub(crate) value: U64,
    }
    default_zeroed!(Register);

    impl Register {
        fn new(number: u32, value: u64) -> Register {
            Register {
                number,
                value: U64(value),
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RegisterArray {
        pub(crate) registers: Array<128, Register>,
        pub(crate) pad: Array<0x100, u8>,

        pub(crate) addr: GpuWeakPointer<Array<128, Register>>,
        pub(crate) count: u16,
        pub(crate) length: u16,
        pub(crate) unk_pad: u32,
    }

    // The G14X and G15 firmware command ABIs both embed this exact object.
    // Keep its non-native 12-byte register stride and trailing self-pointer
    // geometry compile-time locked before generation-specific lists diverge.
    const _: [(); 0x0c] = [(); core::mem::size_of::<Register>()];
    const _: [(); 0x710] = [(); core::mem::size_of::<RegisterArray>()];
    const _: [(); 0x600] = [(); core::mem::offset_of!(RegisterArray, pad)];
    const _: [(); 0x700] = [(); core::mem::offset_of!(RegisterArray, addr)];
    const _: [(); 0x708] = [(); core::mem::offset_of!(RegisterArray, count)];
    const _: [(); 0x70a] = [(); core::mem::offset_of!(RegisterArray, length)];
    const _: [(); 0x70c] = [(); core::mem::offset_of!(RegisterArray, unk_pad)];

    impl RegisterArray {
        pub(crate) fn new(
            self_ptr: GpuWeakPointer<Array<128, Register>>,
            cb: impl FnOnce(&mut RegisterArray),
        ) -> RegisterArray {
            let mut array = RegisterArray {
                registers: Default::default(),
                pad: Default::default(),
                addr: self_ptr,
                count: 0,
                length: 0,
                unk_pad: 0,
            };

            cb(&mut array);

            array
        }

        pub(crate) fn add(&mut self, number: u32, value: u64) {
            self.registers[self.count as usize] = Register::new(number, value);
            self.count += 1;
            self.length += core::mem::size_of::<Register>() as u16;
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct TimestampPointers<'a> {
        pub(crate) start_addr: Option<GpuPointer<'a, AtomicU64>>,
        pub(crate) end_addr: Option<GpuPointer<'a, AtomicU64>>,
    }
}

trivial_gpustruct!(JobTimestamps);
trivial_gpustruct!(RenderTimestamps);

#[derive(Debug)]
pub(crate) struct UserTimestamp {
    pub(crate) mapping: Arc<mmu::KernelMapping>,
    pub(crate) offset: usize,
}

#[derive(Debug, Default)]
pub(crate) struct UserTimestamps {
    pub(crate) start: Option<UserTimestamp>,
    pub(crate) end: Option<UserTimestamp>,
}

impl UserTimestamps {
    pub(crate) fn any(&self) -> bool {
        self.start.is_some() || self.end.is_some()
    }

    pub(crate) fn pointers(&self) -> Result<raw::TimestampPointers<'_>> {
        Ok(raw::TimestampPointers {
            start_addr: self
                .start
                .as_ref()
                .map(|a| GpuPointer::from_mapping(&a.mapping, a.offset))
                .transpose()?,
            end_addr: self
                .end
                .as_ref()
                .map(|a| GpuPointer::from_mapping(&a.mapping, a.offset))
                .transpose()?,
        })
    }
}
