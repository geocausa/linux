// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Common queue functionality.
//!
//! Shared helpers used by the submission logic for multiple command types.

use crate::file;
use crate::fw::job::UserTimestamp;

use core::sync::atomic::{AtomicU64, Ordering};
use kernel::prelude::*;
use kernel::uapi;
use kernel::xarray;

// Apple G15 command descriptors share one kernel-global 64-bit ID counter.
// The host initializes it to 1 and uses an atomic LDADD of 1 for every
// Compute, 3D, and TA descriptor; JobMeta.uuid carries the low 32 bits.
static G15_COMMAND_UUID: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_g15_command_uuid() -> u32 {
    G15_COMMAND_UUID.fetch_add(1, Ordering::Relaxed) as u32
}

pub(super) fn get_timestamp_object(
    objects: Pin<&xarray::XArray<KBox<file::Object>>>,
    timestamp: uapi::drm_asahi_timestamp,
) -> Result<Option<UserTimestamp>> {
    if timestamp.handle == 0 {
        return Ok(None);
    }

    let guard = objects.lock();
    let object = guard
        .get(timestamp.handle.try_into()?)
        .ok_or(ENOENT)?
        .clone();
    core::mem::drop(guard);

    #[allow(irrefutable_let_patterns)]
    if let file::Object::TimestampBuffer(mapping) = object {
        let offset = timestamp.offset;
        if (offset.checked_add(8).ok_or(EINVAL)?) as usize > mapping.size() {
            return Err(ERANGE);
        }
        Ok(Some(UserTimestamp {
            mapping: mapping.clone(),
            offset: offset as usize,
        }))
    } else {
        Err(EINVAL)
    }
}
