// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! System sleep. A Touch ID capture runs on a non-freezable workqueue and keeps
//! talking to the sensor over SPI, so one left running into a suspend is still
//! active while devices go down. A J313 that suspended with the lock screen's
//! capture in flight did not wake. Before the system suspends, end any capture
//! in progress and hold new ones until it has resumed. The hwrng fill thread
//! is not frozen either, so entropy requests are held over the same window.
//! The notifier itself is in `pm_shim.c`; the capture and entropy handling is
//! on [`SepData`].

use core::sync::atomic::{AtomicPtr, Ordering};

use kernel::prelude::*;
use kernel::sync::Arc;
use kernel::types::ForeignOwnable;

use crate::SepData;

/// The notifier has no context pointer, so `SepData` is reached through this
/// global, as in `trusted.rs`. Holds the [`ForeignOwnable::into_foreign`]
/// pointer while registered; `null` otherwise.
static SEP: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());

extern "C" {
    fn sep_pm_register(event: unsafe extern "C" fn(bool)) -> c_int;
    fn sep_pm_unregister();
    fn sep_pm_keep_domains(dev: *mut c_void) -> c_int;
}

/// Attaches every power domain of a multi-domain SEP node and keeps them on
/// across system sleep; see `pm_shim.c`. Must run before the enclave is first
/// spoken to.
pub(crate) fn keep_domains(dev: &kernel::device::Device) -> Result<()> {
    // SAFETY: `dev` is the bound SEP device; the shim ties what it attaches to
    // that device's lifetime through devres.
    let rc = unsafe { sep_pm_keep_domains(dev.as_raw().cast()) };
    if rc != 0 {
        return Err(Error::from_errno(rc));
    }
    Ok(())
}

/// The holds a sleep event takes and releases, on [`SepData`].
trait SleepHolds {
    /// Holds entropy requests, once any in flight is answered.
    fn hold_entropy(&self);
    /// Ends any Touch ID capture and holds new ones.
    fn hold_captures(&self);
    fn release_captures(&self);
    fn release_entropy(&self);
}

impl SleepHolds for SepData {
    fn hold_entropy(&self) {
        self.rng_sleep_prepare();
    }
    fn hold_captures(&self) {
        self.sleep_prepare()
    }
    fn release_captures(&self) {
        self.sleep_finished();
    }
    fn release_entropy(&self) {
        self.rng_sleep_finished();
    }
}

/// `entering`: the system is about to sleep; otherwise it has resumed, or
/// the suspend was aborted.
fn sleep_event(sep: &impl SleepHolds, entering: bool) {
    if entering {
        sep.hold_entropy();
        sep.hold_captures();
    } else {
        sep.release_captures();
        sep.release_entropy();
    }
}

/// # Safety
/// Called by `pm_shim.c` from the PM notifier chain, in process context.
unsafe extern "C" fn pm_event(entering: bool) {
    let ptr = SEP.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // SAFETY: produced by `into_foreign` in `register`, and reclaimed by
    // `unregister` only after `sep_pm_unregister` has waited out any running
    // notifier call, so it is live for this borrow.
    let sep = unsafe { <Arc<SepData> as ForeignOwnable>::borrow(ptr) };
    sleep_event(&*sep, entering);
}

pub(crate) fn register(sep: Arc<SepData>) -> Result<()> {
    let ptr = sep.into_foreign();

    if SEP
        .compare_exchange(
            core::ptr::null_mut(),
            ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        // SAFETY: `ptr` was just produced by `into_foreign` and never published.
        drop(unsafe { <Arc<SepData> as ForeignOwnable>::from_foreign(ptr) });
        return Err(EBUSY);
    }

    // SAFETY: the shim stores the callback and registers a notifier block
    // that lives for the module's lifetime.
    let rc = unsafe { sep_pm_register(pm_event) };
    if rc != 0 {
        let raw = SEP.swap(core::ptr::null_mut(), Ordering::AcqRel);
        if !raw.is_null() {
            // SAFETY: `raw` is the pointer we published; no notifier is wired.
            drop(unsafe { <Arc<SepData> as ForeignOwnable>::from_foreign(raw) });
        }
        return Err(Error::from_errno(rc));
    }
    Ok(())
}

/// Safe to call even if [`register`] never ran. Unregisters the notifier,
/// which waits for a running call to finish, before reclaiming the `Arc`.
pub(crate) fn unregister() {
    if SEP.load(Ordering::Acquire).is_null() {
        return;
    }
    // SAFETY: no preconditions; matched with `sep_pm_register` above.
    unsafe { sep_pm_unregister() };

    let ptr = SEP.swap(core::ptr::null_mut(), Ordering::AcqRel);
    if !ptr.is_null() {
        // SAFETY: `ptr` came from `into_foreign` in `register`, and the
        // notifier can no longer call `pm_event`.
        drop(unsafe { <Arc<SepData> as ForeignOwnable>::from_foreign(ptr) });
    }
}

#[kernel::macros::kunit_tests(apple_sep_pm)]
mod tests {
    use super::*;
    use core::cell::{Cell, RefCell};

    const HOLD_ENTROPY: u8 = 1;
    const HOLD_CAPTURES: u8 = 2;
    const RELEASE_CAPTURES: u8 = 3;
    const RELEASE_ENTROPY: u8 = 4;

    struct Calls {
        made: RefCell<[u8; 8]>,
        count: Cell<usize>,
    }

    impl Calls {
        fn new() -> Self {
            Calls {
                made: RefCell::new([0; 8]),
                count: Cell::new(0),
            }
        }
        fn call(&self, which: u8) {
            let n = self.count.get();
            self.made.borrow_mut()[n] = which;
            self.count.set(n + 1);
        }
        fn made(&self) -> [u8; 8] {
            *self.made.borrow()
        }
    }

    impl SleepHolds for Calls {
        fn hold_entropy(&self) {
            self.call(HOLD_ENTROPY);
        }
        fn hold_captures(&self) {
            self.call(HOLD_CAPTURES);
        }
        fn release_captures(&self) {
            self.call(RELEASE_CAPTURES);
        }
        fn release_entropy(&self) {
            self.call(RELEASE_ENTROPY);
        }
    }

    #[test]
    fn going_to_sleep_holds_entropy_then_captures() {
        let calls = Calls::new();
        sleep_event(&calls, true);
        assert_eq!(
            calls.made(),
            [HOLD_ENTROPY, HOLD_CAPTURES, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn sleep_holds_stay_until_the_post_event() {
        let calls = Calls::new();
        sleep_event(&calls, true);
        assert_eq!(calls.count.get(), 2);
        sleep_event(&calls, false);
        assert_eq!(
            calls.made(),
            [HOLD_ENTROPY, HOLD_CAPTURES, RELEASE_CAPTURES, RELEASE_ENTROPY, 0, 0, 0, 0]
        );
    }

    #[test]
    fn waking_releases_captures_then_entropy() {
        let calls = Calls::new();
        sleep_event(&calls, false);
        assert_eq!(
            calls.made(),
            [RELEASE_CAPTURES, RELEASE_ENTROPY, 0, 0, 0, 0, 0, 0]
        );
    }
}
