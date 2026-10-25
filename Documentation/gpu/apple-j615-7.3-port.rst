MacBook Air M3 J615: Linux 7.3 port candidate
===========================================

This is a local experimental port, not an upstream release or a hardware-qualified
replacement for the retained 7.1.6 Golden / 7.1.12 GPU test kernels. Ubuntu remains
26.04. A later Ubuntu 26.10 migration is explicitly deferred.

Provenance
----------

* Final upstream v7.3: c03c1a14507dfaf8b856c2608f9ac216021fc222.
* Downstream Mac input: omacom/linux-aurora
  7876c8e8349c4f3d0a2f4ce982e3e13ce5bb5d70, based on 7.2.11.
* Remove the optional user-subpage topic introduced by merge
  147bf29afbf848cd8e7514de418330d3c628e93a; native 16 KiB is the first-port scope.
* Integrate the remaining tree against upstream 7.3 using explicit base 7.2.11.
  Retain a coherent downstream Apple device tree and fabric API. Keep upstream
  Nova/Tyr code; those non-Apple GPUs are disabled in the J615 build.

The source/build/conflict records are retained in local lab experiment E1098.
The signed upstream tag was checked against Linus's key from kernel.org WKD,
fingerprint ABAF11C65A2970B130ABE3C479BE3E4300411886.

Port changes and limits
----------------------

Use the independent M3 T8122 runtime, without the separate G15 Pro/Max backend.
Keep the initial pstate ceiling at the previously qualified 338 MHz state. Do not
interpret a successful compilation as validation of higher clocks or suspend.

Adapt downstream Rust users to 7.3 module metadata, parameter values, device table
macros, managed mapping lifetimes, DMA addresses and pin initialization. Preserve
the downstream DRM registration and backing-memory lifetime guards. Remove the
unused DriverFile raw-pointer requirement and its panic placeholder.

Adapt the dockchannel IRQ domain, Wi-Fi management-TX cookie and ASMedia capability
mask APIs. For xHCI retain upstream per-port link tracking and the downstream
suspend-to-idle wake filter, with one root-hub resume after that filter.

The Apple firmware join v1/v2 implementation and downstream MCA clock-consumer
implementation are retained together for the first port. Further reconciliation
of upstream SAE/DPP and MCA refinements is pending. No claim of every unrelated
upstream subsystem improvement or other-Mac qualification is made.

The M3 parser retains the 256-byte render and 64-byte compute payload prefixes
used by the known-working Mesa binary. New nonzero extension fields belong to
other runtimes and remain unsupported here. Actual Mesa/GPU compatibility still
requires a bounded live run.

The zero DCP page-flip sequence and Ubuntu/Mutter timed-commit issue are still
open. The measured nowlts workaround is retained for initial tests; the old
untested synthetic elapsed-time counter patch has not been promoted.

Build and qualification
-----------------------

Use tools/asahi/j615-port-build.sh with an out-of-tree output directory. The
recorded configuration enables Rust, native 16 KiB pages, T8122 runtime and native
display while disabling debug information and unrelated Rust GPU backends.

Shared Rust core, the J615 DTB, GPU/display objects and selected host tests pass.
The full Image/modules build is in progress. No 7.3 test boot or persistent GPU
enablement has occurred. Before boot, pin the completed Image/modules/DTB/loader
artifacts, verify recovery and use a guarded one-shot boot. Then verify storage,
input, USB, native display, compute, render, presentation and actual compositor
acceleration before endurance, suspend/resume or persistent activation.
