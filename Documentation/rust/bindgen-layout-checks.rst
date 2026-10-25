.. SPDX-License-Identifier: GPL-2.0

Checking generated binding layouts
==================================

bindgen can emit compile-time assertions that compare the size, alignment and
field offsets of every generated Rust type with the values Clang computes for
the C type. As upstream, the kernel build passes ``--no-layout-tests`` to
bindgen, so the generated bindings do not contain these assertions and any
supported bindgen version can build the kernel.

To check the layouts of a configuration, build with ``BINDGEN_LAYOUT_TESTS=1``::

    make LLVM=1 BINDGEN_LAYOUT_TESTS=1 prepare

The bindings are then generated with the assertions, which are checked while
the ``bindings`` and ``uapi`` crates are compiled; nothing is executed. A
mismatch stops the build with an ``E0080`` error naming the type and the
property (``Size of``, ``Alignment of`` or ``Offset of``). Changing the
variable regenerates the bindings. The check covers only the configuration
being built.

Known generator limitations
---------------------------

Unmodified bindgen 0.73 does not reproduce every C layout used by the kernel:

* It omits a named anonymous record member, which Clang accepts with
  ``-fms-extensions``, in a declaration such as::

      struct payload { unsigned long words[20]; };
      union container {
          struct payload;
          unsigned long other;
      };

* It can drop the explicit alignment of a defined zero-sized C struct, such
  as ``struct empty {} __attribute__((aligned(64)));``.

With the assertions enabled and stock bindgen 0.73.2, an arm64 Apple
silicon configuration of this tree fails on the size of ``ns_common`` and of
the namespace structures that embed it; bindgen 0.73.1 also fails on
``task_struct``, ``zone`` and ``pglist_data``. In those builds every offset
and alignment assertion passes: bindgen pads each field to the offset Clang
reports, so only the end of the Rust type is short. Rust code that reaches
such a type through a pointer and reads its fields gets the right data, but
must not rely on ``size_of`` for it, copy it by value or index an array of
it. The affected types depend on the bindgen version and the configuration,
which is what the check is for.

Corrected generator
-------------------

``tools/aurora-bindgen/`` contains pinned patches for bindgen 0.73.2 that
repair both cases, a private build recipe and compile-only tests; see
``tools/aurora-bindgen/README.rst``. It installs nothing system-wide. To check
the layouts with it::

    make LLVM=1 BINDGEN=/path/to/bindgen BINDGEN_LAYOUT_TESTS=1 ...

The generated bindings are consumed by the Rust kernel crate and by several
drivers. After changing the bindings generator, rebuild the kernel Image and
all modules together.

The assertions need bindgen 0.71.1 or later (the kernel minimum) together with
the configured ``--rust-target 1.85``: that selects unconditional constant
assertions rather than ``#[test]`` functions, so a build with
``BINDGEN_LAYOUT_TESTS=1`` checks them without a Rust test configuration.
