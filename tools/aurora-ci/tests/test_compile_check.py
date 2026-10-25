# SPDX-License-Identifier: GPL-2.0
from support import TempDirTestCase, load

cc = load("compile-check")

SRC, BUILD = "/work/linux", "/work/out/build"

MAKE_OUTPUT = f"""\
  CC [M]  drivers/gpu/drm/apple/dcp.o
In file included from {SRC}/drivers/gpu/drm/apple/dcp.c:12:
{SRC}/drivers/gpu/drm/apple/dcp.h:137:5: warning: no previous prototype for function 'dcp_x' [-Wmissing-prototypes]
  137 | int dcp_x(void) {{ return 0; }}
      |     ^
  CC      drivers/soc/apple/rtkit.o
{SRC}/drivers/soc/apple/rtkit.c:1507:9: error: use of undeclared identifier 'apple_rtkit_y'
 1507 |         return apple_rtkit_y;
Warning: {SRC}/drivers/soc/apple/rtkit.c:88 function parameter 'rtk' not described in 'apple_rtkit_poll'
make[5]: *** [{SRC}/scripts/Makefile.build:287: drivers/soc/apple/rtkit.o] Error 1
make[4]: *** [{SRC}/scripts/Makefile.build:544: drivers/soc/apple] Error 2
"""


class ParseTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.diags, self.failed, self.misc = cc.parse_make_output(MAKE_OUTPUT, SRC, BUILD)

    def test_compiler_diagnostics_are_attributed_to_their_object(self):
        warning, error = [d for d in self.diags if d["flag"] != "kernel-doc"]
        self.assertEqual((warning["file"], warning["line"], warning["severity"], warning["flag"]),
                         ("drivers/gpu/drm/apple/dcp.h", 137, "warning", "-Wmissing-prototypes"))
        self.assertEqual(warning["object"], "drivers/gpu/drm/apple/dcp.o")
        self.assertEqual(warning["message"], "no previous prototype for function 'dcp_x'")
        self.assertEqual((error["severity"], error["object"]), ("error", "drivers/soc/apple/rtkit.o"))

    def test_kernel_doc_warnings_are_kept(self):
        kdoc = [d for d in self.diags if d["flag"] == "kernel-doc"]
        self.assertEqual([(d["file"], d["line"]) for d in kdoc], [("drivers/soc/apple/rtkit.c", 88)])

    def test_failed_targets_are_files_not_directories(self):
        self.assertEqual(self.failed, ["drivers/soc/apple/rtkit.o"])


class NormaliseTest(TempDirTestCase):
    def test_paths(self):
        self.assertEqual(cc.norm_path(f"{SRC}/drivers/x/epic/../y.h", SRC, BUILD), "drivers/x/y.h")
        self.assertEqual(cc.norm_path(f"{BUILD}/include/generated/autoconf.h", SRC, BUILD),
                         "<build>/include/generated/autoconf.h")
        self.assertEqual(cc.norm_path("include/generated/x.h", SRC, BUILD), "<build>/include/generated/x.h")

    def test_numbers_do_not_make_a_warning_new(self):
        self.assertEqual(cc.norm_msg("array index 4 is past the end of the array (that has type 'u32[3]')"),
                         cc.norm_msg("array index 5 is past the end of the array (that has type 'u32[3]')"))
        self.assertEqual(cc.norm_msg("variable 'x' set but not used"), "variable 'x' set but not used")
