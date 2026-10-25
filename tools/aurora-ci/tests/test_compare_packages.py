# SPDX-License-Identifier: GPL-2.0
import copy

from support import TempDirTestCase, load, write_json

compare = load("compare-packages")

BASE = {
    "schema": 1,
    "kernelrelease": "7.2.10-1-2-ARCH",
    "package": {"file": "linux-aurora-7.2.10.aurora1-2-aarch64.pkg.tar.zst",
                "files": ["usr/lib/modules/<kernelrelease>/vmlinuz"]},
    "headers": {"file": "linux-aurora-headers-7.2.10.aurora1-2-aarch64.pkg.tar.zst", "files": []},
    "toolchain": ["clang18-18.1.8-2-aarch64"],
    "image": {"size": 1000, "page_size": "16K"},
    "dtbs": {"t8103-j293.dtb": "aa", "t6000-j314s.dtb": "bb"},
    "modules": ["kernel/drivers/hid/hid-apple.ko", "kernel/drivers/gpu/drm/asahi/asahi.ko"],
    "builtin": [],
    "config": {"CONFIG_RUST": "y", "CONFIG_DRM_ASAHI": "m"},
}


class CompareTest(TempDirTestCase):
    def head(self):
        return copy.deepcopy(BASE)

    def test_identical_builds_have_no_differences(self):
        result = compare.compare(BASE, self.head(), {})
        self.assertEqual(result["failures"], [])
        self.assertEqual(compare.differences(result), [])

    def test_a_removed_dtb_fails(self):
        head = self.head()
        del head["dtbs"]["t8103-j293.dtb"]
        result = compare.compare(BASE, head, {})
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("t8103-j293.dtb", result["failures"][0])

    def test_a_retired_dtb_is_a_note(self):
        head = self.head()
        del head["dtbs"]["t8103-j293.dtb"]
        result = compare.compare(BASE, head, {"t8103-j293.dtb": "merged into j294"})
        self.assertEqual(result["failures"], [])
        self.assertIn("merged into j294", result["notes"][0])

    def test_added_and_changed_dtbs_are_reported(self):
        head = self.head()
        head["dtbs"]["t6000-j314s.dtb"] = "cc"
        head["dtbs"]["t8122-j613.dtb"] = "dd"
        result = compare.compare(BASE, head, {})
        self.assertEqual(result["dtbs"], {"removed": [], "added": ["t8122-j613.dtb"],
                                          "changed": ["t6000-j314s.dtb"]})
        self.assertEqual(result["failures"], [])

    def test_modules_that_became_built_in(self):
        head = self.head()
        head["modules"] = ["kernel/drivers/gpu/drm/asahi/asahi.ko"]
        head["builtin"] = ["kernel/drivers/hid/hid-apple.ko"]
        result = compare.compare(BASE, head, {})
        self.assertEqual(result["modules"]["became_builtin"], ["kernel/drivers/hid/hid-apple.ko"])
        self.assertEqual(result["modules"]["removed"], [])

    def test_compressed_module_that_became_built_in(self):
        base = copy.deepcopy(BASE)
        base["modules"] = ["kernel/drivers/hid/hid-apple.ko.zst"]
        head = self.head()
        head["modules"], head["builtin"] = [], ["kernel/drivers/hid/hid-apple.ko"]
        result = compare.compare(base, head, {})
        self.assertEqual(result["modules"]["became_builtin"], ["kernel/drivers/hid/hid-apple.ko.zst"])

    def test_removed_module_and_config_change(self):
        head = self.head()
        head["modules"] = ["kernel/drivers/hid/hid-apple.ko"]
        head["config"]["CONFIG_DRM_ASAHI"] = "n"
        result = compare.compare(BASE, head, {})
        self.assertEqual(result["modules"]["removed"], ["kernel/drivers/gpu/drm/asahi/asahi.ko"])
        self.assertEqual(result["config"], [{"symbol": "CONFIG_DRM_ASAHI", "base": "m", "head": "n"}])
        self.assertEqual(result["failures"], [])
        self.assertEqual(len(compare.differences(result)), 2)

    def test_retired_file_format(self):
        path = self.tmp / "retired.txt"
        path.write_text("# comment\n\nt8103-j999.dtb  merged into j998  # trailing\n")
        self.assertEqual(compare.retired_dtbs(path), {"t8103-j999.dtb": "merged into j998"})
        self.assertEqual(compare.retired_dtbs(self.tmp / "missing.txt"), {})

    def test_main_exit_codes(self):
        self.github_env()
        base = write_json(self.tmp / "base.json", BASE)
        changed = self.head()
        changed["config"]["CONFIG_X"] = "y"
        head = write_json(self.tmp / "head.json", changed)
        self.assertEqual(compare.main(["--base", str(base), "--head", str(head)]), 0)
        self.assertEqual(compare.main(["--base", str(base), "--head", str(head), "--expect-identical"]), 1)
        gone = self.head()
        gone["dtbs"] = {}
        self.assertEqual(compare.main(["--base", str(base),
                                       "--head", str(write_json(self.tmp / "gone.json", gone))]), 1)

    def test_rejects_unknown_schema(self):
        self.github_env()
        base = write_json(self.tmp / "base.json", dict(BASE, schema=2))
        self.assertEqual(compare.main(["--base", str(base), "--head", str(base)]), 1)
