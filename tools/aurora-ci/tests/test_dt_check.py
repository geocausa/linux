# SPDX-License-Identifier: GPL-2.0
import json

from support import TempDirTestCase, commit_file, git, load

dt = load("dt-check")


def warning(dtb, key, scope=None):
    return dt.problem("schema", scope or dtb, [dtb, key], f"{key} is not valid")


class MaskTest(TempDirTestCase):
    def test_locations_and_byte_values_are_masked(self):
        text = "arch/arm64/boot/dts/apple/t8103.dtsi:784.3-20: value b'\\x00\\x01' at t8103.dtsi:12"
        self.assertEqual(dt.mask(text), "<loc>: value b'...' at <loc>")

    def test_text_without_locations_is_unchanged(self):
        self.assertEqual(dt.mask("'compatible' is a required property"), "'compatible' is a required property")


class ClassifyTest(TempDirTestCase):
    def test_paths(self):
        cases = {
            "Documentation/devicetree/bindings/iommu/apple,dart.yaml": "binding",
            "Documentation/devicetree/bindings/Makefile": "binding-infra",
            "arch/arm64/boot/dts/apple/t8103.dtsi": "dts",
            "include/dt-bindings/pinctrl/apple.h": "dts",
            "include/uapi/linux/input-event-codes.h": "dts",
            "scripts/dtc/dtc.c": "tooling",
            "scripts/Makefile.dtbs": "tooling",
            "scripts/dtc/dt-check-style": "binding-infra",
            "drivers/gpu/drm/apple/dcp.c": None,
            "README": None,
        }
        targets = {"include/uapi/linux/input-event-codes.h"}
        for path, kind in cases.items():
            with self.subTest(path=path):
                self.assertEqual(dt.classify(path, targets), kind)


class CompareTest(TempDirTestCase):
    def test_only_added_problems_are_new(self):
        base = [warning("a.dtb", "x"), warning("a.dtb", "y")]
        head = [warning("a.dtb", "x"), warning("a.dtb", "z")]
        new, fixed = dt.compare(base, head, {})
        self.assertEqual([p["key"] for p in new], [["a.dtb", "z"]])
        self.assertEqual([p["key"] for p in fixed], [["a.dtb", "y"]])

    def test_duplicates_are_counted(self):
        base = [warning("a.dtb", "x")]
        head = [warning("a.dtb", "x"), warning("a.dtb", "x")]
        new, fixed = dt.compare(base, head, {})
        self.assertEqual((len(new), len(fixed)), (1, 0))

    def test_a_renamed_dtb_keeps_its_problems(self):
        base = [warning("old.dtb", "x", scope="old.dtb")]
        head = [dt.problem("schema", "new.dtb", ["old.dtb", "x"], "x is not valid")]
        new, fixed = dt.compare(base, head, {"old.dtb": "new.dtb"})
        self.assertEqual((new, fixed), ([], []))


class ChangedFilesTest(TempDirTestCase):
    def test_renames_additions_and_deletions(self):
        repo = self.init_repo()
        commit_file(repo, "arch/arm64/boot/dts/apple/a.dtsi", "/ { };\n" * 20, "base")
        commit_file(repo, "drivers/x.c", "int x;\n", "more")
        base = git(repo, "rev-parse", "HEAD")
        git(repo, "mv", "arch/arm64/boot/dts/apple/a.dtsi", "arch/arm64/boot/dts/apple/b.dtsi")
        git(repo, "rm", "--quiet", "drivers/x.c")
        commit_file(repo, "README", "new\n", "head")
        changes = {c["path"] or c["old"]: c for c in dt.changed_files(str(repo), base, "HEAD")}
        self.assertEqual(changes["arch/arm64/boot/dts/apple/b.dtsi"]["status"], "R")
        self.assertEqual(changes["arch/arm64/boot/dts/apple/b.dtsi"]["old"], "arch/arm64/boot/dts/apple/a.dtsi")
        self.assertEqual(changes["drivers/x.c"]["status"], "D")
        self.assertEqual(changes["README"]["status"], "A")

    def test_change_without_device_trees_exits_at_once(self):
        repo = self.init_repo()
        base = commit_file(repo, "arch/arm64/boot/dts/apple/a.dts", "/dts-v1/;\n/ { };\n", "base")
        head = commit_file(repo, "drivers/gpu/drm/apple/dcp.c", "int x;\n", "driver only")
        self.github_env()
        out = self.tmp / "out"
        status = dt.main(["--tree", str(repo), "--base", base, "--head", head, "--out", str(out)])
        report = json.loads((out / "report.json").read_text())
        self.assertEqual((status, report["status"]), (0, "skipped"))
