# SPDX-License-Identifier: GPL-2.0
import json
import shutil
import struct
import subprocess
import unittest
from pathlib import Path

from support import TempDirTestCase, load

inspect = load("inspect-package")

KVER = "7.2.10-1-2.5-ARCH"


def image(page_size_code=2, big_endian=False, magic=b"ARM\x64"):
    flags = (page_size_code << 1) | int(big_endian)
    header = struct.pack("<II QQQ QQQ", 0, 0, 0, 4096, flags, 0, 0, 0) + magic + b"\0" * 4
    return header + b"\0" * 4096


def pack(root, output):
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    subprocess.run(["tar", "--zstd", "-cf", str(output), "-C", str(root), *names], check=True)


@unittest.skipUnless(shutil.which("depmod") and shutil.which("zstd"), "needs kmod's depmod and zstd")
class DescribeTest(TempDirTestCase):
    def make_packages(self, kver=KVER, config="CONFIG_RUST=y\nCONFIG_ARM64_16K_PAGES=y\n",
                      kernel=None, dtbs=("t8103-j293.dtb",)):
        main = self.tmp / "main"
        moddir = main / "usr/lib/modules" / kver
        (moddir / "dtbs").mkdir(parents=True)
        (moddir / "vmlinuz").write_bytes(kernel if kernel is not None else image())
        for dtb in dtbs:
            (moddir / "dtbs" / dtb).write_bytes(b"\xd0\x0d\xfe\xed" + dtb.encode())
        for name in ("modules.dep", "modules.order", "modules.builtin"):
            (moddir / name).write_text("")
        (moddir / "pkgbase").write_text("linux-aurora\n")
        (main / ".PKGINFO").write_text("pkgname = linux-aurora\npkgver = 7.2.10.aurora1-2.5\n")
        (main / ".BUILDINFO").write_text("installed = clang18-18.1.8-2-aarch64\ninstalled = bash-5.3-1-aarch64\n")
        headers = self.tmp / "headers"
        build = headers / "usr/lib/modules" / kver / "build"
        build.mkdir(parents=True)
        (build / ".config").write_text(config)
        (build / "System.map").write_text("ffff800080000000 T _text\n")
        (headers / ".PKGINFO").write_text("pkgname = linux-aurora-headers\n")
        out = self.tmp / "packages"
        out.mkdir()
        pack(main, out / "linux-aurora-7.2.10.aurora1-2.5-aarch64.pkg.tar.zst")
        pack(headers, out / "linux-aurora-headers-7.2.10.aurora1-2.5-aarch64.pkg.tar.zst")
        return out

    def describe(self, packages):
        work = self.tmp / "work"
        work.mkdir()
        return inspect.describe(*inspect.find_packages(packages), work)

    def failed(self, meta):
        return [c["name"] for c in meta["checks"] if not c["ok"]]

    def test_a_good_package_passes(self):
        meta = self.describe(self.make_packages())
        self.assertEqual(self.failed(meta), [])
        self.assertEqual(meta["kernelrelease"], KVER)
        self.assertEqual(meta["image"]["page_size"], "16K")
        self.assertEqual(list(meta["dtbs"]), ["t8103-j293.dtb"])
        self.assertEqual(meta["toolchain"], ["clang18-18.1.8-2-aarch64"])
        self.assertIn("usr/lib/modules/<kernelrelease>/vmlinuz", meta["package"]["files"])

    def test_release_without_arch_suffix_fails(self):
        meta = self.describe(self.make_packages(kver="7.2.10-1-2"))
        self.assertEqual(self.failed(meta), ["kernel release ends in -ARCH"])

    def test_page_size_mismatch_fails(self):
        meta = self.describe(self.make_packages(kernel=image(page_size_code=1)))
        self.assertEqual(self.failed(meta), ["Image page size matches the configuration"])

    def test_rust_disabled_fails(self):
        meta = self.describe(self.make_packages(config="# CONFIG_RUST is not set\nCONFIG_ARM64_16K_PAGES=y\n"))
        self.assertEqual(self.failed(meta), ["CONFIG_RUST=y"])

    def test_no_device_trees_fails(self):
        meta = self.describe(self.make_packages(dtbs=()))
        self.assertEqual(self.failed(meta), ["package carries device trees"])

    def test_not_an_image_is_an_error(self):
        with self.assertRaises(inspect.PackageError):
            self.describe(self.make_packages(kernel=image(magic=b"NOPE")))

    def test_main_writes_metadata_and_unpacks(self):
        self.github_env()
        packages = self.make_packages()
        output, unpacked = self.tmp / "meta.json", self.tmp / "root"
        self.assertEqual(inspect.main(["--packages", str(packages), "--output", str(output),
                                       "--unpacked", str(unpacked)]), 0)
        self.assertEqual(json.loads(output.read_text())["schema"], inspect.SCHEMA)
        self.assertTrue((unpacked / "usr/lib/modules" / KVER / "vmlinuz").is_file())


class HelpersTest(TempDirTestCase):
    def test_parse_config(self):
        config = inspect.parse_config("CONFIG_A=y\n# CONFIG_B is not set\nCONFIG_C=\"x=y\"\n# comment\n")
        self.assertEqual(config, {"CONFIG_A": "y", "CONFIG_B": "n", "CONFIG_C": '"x=y"'})

    def test_find_packages_needs_exactly_one_pair(self):
        (self.tmp / "linux-aurora-7.2-1-aarch64.pkg.tar.zst").write_bytes(b"")
        with self.assertRaises(inspect.PackageError):
            inspect.find_packages(self.tmp)
        (self.tmp / "linux-aurora-headers-7.2-1-aarch64.pkg.tar.zst").write_bytes(b"")
        main, headers = inspect.find_packages(self.tmp)
        self.assertEqual((main.name, headers.name), ("linux-aurora-7.2-1-aarch64.pkg.tar.zst",
                                                     "linux-aurora-headers-7.2-1-aarch64.pkg.tar.zst"))

    def test_image_header(self):
        path = Path(self.tmp / "Image")
        path.write_bytes(image(page_size_code=3, big_endian=True))
        self.assertEqual(inspect.image_header(path)["page_size"], "64K")
        self.assertEqual(inspect.image_header(path)["endian"], "be")
