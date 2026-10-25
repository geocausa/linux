# SPDX-License-Identifier: GPL-2.0
import io
import json
import stat

from support import TOOLS, TempDirTestCase, load

boot = load("boot-test")
KVER = "7.2.10-1-2-ARCH"


def read_cpio(data):
    """Parse a newc archive back into {name: (mode, data, rdev)}."""
    entries, offset = {}, 0
    while True:
        header = data[offset:offset + 110]
        assert header[:6] == b"070701", header[:6]
        fields = [int(header[6 + 8 * i:14 + 8 * i], 16) for i in range(13)]
        mode, size, rdev, namesize = fields[1], fields[6], (fields[9], fields[10]), fields[11]
        name = data[offset + 110:offset + 110 + namesize - 1].decode()
        offset += 110 + namesize
        offset += -offset % 4
        body = data[offset:offset + size]
        offset += size
        offset += -offset % 4
        if name == "TRAILER!!!":
            return entries
        entries[name] = (mode, body, rdev)


class LoadOrderTest(TempDirTestCase):
    def moddir(self, dep, order):
        path = self.tmp / "modules"
        path.mkdir()
        (path / "modules.dep").write_text(dep)
        (path / "modules.order").write_text(order)
        return path

    def test_dependencies_load_first(self):
        moddir = self.moddir("a.ko: b.ko c.ko\nb.ko: c.ko\nc.ko:\nd.ko:\n", "a.ko\nd.ko\nb.ko\nc.ko\n")
        order, skipped = boot.load_order(moddir, {})
        self.assertEqual(order, ["c.ko", "b.ko", "a.ko", "d.ko"])
        self.assertEqual(skipped, {})

    def test_skipping_a_module_skips_what_needs_it(self):
        moddir = self.moddir("a.ko: b.ko\nb.ko:\nc.ko:\n", "a.ko\nb.ko\nc.ko\n")
        order, skipped = boot.load_order(moddir, {"b.ko": "noisy"})
        self.assertEqual(order, ["c.ko"])
        self.assertIn("noisy", skipped["b.ko"])
        self.assertIn("b.ko", skipped["a.ko"])

    def test_modules_missing_from_order_still_load(self):
        moddir = self.moddir("a.ko:\nextra.ko:\n", "a.ko\n")
        self.assertEqual(boot.load_order(moddir, {})[0], ["a.ko", "extra.ko"])


class InitramfsTest(TempDirTestCase):
    def test_archive_contents(self):
        moddir = self.tmp / "root/usr/lib/modules" / KVER
        (moddir / "kernel/drivers").mkdir(parents=True)
        (moddir / "kernel/drivers/a.ko").write_bytes(b"module-a")
        (moddir / "modules.dep").write_text("kernel/drivers/a.ko:\n")
        (moddir / "modules.order").write_text("kernel/drivers/a.ko\n")
        (moddir / "vmlinuz").write_bytes(b"image")
        busybox = self.tmp / "busybox"
        busybox.write_bytes(b"busybox")
        output = self.tmp / "initramfs.cpio"
        kver, image, order, skipped = boot.build_initramfs(self.tmp / "root", busybox, output, {})
        self.assertEqual((kver, order, skipped), (KVER, ["kernel/drivers/a.ko"], {}))
        entries = read_cpio(output.read_bytes())
        self.assertEqual(entries[f"lib/modules/{KVER}/kernel/drivers/a.ko"][1], b"module-a")
        self.assertTrue(stat.S_ISDIR(entries[f"lib/modules/{KVER}/kernel/drivers"][0]))
        self.assertTrue(stat.S_ISCHR(entries["dev/console"][0]))
        self.assertEqual(entries["dev/console"][2], (5, 1))
        self.assertEqual(entries["bin/busybox"][0] & 0o777, 0o755)
        self.assertIn(f"/lib/modules/{KVER}/$module", entries["init"][1].decode())
        self.assertEqual(entries["load-order"][1], b"kernel/drivers/a.ko\n")

    def test_writer_pads_to_four_bytes(self):
        stream = io.BytesIO()
        writer = boot.CpioWriter(stream)
        writer.file("x", b"12345")
        writer.close()
        self.assertEqual(len(stream.getvalue()) % 4, 0)
        self.assertEqual(read_cpio(stream.getvalue())["x"][1], b"12345")


GOOD_LOG = f"""\
[    0.000000] Booting Linux on physical CPU 0x0000000000
AURORA-CI-BOOT kernel={KVER}
AURORA-CI-INSMOD ok kernel/a.ko
AURORA-CI-INSMOD fail kernel/nodev.ko No such device
AURORA-CI-INSMOD fail kernel/param.ko Invalid argument
AURORA-CI-TAINT 0
AURORA-CI-LOADED 1
AURORA-CI-DMESG-BEGIN
[    0.000000] Booting Linux on physical CPU 0x0000000000
[    1.000000] a: registered
AURORA-CI-DMESG-END
AURORA-CI-DONE
"""
EXPECTED = {"insmod_errors": {"kernel/param.ko": {"error": "Invalid argument", "reason": "needs params"}}}


class ParseTest(TempDirTestCase):
    def test_a_clean_boot(self):
        report = boot.parse(GOOD_LOG, KVER, EXPECTED)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["modules"]["ok"], ["kernel/a.ko"])
        self.assertEqual(report["modules"]["no_device"], ["kernel/nodev.ko"])
        self.assertEqual(report["modules"]["expected"], ["kernel/param.ko"])

    def test_unexpected_or_changed_error_fails(self):
        log = GOOD_LOG.replace("param.ko Invalid argument", "param.ko Unknown symbol in module")
        report = boot.parse(log, KVER, EXPECTED)
        self.assertEqual(report["failures"], ["kernel/param.ko failed to load: Unknown symbol in module"])

    def test_splats_fail_once_each(self):
        splat = "[    2.000000] WARNING: drivers/x.c:10 at x_probe+0x10/0x20, CPU#0: insmod/12"
        log = GOOD_LOG.replace("AURORA-CI-INSMOD ok kernel/a.ko\n", f"AURORA-CI-INSMOD ok kernel/a.ko\n{splat}\n")
        log = log.replace("[    1.000000] a: registered\n", f"[    1.000000] a: registered\n{splat}\n")
        report = boot.parse(log, KVER, EXPECTED)
        self.assertEqual(report["failures"], [f"kernel log: {splat}"])

    def test_allowed_splat(self):
        log = GOOD_LOG.replace("a: registered", "WARNING: known noise")
        expected = dict(EXPECTED, dmesg_allow=[{"pattern": "known noise", "reason": "test"}])
        self.assertEqual(boot.parse(log, KVER, expected)["failures"], [])

    def test_bad_taint(self):
        report = boot.parse(GOOD_LOG.replace("TAINT 0", "TAINT 512"), KVER, EXPECTED)
        self.assertEqual(report["failures"], ["kernel tainted 512: W (warning)"])
        staging = boot.parse(GOOD_LOG.replace("TAINT 0", "TAINT 1024"), KVER, EXPECTED)
        self.assertEqual(staging["failures"], [])

    def test_no_boot_wrong_release_and_early_stop(self):
        self.assertIn("did not reach userspace", boot.parse("[ 0.0] Kernel panic", KVER, EXPECTED)["failures"][0])
        wrong = boot.parse(GOOD_LOG.replace(f"kernel={KVER}", "kernel=6.0-ARCH"), KVER, EXPECTED)
        self.assertIn("guest reports kernel 6.0-ARCH", wrong["failures"][0])
        early = boot.parse(GOOD_LOG.replace("AURORA-CI-DONE\n", ""), KVER, EXPECTED)
        self.assertEqual(early["failures"], ["the guest stopped before finishing the module run"])

    def test_wrapped_log_fails(self):
        log = GOOD_LOG.replace("[    0.000000] Booting Linux on physical CPU 0x0000000000\n[    1.000000]",
                               "[    1.000000]")
        report = boot.parse(log, KVER, EXPECTED)
        self.assertEqual(report["failures"], ["the kernel log wrapped before it was read; raise log_buf_len"])

    def test_stale_expectation_is_reported_not_failed(self):
        log = GOOD_LOG.replace("fail kernel/param.ko Invalid argument", "ok kernel/param.ko")
        report = boot.parse(log, KVER, EXPECTED)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["stale_expectations"], ["kernel/param.ko"])


class ExpectedFileTest(TempDirTestCase):
    def test_every_entry_has_a_reason(self):
        expected = json.loads((TOOLS / "boot-expected.json").read_text())
        self.assertTrue(all(reason.strip() for reason in expected["skip"].values()))
        for module, entry in expected["insmod_errors"].items():
            with self.subTest(module=module):
                self.assertTrue(entry["error"] and entry["reason"].strip())
                self.assertNotIn(entry["error"], boot.NO_DEVICE)
        for entry in expected["dmesg_allow"]:
            self.assertTrue(entry["pattern"] and entry["reason"].strip())
