#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Boot the packaged kernel in a QEMU virt machine and load its modules.

This is the shipped kernel, not a test build: the Image and modules come from
the unpacked linux-aurora package. The guest runs a busybox initramfs that
loads every module the package ships (dependencies first, as modules.dep
orders them) and then prints the kernel log. The test fails when:

  - the kernel does not reach userspace, or reports a different release;
  - the log shows a crash or warning splat (WARNING:, BUG:, Oops, a panic,
    UBSAN/KASAN/KFENCE reports, list or refcount corruption, stalls);
  - the kernel taints itself with W, D, B or L;
  - a module fails to load for any reason other than missing hardware
    (ENODEV/ENXIO), unless boot-expected.json lists that exact error.

The VM has none of the Apple hardware, so drivers register and wait without
probing. That exercises symbol resolution, module init and exit paths, and
built-in initcalls on every boot; driver behaviour needs real Macs.
"""

import argparse
import json
import lzma
import gzip
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

NO_DEVICE = {"No such device", "No such device or address"}
SPLAT = re.compile(
    r"(WARNING:|BUG:|Oops|Unable to handle kernel|Internal error:|Kernel panic|"
    r"kernel BUG at|UBSAN:|KASAN:|KFENCE:|refcount_t:|list_add corruption|list_del corruption|"
    r"detected stall|soft lockup|hard LOCKUP|blocked for more than|Unknown symbol|"
    r"disagrees about version of symbol|possible circular locking|possible recursive locking)")
# Taint flags that mean the kernel hit a problem (Documentation/admin-guide/tainted-kernels.rst).
BAD_TAINT = {5: "B (bad page)", 7: "D (died: oops or BUG)", 9: "W (warning)", 14: "L (soft lockup)"}
MARK = "AURORA-CI"

INIT = """#!/bin/busybox sh
/bin/busybox --install -s /bin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null
echo "{mark}-BOOT kernel=$(uname -r)"
while read -r module; do
  if out=$(insmod "/lib/modules/{kver}/$module" 2>&1); then
    echo "{mark}-INSMOD ok $module"
  else
    echo "{mark}-INSMOD fail $module ${{out##*: }}"
  fi
done < /load-order
echo "{mark}-TAINT $(cat /proc/sys/kernel/tainted)"
echo "{mark}-LOADED $(wc -l < /proc/modules)"
echo "{mark}-DMESG-BEGIN"
dmesg
echo "{mark}-DMESG-END"
echo "{mark}-DONE"
poweroff -f
"""


class CpioWriter:
    """Minimal newc (SVR4, no CRC) archive writer for an initramfs."""

    def __init__(self, stream):
        self.stream, self.ino = stream, 0

    def _entry(self, name, mode, data=b"", rdev=(0, 0)):
        self.ino += 1
        encoded = name.encode() + b"\0"
        fields = (self.ino, mode, 0, 0, 1, 0, len(data), 0, 0, rdev[0], rdev[1], len(encoded), 0)
        self.stream.write(b"070701" + b"".join(b"%08x" % f for f in fields) + encoded)
        self._pad(110 + len(encoded))
        self.stream.write(data)
        self._pad(len(data))

    def _pad(self, length):
        self.stream.write(b"\0" * (-length % 4))

    def directory(self, name):
        self._entry(name, stat.S_IFDIR | 0o755)

    def file(self, name, data, mode=0o644):
        self._entry(name, stat.S_IFREG | mode, data)

    def char_device(self, name, major, minor):
        self._entry(name, stat.S_IFCHR | 0o600, rdev=(major, minor))

    def close(self):
        self._entry("TRAILER!!!", 0)


def read_module(path):
    data = path.read_bytes()
    if path.suffix == ".zst":
        return subprocess.run(["zstd", "-dc", str(path)], check=True, stdout=subprocess.PIPE).stdout
    if path.suffix == ".xz":
        return lzma.decompress(data)
    if path.suffix == ".gz":
        return gzip.decompress(data)
    return data


def load_order(moddir, skip):
    """Modules in modules.order, each after its dependencies; drops skipped
    modules and every module that needs one of them."""
    deps = {}
    for line in (moddir / "modules.dep").read_text().splitlines():
        module, _, rest = line.partition(":")
        deps[module] = rest.split()
    order = [m for m in (moddir / "modules.order").read_text().split() if m in deps]
    order += sorted(set(deps) - set(order))

    skipped = {m: f"listed in boot-expected.json: {skip[m]}" for m in skip if m in deps}
    for module, needs in deps.items():
        blocked = [d for d in needs if d in skip]
        if blocked and module not in skipped:
            skipped[module] = f"needs skipped module {blocked[0]}"

    result, seen = [], set()

    def visit(module):
        if module in seen or module in skipped:
            return
        seen.add(module)
        for dependency in reversed(deps.get(module, [])):
            visit(dependency)
        result.append(module)

    for module in order:
        visit(module)
    return result, skipped


def build_initramfs(root, busybox, output, skip):
    moddir_list = [p for p in (root / "usr/lib/modules").iterdir() if p.is_dir()]
    if len(moddir_list) != 1:
        raise RuntimeError("expected one kernel release in the unpacked package")
    moddir = moddir_list[0]
    kver = moddir.name
    order, skipped = load_order(moddir, skip)

    with open(output, "wb") as stream:
        cpio = CpioWriter(stream)
        for directory in ("bin", "dev", "proc", "sys", "tmp", "lib", "lib/modules", f"lib/modules/{kver}"):
            cpio.directory(directory)
        cpio.char_device("dev/console", 5, 1)
        cpio.file("bin/busybox", Path(busybox).read_bytes(), 0o755)
        cpio.file("init", INIT.format(mark=MARK, kver=kver).encode(), 0o755)
        cpio.file("load-order", ("\n".join(order) + "\n").encode())
        for name in sorted(p.name for p in moddir.glob("modules.*")):
            cpio.file(f"lib/modules/{kver}/{name}", (moddir / name).read_bytes())
        made = set()
        for module in order:
            parent = str(Path(module).parent)
            for depth in range(1, len(Path(parent).parts) + 1):
                directory = str(Path(*Path(parent).parts[:depth]))
                if directory not in made:
                    cpio.directory(f"lib/modules/{kver}/{directory}")
                    made.add(directory)
            cpio.file(f"lib/modules/{kver}/{module}", read_module(moddir / module))
        cpio.close()
    return kver, moddir / "vmlinuz", order, skipped


def kvm_usable():
    """KVM runs an arm64 guest only on an arm64 host."""
    return platform.machine() in ("aarch64", "arm64") and os.access("/dev/kvm", os.R_OK | os.W_OK)


def cpu_model(accel, requested):
    """The -cpu value: the host CPU under KVM, and under TCG a fixed model
    rather than "max". "max" turns on every feature the QEMU version
    emulates, which on QEMU 8 means pointer authentication slow enough to
    stall boot and page clearing that faults; a named model is the same on
    every QEMU. Neoverse N1 has the 16K granule this kernel needs."""
    if requested:
        return requested
    return "neoverse-n1" if accel == "tcg" else "host"


def run_qemu(image, initramfs, log, accel, memory, cpus, timeout, cpu=None):
    command = [
        "qemu-system-aarch64", "-M", "virt,gic-version=max", "-cpu", cpu_model(accel, cpu), "-accel", accel,
        "-m", str(memory), "-smp", str(cpus), "-nodefaults", "-display", "none",
        "-serial", f"file:{log}", "-no-reboot",
        "-kernel", str(image), "-initrd", str(initramfs),
        "-append", "console=ttyAMA0 earlycon rdinit=/init panic=-1 loglevel=4 log_buf_len=16M",
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return result.returncode, result.stdout, time.monotonic() - started, False
    except subprocess.TimeoutExpired as expired:
        output = expired.stdout or ""
        return None, output if isinstance(output, str) else output.decode(), timeout, True


def parse(log_text, kver, expected):
    report = {"booted": False, "release": None, "done": False, "taint": None, "loaded": None,
              "log_complete": False,
              "modules": {"ok": [], "no_device": [], "expected": [], "unexpected": []},
              "splats": [], "failures": [], "stale_expectations": []}
    errors = expected.get("insmod_errors", {})
    allowed = [(re.compile(a["pattern"]), a["reason"]) for a in expected.get("dmesg_allow", [])]
    in_dmesg = first_dmesg_line = False
    for line in log_text.splitlines():
        if f"{MARK}-BOOT kernel=" in line:
            report["booted"] = True
            report["release"] = line.split("kernel=", 1)[1].strip()
        elif f"{MARK}-INSMOD " in line:
            fields = line.split(f"{MARK}-INSMOD ", 1)[1].split(" ", 2)
            status, module = fields[0], fields[1]
            error = fields[2].strip() if len(fields) > 2 else ""
            if status == "ok":
                report["modules"]["ok"].append(module)
            elif error in NO_DEVICE:
                report["modules"]["no_device"].append(module)
            elif errors.get(module, {}).get("error") == error:
                report["modules"]["expected"].append(module)
            else:
                report["modules"]["unexpected"].append({"module": module, "error": error})
        elif f"{MARK}-TAINT " in line:
            report["taint"] = int(line.split(f"{MARK}-TAINT ", 1)[1].split()[0])
        elif f"{MARK}-LOADED " in line:
            report["loaded"] = int(line.split(f"{MARK}-LOADED ", 1)[1].split()[0])
        elif f"{MARK}-DMESG-BEGIN" in line:
            in_dmesg = True
            first_dmesg_line = True
            continue
        elif f"{MARK}-DMESG-END" in line:
            in_dmesg = False
        elif f"{MARK}-DONE" in line:
            report["done"] = True
        if in_dmesg and first_dmesg_line:
            # A log that still starts at boot has not wrapped, so no early
            # splat can have been pushed out of it.
            report["log_complete"] = "Booting Linux" in line
            first_dmesg_line = False
        if (in_dmesg or not report["done"]) and SPLAT.search(line) and MARK not in line:
            if not any(pattern.search(line) for pattern, _ in allowed):
                report["splats"].append(line.strip())
    report["splats"] = list(dict.fromkeys(report["splats"]))

    if not report["booted"]:
        report["failures"].append("the kernel did not reach userspace")
    elif report["release"] != kver:
        report["failures"].append(f"guest reports kernel {report['release']}, package ships {kver}")
    if report["booted"] and not report["done"]:
        report["failures"].append("the guest stopped before finishing the module run")
    elif report["done"] and not report["log_complete"]:
        report["failures"].append("the kernel log wrapped before it was read; raise log_buf_len")
    for item in report["modules"]["unexpected"]:
        report["failures"].append(f"{item['module']} failed to load: {item['error']}")
    for splat in report["splats"]:
        report["failures"].append(f"kernel log: {splat}")
    if report["taint"]:
        bad = [name for bit, name in BAD_TAINT.items() if report["taint"] & (1 << bit)]
        if bad:
            report["failures"].append(f"kernel tainted {report['taint']}: {', '.join(bad)}")
    loaded = set(report["modules"]["ok"])
    report["stale_expectations"] = sorted(m for m in errors if m in loaded)
    return report


def summary(report, meta):
    mods = report["modules"]
    lines = ["### Boot test (QEMU virt, shipped kernel)", "",
             f"Accelerator `{meta['accel']}` (`-cpu {meta['cpu']}`), {meta['seconds']:.0f} s, "
             f"kernel `{meta['kver']}`.", "",
             f"Modules: {len(mods['ok'])} loaded, {len(mods['no_device'])} found no device, "
             f"{len(mods['expected'])} failed as expected, {len(mods['unexpected'])} failed, "
             f"{len(meta['skipped'])} skipped. Taint {report['taint']}.", ""]
    if report["failures"]:
        lines += ["**Failures**", ""] + [f"- {f}" for f in report["failures"][:60]] + [""]
    if report["stale_expectations"]:
        lines += ["Now load cleanly (remove from boot-expected.json): "
                  + ", ".join(f"`{m}`" for m in report["stale_expectations"]), ""]
    if meta["skipped"]:
        lines += ["<details><summary>Skipped modules</summary>", ""]
        lines += [f"- `{m}`: {why}" for m, why in sorted(meta["skipped"].items())] + ["", "</details>", ""]
    if mods["no_device"]:
        lines += ["<details><summary>No device in the VM</summary>", ""]
        lines += [f"- `{m}`" for m in mods["no_device"]] + ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="unpacked linux-aurora package")
    parser.add_argument("--busybox", type=Path, required=True, help="static aarch64 busybox")
    parser.add_argument("--expected", type=Path, required=True, help="boot-expected.json")
    parser.add_argument("--out", type=Path, required=True, help="directory for the log and report")
    parser.add_argument("--accel", choices=("auto", "kvm", "tcg"), default="auto")
    parser.add_argument("--memory", type=int, default=4096, help="guest memory in MiB")
    parser.add_argument("--cpus", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=1200, help="seconds before the guest is killed")
    parser.add_argument("--cpu", help="QEMU -cpu value (default: host under KVM, neoverse-n1 under TCG)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expected.read_text())
    initramfs = args.out / "initramfs.cpio"
    kver, image, order, skipped = build_initramfs(args.root, args.busybox, initramfs, expected.get("skip", {}))

    accels = {"auto": ["kvm", "tcg"] if kvm_usable() else ["tcg"], "kvm": ["kvm"], "tcg": ["tcg"]}[args.accel]
    log = args.out / "console.log"
    for accel in accels:
        log.unlink(missing_ok=True)
        status, qemu_output, seconds, timed_out = run_qemu(image, initramfs, log, accel, args.memory,
                                                           args.cpus, args.timeout, args.cpu)
        log_text = log.read_text(errors="replace") if log.exists() else ""
        # A QEMU that cannot start with KVM exits before the guest prints anything.
        if accel == "kvm" and not log_text.strip() and status not in (0, None):
            print(f"KVM start failed ({qemu_output.strip()[:300]}); retrying with TCG", file=sys.stderr)
            continue
        break
    initramfs.unlink()

    report = parse(log_text, kver, expected)
    if timed_out:
        report["failures"].insert(0, f"the guest did not power off within {args.timeout} s")
    elif status != 0:
        report["failures"].insert(0, f"QEMU exited with status {status}: {qemu_output.strip()[:300]}")
    meta = {"kver": kver, "accel": accel, "cpu": cpu_model(accel, args.cpu), "seconds": seconds,
            "skipped": skipped, "planned": len(order)}
    (args.out / "report.json").write_text(json.dumps(dict(report, **meta), indent=1) + "\n")

    text = summary(report, meta)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    for failure in report["failures"][:50]:
        print(f"::error title=Boot test::{failure}", file=sys.stderr)
    if report["failures"] and not report["booted"]:
        print("Last console lines:\n" + "\n".join(log_text.splitlines()[-40:]), file=sys.stderr)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
