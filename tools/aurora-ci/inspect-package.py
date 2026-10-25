#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Describe a built linux-aurora package pair and check what installs rely on.

Reads linux-aurora-<ver>-aarch64.pkg.tar.zst and its -headers package, unpacks
them, and writes a metadata record (kernel release, image header, device
trees, modules, resolved configuration, toolchain from .BUILDINFO) that
compare-packages.py diffs against another build. It fails when the package
would break an install:

  - the kernel release must end in -ARCH (boot bundle DTB discovery);
  - vmlinuz must be a little-endian arm64 Image whose page size matches the
    configuration;
  - the package must carry device trees, modules.dep and pkgbase;
  - every module symbol must resolve (depmod -e against System.map);
  - the configuration must keep CONFIG_RUST=y (Rust drivers ship).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = 1
KVER_TOKEN = "<kernelrelease>"
PAGE_SIZES = {1: "4K", 2: "16K", 3: "64K"}
CONFIG_PAGE_SIZE = {"CONFIG_ARM64_4K_PAGES": "4K", "CONFIG_ARM64_16K_PAGES": "16K",
                    "CONFIG_ARM64_64K_PAGES": "64K"}


class PackageError(Exception):
    pass


def find_packages(directory):
    main, headers = [], []
    for path in sorted(directory.glob("*.pkg.tar.zst")):
        if path.name.startswith("linux-aurora-headers-"):
            headers.append(path)
        elif re.match(r"linux-aurora-[0-9]", path.name):
            main.append(path)
    if len(main) != 1 or len(headers) != 1:
        raise PackageError(f"expected one linux-aurora and one linux-aurora-headers package in "
                           f"{directory}, found {[p.name for p in main + headers]}")
    return main[0], headers[0]


def unpack(package, destination):
    """Unpack a .pkg.tar.zst with the system zstd and tar; return member names."""
    destination.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(["tar", "--zstd", "-tf", str(package)], check=True, text=True,
                           stdout=subprocess.PIPE).stdout.splitlines()
    subprocess.run(["tar", "--zstd", "-xf", str(package), "-C", str(destination)], check=True)
    return names


def parse_pkginfo(text):
    info = {}
    for line in text.splitlines():
        if " = " in line and not line.startswith("#"):
            key, value = line.split(" = ", 1)
            info.setdefault(key, []).append(value)
    return info


def parse_config(text):
    config = {}
    for line in text.splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            key, value = line.split("=", 1)
            config[key] = value
        else:
            match = re.match(r"# (CONFIG_\w+) is not set$", line)
            if match:
                config[match.group(1)] = "n"
    return config


def image_header(path):
    """Decode the arm64 Image header (Documentation/arch/arm64/booting.rst)."""
    with open(path, "rb") as image:
        header = image.read(64)
    if len(header) < 64 or header[56:60] != b"ARM\x64":
        raise PackageError(f"{path.name} is not an arm64 Image (no ARM\\x64 magic)")
    text_offset, image_size, flags = struct.unpack_from("<QQQ", header, 8)
    return {
        "text_offset": text_offset,
        "image_size": image_size,
        "endian": "be" if flags & 1 else "le",
        "page_size": PAGE_SIZES.get((flags >> 1) & 3, "unspecified"),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(names, kernelrelease):
    return sorted(name.replace(kernelrelease, KVER_TOKEN) for name in names
                  if name and not name.endswith("/"))


def run_depmod(root, kernelrelease, system_map):
    """Return the module symbol problems depmod reports, as plain lines."""
    result = subprocess.run(
        ["depmod", "-b", str(root / "usr"), "-e", "-F", str(system_map), "-n", kernelrelease],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    if result.returncode:
        lines.append(f"depmod exited with status {result.returncode}")
    return [line.replace(str(root), "") for line in lines]


def describe(main_pkg, headers_pkg, workdir):
    root = workdir / "root"
    main_names = unpack(main_pkg, root)
    headers_root = workdir / "headers"
    headers_names = unpack(headers_pkg, headers_root)

    module_dirs = [p for p in (root / "usr/lib/modules").iterdir() if p.is_dir()]
    if len(module_dirs) != 1:
        raise PackageError(f"expected one kernel release under usr/lib/modules, found "
                           f"{[p.name for p in module_dirs]}")
    moddir = module_dirs[0]
    kernelrelease = moddir.name
    build = headers_root / "usr/lib/modules" / kernelrelease / "build"
    config_path = build / ".config"
    if not config_path.is_file():
        raise PackageError(f"headers package has no build/.config for {kernelrelease}")
    config = parse_config(config_path.read_text())

    pkginfo = parse_pkginfo((root / ".PKGINFO").read_text())
    buildinfo = parse_pkginfo((root / ".BUILDINFO").read_text()) if (root / ".BUILDINFO").exists() else {}
    toolchain = sorted(p for p in buildinfo.get("installed", [])
                       if re.match(r"(clang|llvm|lld|rust|rustup|pahole|dtc|gcc|binutils|kmod)[0-9]*-", p))

    image = moddir / "vmlinuz"
    header = image_header(image) if image.is_file() else None
    dtbs = {p.name: sha256(p) for p in sorted((moddir / "dtbs").glob("*.dtb"))}
    order = (moddir / "modules.order").read_text().split() if (moddir / "modules.order").exists() else []
    modules = sorted(str(p.relative_to(moddir)) for p in moddir.rglob("*.ko*"))
    builtin = (moddir / "modules.builtin").read_text().split() if (moddir / "modules.builtin").exists() else []
    depmod = run_depmod(root, kernelrelease, build / "System.map") if (build / "System.map").exists() \
        else ["System.map missing from the headers package"]

    page_config = [size for key, size in CONFIG_PAGE_SIZE.items() if config.get(key) == "y"]
    checks = [
        ("kernel release ends in -ARCH", kernelrelease.endswith("-ARCH"), kernelrelease),
        ("vmlinuz is a little-endian arm64 Image", bool(header) and header["endian"] == "le",
         json.dumps(header) if header else "missing"),
        ("Image page size matches the configuration",
         bool(header) and page_config == [header["page_size"]],
         f"Image {header['page_size'] if header else '?'}, config {page_config}"),
        ("package carries device trees", bool(dtbs), f"{len(dtbs)} DTBs"),
        ("modules.dep was generated", (moddir / "modules.dep").is_file(), ""),
        ("pkgbase is linux-aurora",
         (moddir / "pkgbase").is_file() and (moddir / "pkgbase").read_text().strip() == "linux-aurora", ""),
        ("every module symbol resolves", not depmod, "; ".join(depmod[:20])),
        ("CONFIG_RUST=y", config.get("CONFIG_RUST") == "y", config.get("CONFIG_RUST", "absent")),
        ("modules.order lists only shipped modules",
         all(entry in modules for entry in order), f"{len(order)} in modules.order"),
    ]

    return {
        "schema": SCHEMA,
        "kernelrelease": kernelrelease,
        "package": {
            "file": main_pkg.name,
            "version": pkginfo.get("pkgver", ["?"])[0],
            "size": main_pkg.stat().st_size,
            "sha256": sha256(main_pkg),
            "files": normalise(main_names, kernelrelease),
        },
        "headers": {
            "file": headers_pkg.name,
            "size": headers_pkg.stat().st_size,
            "sha256": sha256(headers_pkg),
            "files": normalise(headers_names, kernelrelease),
        },
        "toolchain": toolchain,
        "image": dict(header or {}, size=image.stat().st_size if image.is_file() else 0,
                      sha256=sha256(image) if image.is_file() else None),
        "dtbs": dtbs,
        "modules": modules,
        "builtin": builtin,
        "config": config,
        "depmod": depmod,
        "checks": [{"name": n, "ok": bool(ok), "detail": d} for n, ok, d in checks],
    }


def summary(meta):
    lines = [f"### Package `{meta['package']['file']}`", "",
             f"Kernel release `{meta['kernelrelease']}`; {len(meta['modules'])} modules, "
             f"{len(meta['dtbs'])} device trees, image {meta['image']['size'] // (1 << 20)} MiB "
             f"({meta['image'].get('page_size', '?')} pages).", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    for check in meta["checks"]:
        detail = (check["detail"] or "").replace("|", "\\|")
        lines.append(f"| {check['name']} | {'pass' if check['ok'] else '**FAIL**'} | {detail} |")
    if meta["toolchain"]:
        lines += ["", "Toolchain: " + ", ".join(f"`{p}`" for p in meta["toolchain"])]
    return "\n".join(lines) + "\n\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packages", type=Path, required=True,
                        help="directory holding the two .pkg.tar.zst files")
    parser.add_argument("--output", type=Path, required=True, help="metadata JSON to write")
    parser.add_argument("--unpacked", type=Path,
                        help="keep the unpacked kernel package here (for the boot test)")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="aurora-ci-inspect-") as scratch:
        try:
            main_pkg, headers_pkg = find_packages(args.packages)
            meta = describe(main_pkg, headers_pkg, Path(scratch))
        except (PackageError, OSError, subprocess.CalledProcessError) as error:
            print(f"::error title=Package inspection::{error}", file=sys.stderr)
            return 1
        if args.unpacked:
            if args.unpacked.exists():
                shutil.rmtree(args.unpacked)
            shutil.move(str(Path(scratch) / "root"), str(args.unpacked))

    args.output.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    text = summary(meta)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    failed = [c for c in meta["checks"] if not c["ok"]]
    for check in failed:
        print(f"::error title=Package check failed::{check['name']}: {check['detail']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
