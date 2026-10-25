#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Build the Apple KUnit suites into one arm64 kernel, run it, check the results.

aurora.kunitconfig selects every KUnit suite for this tree's Apple, Asahi and
Aurora code (drivers, their Rust code, and the kernel crate they use), with
the 16K pages and 48-bit VA the shipped kernel has. tools/testing/kunit/
kunit.py builds it with LLVM and boots it in QEMU; it uses KVM when /dev/kvm
works and TCG otherwise.

kunit.py's own verdict is not used: it fails on any failed case, expected or
not, and does not see kernel warnings inside cases that still pass. This
script compares every non-passing case and every splat with
kunit-expected.json instead, and fails on anything new, changed or missing.

  --install-toolchain   (Ubuntu) LLVM 18 and QEMU from apt, Rust 1.85.0 with
                        rust-src and rustfmt, bindgen-cli 0.71.1 from crates.io
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KUNITCONFIG = HERE / "aurora.kunitconfig"
EXPECTED = HERE / "kunit-expected.json"
RUST = "1.85.0"
BINDGEN = "0.71.1"
LLVM_DIR = Path("/usr/lib/llvm-18")
APT = ["bc", "bison", "build-essential", "clang-18", "flex", "libclang-18-dev", "libelf-dev",
       "libssl-dev", "lld-18", "llvm-18", "python3", "qemu-system-arm"]


def bindgen_root():
    return Path(os.environ.get("RUNNER_TEMP", Path.home() / ".cache")) / f"bindgen-{BINDGEN}"


def install_toolchain():
    subprocess.run(["sudo", "apt-get", "update", "-q"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "-q", "--no-install-recommends", *APT], check=True)
    subprocess.run(["rustup", "toolchain", "install", RUST, "--profile", "minimal",
                    "--component", "rust-src", "--component", "rustfmt"], check=True)
    subprocess.run(["cargo", f"+{RUST}", "install", "--locked", "bindgen-cli", "--version", BINDGEN,
                    "--root", str(bindgen_root())], check=True)


def kernel_env():
    env = dict(os.environ, RUSTUP_TOOLCHAIN=RUST, LIBCLANG_PATH=str(LLVM_DIR / "lib"))
    # RUST_LOG makes bindgen print an extra version line, which the kernel's
    # Rust availability check then misreads as a libclang mismatch.
    env.pop("RUST_LOG", None)
    return env


def make_options():
    return [f"LLVM={LLVM_DIR}/bin/", f"BINDGEN={bindgen_root() / 'bin/bindgen'}"]


# --- result checking -------------------------------------------------------

TIME = re.compile(r"^\[\s*\d+\.\d+\]\s?")
SUBTEST = re.compile(r"^(\s*)# Subtest: (\S+)")
RESULT = re.compile(r"^(\s*)(?:not )?ok \d+ (\S+)")
SPLATS = [
    (re.compile(r"^WARNING: (?:CPU: \d+ PID: \d+ at )?(\S+?)(?::\d+)? (?:at )?([A-Za-z0-9_.]+)\+"),
     lambda m: f"{m.group(1)}:{m.group(2)}"),
    (re.compile(r"^(kernel BUG at \S+?)(?::\d+)?!"), lambda m: m.group(1)),
    (re.compile(r"^(BUG: [^\n]*?)(?: in | at |$)"), lambda m: m.group(1).strip()),
    (re.compile(r"^(Unable to handle kernel [a-zA-Z ]+)"), lambda m: m.group(1).strip()),
    (re.compile(r"^(Internal error: [^:]+)"), lambda m: m.group(1)),
    (re.compile(r"^(UBSAN: [a-z-]+)"), lambda m: m.group(1)),
    (re.compile(r"^(KASAN: [a-z-]+)"), lambda m: m.group(1)),
    (re.compile(r"^(INFO: task \S+ blocked)"), lambda m: "INFO: task blocked"),
]


def flatten(group, path=(), results=None, suites=None):
    """kunit.py --json tree -> {suite.test[.param]: status}, {suite: Counter}."""
    results = {} if results is None else results
    suites = {} if suites is None else suites
    for sub in group.get("sub_groups", []):
        if not path:
            suites.setdefault(sub["name"], Counter())
        flatten(sub, path + (sub["name"],), results, suites)
    for case in group.get("test_cases", []):
        results[".".join(path + (case["name"],))] = case["status"]
        if path:
            suites.setdefault(path[0], Counter())[case["status"]] += 1
    return results, suites


BOOT = "(during boot, before the first suite)"


def splats_by_test(log_text):
    """Kernel splats in kunit.py's test.log, attributed to the case whose
    result line follows them, or to BOOT when no suite has started yet:
    {test id: Counter(call site)}."""
    found, pending, stack = {}, [], []
    for raw in log_text.splitlines():
        line = TIME.sub("", raw)
        match = SUBTEST.match(line)
        if match:
            depth = len(match.group(1)) // 4
            del stack[depth:]
            stack.extend([""] * (depth - len(stack)))
            stack.append(match.group(2))
            continue
        match = RESULT.match(line)
        if match:
            depth = len(match.group(1)) // 4
            if pending:
                test = ".".join(stack[1:depth + 1] + [match.group(2)]) if depth else match.group(2)
                found.setdefault(test, Counter()).update(pending)
                pending = []
            continue
        for pattern, site in SPLATS:
            match = pattern.match(line)
            if match:
                if stack:
                    pending.append(site(match))
                else:
                    found.setdefault(BOOT, Counter())[site(match)] += 1
                break
    if pending:
        found.setdefault("(after the last result)", Counter()).update(pending)
    return found


def compare(results, splats, expected):
    """Every difference from kunit-expected.json, as text."""
    want, problems = expected["results"], []
    for test, status in sorted(results.items()):
        entry = want.get(test)
        if status == "PASS" and entry:
            problems.append(f"now passes (expected {entry['status']}): {test}")
        elif status != "PASS" and not entry:
            problems.append(f"{status}: {test}")
        elif status != "PASS" and entry["status"] != status:
            problems.append(f"{status} (expected {entry['status']}): {test}")
    problems += [f"did not run (expected {want[t]['status']}): {t}" for t in sorted(want) if t not in results]
    warned = {t: {e["site"]: e["count"] for e in entries} for t, entries in expected.get("warnings", {}).items()}
    for test in sorted(set(splats) | set(warned)):
        seen, known = splats.get(test, Counter()), warned.get(test, {})
        for site in sorted(set(seen) | set(known)):
            if seen.get(site, 0) != known.get(site, 0):
                problems.append(f"kernel splat {site} in {test}: {seen.get(site, 0)} seen, "
                                f"{known.get(site, 0)} expected")
    return problems


def summary(results, suites, splats, problems, expected):
    totals = Counter()
    for counts in suites.values():
        totals.update(counts)
    lines = ["### KUnit (Apple suites)", "",
             f"{len(suites)} suites, {len(results)} cases: {totals['PASS']} pass, {totals['FAIL']} fail, "
             f"{totals['SKIP']} skip, {totals['ERROR']} error; "
             f"{sum(sum(c.values()) for c in splats.values())} kernel splats inside cases.", ""]
    if problems:
        lines += ["**Differences from kunit-expected.json**", ""] + [f"- {p}" for p in problems] + [""]
    known = [t for t, s in sorted(results.items()) if s != "PASS" and t in expected["results"]]
    if known:
        lines += ["<details><summary>Expected non-passing cases</summary>", ""]
        lines += [f"- `{t}` {results[t]}: {expected['results'][t]['reason']}" for t in known] + ["", "</details>", ""]
    lines += ["<details><summary>Per suite</summary>", "", "| Suite | Pass | Fail | Skip | Error |",
              "|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | {c['PASS']} | {c['FAIL']} | {c['SKIP']} | {c['ERROR']} |" for name, c in sorted(suites.items())]
    return "\n".join(lines + ["", "</details>", ""]) + "\n"


def check(results_json, log_path, expected):
    tree = json.loads(Path(results_json).read_text())
    results, suites = flatten(tree)
    if not results:
        return results, suites, {}, ["no test results: the kernel did not build, boot or print KTAP"]
    log_text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    splats = splats_by_test(log_text)
    return results, suites, splats, compare(results, splats, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-toolchain", action="store_true", help="install the toolchain (Ubuntu) and exit")
    parser.add_argument("--output", type=Path, help="build directory; results and logs land here")
    parser.add_argument("--check-only", action="store_true",
                        help="only compare an earlier run's kunit-results.json and test.log in --output")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    if args.install_toolchain:
        install_toolchain()
        return 0
    if not args.output:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    results_json, log = args.output / "kunit-results.json", args.output / "test.log"

    if not args.check_only:
        env = kernel_env()
        options = make_options()
        tree = HERE.parent.parent
        subprocess.run(["make", "ARCH=arm64", f"O={args.output}", *options, "rustavailable"],
                       cwd=tree, env=env, check=True)
        command = [sys.executable, "tools/testing/kunit/kunit.py", "run", "--arch=arm64",
                   f"--build_dir={args.output}", f"--kunitconfig={KUNITCONFIG}", f"--jobs={args.jobs}",
                   "--timeout=1800", f"--json={results_json}", f"--junit={args.output / 'kunit-results.xml'}"]
        for option in options:
            command += ["--make_options", option]
        if not (platform.machine() in ("aarch64", "arm64") and os.access("/dev/kvm", os.R_OK | os.W_OK)):
            # kunit.py's arm64 default is -cpu max; under TCG a fixed model
            # behaves the same on every QEMU (see boot-test.py, cpu_model).
            command.append("--qemu_args=-cpu neoverse-n1")
        results_json.unlink(missing_ok=True)
        status = subprocess.run(command, cwd=tree, env=env).returncode
        print(f"kunit.py exited with status {status}; checking against {EXPECTED.name}")
        if not results_json.exists():
            print("::error title=KUnit::kunit.py wrote no results; see the build or boot output above",
                  file=sys.stderr)
            return 1

    expected = json.loads(EXPECTED.read_text())
    results, suites, splats, problems = check(results_json, log, expected)
    text = summary(results, suites, splats, problems, expected)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    for problem in problems[:60]:
        print(f"::error title=KUnit::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
