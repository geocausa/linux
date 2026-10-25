#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Run the tree's hermetic host test suites, as listed in host-tests.json.

These are the suites under tools/ that need no Apple hardware, root, network
or running kernel: Python and Rust tests of driver logic built against the
real sources, their mutation checks, and the tools' own tests. Each entry
runs from the top of the tree:

  argv           the command; {output} is a fresh path that does not exist
                 yet (its parent does), for suites that write results
  setup          commands run first, same substitution (e.g. compile a test)
  env            extra environment, same substitution
  expect         "pass" (exit 0, the default) or "fail" (non-zero: some
                 mutation checks report a caught mutant that way)
  rust           the toolchain, passed as RUSTUP_TOOLCHAIN
  commits        history the suite reads (--fetch-commits fetches it)
  known_failure  why the suite fails today; it runs and is reported, but does
                 not fail the job, and passing again is reported too
  serial         the suite measures time, so it runs alone after the others

Suites run in a controlled environment: TMPDIR outside /home, no
XDG_CONFIG_HOME, no bytecode files, and a sudo that always refuses, so no
suite can change the machine it runs on. The tree must be as clean after the
run as before it. Suites that cannot run here are listed under "excluded"
with the reason; the unit tests check that every test-like file under the
"discovery" roots is listed in one place or the other.
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MANIFEST = Path(__file__).with_name("host-tests.json")
SUDO_STUB = "#!/bin/sh\necho \"sudo is not available to host test suites: $*\" >&2\nexit 1\n"


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    names = [suite["name"] for suite in manifest["suites"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: suite names must be unique")
    for suite in manifest["suites"]:
        if suite.get("expect", "pass") not in ("pass", "fail"):
            raise ValueError(f"{suite['name']}: expect must be pass or fail")
    return manifest


def select(manifest, groups=None, only=None):
    suites = [s for s in manifest["suites"] if not groups or s.get("group", "tools") in groups]
    if only:
        unknown = set(only) - {s["name"] for s in suites}
        if unknown:
            raise ValueError(f"unknown suite(s): {', '.join(sorted(unknown))}")
        suites = [s for s in suites if s["name"] in only]
    return suites


def install(manifest, suites):
    packages = manifest.get("apt", [])
    if packages:
        subprocess.run(["sudo", "apt-get", "update", "-q"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-q", "--no-install-recommends", *packages], check=True)
    for version in sorted({s["rust"] for s in suites if s.get("rust")}):
        subprocess.run(["rustup", "toolchain", "install", version, "--profile", "minimal"], check=True)
    # Ubuntu 24.04 keeps unprivileged user namespaces from bubblewrap, which
    # the desktop-session suites run their helpers in.
    sysctl = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    if sysctl.exists() and sysctl.read_text().strip() != "0":
        subprocess.run(["sudo", "sysctl", "-w", "kernel.apparmor_restrict_unprivileged_userns=0"], check=True)


def fetch_commits(tree, suites, remote="origin"):
    """Fetch the history the suites read. Returns (fetched, unavailable)."""
    def present(commit):
        return subprocess.run(["git", "-C", str(tree), "cat-file", "-e", f"{commit}^{{commit}}"],
                              stderr=subprocess.DEVNULL).returncode == 0

    def fetch(*commits):
        return subprocess.run(["git", "-C", str(tree), "fetch", "--quiet", "--no-tags", "--depth=1", remote,
                               *commits], stderr=subprocess.DEVNULL).returncode == 0

    missing = [c for c in sorted({c for s in suites for c in s.get("commits", [])}) if not present(c)]
    if missing and not fetch(*missing):
        # One unknown commit fails the whole fetch; find which.
        for commit in missing:
            fetch(commit)
    return [c for c in missing if present(c)], {c for c in missing if not present(c)}


def tree_state(tree):
    result = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=all"],
                            check=True, text=True, stdout=subprocess.PIPE)
    return set(result.stdout.splitlines())


def base_environment(scratch):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("XDG_CONFIG_HOME", None)
    tmp = scratch / "tmp"
    stub = scratch / "bin"
    tmp.mkdir()
    stub.mkdir()
    (stub / "sudo").write_text(SUDO_STUB)
    (stub / "sudo").chmod(0o755)
    env.update(TMPDIR=str(tmp), PATH=f"{stub}{os.pathsep}{env.get('PATH', '')}")
    return env


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:120]


def run_suite(suite, tree, output, env, unavailable=frozenset()):
    workdir = output / safe_name(suite["name"])
    workdir.mkdir(parents=True)
    lacking = sorted(set(suite.get("commits", [])) & unavailable)
    if lacking:
        note = f"history unavailable: {', '.join(c[:12] for c in lacking)}"
        (workdir / "log.txt").write_text(note + "\n")
        return {"name": suite["name"], "status": "skipped", "seconds": 0.0,
                "known_failure": suite.get("known_failure"), "log": note}
    fresh = workdir / "out"
    fill = lambda text: text.replace("{output}", str(fresh))  # noqa: E731
    suite_env = dict(env, **{k: fill(v) for k, v in suite.get("env", {}).items()})
    if suite.get("rust"):
        suite_env["RUSTUP_TOOLCHAIN"] = suite["rust"]
    log, started, status = [], time.monotonic(), None
    timeout = suite.get("timeout", 1200)
    for step in suite.get("setup", []) + [suite["argv"]]:
        command = [fill(part) for part in step]
        log.append("$ " + " ".join(command))
        try:
            result = subprocess.run(command, cwd=tree, env=suite_env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as expired:
            output_text = expired.stdout or ""
            log.append(output_text if isinstance(output_text, str) else output_text.decode(errors="replace"))
            status = "timeout"
            break
        except OSError as error:
            log.append(str(error))
            status = "could not start"
            break
        log.append(result.stdout)
        if step is not suite["argv"] and result.returncode:
            status = f"setup failed ({result.returncode})"
            break
        if step is suite["argv"]:
            want_pass = suite.get("expect", "pass") == "pass"
            ok = (result.returncode == 0) == want_pass
            status = "pass" if ok else f"exit {result.returncode}, expected {'0' if want_pass else 'non-zero'}"
    text = "\n".join(log)
    (workdir / "log.txt").write_text(text)
    return {"name": suite["name"], "status": status, "seconds": round(time.monotonic() - started, 1),
            "known_failure": suite.get("known_failure"), "log": text}


def verdict(result):
    if result["status"] == "skipped":
        return "skipped"
    if result["known_failure"]:
        return "known failure, now passes" if result["status"] == "pass" else "known failure"
    return "pass" if result["status"] == "pass" else "FAIL"


def report(results, dirtied, fetched, unavailable=frozenset()):
    failed = [r for r in results if verdict(r) == "FAIL"]
    known = [r for r in results if r["known_failure"] and verdict(r) != "skipped"]
    skipped = [r for r in results if verdict(r) == "skipped"]
    lines = ["### Host test suites", "",
             f"{len(results)} suites: {sum(verdict(r) == 'pass' for r in results)} passed, {len(failed)} failed, "
             f"{len(known)} known failures, {len(skipped)} skipped; "
             f"{sum(r['seconds'] for r in results):.0f} s of suite time.", ""]
    if fetched:
        lines += [f"Fetched {len(fetched)} commit(s) the suites read.", ""]
    if unavailable:
        lines += [f"**{len(unavailable)} commit(s) the suites read are not in this repository**, so "
                  f"{len(skipped)} suite(s) were skipped: " + ", ".join(f"`{c[:12]}`" for c in sorted(unavailable)), ""]
    if failed:
        lines += ["**Failed**", ""] + [f"- `{r['name']}`: {r['status']}" for r in failed] + [""]
    if known:
        lines += ["Known failures (not blocking):", ""]
        lines += [f"- `{r['name']}`: {r['known_failure']}"
                  + (" **Now passes: remove `known_failure`.**" if r["status"] == "pass" else "") for r in known]
        lines.append("")
    if dirtied:
        lines += ["**The suites changed the tree:**", ""] + [f"- `{line}`" for line in dirtied[:40]] + [""]
    lines += ["<details><summary>Every suite</summary>", "", "| Suite | Result | Seconds |", "|---|---|---|"]
    lines += [f"| `{r['name']}` | {verdict(r)} | {r['seconds']} |" for r in results]
    return "\n".join(lines + ["", "</details>", ""]) + "\n", failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--tree", type=Path, default=Path("."))
    parser.add_argument("--group", action="append", help="run only this group (repeatable)")
    parser.add_argument("--only", action="append", help="run only this suite (repeatable)")
    parser.add_argument("--output", type=Path, help="directory for each suite's output and log")
    parser.add_argument("--jobs", type=int, default=1, help="suites to run at once")
    parser.add_argument("--list", action="store_true", help="list the selected suites and exit")
    parser.add_argument("--install", action="store_true",
                        help="install what the selected suites need (apt, Rust, sysctl) and exit")
    parser.add_argument("--fetch-commits", action="store_true",
                        help="fetch the history the selected suites read before running them")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        suites = select(manifest, args.group, args.only)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for suite in suites:
            print(f"{suite['name']}: {' '.join(suite['argv'])}")
        return 0
    if args.install:
        install(manifest, suites)
        return 0
    if not args.output:
        parser.error("--output is required to run suites")
    tree = args.tree.resolve()
    fetched, unavailable = fetch_commits(tree, suites) if args.fetch_commits else ([], set())
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    output = args.output.resolve()

    in_ci = bool(os.environ.get("GITHUB_ACTIONS"))
    before = tree_state(tree)
    with tempfile.TemporaryDirectory(prefix="aurora-ci-host-", dir="/tmp") as scratch:
        env = base_environment(Path(scratch))
        # Serial suites measure time; they run alone, after the parallel batch.
        parallel = [s for s in suites if not s.get("serial")]
        serial = [s for s in suites if s.get("serial")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(run_suite, suite, tree, output, env, unavailable) for suite in parallel]
            results = []
            for future in futures + [None] * len(serial):
                result = future.result() if future else run_suite(serial[len(results) - len(parallel)],
                                                                  tree, output, env, unavailable)
                results.append(result)
                if in_ci:
                    print(f"::group::{result['name']}: {verdict(result)} ({result['seconds']} s)")
                    print(result["log"])
                    print("::endgroup::")
                else:
                    print(f"{result['name']}: {verdict(result)} ({result['seconds']} s)")
    dirtied = sorted(tree_state(tree) - before)

    text, failed = report(results, dirtied, fetched, unavailable)
    if not in_ci:
        print(text.split("<details>")[0], end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    (output / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "log"} for r in results], indent=1) + "\n")
    for r in failed:
        print(f"::error title=Host test failed::{r['name']}: {r['status']}", file=sys.stderr)
    for r in results:
        if r["known_failure"] and r["status"] == "pass":
            print(f"::warning title=Known failure now passes::{r['name']}: remove known_failure", file=sys.stderr)
    if unavailable:
        print(f"::warning title=History unavailable::{len(unavailable)} commit(s) the suites read could not be "
              "fetched; the suites that need them were skipped", file=sys.stderr)
    if dirtied:
        print(f"::error title=Host tests changed the tree::{len(dirtied)} paths, first {dirtied[0]}", file=sys.stderr)
    return 1 if failed or dirtied else 0


if __name__ == "__main__":
    sys.exit(main())
