#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Check the commits a pull request or merge group adds.

For every non-merge commit in base..head:

  Signed-off-by   at least one trailer is required (blocking). Carried
                  commits keep their author and gain the carrier's sign-off,
                  so the author's own sign-off is not required here.
  checkpatch      scripts/checkpatch.pl findings, reported for review
                  (advisory: the job does not fail on them).

and for the range as a whole:

  UAPI            a change under include/uapi/ is flagged: userspace that
                  builds against those headers (Mesa for the GPU, for one)
                  needs a matching update before the kernel ships.

Ranges of more than --max-commits commits (rebases onto a new upstream,
merges of other branches) are reported but never fail: their commits come
from upstream trees with their own rules.

The checkout may be shallow; history is fetched until base and head meet.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SIGNED_OFF = re.compile(r"^Signed-off-by:\s*\S.*$", re.M)
CHECKPATCH_LINE = re.compile(r"^(?P<file>[^:\s]+):(?P<line>\d+): (?P<level>ERROR|WARNING|CHECK):(?P<type>\w+): (?P<message>.*)$")
DEPTHS = (64, 256, 1024, 4096)


def git(repo, *args, check=True, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check, text=text,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def range_history_complete(repo, base, head):
    if git(repo, "merge-base", base, head, check=False).returncode != 0:
        return False
    shallow = Path(git(repo, "rev-parse", "--path-format=absolute",
                       "--git-path", "shallow").stdout.strip())
    if not shallow.exists():
        return True
    # A recent merge can expose base while an older topic parent is still
    # shallow. Such a boundary looks like a root commit to format-patch.
    boundaries = set(shallow.read_text().split())
    added = set(git(repo, "rev-list", f"{base}..{head}").stdout.split())
    return boundaries.isdisjoint(added)


def ensure_history(repo, remote, base, head):
    """Fetch base and head, deepening until they share history.

    Also require every commit added by the range to have its parents.
    Returns False if either condition remains unmet at the deepest fetch."""
    for depth in DEPTHS:
        if range_history_complete(repo, base, head):
            return True
        git(repo, "fetch", "--quiet", "--no-tags", f"--depth={depth}", remote, base, head)
    return range_history_complete(repo, base, head)


def commits_in(repo, base, head):
    out = git(repo, "rev-list", "--reverse", "--no-merges", f"{base}..{head}").stdout
    return out.split()


def describe(repo, sha):
    fields = git(repo, "show", "-s", "--format=%an <%ae>%x00%s%x00%B", sha).stdout.split("\x00", 2)
    return {"sha": sha, "author": fields[0], "subject": fields[1], "message": fields[2]}


def run_checkpatch(repo, sha):
    """checkpatch.pl findings for one commit, as dicts."""
    # A patch may contain non-UTF-8 source text. Preserve its bytes for perl;
    # only diagnostic output is decoded for the report.
    patch = git(repo, "format-patch", "-1", "--stdout", sha, text=False).stdout
    result = subprocess.run(
        ["perl", "scripts/checkpatch.pl", "--terse", "--showfile", "--show-types", "--no-summary",
         "--color=never", "-"], cwd=repo, input=patch,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    findings = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        match = CHECKPATCH_LINE.match(line.strip())
        if match:
            findings.append(match.groupdict())
    return findings


def changed_uapi(repo, base, head):
    out = git(repo, "diff", "--name-only", f"{base}...{head}", "--", "include/uapi").stdout
    return out.split()


def escape(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def check(repo, remote, base, head, max_commits, checkpatch=True):
    report = {"base": base, "head": head, "commits": [], "uapi": [], "failures": [],
              "mode": "full", "notes": []}
    if not ensure_history(repo, remote, base, head):
        report["mode"] = "unrelated"
        report["notes"].append(f"Complete added history for {base[:12]}..{head[:12]} "
                               f"was not established within {DEPTHS[-1]} commits; "
                               "commit checks skipped")
        return report

    shas = commits_in(repo, base, head)
    report["uapi"] = changed_uapi(repo, base, head)
    if len(shas) > max_commits:
        report["mode"] = "integration"
        report["notes"].append(f"{len(shas)} commits is integration-sized (over {max_commits}); "
                               "sign-offs are reported, not enforced, and checkpatch is skipped")
    for sha in shas:
        entry = describe(repo, sha)
        entry["signed_off"] = bool(SIGNED_OFF.search(entry["message"]))
        del entry["message"]
        entry["checkpatch"] = run_checkpatch(repo, sha) if checkpatch and report["mode"] == "full" else []
        report["commits"].append(entry)
        if not entry["signed_off"] and report["mode"] == "full":
            report["failures"].append(f"{sha[:12]} (\"{entry['subject']}\") has no Signed-off-by trailer")
    return report


def summary(report):
    lines = ["### Commit checks", ""]
    commits = report["commits"]
    lines.append(f"{len(commits)} commits in `{report['base'][:12]}..{report['head'][:12]}`"
                 + (f" ({report['mode']} range)" if report["mode"] != "full" else "") + ".")
    lines += [""] + [f"- {note}" for note in report["notes"]] + [""]
    if report["uapi"]:
        lines += ["**UAPI headers changed** — userspace built against them needs a matching update "
                  "before this kernel ships:", ""] + [f"- `{path}`" for path in report["uapi"]] + [""]
    if commits:
        lines += ["| Commit | Subject | Signed-off-by | checkpatch |", "|---|---|---|---|"]
        for c in commits[:200]:
            counts = {level: sum(1 for f in c["checkpatch"] if f["level"] == level)
                      for level in ("ERROR", "WARNING", "CHECK")}
            cp = ", ".join(f"{n} {level.lower()}" for level, n in counts.items() if n) or "clean"
            if report["mode"] != "full":
                cp = "skipped"
            subject = c["subject"].replace("|", "\\|")[:90]
            lines.append(f"| `{c['sha'][:12]}` | {subject} | {'yes' if c['signed_off'] else '**no**'} | {cp} |")
        if len(commits) > 200:
            lines.append(f"| … {len(commits) - 200} more | | | |")
        lines.append("")
    if report["failures"]:
        lines += ["**Failures**", ""] + [f"- {f}" for f in report["failures"]] + [""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--max-commits", type=int, default=100)
    parser.add_argument("--no-checkpatch", action="store_true")
    parser.add_argument("--report", type=Path, help="write the findings as JSON")
    args = parser.parse_args(argv)

    try:
        report = check(args.repo, args.remote, args.base, args.head, args.max_commits,
                       checkpatch=not args.no_checkpatch)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        print(f"::error title=Commit checks::{' '.join(error.cmd)} failed: {escape(stderr)}",
              file=sys.stderr)
        return 1
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")

    text = summary(report)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    if report["uapi"]:
        print("::warning title=UAPI headers changed::" + escape(
            "This range changes " + ", ".join(report["uapi"])
            + ". Userspace built against these headers needs a matching update before the kernel ships."))
    for c in report["commits"]:
        for f in c["checkpatch"]:
            if f["level"] == "ERROR":
                print(f"::warning file={f['file']},line={f['line']},title=checkpatch {f['type']} "
                      f"({c['sha'][:12]})::{escape(f['message'])}")
    for failure in report["failures"]:
        print(f"::error title=Signed-off-by missing::{escape(failure)}", file=sys.stderr)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
