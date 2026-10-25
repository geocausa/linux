#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Compare two linux-aurora package descriptions from inspect-package.py.

Reports how a build differs from a baseline (the target branch, or a package
that already ships): resolved configuration, device trees, modules, image size
and package contents. Kernel releases differ between any two builds, so file
names are compared with the release replaced by a placeholder.

It fails only for changes that strand installed Macs:

  - a device tree the baseline shipped is gone. The boot bundle picks a DTB by
    name, so a Mac whose DTB disappears stops booting the new kernel. A
    deliberate retirement is listed, with its reason, in retired-dtbs.txt in
    the same change.

Everything else (modules that became built-in or went away, configuration
changes) is reported for the reviewer. --expect-identical turns every
difference into a failure; it is how CI proves a build matches what ships.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

LIST_LIMIT = 40


def load(path):
    meta = json.loads(Path(path).read_text())
    if meta.get("schema") != 1:
        raise ValueError(f"{path}: unsupported metadata schema {meta.get('schema')!r}")
    return meta


def retired_dtbs(path):
    """Names listed in retired-dtbs.txt: '<name>.dtb  <reason>' per line."""
    if not path or not Path(path).is_file():
        return {}
    retired = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, reason = line.partition(" ")
            retired[name] = reason.strip()
    return retired


def diff_sets(base, head):
    base, head = set(base), set(head)
    return sorted(base - head), sorted(head - base)


def compare(base, head, retired):
    result = {"failures": [], "notes": []}

    removed, added = diff_sets(base["dtbs"], head["dtbs"])
    changed = sorted(n for n in set(base["dtbs"]) & set(head["dtbs"]) if base["dtbs"][n] != head["dtbs"][n])
    stranded = [n for n in removed if n not in retired]
    result["dtbs"] = {"removed": removed, "added": added, "changed": changed}
    for name in stranded:
        result["failures"].append(
            f"device tree {name} is no longer built; Macs that boot it would be stranded "
            "(list it in tools/aurora-ci/retired-dtbs.txt with a reason if this is deliberate)")
    for name in removed:
        if name in retired:
            result["notes"].append(f"device tree {name} retired: {retired[name]}")

    removed, added = diff_sets(base["modules"], head["modules"])
    builtin = set(head.get("builtin", []))
    # modules.builtin names the .ko a built-in driver would have been; a
    # shipped module may carry a compression suffix (.ko.zst).
    became_builtin = [m for m in removed if re.sub(r"\.ko\.\w+$", ".ko", m) in builtin]
    gone = [m for m in removed if m not in became_builtin]
    result["modules"] = {"removed": gone, "became_builtin": became_builtin, "added": added}

    config_changes = []
    for key in sorted(set(base["config"]) | set(head["config"])):
        old, new = base["config"].get(key, "absent"), head["config"].get(key, "absent")
        if old != new:
            config_changes.append({"symbol": key, "base": old, "head": new})
    result["config"] = config_changes

    removed, added = diff_sets(base["package"]["files"], head["package"]["files"])
    result["package_files"] = {"removed": removed, "added": added}
    removed, added = diff_sets(base["headers"]["files"], head["headers"]["files"])
    result["headers_files"] = {"removed": removed, "added": added}
    result["image"] = {"base": base["image"]["size"], "head": head["image"]["size"],
                       "page_size": [base["image"].get("page_size"), head["image"].get("page_size")]}
    result["toolchain"] = {"removed": diff_sets(base["toolchain"], head["toolchain"])[0],
                           "added": diff_sets(base["toolchain"], head["toolchain"])[1]}
    return result


def differences(result):
    """Every difference, as text, for --expect-identical."""
    found = []
    for kind in ("removed", "added", "changed"):
        found += [f"DTB {kind}: {n}" for n in result["dtbs"][kind]]
    for kind in ("removed", "became_builtin", "added"):
        found += [f"module {kind}: {n}" for n in result["modules"][kind]]
    found += [f"config {c['symbol']}: {c['base']} -> {c['head']}" for c in result["config"]]
    for section in ("package_files", "headers_files"):
        for kind in ("removed", "added"):
            found += [f"{section} {kind}: {n}" for n in result[section][kind]]
    if result["image"]["page_size"][0] != result["image"]["page_size"][1]:
        found.append(f"image page size: {result['image']['page_size']}")
    return found


def bullet_list(items, limit=LIST_LIMIT):
    lines = [f"- `{item}`" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"- … and {len(items) - limit} more (see the JSON report)")
    return lines


def render(base, head, result, title):
    lines = [f"### {title}", "",
             f"`{base['package']['file']}` → `{head['package']['file']}`", ""]
    delta = result["image"]["head"] - result["image"]["base"]
    lines.append(f"Image: {result['image']['base']:,} → {result['image']['head']:,} bytes ({delta:+,}).")
    dt = result["dtbs"]
    lines.append(f"Device trees: {len(dt['removed'])} removed, {len(dt['added'])} added, "
                 f"{len(dt['changed'])} changed.")
    mods = result["modules"]
    lines.append(f"Modules: {len(mods['removed'])} removed, {len(mods['became_builtin'])} now built in, "
                 f"{len(mods['added'])} added.")
    lines.append(f"Configuration: {len(result['config'])} symbols differ.")
    lines.append("")
    if result["failures"]:
        lines += ["**Failures**", ""] + [f"- {f}" for f in result["failures"]] + [""]
    for heading, items in (("Device trees removed", dt["removed"]), ("Device trees added", dt["added"]),
                           ("Device trees changed", dt["changed"]),
                           ("Modules removed", mods["removed"]), ("Modules now built in", mods["became_builtin"]),
                           ("Modules added", mods["added"])):
        if items:
            lines += [f"<details><summary>{heading} ({len(items)})</summary>", ""] + bullet_list(items) + ["", "</details>", ""]
    if result["config"]:
        rows = [f"| `{c['symbol']}` | `{c['base']}` | `{c['head']}` |" for c in result["config"][:LIST_LIMIT]]
        more = [f"| … {len(result['config']) - LIST_LIMIT} more | | |"] if len(result["config"]) > LIST_LIMIT else []
        lines += [f"<details><summary>Configuration changes ({len(result['config'])})</summary>", "",
                  "| Symbol | Base | Head |", "|---|---|---|"] + rows + more + ["", "</details>", ""]
    if result["toolchain"]["removed"] or result["toolchain"]["added"]:
        lines += ["Toolchain changed: " + ", ".join(f"`-{p}`" for p in result["toolchain"]["removed"])
                  + " " + ", ".join(f"`+{p}`" for p in result["toolchain"]["added"]), ""]
    for note in result["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="baseline metadata JSON")
    parser.add_argument("--head", type=Path, required=True, help="metadata JSON of the build under test")
    parser.add_argument("--retired", type=Path, help="retired-dtbs.txt from the tree under test")
    parser.add_argument("--report", type=Path, help="write the comparison as JSON")
    parser.add_argument("--title", default="Compared with the target branch")
    parser.add_argument("--expect-identical", action="store_true",
                        help="fail on any difference (release names aside)")
    args = parser.parse_args(argv)

    try:
        base, head = load(args.base), load(args.head)
    except (OSError, ValueError) as error:
        print(f"::error title=Package comparison::{error}", file=sys.stderr)
        return 1
    result = compare(base, head, retired_dtbs(args.retired))
    if args.expect_identical:
        result["failures"] += [f"differs from the baseline: {d}" for d in differences(result)]
    if args.report:
        args.report.write_text(json.dumps(result, indent=1) + "\n")

    text = render(base, head, result, args.title)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
            out.write(text)
    for failure in result["failures"][:50]:
        print(f"::error title=Package comparison::{failure}", file=sys.stderr)
    for name in result["modules"]["removed"][:20]:
        print(f"::warning title=Module no longer built::{name}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
