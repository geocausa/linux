#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Point the packaging recipe for linux-aurora at the commit under test.

CI builds the kernel with the recipe that ships it: omacom/omarchy-pkgs
pkgbuilds/linux-aurora, run through that repository's own bin/build. This
script makes the smallest change that lets the recipe build an unpublished
commit, and refuses to guess when the recipe no longer looks the way it
expects:

  _commit   the commit under test
  pkgrel    the recipe's pkgrel plus a CI subrelease (2 -> 2.<revision>), so a
            CI package never has the version of a published one and bin/build
            does not skip it as already published
  source    the kernel entry reads a bare repository placed next to the
            PKGBUILD (aurora-ci-source) instead of cloning GitHub again

Every other line of the recipe, the builder image and the build tooling stay
exactly as they are on omarchy-pkgs.
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = "aurora-ci-source"
SOURCE_REF = "refs/heads/aurora-ci"
UPSTREAM_SOURCE = "linux::git+https://github.com/omacom/linux-aurora.git#commit=${_commit}"
LOCAL_SOURCE = "linux::git+file://${startdir}/" + SOURCE_DIR + "#commit=${_commit}"

COMMIT_LINE = re.compile(r"^_commit=([0-9a-f]{40})$", re.M)
PKGREL_LINE = re.compile(r"^pkgrel=([0-9]+)(\.[0-9]+)?$", re.M)


class RecipeError(Exception):
    """The recipe does not have the shape this script knows how to change."""


def single_match(pattern, text, what):
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RecipeError(f"expected exactly one {what} line in PKGBUILD, found {len(matches)}")
    return matches[0]


def rewrite(text, commit, revision):
    """Return the rewritten PKGBUILD text and a record of what changed."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RecipeError(f"commit must be a full 40-character SHA, got {commit!r}")
    if not re.fullmatch(r"[1-9][0-9]*", str(revision)):
        raise RecipeError(f"revision must be a positive integer, got {revision!r}")

    old_commit = single_match(COMMIT_LINE, text, "_commit=<sha>")
    old_pkgrel = single_match(PKGREL_LINE, text, "pkgrel=")
    if old_pkgrel.group(2):
        # pacman allows one subrelease; adding ours would not be a valid pkgrel.
        raise RecipeError(f"pkgrel {old_pkgrel.group(0)!r} already has a subrelease")
    if text.count(UPSTREAM_SOURCE) != 1:
        raise RecipeError(f"expected the kernel source entry {UPSTREAM_SOURCE!r} exactly once")

    pkgrel = f"{old_pkgrel.group(1)}.{revision}"
    text = COMMIT_LINE.sub(f"_commit={commit}", text, count=1)
    text = PKGREL_LINE.sub(f"pkgrel={pkgrel}", text, count=1)
    text = text.replace(UPSTREAM_SOURCE, LOCAL_SOURCE)
    return text, {
        "recipe_commit": old_commit.group(1),
        "commit": commit,
        "recipe_pkgrel": old_pkgrel.group(1),
        "pkgrel": pkgrel,
        "source": LOCAL_SOURCE,
    }


def git(*args, cwd=None):
    result = subprocess.run(["git", *args], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RecipeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def make_source(recipe, kernel, commit):
    """Create a shallow bare repository holding only the commit under test."""
    source = recipe / SOURCE_DIR
    if source.exists():
        raise RecipeError(f"{source} already exists; use a fresh recipe checkout")
    git("init", "--quiet", "--bare", str(source))
    git("symbolic-ref", "HEAD", SOURCE_REF, cwd=source)
    git("fetch", "--quiet", "--no-tags", "--depth=1", str(kernel.resolve()),
        f"{commit}:{SOURCE_REF}", cwd=source)
    fetched = git("rev-parse", "--verify", f"{SOURCE_REF}^{{commit}}", cwd=source)
    if fetched != commit:
        raise RecipeError(f"source repository holds {fetched}, expected {commit}")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", type=Path, required=True,
                        help="pkgbuilds/linux-aurora in an omarchy-pkgs checkout")
    parser.add_argument("--kernel", type=Path, required=True,
                        help="kernel checkout that contains the commit")
    parser.add_argument("--commit", required=True, help="full SHA of the commit to build")
    parser.add_argument("--revision", required=True,
                        help="CI subrelease appended to pkgrel (e.g. the run number)")
    parser.add_argument("--record", type=Path,
                        help="write what changed, as JSON, to this file")
    args = parser.parse_args(argv)

    pkgbuild = args.recipe / "PKGBUILD"
    try:
        original = pkgbuild.read_text()
        text, record = rewrite(original, args.commit, args.revision)
        make_source(args.recipe, args.kernel, args.commit)
    except (RecipeError, OSError) as error:
        print(f"::error title=Recipe override::{error}", file=sys.stderr)
        return 1
    pkgbuild.write_text(text)

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), text.splitlines(keepends=True),
        "PKGBUILD (omarchy-pkgs)", "PKGBUILD (CI)"))
    record["diff"] = diff
    print(diff, end="")
    if args.record:
        args.record.write_text(json.dumps(record, indent=2) + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as out:
            out.write("### Recipe override\n\n"
                      f"Recipe pin `{record['recipe_commit']}` -> `{args.commit}`, "
                      f"pkgrel `{record['recipe_pkgrel']}` -> `{record['pkgrel']}`; "
                      "every other line is omarchy-pkgs' own.\n\n"
                      f"```diff\n{diff}```\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
