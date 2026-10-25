# SPDX-License-Identifier: GPL-2.0
import json
import subprocess

from support import TempDirTestCase, commit_file, git, load

checks = load("check-commits")

SIGNED = "\n\nSigned-off-by: Test Author <author@example.org>"


class RangeTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.origin = self.init_repo("origin")
        self.base = commit_file(self.origin, "README", "base\n", "base" + SIGNED)
        for i in range(10):
            self.base = commit_file(self.origin, "README", f"base {i}\n", f"base {i}" + SIGNED)

    def shallow_clone(self, head):
        clone = self.tmp / "clone"
        git(self.tmp, "clone", "--quiet", "--depth=1", "--no-local", f"file://{self.origin}", str(clone))
        git(clone, "fetch", "--quiet", "--depth=1", "origin", head)
        return clone

    def test_signed_series_passes_from_a_shallow_clone(self):
        commit_file(self.origin, "drivers/a.c", "a\n", "drivers: add a" + SIGNED)
        head = commit_file(self.origin, "drivers/b.c", "b\n", "drivers: add b" + SIGNED)
        clone = self.shallow_clone(head)
        report = checks.check(clone, "origin", self.base, head, max_commits=100, checkpatch=False)
        self.assertEqual(report["mode"], "full")
        self.assertEqual([c["subject"] for c in report["commits"]], ["drivers: add a", "drivers: add b"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["uapi"], [])

    def test_non_utf8_patch_reaches_checkpatch_unchanged(self):
        source = self.origin / "fixture.c"
        source.write_bytes(b"/* legacy byte: \xc0 */\n")
        git(self.origin, "add", "fixture.c")
        git(self.origin, "commit", "--quiet", "-m", "legacy source" + SIGNED)
        head = git(self.origin, "rev-parse", "HEAD")
        scripts = self.origin / "scripts"
        scripts.mkdir()
        (scripts / "checkpatch.pl").write_text(
            'use strict; use warnings;\n'
            'binmode STDIN; binmode STDOUT;\n'
            'local $/; my $patch = <STDIN>;\n'
            'open my $out, ">:raw", "seen.patch" or die $!;\n'
            'print {$out} $patch; close $out or die $!;\n'
            'print "fixture.c:1: WARNING:NON_UTF8: legacy byte \\xc0\\n";\n'
        )
        expected = subprocess.check_output(
            ["git", "-C", str(self.origin), "format-patch", "-1", "--stdout", head])
        self.assertIn(b"\xc0", expected)
        report = checks.check(self.origin, "origin", self.base, head,
                              max_commits=100)
        self.assertEqual((self.origin / "seen.patch").read_bytes(), expected)
        self.assertEqual(report["failures"], [])
        finding, = report["commits"][0]["checkpatch"]
        self.assertEqual(finding["type"], "NON_UTF8")
        self.assertEqual(finding["message"], "legacy byte \ufffd")

    def test_shallow_merge_fetches_the_whole_added_branch(self):
        git(self.origin, "checkout", "--quiet", "-b", "topic")
        for i in range(70):
            commit_file(self.origin, "drivers/a.c", f"{i}\n",
                        f"topic {i}" + SIGNED)
        topic = git(self.origin, "rev-parse", "HEAD")
        git(self.origin, "checkout", "--quiet", "main")
        base = commit_file(self.origin, "README", "main update\n",
                           "main update" + SIGNED)
        git(self.origin, "merge", "--quiet", "--no-ff", "-m", "Merge topic", topic)
        head = git(self.origin, "rev-parse", "HEAD")
        clone = self.shallow_clone(head)
        report = checks.check(clone, "origin", base, head,
                              max_commits=100, checkpatch=False)
        self.assertEqual([c["subject"] for c in report["commits"]],
                         [f"topic {i}" for i in range(70)])
        self.assertEqual(report["failures"], [])

    def test_unsigned_commit_fails(self):
        unsigned = commit_file(self.origin, "drivers/a.c", "a\n", "drivers: add a\n\nNo trailer here.")
        head = commit_file(self.origin, "drivers/b.c", "b\n", "drivers: add b" + SIGNED)
        report = checks.check(self.origin, "origin", self.base, head, max_commits=100, checkpatch=False)
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn(unsigned[:12], report["failures"][0])

    def test_merges_are_not_checked(self):
        git(self.origin, "checkout", "--quiet", "-b", "topic")
        topic = commit_file(self.origin, "drivers/a.c", "a\n", "drivers: add a" + SIGNED)
        git(self.origin, "checkout", "--quiet", "main")
        commit_file(self.origin, "drivers/b.c", "b\n", "drivers: add b" + SIGNED)
        git(self.origin, "merge", "--quiet", "--no-ff", "-m", "Merge topic (no sign-off)", topic)
        head = git(self.origin, "rev-parse", "HEAD")
        report = checks.check(self.origin, "origin", self.base, head, max_commits=100, checkpatch=False)
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["commits"]), 2)

    def test_integration_sized_range_reports_without_failing(self):
        for i in range(4):
            head = commit_file(self.origin, "drivers/a.c", f"{i}\n", f"upstream {i}")
        report = checks.check(self.origin, "origin", self.base, head, max_commits=3, checkpatch=False)
        self.assertEqual(report["mode"], "integration")
        self.assertEqual(report["failures"], [])
        self.assertTrue(all(not c["signed_off"] for c in report["commits"]))

    def test_uapi_changes_are_flagged(self):
        head = commit_file(self.origin, "include/uapi/drm/asahi_drm.h", "x\n", "drm/asahi: uapi" + SIGNED)
        report = checks.check(self.origin, "origin", self.base, head, max_commits=100, checkpatch=False)
        self.assertEqual(report["uapi"], ["include/uapi/drm/asahi_drm.h"])

    def test_main_reports_and_exits(self):
        self.github_env()
        head = commit_file(self.origin, "a.c", "a\n", "unsigned")
        report = self.tmp / "report.json"
        status = checks.main(["--repo", str(self.origin), "--base", self.base, "--head", head,
                              "--no-checkpatch", "--report", str(report)])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(report.read_text())["commits"][0]["signed_off"], False)


class CheckpatchParseTest(TempDirTestCase):
    def test_terse_showfile_line(self):
        line = "drivers/gpu/drm/apple/dcp.c:120: WARNING:LONG_LINE: line length of 120 exceeds 100 columns"
        match = checks.CHECKPATCH_LINE.match(line)
        self.assertEqual(match["file"], "drivers/gpu/drm/apple/dcp.c")
        self.assertEqual((match["line"], match["level"], match["type"]), ("120", "WARNING", "LONG_LINE"))
