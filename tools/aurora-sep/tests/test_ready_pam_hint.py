#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""aurora-sep-ready names the authselect step only when fingerprints are off."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

READY = Path(__file__).resolve().parents[1] / "aurora-sep-ready"
COMMAND = "sudo authselect enable-feature with-fingerprint"


def function(source, name):
    match = re.search(r"^%s\(\) \{\n.*?^\}\n" % name, source, re.M | re.S)
    if match is None:
        raise AssertionError("no %s() in aurora-sep-ready" % name)
    return match.group(0)


class PamHintTest(unittest.TestCase):
    def hint(self, authselect):
        """Runs pam_hint() with `authselect` as the stub's body, or none."""
        source = READY.read_text()
        script = function(source, "log") + function(source, "pam_hint") + "pam_hint\n"
        with tempfile.TemporaryDirectory() as bindir:
            if authselect is not None:
                stub = Path(bindir) / "authselect"
                stub.write_text("#!/bin/sh\n" + authselect)
                stub.chmod(0o755)
            env = dict(os.environ, PATH=bindir + ":/usr/bin:/bin")
            if authselect is None:
                env["PATH"] = bindir
            result = subprocess.run(
                ["/bin/sh", "-c", script], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        return result.stderr

    def test_off_names_the_command_once(self):
        lines = self.hint('echo "local with-silent-lastlog with-mdns4"\n').splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("aurora-sep-ready: "))
        self.assertTrue(lines[0].endswith(COMMAND))

    def test_on_is_silent(self):
        self.assertEqual(self.hint('echo "local with-fingerprint with-mdns4"\n'), "")
        self.assertEqual(self.hint('echo "sssd with-mdns4 with-fingerprint"\n'), "")

    def test_a_similar_feature_is_not_mistaken_for_it(self):
        self.assertIn(COMMAND, self.hint('echo "local with-fingerprint-x"\n'))

    def test_no_authselect_configuration_is_silent(self):
        self.assertEqual(self.hint('echo "No existing configuration detected." >&2\nexit 2\n'), "")

    def test_no_authselect_is_silent(self):
        self.assertEqual(self.hint(None), "")


if __name__ == "__main__":
    unittest.main()
