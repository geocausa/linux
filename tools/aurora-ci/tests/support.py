# SPDX-License-Identifier: GPL-2.0
"""Helpers shared by the CI tools' unit tests."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
TREE = TOOLS.parent.parent


def load(script):
    """Import tools/aurora-ci/<script>.py (the names contain dashes)."""
    path = TOOLS / f"{script}.py"
    spec = importlib.util.spec_from_file_location(script.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo, *args, env=None):
    full_env = dict(os.environ, GIT_AUTHOR_NAME="Test Author", GIT_AUTHOR_EMAIL="author@example.org",
                    GIT_COMMITTER_NAME="Test Committer", GIT_COMMITTER_EMAIL="committer@example.org",
                    GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", **(env or {}))
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full_env).stdout.strip()


def commit_file(repo, path, text, message):
    target = Path(repo) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    git(repo, "add", path)
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "HEAD")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="aurora-ci-test-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # The scripts print GitHub workflow commands (::error::...); keep them
        # out of the test log, where the runner would turn them into
        # annotations on the CI run itself.
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        for redirect in (contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr)):
            redirect.__enter__()
            self.addCleanup(redirect.__exit__, None, None, None)

    def init_repo(self, name="repo", bare=False):
        path = self.tmp / name
        path.mkdir()
        git(path, "init", "--quiet", "--initial-branch=main", *(["--bare"] if bare else []))
        return path

    def github_env(self):
        """Point GITHUB_OUTPUT and GITHUB_STEP_SUMMARY at files in the temp dir."""
        output, summary = self.tmp / "github-output", self.tmp / "github-summary"
        patcher = unittest.mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output),
                                                        "GITHUB_STEP_SUMMARY": str(summary)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return output, summary


def read_outputs(path):
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value))
    return path
