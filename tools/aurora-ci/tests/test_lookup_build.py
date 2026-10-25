# SPDX-License-Identifier: GPL-2.0
from support import TempDirTestCase, commit_file, git, load

lookup = load("lookup-build")


class FakeApi:
    def __init__(self, artifacts=None):
        self.artifacts, self.asked = artifacts or {}, []

    def newest_artifact(self, name):
        self.asked.append(name)
        return self.artifacts.get(name, "")


class LookupTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.kernel = self.init_repo("kernel")
        self.base = commit_file(self.kernel, "Makefile", "VERSION = 7\n", "kernel")
        commit_file(self.kernel, "tools/lib/x.c", "int x;\n", "a tool")
        self.recipe = self.init_repo("recipe")
        for part in lookup.RECIPE_PARTS:
            path = part if part.startswith(("bin/", "helpers/")) else f"{part}/file"
            commit_file(self.recipe, path, part, f"add {part}")
        commit_file(self.recipe, "bin/omarchy-release", "x", "release tooling")
        commit_file(self.recipe, "pkgbuilds/other/PKGBUILD", "x", "another package")

    def key(self):
        tree = lookup.kernel_id(self.kernel, "HEAD")
        return lookup.build_key(tree, {p: git(self.recipe, "rev-parse", f"HEAD:{p}") for p in lookup.RECIPE_PARTS})

    def test_finds_earlier_build_verification_and_baseline(self):
        key = self.key()
        base_id = lookup.kernel_id(self.kernel, self.base)
        api = FakeApi({f"kernel-package-{key}": "11", f"kernel-verified-{key}": "12",
                       f"kernel-metadata-{base_id}": "13"})
        result = lookup.lookup(api, self.kernel, self.recipe, self.base, force=False)
        self.assertEqual((result["key"], result["package_run"], result["verified_run"]), (key, "11", "12"))
        self.assertEqual((result["base_tree"], result["baseline_run"]), (base_id, "13"))
        self.assertEqual(result["recipe_commit"], git(self.recipe, "rev-parse", "HEAD"))

    def test_force_ignores_earlier_results(self):
        api = FakeApi({f"kernel-package-{self.key()}": "11"})
        result = lookup.lookup(api, self.kernel, self.recipe, "", force=True)
        self.assertEqual((result["package_run"], result["verified_run"], result["baseline_run"]), ("", "", ""))
        self.assertEqual(api.asked, [])

    def test_other_packages_and_tooling_do_not_change_the_key(self):
        before = self.key()
        commit_file(self.recipe, "pkgbuilds/other/PKGBUILD", "y", "bump another package")
        commit_file(self.recipe, "bin/omarchy-release", "y", "release tooling")
        commit_file(self.recipe, "helpers/upstream-watch.py", "y", "an unrelated helper")
        self.assertEqual(self.key(), before)
        commit_file(self.recipe, "helpers/docker-helpers.sh", "changed", "change the builder")
        self.assertNotEqual(self.key(), before)
        before = self.key()
        commit_file(self.recipe, "pkgbuilds/linux-aurora/file", "changed", "change the kernel recipe")
        self.assertNotEqual(self.key(), before)

    def test_kernel_tree_changes_the_key(self):
        for path in ("README", "tools/lib/y.c", "tools/aurora-sep/z.py"):
            with self.subTest(path=path):
                before = self.key()
                commit_file(self.kernel, path, "x\n", "change")
                self.assertNotEqual(self.key(), before)

    def test_ci_changes_do_not_change_the_key(self):
        before = self.key()
        commit_file(self.kernel, ".github/workflows/kernel-ci.yml", "on: push\n", "workflow")
        commit_file(self.kernel, "tools/aurora-ci/plan.py", "print()\n", "ci tool")
        self.assertEqual(self.key(), before)


REPO = 1000


def artifact(run, created, head_repo=REPO, expired=False, name="kernel-package-x"):
    return {"name": name, "expired": expired, "created_at": created,
            "workflow_run": {"id": run, "repository_id": REPO, "head_repository_id": head_repo}}


QUEUE = "gh-readonly-queue/integration/aurora/pr-7-0123456789abcdef0123456789abcdef01234567"


def run(event="push", path=".github/workflows/kernel-ci.yml", head_repo=REPO, branch=None):
    if branch is None:
        branch = QUEUE if event == "merge_group" else "integration/aurora"
    return {"event": event, "path": path, "head_branch": branch,
            "repository": {"id": REPO}, "head_repository": {"id": head_repo}}


class ApiTest(TempDirTestCase):
    def api(self, artifacts, runs):
        api = lookup.Api("https://api.example", "o/r", "token", str(REPO))
        api.calls = []

        def get(path, **query):
            api.calls.append(path)
            if path == "actions/artifacts":
                return {"artifacts": [a for a in artifacts if a["name"] == query["name"]]}
            return runs[int(path.rsplit("/", 1)[1])]
        api.get = get
        return api

    def test_newest_unexpired_artifact_from_a_trusted_run(self):
        api = self.api([artifact(1, "2026-10-20", expired=True), artifact(2, "2026-10-18"),
                        artifact(3, "2026-10-19"), artifact(4, "2026-10-21", name="other")],
                       {2: run(), 3: run("merge_group")})
        self.assertEqual(api.newest_artifact("kernel-package-x"), "3")
        self.assertEqual(self.api([], {}).newest_artifact("kernel-package-x"), "")

    def test_a_fork_run_is_never_reused(self):
        # A fork's pull request uploads a newer artifact under the same name.
        api = self.api([artifact(2, "2026-10-18"), artifact(9, "2026-10-20", head_repo=2000)],
                       {2: run(), 9: run("pull_request", head_repo=2000)})
        self.assertEqual(api.newest_artifact("kernel-package-x"), "2")
        self.assertNotIn("actions/runs/9", api.calls)

    def test_the_run_itself_must_be_trusted(self):
        # The artifact claims this repository, but the run says otherwise.
        cases = {
            "fork head": run("pull_request", head_repo=2000),
            "another workflow": run(path=".github/workflows/something-else.yml"),
            "an untrusted event": run("pull_request_target"),
            "a pull request from this repository": run("pull_request", branch="topic"),
            "a labelled build": run("pull_request", ".github/workflows/kernel-ci-label.yml", branch="topic"),
            "a manual run of another branch": run("workflow_dispatch", branch="topic"),
            "a push to another branch": run("push", branch="topic"),
            "a queue for another branch": run("merge_group", branch="gh-readonly-queue/topic/pr-7-0123"),
            "the branch name as a queue": run("merge_group", branch="integration/aurora"),
        }
        for name, untrusted in cases.items():
            with self.subTest(name):
                api = self.api([artifact(5, "2026-10-20")], {5: untrusted})
                self.assertEqual(api.newest_artifact("kernel-package-x"), "")

    def test_every_trusted_event_on_the_branch_or_its_queue(self):
        for event in sorted(lookup.TRUSTED_EVENTS):
            with self.subTest(event):
                api = self.api([artifact(5, "2026-10-20")], {5: run(event)})
                self.assertEqual(api.newest_artifact("kernel-package-x"), "5")

    def test_an_unreviewed_run_is_skipped_for_an_older_reviewed_one(self):
        # A writer's pull request uploads a newer artifact under the same name.
        api = self.api([artifact(2, "2026-10-18"), artifact(9, "2026-10-20")],
                       {2: run(), 9: run("pull_request", branch="topic")})
        self.assertEqual(api.newest_artifact("kernel-package-x"), "2")
