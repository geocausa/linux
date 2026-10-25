# SPDX-License-Identifier: GPL-2.0
import json

from support import TempDirTestCase, load

result = load("result")


def needs(tier="light", verified="", **results):
    jobs = {"plan": {"result": "success", "outputs": {"tier": tier, "reason": "test",
                                                       "verified_run": verified}}}
    defaults = {"commits": "success", "compile": "success", "devicetree": "success", "host-tests": "success", "self-test": "success",
                "package": "skipped", "kunit": "skipped", "boot": "skipped"}
    if tier == "full":
        defaults.update(package="success", kunit="success", boot="success")
    defaults.update({k.replace("_", "-"): v for k, v in results.items()})
    jobs.update({job: {"result": value} for job, value in defaults.items()})
    return jobs


class EvaluateTest(TempDirTestCase):
    def test_light_pull_request(self):
        self.assertTrue(result.evaluate(needs(), "pull_request")[0])

    def test_light_run_must_not_build(self):
        self.assertFalse(result.evaluate(needs(package="success"), "pull_request")[0])

    def test_any_failed_light_job_fails(self):
        for job in ("commits", "compile", "devicetree", "host_tests", "self_test"):
            with self.subTest(job=job):
                self.assertFalse(result.evaluate(needs(**{job: "failure"}), "pull_request")[0])

    def test_cancelled_counts_as_failed(self):
        self.assertFalse(result.evaluate(needs(host_tests="cancelled"), "pull_request")[0])

    def test_full_run_needs_every_job(self):
        self.assertTrue(result.evaluate(needs("full"), "merge_group")[0])
        for job in ("package", "kunit", "boot"):
            with self.subTest(job=job):
                self.assertFalse(result.evaluate(needs("full", **{job: "failure"}), "merge_group")[0])
                self.assertFalse(result.evaluate(needs("full", **{job: "skipped"}), "merge_group")[0])

    def test_verified_key_skips_kunit_and_boot_only(self):
        verified = needs("full", verified="123", kunit="skipped", boot="skipped")
        self.assertTrue(result.evaluate(verified, "merge_group")[0])
        self.assertFalse(result.evaluate(dict(verified, package={"result": "skipped"}), "merge_group")[0])

    def test_commit_checks_only_for_ranges(self):
        push = needs("full", commits="skipped", compile="skipped", devicetree="skipped")
        self.assertTrue(result.evaluate(push, "push")[0])
        self.assertFalse(result.evaluate(push, "merge_group")[0])

    def test_failed_plan_fails(self):
        jobs = needs()
        jobs["plan"]["result"] = "failure"
        self.assertFalse(result.evaluate(jobs, "pull_request")[0])

    def test_main(self):
        output, summary = self.github_env()
        self.assertEqual(result.main(["--needs", json.dumps(needs()), "--event-name", "pull_request"]), 0)
        self.assertIn("ok=true", output.read_text())
        self.assertIn("merge queue builds", summary.read_text())
        self.assertEqual(result.main(["--needs", json.dumps(needs(boot="success")),
                                      "--event-name", "pull_request"]), 1)
