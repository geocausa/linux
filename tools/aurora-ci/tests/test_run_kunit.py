# SPDX-License-Identifier: GPL-2.0
import json

from support import TOOLS, TempDirTestCase, load

kunit = load("run-kunit")

LOG = """\
[    0.900000] WARNING: drivers/soc/apple/rtkit.c:1505 at apple_rtkit_ci_control+0x1c/0x2c, CPU#0: swapper/0/1
[    1.000000] KTAP version 1
[    1.000001] 1..2
[    1.000002]     # Subtest: apple-dcp-fabric
[    1.000003]     1..2
[    1.100000] WARNING: kernel/workqueue.c:4180 at __flush_work+0x2c/0x40, CPU#0: kunit_try_catch/12
[    1.100001]     ok 1 fabric_unbind_test
[    1.100002]     not ok 2 fabric_retry_test
[    1.100003] not ok 1 apple-dcp-fabric
"""

TREE = {"name": "", "sub_groups": [
    {"name": "apple-dcp-fabric", "sub_groups": [], "test_cases": [
        {"name": "fabric_unbind_test", "status": "PASS"},
        {"name": "fabric_retry_test", "status": "FAIL"}]},
    {"name": "typec_mux_lookup", "sub_groups": [
        {"name": "params", "sub_groups": [], "test_cases": [{"name": "one", "status": "SKIP"}]}],
     "test_cases": []}]}


class ParseTest(TempDirTestCase):
    def test_flatten(self):
        results, suites = kunit.flatten(TREE)
        self.assertEqual(results, {"apple-dcp-fabric.fabric_unbind_test": "PASS",
                                   "apple-dcp-fabric.fabric_retry_test": "FAIL",
                                   "typec_mux_lookup.params.one": "SKIP"})
        self.assertEqual(suites["apple-dcp-fabric"]["FAIL"], 1)

    def test_splats_are_attributed_to_the_case_or_to_boot(self):
        splats = kunit.splats_by_test(LOG)
        self.assertEqual(dict(splats["apple-dcp-fabric.fabric_unbind_test"]), {"kernel/workqueue.c:__flush_work": 1})
        self.assertEqual(dict(splats[kunit.BOOT]), {"drivers/soc/apple/rtkit.c:apple_rtkit_ci_control": 1})


class CompareTest(TempDirTestCase):
    def expected(self, **results):
        return {"results": results, "warnings": {
            "apple-dcp-fabric.fabric_unbind_test": [{"site": "kernel/workqueue.c:__flush_work", "count": 1}]}}

    def test_expected_results_match(self):
        results, _ = kunit.flatten(TREE)
        expected = self.expected(**{"apple-dcp-fabric.fabric_retry_test": {"status": "FAIL"},
                                    "typec_mux_lookup.params.one": {"status": "SKIP"}})
        splats = {"apple-dcp-fabric.fabric_unbind_test": kunit.Counter({"kernel/workqueue.c:__flush_work": 1})}
        self.assertEqual(kunit.compare(results, splats, expected), [])

    def test_every_kind_of_difference(self):
        results, _ = kunit.flatten(TREE)
        expected = self.expected(**{"typec_mux_lookup.params.one": {"status": "FAIL"},
                                    "apple-dcp-fabric.fabric_unbind_test": {"status": "SKIP"},
                                    "gone.case": {"status": "SKIP"}})
        problems = kunit.compare(results, kunit.splats_by_test(LOG), expected)
        self.assertEqual(problems, [
            "FAIL: apple-dcp-fabric.fabric_retry_test",
            "now passes (expected SKIP): apple-dcp-fabric.fabric_unbind_test",
            "SKIP (expected FAIL): typec_mux_lookup.params.one",
            "did not run (expected SKIP): gone.case",
            f"kernel splat drivers/soc/apple/rtkit.c:apple_rtkit_ci_control in {kunit.BOOT}: 1 seen, 0 expected",
        ])

    def test_no_results_is_a_failure(self):
        path = self.tmp / "results.json"
        path.write_text(json.dumps({"name": "", "sub_groups": [], "test_cases": []}))
        *_, problems = kunit.check(path, self.tmp / "missing.log", self.expected())
        self.assertEqual(len(problems), 1)


class ExpectedFileTest(TempDirTestCase):
    def test_every_entry_has_a_reason(self):
        expected = json.loads((TOOLS / "kunit-expected.json").read_text())
        for test, entry in expected["results"].items():
            with self.subTest(test=test):
                self.assertIn(entry["status"], ("SKIP", "FAIL"))
                self.assertTrue(entry["reason"].strip())
        for test, entries in expected["warnings"].items():
            for entry in entries:
                with self.subTest(test=test, site=entry["site"]):
                    self.assertGreater(entry["count"], 0)
                    self.assertTrue(entry["reason"].strip())
