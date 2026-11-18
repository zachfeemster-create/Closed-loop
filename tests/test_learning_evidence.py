"""Evidence builder regression tests: distinct records, missing feeds, immutable sources."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "closedloop"))
from learning_evidence import METHOD_PATHS, build, context, sha


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for relative in METHOD_PATHS:
            p = self.root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("original mathematics\n")
        (self.root / "autorun/data").mkdir(parents=True)
        (self.root / "autorun/data/test.csv").write_text("x,y\n1,2\n3,4\n")
        (self.root / "autorun/results/nightly").mkdir(parents=True)

    def day(self, date, names, skipped=()):
        objects = [
            {"name": name, "lens": "q4 Inverse -theta", "eq": "flat"}
            for name in names
        ]
        path = self.root / "autorun/results/nightly" / (date + ".json")
        path.write_text(json.dumps({
            "ran_at": date, "survivors": objects, "luck_pile": [{"name": "fake"}],
            "log": [{"fake_arm_passed": False}], "feeds_ok": ["fred"],
            "feeds_skipped": list(skipped),
        }))

    def test_actual_transition_and_deduplication(self):
        self.day("2026-11-15", ["A", "A", "B"])
        self.day("2026-11-16", ["A", "B", "C"])
        self.day("2026-11-17", ["A", "C"], ["worldbank timeout"])
        report = build(self.root)
        self.assertEqual(report["historical_run_count"], 3)
        self.assertEqual(report["nightly_history"][0]["distinct_survivors"], 2)
        self.assertEqual(report["nightly_history"][1]["new_vs_previous"], ["C | q4 Inverse -theta | flat"])
        self.assertEqual(report["latest"]["lost_vs_previous"], ["B | q4 Inverse -theta | flat"])
        self.assertEqual(report["latest"]["streaks"][0]["consecutive_recorded_days"], 3)
        self.assertEqual(report["latest"]["feeds_skipped"], ["worldbank timeout"])
        self.assertIn("For Smoke Signal", context(report))
        self.assertIn("For Quantum Arcane", context(report))

    def test_source_hashes_change_but_source_does_not(self):
        self.day("2026-11-17", ["A"])
        old = sha(self.root / "autorun/data/test.csv")
        before = build(self.root)
        self.assertEqual(before["datasets"][0]["rows"], 2)
        self.assertEqual(old, before["datasets"][0]["sha256"])
        (self.root / "autorun/data/test.csv").write_text("x,y\n1,2\n3,5\n")
        after = build(self.root)
        self.assertNotEqual(old, after["datasets"][0]["sha256"])
        self.assertEqual(after["latest"]["survivors"], 1)

    def test_empty_history_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "No historical"):
            build(self.root)

    def test_bad_history_fails_loudly(self):
        bad = self.root / "autorun/results/nightly/2026-11-17.json"
        bad.write_text('{"survivors": {"not": "a list"}}')
        with self.assertRaisesRegex(ValueError, "survivors must be a list"):
            build(self.root)


if __name__ == "__main__":
    unittest.main()
