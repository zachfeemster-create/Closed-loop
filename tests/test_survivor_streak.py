import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/survivor_streak"))
from matched_null import counts, evaluate, longest, read_history, shuffle
import random


class HistoricalNullTests(unittest.TestCase):
    def test_mixed_controls_preserve_both_degrees_and_actual_longest(self):
        dates = ["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17"]
        rows = [{"A", "B"}, {"A", "C"}, {"A", "D"}, {"A", "B"}]
        self.assertEqual(longest(dates, rows)[0], 4)
        fake, n = shuffle(rows, random.Random(17), 100)
        self.assertGreater(n, 0)
        self.assertEqual([len(x) for x in fake], [2, 2, 2, 2])
        self.assertEqual(counts(fake), counts(rows))
        report = evaluate(dates, rows, [{"path": str(i)} for i in range(4)], fakes=30, swaps=100)
        self.assertEqual(report["observed_global_max_consecutive_days"], 4)
        self.assertTrue(0 < report["null_control"]["p_ge_observed_plus_one"] <= 1)

    def test_missing_date_does_not_count_as_consecutive_day(self):
        dates = ["2026-11-14", "2026-11-16", "2026-11-17"]
        self.assertEqual(longest(dates, [{"A"}, {"A"}, {"A"}])[0], 2)

    def test_actual_dated_sources_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "autorun/results/nightly"
            p.mkdir(parents=True)
            for date in ["2026-11-14", "2026-11-15", "2026-11-16"]:
                (p / (date + ".json")).write_text(json.dumps({
                    "kind": "nightly", "survivors": [
                        {"name": "FRED INDPRO", "lens": "q4", "eq": "flat"},
                        {"name": "FRED INDPRO", "lens": "q4", "eq": "flat"},
                    ], "feeds_skipped": ["worldbank"],
                }))
            dates, rows, sources = read_history(Path(root))
            self.assertEqual(len(dates), 3)
            self.assertEqual([len(s) for s in rows], [1, 1, 1])
            self.assertEqual(sources[0]["raw"], 2)
            self.assertEqual(sources[0]["distinct"], 1)

    def test_no_significance_claim_if_all_keys_immutable(self):
        dates = ["2026-11-14", "2026-11-15", "2026-11-16"]
        with self.assertRaisesRegex(ValueError, "could not be shuffled"):
            evaluate(dates, [{"A"}, {"A"}, {"A"}], [{}, {}, {}], fakes=20, swaps=40)


if __name__ == "__main__":
    unittest.main()
