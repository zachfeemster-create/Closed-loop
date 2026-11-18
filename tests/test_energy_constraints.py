"""Contract tests for an energy solver; no quantum or food deployment."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/energy_constraints"))
from pilot import checked_edges, descend, exact_feasible, fixtures, run, violated_constraints


class EnergyConstraintTests(unittest.TestCase):
    def test_exact_oracle_rejects_undercolored_triangle(self):
        edges = checked_edges(3, [(0, 1), (1, 2), (2, 0)])
        self.assertIsNone(exact_feasible(3, edges, 2))
        self.assertIsNotNone(exact_feasible(3, edges, 3))

    def test_early_stopped_candidate_must_pass_independent_exact_rules(self):
        graph = checked_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        out = descend(5, graph, 2, seed=7, restarts=32, moves=300)
        self.assertEqual(out["violations"], 0)
        self.assertTrue(out["stopped_early"])
        self.assertEqual(violated_constraints(out["candidate"], graph, 2), 0)

    def test_no_fake_success_on_impossible_conflicts(self):
        graph = checked_edges(4, [(a, b) for a in range(4) for b in range(a + 1, 4)])
        result = descend(4, graph, 3, seed=4, restarts=3, moves=100)
        self.assertGreater(result["violations"], 0)
        self.assertFalse(result["stopped_early"])

    def test_invalid_model_states_and_self_loops_rejected(self):
        with self.assertRaises(ValueError):
            checked_edges(3, [(0, 0)])
        with self.assertRaises(ValueError):
            violated_constraints([1, 3, 0], [(0, 1)], 3)
        with self.assertRaises(ValueError):
            checked_edges(10, [])

    def test_full_experiment_never_promotes_unverified_solution(self):
        res = run(seed=20261117, restarts=24, moves=300)
        self.assertEqual(len(res["records"]), 7)
        self.assertEqual(res["false_accept_count"], 0)
        self.assertEqual(res["claim"], "HAND_SPECIFIED_QUBO_STYLE_ENERGY_NOT_TRAINED_CRAM")
        self.assertTrue(all(item["exact_feasible_at_k"] or not item["energy_candidate_valid_at_k"]
                            for item in res["records"]))
        self.assertTrue(any(not item["exact_feasible_at_k"] for item in res["records"]))
        self.assertTrue(any(item["exact_feasible_at_k"] for item in res["records"]))
        self.assertTrue(all(item["energy_wall_seconds"] >= 0 for item in res["records"]))


if __name__ == "__main__":
    unittest.main()
