"""Regression checks for the corrected number-menu test (synthetic inputs)."""
import unittest
from unittest.mock import patch

import numpy as np

from closedloop import menu as M


class MenuTest(unittest.TestCase):
    def test_eight_raw_hits_are_three_distinct_values(self):
        constants = {f"same_{i}": 0.078125 for i in range(6)}
        constants.update(lower=0.0775, upper=0.0785)
        result = M.menu_test(constants, 0.078, ratios=False, n_fakes=20)
        self.assertEqual(result["raw_hits"], 8)
        self.assertEqual(result["distinct_hits"], 3)
        self.assertEqual([v for _, v in result["hits"]], [0.078125, 0.0775, 0.0785])

    def test_equivalent_ratios_count_once_and_seed_reproduces(self):
        constants = dict(ten=10, five=5, large=128, small=64)
        first = M.menu_test(constants, 0.078, n_fakes=50)
        self.assertEqual(first["candidates"], 16)
        self.assertEqual(first["raw_hits"], 2)
        self.assertEqual(first["distinct_hits"], 1)
        self.assertEqual(first, M.menu_test(constants, 0.078, n_fakes=50))
        no_ratios = M.menu_test(constants, 0.078, ratios=False, n_fakes=50)
        self.assertEqual(no_ratios["candidates"], 4)
        self.assertEqual(no_ratios["distinct_hits"], 0)

    def test_every_fake_preserves_lattice_and_equal_constants(self):
        original = M._candidates
        draws = []

        def capture(vals, names, ratios):
            draws.append(vals.copy())
            return original(vals, names, ratios)

        with patch.object(M, "_candidates", side_effect=capture):
            M.menu_test(dict(a=10, b=10, power=128, real=3.14), 0.078, n_fakes=40)
        self.assertEqual(len(draws), 41)
        for a, b, power, real in draws[1:]:
            self.assertEqual(a, b)
            self.assertTrue(float(a).is_integer())
            self.assertEqual(np.log2(power), round(np.log2(power)))
            self.assertGreater(real, 0)

    def test_midrank_percentile_and_strict_pass_bar(self):
        # A real hit and fake counts 0, 1, 1, 2: percentile = 25 + half of 50.
        values = [np.array([x], float) for x in (1, 3, 1, 1, 2)]
        values[-1] = np.array([1, 1.005])
        with patch.object(M, "_candidates", side_effect=[(v, [str(i) for i in range(len(v))]) for v in values]):
            result = M.menu_test(dict(a=1), 1, n_fakes=4, bar_pct=50)
        self.assertEqual(result["distinct_hits"], 1)
        self.assertEqual(result["fake_bar"], 1)
        self.assertEqual(result["percentile"], 50)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
