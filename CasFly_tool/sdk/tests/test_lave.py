from __future__ import annotations

import unittest

from casfly_sdk import Edge, LagAwareViterbi, TPHG, map_lag_bin_weight


class TestLAVE(unittest.TestCase):
    def test_lave_prefers_higher_probability_path(self) -> None:
        tphg = TPHG(
            edges=[
                Edge("A", "X", probability=0.8, lag_days=1.0, lag_bin="0-7 days"),
                Edge("B", "X", probability=0.6, lag_days=1.0, lag_bin="0-7 days"),
                Edge("C", "A", probability=0.9, lag_days=1.0, lag_bin="0-7 days"),
                Edge("D", "B", probability=0.9, lag_days=1.0, lag_bin="0-7 days"),
            ]
        )

        result = LagAwareViterbi(max_depth=4).expand(tphg, "X")
        self.assertEqual(result.best_path, ("C", "A", "X"))
        self.assertAlmostEqual(result.probability, 0.72)

    def test_lag_weight_factor_is_applied(self) -> None:
        tphg = TPHG(
            edges=[
                Edge("A", "X", probability=0.8, lag_days=1.0, lag_bin="0-7 days"),
            ]
        )
        weight_fn = map_lag_bin_weight({"0-7 days": 0.5})
        result = LagAwareViterbi(max_depth=2, lag_weight_fn=weight_fn).expand(tphg, "X")
        self.assertEqual(result.best_path, ("A", "X"))
        self.assertAlmostEqual(result.raw_probability, 0.8)
        self.assertAlmostEqual(result.lag_weight_product, 0.5)
        self.assertAlmostEqual(result.probability, 0.4)


if __name__ == "__main__":
    unittest.main()
