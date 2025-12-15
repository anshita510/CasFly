from __future__ import annotations

import unittest

from casfly_sdk import (
    CasFlyNode,
    CasFlyOrchestrator,
    Edge,
    ProbabilisticLookupTable,
    TPHG,
)


class TestOrchestrator(unittest.TestCase):
    def test_orchestrator_moves_across_devices(self) -> None:
        tphg_a = TPHG(edges=[Edge("Elevated Heart Rate", "Heart Attack", 0.9, 0.2, "0-7 days")])
        tphg_b = TPHG(edges=[Edge("Hypertension", "Elevated Heart Rate", 0.85, 0.4, "0-7 days")])

        plt = ProbabilisticLookupTable([
            ("Elevated Heart Rate", "device_b", 0.9),
        ])

        node_a = CasFlyNode("device_a", tphg_a, plt)
        node_b = CasFlyNode("device_b", tphg_b, plt)

        orchestrator = CasFlyOrchestrator(max_hops=4)
        orchestrator.register(node_a)
        orchestrator.register(node_b)

        result = orchestrator.trace(start_device="device_a", start_event="Heart Attack")

        self.assertIn("device_a", result.visited_devices)
        self.assertIn("device_b", result.visited_devices)
        self.assertGreaterEqual(len(result.hops), 2)
        audit = result.confidence_audit()
        self.assertIn("weighted_confidence", audit)
        self.assertIn("theorem_lower_bound", audit)


if __name__ == "__main__":
    unittest.main()
