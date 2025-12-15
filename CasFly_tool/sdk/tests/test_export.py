from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from casfly_sdk.models import ChainHop, ChainResult


def _single_hop_result() -> ChainResult:
    return ChainResult(
        hops=[ChainHop(
            source_device="device_a",
            target_device="device_b",
            trigger_event="Heart Attack",
            best_path=("Hypertension", "Heart Attack"),
            path_probability=0.72,
            cumulative_lag_days=2.0,
            raw_path_probability=0.80,
            lag_weight_product=0.90,
        )],
        visited_devices=["device_a"],
        visited_edges={("Hypertension", "Heart Attack")},
    )


def _multi_hop_result() -> ChainResult:
    return ChainResult(
        hops=[
            ChainHop(
                source_device="hospital",
                target_device="bp_cuff",
                trigger_event="Heart Attack",
                best_path=("Elevated Heart Rate", "Heart Attack"),
                path_probability=0.85,
                cumulative_lag_days=1.0,
                raw_path_probability=0.85,
                lag_weight_product=1.0,
            ),
            ChainHop(
                source_device="bp_cuff",
                target_device=None,
                trigger_event="Elevated Heart Rate",
                best_path=("Hypertension", "Elevated Heart Rate"),
                path_probability=0.70,
                cumulative_lag_days=3.0,
                raw_path_probability=0.70,
                lag_weight_product=1.0,
            ),
        ],
        visited_devices=["hospital", "bp_cuff"],
        visited_edges={
            ("Elevated Heart Rate", "Heart Attack"),
            ("Hypertension", "Elevated Heart Rate"),
        },
    )


class TestChainResultJson(unittest.TestCase):
    def setUp(self) -> None:
        self.result = _single_hop_result()

    def test_to_json_is_valid_json(self) -> None:
        data = json.loads(self.result.to_json())
        self.assertIsInstance(data, dict)

    def test_to_json_contains_required_fields(self) -> None:
        data = json.loads(self.result.to_json())
        for field in ("chain_confidence", "visited_devices", "visited_edges", "audit", "hops"):
            self.assertIn(field, data)

    def test_to_json_hop_fields(self) -> None:
        data = json.loads(self.result.to_json())
        hop = data["hops"][0]
        self.assertEqual(hop["source_device"], "device_a")
        self.assertEqual(hop["best_path"], ["Hypertension", "Heart Attack"])
        self.assertAlmostEqual(hop["path_probability"], 0.72)

    def test_to_json_confidence_matches(self) -> None:
        data = json.loads(self.result.to_json())
        self.assertAlmostEqual(data["chain_confidence"], self.result.chain_confidence)

    def test_to_json_audit_fields(self) -> None:
        data = json.loads(self.result.to_json())
        for key in ("weighted_confidence", "raw_confidence", "lag_weight_product", "theorem_lower_bound"):
            self.assertIn(key, data["audit"])

    def test_to_json_visited_edges_serialised_as_list(self) -> None:
        data = json.loads(self.result.to_json())
        self.assertIsInstance(data["visited_edges"], list)


class TestChainResultCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.result = _single_hop_result()

    def test_to_csv_is_parseable(self) -> None:
        rows = list(csv.DictReader(self.result.to_csv().splitlines()))
        self.assertEqual(len(rows), 1)

    def test_to_csv_header_columns(self) -> None:
        rows = list(csv.DictReader(self.result.to_csv().splitlines()))
        self.assertIn("source_device", rows[0])
        self.assertIn("best_path", rows[0])
        self.assertIn("path_probability", rows[0])

    def test_to_csv_hop_values(self) -> None:
        rows = list(csv.DictReader(self.result.to_csv().splitlines()))
        self.assertEqual(rows[0]["source_device"], "device_a")
        self.assertEqual(rows[0]["best_path"], "Hypertension -> Heart Attack")

    def test_to_csv_multi_hop_row_count(self) -> None:
        rows = list(csv.DictReader(_multi_hop_result().to_csv().splitlines()))
        self.assertEqual(len(rows), 2)

    def test_to_csv_hop_numbers(self) -> None:
        rows = list(csv.DictReader(_multi_hop_result().to_csv().splitlines()))
        self.assertEqual(rows[0]["hop"], "1")
        self.assertEqual(rows[1]["hop"], "2")


class TestChainResultDot(unittest.TestCase):
    def setUp(self) -> None:
        self.result = _single_hop_result()

    def test_to_dot_starts_with_digraph(self) -> None:
        self.assertTrue(self.result.to_dot().startswith("digraph CasFlyChain {"))

    def test_to_dot_contains_device_label(self) -> None:
        self.assertIn("device_a", self.result.to_dot())

    def test_to_dot_contains_event_labels(self) -> None:
        dot = self.result.to_dot()
        self.assertIn("Hypertension", dot)
        self.assertIn("Heart Attack", dot)

    def test_to_dot_multi_hop_has_dashed_inter_hop_edge(self) -> None:
        dot = _multi_hop_result().to_dot()
        self.assertIn("dashed", dot)

    def test_to_dot_multi_hop_both_devices_present(self) -> None:
        dot = _multi_hop_result().to_dot()
        self.assertIn("hospital", dot)
        self.assertIn("bp_cuff", dot)

    def test_to_dot_hop_qualified_node_ids_avoid_collision(self) -> None:
        # Both hops have different events; hop-qualified IDs should both appear
        dot = _multi_hop_result().to_dot()
        self.assertIn("hop0_", dot)
        self.assertIn("hop1_", dot)

    def test_to_dot_custom_title(self) -> None:
        dot = self.result.to_dot(title="My Chain")
        self.assertIn("My Chain", dot)

    def test_to_dot_closes_brace(self) -> None:
        self.assertTrue(self.result.to_dot().rstrip().endswith("}"))


class TestChainResultExport(unittest.TestCase):
    def setUp(self) -> None:
        self.result = _single_hop_result()

    def test_export_json_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.json"
            self.result.export(path)
            data = json.loads(path.read_text())
        self.assertIn("chain_confidence", data)

    def test_export_csv_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.csv"
            self.result.export(path)
            rows = list(csv.DictReader(path.read_text().splitlines()))
        self.assertEqual(len(rows), 1)

    def test_export_dot_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.dot"
            self.result.export(path)
            content = path.read_text()
        self.assertTrue(content.startswith("digraph"))

    def test_export_gv_extension(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.gv"
            self.result.export(path)
            self.assertTrue(path.read_text().startswith("digraph"))

    def test_export_explicit_fmt_overrides_extension(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chain.txt"
            self.result.export(path, fmt="json")
            data = json.loads(path.read_text())
        self.assertIn("chain_confidence", data)

    def test_export_unknown_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.result.export(Path(td) / "chain.xyz")


if __name__ == "__main__":
    unittest.main()
