"""Unit tests for CasFlyDevice — the real UDP-based distributed device."""
from __future__ import annotations

import os
import tempfile
import unittest

import joblib
import networkx as nx
import pandas as pd


class TestCasFlyDevice(unittest.TestCase):
    """Tests for CasFlyDevice using synthetic .pkl and CSV fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temp directory with minimal TPHG and lookup CSV files."""
        cls.tmpdir = tempfile.mkdtemp()

        # Build synthetic TPHG: Hypertension → Elevated Heart Rate
        g = nx.MultiDiGraph()
        g.add_edge("Hypertension", "Elevated Heart Rate", probability=0.85, lag_bin=7.0)
        tphg_dir = os.path.join(cls.tmpdir, "Device1")
        os.makedirs(tphg_dir, exist_ok=True)
        joblib.dump(g, os.path.join(tphg_dir, "patient_001_tphg.pkl"))
        cls.tphg_dir = tphg_dir

        # Lookup CSV
        lookup_path = os.path.join(cls.tmpdir, "lookup.csv")
        pd.DataFrame(
            [
                {
                    "device": "Device1",
                    "ip": "127.0.0.1",
                    "port": 15001,
                    "cause_type": "['Hypertension']",
                    "effect_type": "['Elevated Heart Rate']",
                    "probability_list": "[0.85]",
                },
                {
                    "device": "Device2",
                    "ip": "127.0.0.1",
                    "port": 15002,
                    "cause_type": "['Elevated Heart Rate']",
                    "effect_type": "['Arrhythmia Alert']",
                    "probability_list": "[0.90]",
                },
            ]
        ).to_csv(lookup_path, index=False)
        cls.lookup_csv = lookup_path

        # Filtered patients CSV
        fp_path = os.path.join(cls.tmpdir, "filtered_patients.csv")
        pd.DataFrame(
            [{"PATIENT": "patient_001", "RELEVANT_DEVICES": "Device1,Device2"}]
        ).to_csv(fp_path, index=False)
        cls.fp_csv = fp_path

    def _make_device(self):
        from casfly_sdk import CasFlyDevice

        return CasFlyDevice(
            device_id="Device1",
            lookup_csv=self.lookup_csv,
            tphg_dir=self.tphg_dir,
            filtered_patients_csv=self.fp_csv,
            ip="127.0.0.1",
            port=15001,
            metrics_dir=os.path.join(self.tmpdir, "logs"),
        )

    def test_load_tphg_returns_multigraph(self) -> None:
        device = self._make_device()
        tphg = device.load_tphg("patient_001")
        self.assertIsInstance(tphg, (nx.Graph, nx.MultiDiGraph))
        self.assertIn("Hypertension", tphg.nodes)
        self.assertIn("Elevated Heart Rate", tphg.nodes)

    def test_load_tphg_missing_returns_none(self) -> None:
        device = self._make_device()
        result = device.load_tphg("no_such_patient")
        self.assertIsNone(result)

    def test_viterbi_expand_backward_known_event(self) -> None:
        device = self._make_device()
        tphg = device.load_tphg("patient_001")
        paths = device.viterbi_expand_backward("Elevated Heart Rate", tphg)
        self.assertTrue(len(paths) > 0)
        # Hypertension should appear somewhere in the expanded paths
        all_nodes = {node for path in paths for node in path}
        self.assertIn("Hypertension", all_nodes)

    def test_viterbi_expand_backward_unknown_event_uses_fallback(self) -> None:
        device = self._make_device()
        tphg = device.load_tphg("patient_001")
        paths = device.viterbi_expand_backward("UnknownEvent", tphg)
        self.assertTrue(len(paths) > 0)

    def test_expand_causal_chain_populates_chain(self) -> None:
        device = self._make_device()
        chain = []
        device._expand_causal_chain("Elevated Heart Rate", chain, set(), "patient_001")
        self.assertTrue(len(chain) > 0)
        # Each chain entry: [pred, effect, prob, lag, device_id]
        self.assertEqual(len(chain[0]), 5)

    def test_find_next_device_routes_correctly(self) -> None:
        device = self._make_device()
        result = device._find_next_device(
            "Arrhythmia Alert", "patient_001", visited_devices={"Device1"}
        )
        # Should route to Device2 which has Arrhythmia Alert in effect_type
        self.assertIn("Device2", result)
        self.assertEqual(result["Device2"]["port"], 15002)

    def test_device_port_read_from_lookup_csv(self) -> None:
        device = self._make_device()
        self.assertEqual(device.port, 15001)
        self.assertEqual(device.ip, "127.0.0.1")

    def test_relevant_devices_filtered_per_patient(self) -> None:
        device = self._make_device()
        rel = device._get_relevant_devices("patient_001")
        self.assertIn("Device1", rel)
        self.assertIn("Device2", rel)

    def test_relevant_devices_empty_for_unknown_patient(self) -> None:
        device = self._make_device()
        rel = device._get_relevant_devices("unknown_patient")
        self.assertEqual(rel, [])


if __name__ == "__main__":
    unittest.main()
