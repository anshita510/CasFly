from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from casfly_sdk.cli import _merge_cohort, build_parser


# ---------------------------------------------------------------------------
# Helpers — write tiny temp fixture files
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _events_csv(path: Path) -> None:
    _write_csv(path, [
        {"PATIENT": "P001", "DATE": "2023-01-06", "DESCRIPTION": "Hypertension"},
        {"PATIENT": "P001", "DATE": "2023-01-10", "DESCRIPTION": "Heart Attack"},
        {"PATIENT": "P002", "DATE": "2023-02-01", "DESCRIPTION": "Hypertension"},
        {"PATIENT": "P002", "DATE": "2023-02-05", "DESCRIPTION": "Heart Attack"},
    ])


def _routing_csv(path: Path) -> None:
    _write_csv(path, [
        {"event": "Hypertension", "device": "bp_cuff", "probability": "0.9"},
        {"event": "Heart Attack", "device": "hospital", "probability": "0.8"},
    ])


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_parser_has_trace_subcommand(self) -> None:
        # Should not raise
        args = self.parser.parse_args([
            "trace",
            "--device", "sw", "events.csv",
            "--routing", "routing.csv",
            "--patient", "P001",
            "--event", "Heart Attack",
            "--start-device", "sw",
        ])
        self.assertEqual(args.command, "trace")

    def test_parser_has_validate_subcommand(self) -> None:
        args = self.parser.parse_args([
            "validate",
            "--device", "sw", "events.csv",
            "--routing", "routing.csv",
        ])
        self.assertEqual(args.command, "validate")

    def test_parser_has_build_cache_subcommand(self) -> None:
        args = self.parser.parse_args([
            "build-cache",
            "--device", "sw", "events.csv",
            "--output", "/tmp/cache",
        ])
        self.assertEqual(args.command, "build-cache")

    def test_trace_defaults(self) -> None:
        args = self.parser.parse_args([
            "trace",
            "--device", "sw", "events.csv",
            "--routing", "routing.csv",
            "--patient", "P001",
            "--event", "Heart Attack",
            "--start-device", "sw",
        ])
        self.assertEqual(args.max_hops, 8)
        self.assertEqual(args.max_depth, 10)
        self.assertEqual(args.fallback_depth, 3)
        self.assertEqual(args.max_lag_days, 730.0)
        self.assertEqual(args.format, "text")
        self.assertIsNone(args.output)

    def test_trace_format_choices(self) -> None:
        for fmt in ("text", "json", "csv", "dot"):
            args = self.parser.parse_args([
                "trace",
                "--device", "sw", "events.csv",
                "--routing", "routing.csv",
                "--patient", "P001",
                "--event", "E",
                "--start-device", "sw",
                "--format", fmt,
            ])
            self.assertEqual(args.format, fmt)

    def test_trace_multiple_devices(self) -> None:
        args = self.parser.parse_args([
            "trace",
            "--device", "sw", "sw.csv",
            "--device", "bp", "bp.csv",
            "--routing", "routing.csv",
            "--patient", "P001",
            "--event", "E",
            "--start-device", "sw",
        ])
        self.assertEqual(len(args.device), 2)

    def test_validate_tphg_dir(self) -> None:
        args = self.parser.parse_args([
            "validate",
            "--device", "sw", "events.csv",
            "--routing", "routing.csv",
            "--tphg-dir", "/some/dir",
        ])
        self.assertEqual(args.tphg_dir, "/some/dir")

    def test_no_subcommand_exits(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])


# ---------------------------------------------------------------------------
# _merge_cohort
# ---------------------------------------------------------------------------

class TestMergeCohort(unittest.TestCase):
    def test_single_device(self) -> None:
        device_events = {"sw": {"P001": [1, 2], "P002": [3]}}
        cohort = _merge_cohort(device_events)
        self.assertEqual(cohort["P001"], [1, 2])
        self.assertEqual(cohort["P002"], [3])

    def test_merges_across_devices(self) -> None:
        device_events = {
            "sw": {"P001": [1, 2]},
            "bp": {"P001": [3, 4], "P002": [5]},
        }
        cohort = _merge_cohort(device_events)
        self.assertEqual(cohort["P001"], [1, 2, 3, 4])
        self.assertEqual(cohort["P002"], [5])

    def test_empty_input(self) -> None:
        self.assertEqual(_merge_cohort({}), {})


# ---------------------------------------------------------------------------
# casfly validate (integration via parser + _cmd_validate)
# ---------------------------------------------------------------------------

class TestCmdValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.events_csv = self.root / "events.csv"
        self.routing_csv = self.root / "routing.csv"
        _events_csv(self.events_csv)
        _routing_csv(self.routing_csv)

    def tearDown(self) -> None:
        self.td.cleanup()

    def _run_validate(self, extra_args: list[str] = []) -> int:
        parser = build_parser()
        args = parser.parse_args([
            "validate",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.routing_csv),
        ] + extra_args)
        try:
            args.func(args)
            return 0
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0

    def test_clean_setup_exits_zero(self) -> None:
        exit_code = self._run_validate()
        self.assertEqual(exit_code, 0)

    def test_validate_prints_report(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "validate",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.routing_csv),
        ])
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            try:
                args.func(args)
            except SystemExit:
                pass
        output = mock_out.getvalue()
        self.assertTrue(len(output) > 0)

    def test_missing_tphg_dir_gives_error_exit(self) -> None:
        exit_code = self._run_validate(["--tphg-dir", "/nonexistent/tphg/"])
        self.assertEqual(exit_code, 1)

    def test_bad_routing_file_exits_nonzero(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "validate",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.root / "nonexistent.csv"),
        ])
        with self.assertRaises(SystemExit) as ctx:
            args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)


# ---------------------------------------------------------------------------
# casfly trace (integration via parser + _cmd_trace)
# ---------------------------------------------------------------------------

class TestCmdTrace(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.events_csv = self.root / "events.csv"
        self.routing_csv = self.root / "routing.csv"
        _events_csv(self.events_csv)
        _routing_csv(self.routing_csv)

    def tearDown(self) -> None:
        self.td.cleanup()

    def _parse_trace(self, extra_args: list[str] = []) -> object:
        parser = build_parser()
        return parser.parse_args([
            "trace",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.routing_csv),
            "--patient", "P001",
            "--event", "Heart Attack",
            "--start-device", "smartwatch",
        ] + extra_args)

    def test_trace_text_output_runs(self) -> None:
        args = self._parse_trace()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            args.func(args)
        output = mock_out.getvalue()
        self.assertIn("confidence", output.lower())

    def test_trace_json_format_is_valid(self) -> None:
        args = self._parse_trace(["--format", "json"])
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            args.func(args)
        output = mock_out.getvalue()
        # Strip the "Tracing..." preamble line, then find the JSON block
        json_part = output[output.index("{"):]
        data = json.loads(json_part)
        self.assertIn("chain_confidence", data)

    def test_trace_csv_format_has_hop_column(self) -> None:
        args = self._parse_trace(["--format", "csv"])
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            args.func(args)
        output = mock_out.getvalue()
        csv_part = output[output.index("hop"):]
        rows = list(csv.DictReader(csv_part.splitlines()))
        self.assertIn("hop", rows[0])

    def test_trace_dot_format_starts_with_digraph(self) -> None:
        args = self._parse_trace(["--format", "dot"])
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            args.func(args)
        output = mock_out.getvalue()
        self.assertIn("digraph", output)

    def test_trace_writes_json_output_file(self) -> None:
        out = self.root / "result.json"
        args = self._parse_trace(["--output", str(out)])
        with patch("sys.stdout", new_callable=StringIO):
            args.func(args)
        data = json.loads(out.read_text())
        self.assertIn("chain_confidence", data)

    def test_trace_writes_csv_output_file(self) -> None:
        out = self.root / "result.csv"
        args = self._parse_trace(["--output", str(out)])
        with patch("sys.stdout", new_callable=StringIO):
            args.func(args)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        self.assertGreater(len(rows), 0)

    def test_trace_writes_dot_output_file(self) -> None:
        out = self.root / "result.dot"
        args = self._parse_trace(["--output", str(out)])
        with patch("sys.stdout", new_callable=StringIO):
            args.func(args)
        self.assertTrue(out.read_text().startswith("digraph"))

    def test_trace_unknown_patient_exits(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "trace",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.routing_csv),
            "--patient", "P999",
            "--event", "Heart Attack",
            "--start-device", "smartwatch",
        ])
        with self.assertRaises(SystemExit) as ctx:
            args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)

    def test_trace_unknown_start_device_exits(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "trace",
            "--device", "smartwatch", str(self.events_csv),
            "--routing", str(self.routing_csv),
            "--patient", "P001",
            "--event", "Heart Attack",
            "--start-device", "nonexistent_device",
        ])
        with self.assertRaises(SystemExit) as ctx:
            args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)

    def test_trace_bad_events_file_exits(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "trace",
            "--device", "smartwatch", str(self.root / "missing.csv"),
            "--routing", str(self.routing_csv),
            "--patient", "P001",
            "--event", "Heart Attack",
            "--start-device", "smartwatch",
        ])
        with self.assertRaises(SystemExit) as ctx:
            args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)


# ---------------------------------------------------------------------------
# casfly build-cache
# ---------------------------------------------------------------------------

class TestCmdBuildCache(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.events_csv = self.root / "events.csv"
        _events_csv(self.events_csv)

    def tearDown(self) -> None:
        self.td.cleanup()

    def test_build_cache_with_joblib(self) -> None:
        pytest = __import__("importlib").util.find_spec("joblib")
        if pytest is None:
            self.skipTest("joblib not installed")

        out_dir = self.root / "cache"
        parser = build_parser()
        args = parser.parse_args([
            "build-cache",
            "--device", "smartwatch", str(self.events_csv),
            "--output", str(out_dir),
        ])
        with patch("sys.stdout", new_callable=StringIO):
            args.func(args)
        self.assertTrue((out_dir / "smartwatch").is_dir())

    def test_build_cache_without_joblib_exits(self) -> None:
        out_dir = self.root / "cache"
        parser = build_parser()
        args = parser.parse_args([
            "build-cache",
            "--device", "smartwatch", str(self.events_csv),
            "--output", str(out_dir),
        ])
        # Mock build_tphg_cache to raise ImportError (simulates missing joblib)
        with patch("casfly_sdk.utils.build_tphg_cache", side_effect=ImportError("joblib")):
            with patch("sys.stdout", new_callable=StringIO):
                with self.assertRaises(SystemExit) as ctx:
                    args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)

    def test_build_cache_no_devices_exits(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "build-cache",
            "--output", str(self.root / "cache"),
        ])
        args.device = None  # simulate no --device provided
        with self.assertRaises(SystemExit) as ctx:
            args.func(args)
        self.assertEqual(int(ctx.exception.code), 1)


if __name__ == "__main__":
    unittest.main()
