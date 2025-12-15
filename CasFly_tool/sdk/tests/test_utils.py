from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from datetime import datetime
from pathlib import Path

from casfly_sdk import ConditionalProbabilityTable, ProbabilisticLookupTable, TimedEvent
from casfly_sdk.utils import count_transitions, load_events_by_patient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TimedEvent.from_csv
# ---------------------------------------------------------------------------

class TestTimedEventFromCsv(unittest.TestCase):
    def test_loads_events_with_default_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", """\
                DATE,DESCRIPTION
                2023-01-10,Hypertension
                2023-01-15,Heart Attack
            """)
            events = TimedEvent.from_csv(f)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "Hypertension")
        self.assertEqual(events[0].timestamp, datetime(2023, 1, 10))
        self.assertEqual(events[1].name, "Heart Attack")

    def test_custom_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", """\
                ts,event
                2023-06-01,Elevated Heart Rate
            """)
            events = TimedEvent.from_csv(f, name_col="event", timestamp_col="ts")
        self.assertEqual(events[0].name, "Elevated Heart Rate")
        self.assertEqual(events[0].timestamp, datetime(2023, 6, 1))

    def test_custom_timestamp_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", """\
                DATE,DESCRIPTION
                15/01/2023,Hypertension
            """)
            events = TimedEvent.from_csv(f, timestamp_fmt="%d/%m/%Y")
        self.assertEqual(events[0].timestamp, datetime(2023, 1, 15))

    def test_empty_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", "DATE,DESCRIPTION\n")
            events = TimedEvent.from_csv(f)
        self.assertEqual(events, [])


# ---------------------------------------------------------------------------
# TimedEvent.from_json
# ---------------------------------------------------------------------------

class TestTimedEventFromJson(unittest.TestCase):
    def test_loads_flat_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = [
                {"DATE": "2023-01-10", "DESCRIPTION": "Hypertension"},
                {"DATE": "2023-01-15", "DESCRIPTION": "Heart Attack"},
            ]
            f = Path(td) / "ev.json"
            f.write_text(json.dumps(data), encoding="utf-8")
            events = TimedEvent.from_json(f)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "Hypertension")
        self.assertEqual(events[1].timestamp, datetime(2023, 1, 15))

    def test_raises_on_non_array(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "bad.json"
            f.write_text('{"key": "value"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                TimedEvent.from_json(f)


# ---------------------------------------------------------------------------
# ProbabilisticLookupTable.from_csv
# ---------------------------------------------------------------------------

class TestPLTFromCsv(unittest.TestCase):
    def test_loads_routing_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "plt.csv", """\
                event,device,probability
                Elevated Heart Rate,smartwatch,0.9
                Hypertension,bp_cuff,0.85
            """)
            plt = ProbabilisticLookupTable.from_csv(f)
        self.assertEqual(plt.next_device("Elevated Heart Rate"), "smartwatch")
        self.assertEqual(plt.next_device("Hypertension"), "bp_cuff")
        self.assertIsNone(plt.next_device("Unknown Event"))

    def test_custom_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "plt.csv", """\
                ev,dev,prob
                Heart Attack,hospital,0.95
            """)
            plt = ProbabilisticLookupTable.from_csv(
                f, event_col="ev", device_col="dev", probability_col="prob"
            )
        self.assertEqual(plt.next_device("Heart Attack"), "hospital")

    def test_excludes_visited_device(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "plt.csv", """\
                event,device,probability
                Hypertension,bp_cuff,0.9
                Hypertension,smartwatch,0.6
            """)
            plt = ProbabilisticLookupTable.from_csv(f)
        self.assertEqual(plt.next_device("Hypertension", exclude={"bp_cuff"}), "smartwatch")

    def test_invalid_probability_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "plt.csv", """\
                event,device,probability
                Hypertension,bp_cuff,1.5
            """)
            with self.assertRaises(ValueError):
                ProbabilisticLookupTable.from_csv(f)


# ---------------------------------------------------------------------------
# load_events_by_patient — CSV
# ---------------------------------------------------------------------------

class TestLoadEventsByPatientCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.f = _write(Path(self.tmp), "ev.csv", """\
            PATIENT,DATE,DESCRIPTION
            P001,2023-01-10,Hypertension
            P001,2023-01-15,Heart Attack
            P002,2023-02-01,Elevated Heart Rate
        """)

    def test_groups_events_by_patient(self) -> None:
        result = load_events_by_patient(self.f)
        self.assertIn("P001", result)
        self.assertIn("P002", result)
        self.assertEqual(len(result["P001"]), 2)
        self.assertEqual(len(result["P002"]), 1)

    def test_event_names_and_timestamps(self) -> None:
        result = load_events_by_patient(self.f)
        names = {e.name for e in result["P001"]}
        self.assertIn("Hypertension", names)
        self.assertIn("Heart Attack", names)

    def test_auto_detect_csv_extension(self) -> None:
        result = load_events_by_patient(self.f, fmt="auto")
        self.assertIn("P001", result)

    def test_explicit_fmt_csv(self) -> None:
        result = load_events_by_patient(self.f, fmt="csv")
        self.assertIn("P002", result)

    def test_custom_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", """\
                pid,ts,event
                P001,2023-01-10,Hypertension
            """)
            result = load_events_by_patient(
                f, patient_col="pid", name_col="event", timestamp_col="ts"
            )
        self.assertEqual(result["P001"][0].name, "Hypertension")


# ---------------------------------------------------------------------------
# load_events_by_patient — JSON
# ---------------------------------------------------------------------------

class TestLoadEventsByPatientJson(unittest.TestCase):
    def test_flat_array(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = [
                {"PATIENT": "P001", "DATE": "2023-01-10", "DESCRIPTION": "Hypertension"},
                {"PATIENT": "P001", "DATE": "2023-01-15", "DESCRIPTION": "Heart Attack"},
                {"PATIENT": "P002", "DATE": "2023-02-01", "DESCRIPTION": "Elevated Heart Rate"},
            ]
            f = Path(td) / "ev.json"
            f.write_text(json.dumps(data), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(len(result["P001"]), 2)
        self.assertEqual(len(result["P002"]), 1)

    def test_nested_by_patient(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = {
                "P001": [
                    {"DATE": "2023-01-10", "DESCRIPTION": "Hypertension"},
                    {"DATE": "2023-01-15", "DESCRIPTION": "Heart Attack"},
                ],
                "P002": [
                    {"DATE": "2023-02-01", "DESCRIPTION": "Elevated Heart Rate"},
                ],
            }
            f = Path(td) / "ev.json"
            f.write_text(json.dumps(data), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(len(result["P001"]), 2)
        self.assertEqual(result["P002"][0].name, "Elevated Heart Rate")

    def test_raises_on_invalid_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ev.json"
            f.write_text('"just a string"', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_events_by_patient(f)


# ---------------------------------------------------------------------------
# load_events_by_patient — FHIR R4 Bundle
# ---------------------------------------------------------------------------

class TestLoadEventsByPatientFhir(unittest.TestCase):
    def _bundle(self, entries: list[dict]) -> dict:
        return {"resourceType": "Bundle", "type": "collection", "entry": entries}

    def _condition(self, patient: str, text: str, date: str) -> dict:
        return {"resource": {
            "resourceType": "Condition",
            "subject": {"reference": f"Patient/{patient}"},
            "code": {"text": text},
            "recordedDate": date,
        }}

    def _observation(self, patient: str, text: str, date: str) -> dict:
        return {"resource": {
            "resourceType": "Observation",
            "subject": {"reference": f"Patient/{patient}"},
            "code": {"text": text},
            "effectiveDateTime": date,
        }}

    def test_loads_conditions_and_observations(self) -> None:
        bundle = self._bundle([
            self._condition("P001", "Hypertension", "2023-01-06"),
            self._observation("P001", "Elevated Blood Pressure", "2023-01-07"),
            self._condition("P002", "Hypertension", "2023-02-01"),
        ])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "data.fhir.json"
            f.write_text(json.dumps(bundle), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(len(result["P001"]), 2)
        self.assertEqual(len(result["P002"]), 1)
        names = {e.name for e in result["P001"]}
        self.assertIn("Hypertension", names)
        self.assertIn("Elevated Blood Pressure", names)

    def test_skips_unsupported_resource_types(self) -> None:
        bundle = self._bundle([
            {"resource": {"resourceType": "Patient", "id": "P001"}},
            self._condition("P001", "Hypertension", "2023-01-06"),
        ])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "data.fhir.json"
            f.write_text(json.dumps(bundle), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(len(result["P001"]), 1)

    def test_raises_on_non_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "data.fhir.json"
            f.write_text(json.dumps({"resourceType": "Patient"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_events_by_patient(f)

    def test_effectiveperiod_start_used_as_timestamp(self) -> None:
        bundle = self._bundle([{"resource": {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/P001"},
            "code": {"text": "Heart Rate"},
            "effectivePeriod": {"start": "2023-03-01"},
        }}])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "data.fhir.json"
            f.write_text(json.dumps(bundle), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(result["P001"][0].timestamp, datetime(2023, 3, 1))

    def test_code_coding_display_fallback(self) -> None:
        bundle = self._bundle([{"resource": {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/P001"},
            "code": {"coding": [{"display": "Hypertension (disorder)"}]},
            "recordedDate": "2023-01-06",
        }}])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "data.fhir.json"
            f.write_text(json.dumps(bundle), encoding="utf-8")
            result = load_events_by_patient(f)
        self.assertEqual(result["P001"][0].name, "Hypertension (disorder)")


# ---------------------------------------------------------------------------
# load_events_by_patient — format detection
# ---------------------------------------------------------------------------

class TestFormatDetection(unittest.TestCase):
    def test_unknown_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "events.txt"
            f.write_text("data", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_events_by_patient(f)

    def test_unknown_explicit_fmt_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = _write(Path(td), "ev.csv", "PATIENT,DATE,DESCRIPTION\n")
            with self.assertRaises(ValueError):
                load_events_by_patient(f, fmt="xml")


# ---------------------------------------------------------------------------
# count_transitions
# ---------------------------------------------------------------------------

class TestCountTransitions(unittest.TestCase):
    def _make_events(self, *args: tuple[str, str]) -> list[TimedEvent]:
        """args: (name, iso_date) pairs"""
        return [TimedEvent(name, datetime.fromisoformat(d)) for name, d in args]

    def test_counts_ordered_pairs(self) -> None:
        events = {
            "P001": self._make_events(
                ("Hypertension", "2023-01-06"),
                ("Heart Attack", "2023-01-10"),
            )
        }
        counts = count_transitions(events)
        self.assertEqual(counts[("Hypertension", "Heart Attack", "0-7 days")], 1)

    def test_accumulates_across_patients(self) -> None:
        events = {
            "P001": self._make_events(
                ("Hypertension", "2023-01-06"), ("Heart Attack", "2023-01-10")
            ),
            "P002": self._make_events(
                ("Hypertension", "2023-02-01"), ("Heart Attack", "2023-02-05")
            ),
        }
        counts = count_transitions(events)
        self.assertEqual(counts[("Hypertension", "Heart Attack", "0-7 days")], 2)

    def test_respects_max_lag_days(self) -> None:
        events = {
            "P001": self._make_events(
                ("A", "2023-01-01"),
                ("B", "2023-06-01"),  # ~150 days — beyond a 30-day window
            )
        }
        counts = count_transitions(events, max_lag_days=30.0)
        self.assertNotIn(("A", "B", "30-90 days"), counts)

    def test_correct_lag_bin_assignment(self) -> None:
        events = {
            "P001": self._make_events(
                ("A", "2023-01-01"),
                ("B", "2023-01-20"),  # 19 days → "14-30 days"
            )
        }
        counts = count_transitions(events)
        self.assertIn(("A", "B", "14-30 days"), counts)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(count_transitions({}), {})

    def test_single_event_patient_produces_no_transitions(self) -> None:
        events = {"P001": self._make_events(("Hypertension", "2023-01-01"))}
        self.assertEqual(count_transitions(events), {})

    def test_feeds_into_cpt(self) -> None:
        events = {
            "P001": self._make_events(
                ("Hypertension", "2023-01-06"), ("Heart Attack", "2023-01-10")
            )
        }
        counts = count_transitions(events)
        cpt = ConditionalProbabilityTable.from_counts(counts)
        prob = cpt.get("Hypertension", "Heart Attack", "0-7 days")
        self.assertIsNotNone(prob)
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
