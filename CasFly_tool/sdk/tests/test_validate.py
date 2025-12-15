from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from casfly_sdk import ConditionalProbabilityTable, ProbabilisticLookupTable
from casfly_sdk.models import TimedEvent
from casfly_sdk.validate import (
    ValidationReport,
    validate_all,
    validate_cpt_coverage,
    validate_plt_coverage,
    validate_tphg_cache,
)


def _ev(name: str, date: str) -> TimedEvent:
    return TimedEvent(name=name, timestamp=datetime.fromisoformat(date))


def _make_cohort() -> dict[str, list[TimedEvent]]:
    return {
        "P001": [
            _ev("Hypertension", "2023-01-06"),
            _ev("Heart Attack", "2023-01-10"),
        ],
        "P002": [
            _ev("Hypertension", "2023-02-01"),
            _ev("Heart Attack", "2023-02-05"),
        ],
    }


def _make_cpt() -> ConditionalProbabilityTable:
    return ConditionalProbabilityTable.from_counts({
        ("Hypertension", "Heart Attack", "0-7 days"): 10,
    })


def _make_plt(*routes) -> ProbabilisticLookupTable:
    return ProbabilisticLookupTable(list(routes))


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport(unittest.TestCase):
    def test_empty_report_has_no_errors(self) -> None:
        r = ValidationReport()
        self.assertFalse(r.has_errors())
        self.assertFalse(r.has_warnings())

    def test_has_errors_detects_error_level(self) -> None:
        r = ValidationReport()
        r._add("ERROR", "check", "something broke")
        self.assertTrue(r.has_errors())
        self.assertFalse(r.has_warnings())

    def test_has_warnings_detects_warning_level(self) -> None:
        r = ValidationReport()
        r._add("WARNING", "check", "something odd")
        self.assertTrue(r.has_warnings())
        self.assertFalse(r.has_errors())

    def test_errors_filters_correctly(self) -> None:
        r = ValidationReport()
        r._add("ERROR", "c", "e")
        r._add("WARNING", "c", "w")
        self.assertEqual(len(r.errors()), 1)
        self.assertEqual(len(r.warnings()), 1)

    def test_summary_pass(self) -> None:
        r = ValidationReport()
        r._add("INFO", "c", "ok")
        self.assertIn("PASS", r.summary())

    def test_summary_fail(self) -> None:
        r = ValidationReport()
        r._add("ERROR", "c", "bad")
        self.assertIn("FAIL", r.summary())

    def test_str_shows_issues(self) -> None:
        r = ValidationReport()
        r._add("WARNING", "plt_coverage", "Event X has no route")
        self.assertIn("plt_coverage", str(r))
        self.assertIn("Event X", str(r))

    def test_str_empty_report(self) -> None:
        self.assertIn("OK", str(ValidationReport()))


# ---------------------------------------------------------------------------
# validate_plt_coverage
# ---------------------------------------------------------------------------

class TestValidatePltCoverage(unittest.TestCase):
    def test_all_events_covered_gives_info(self) -> None:
        plt = _make_plt(
            ("Hypertension", "bp_cuff", 0.9),
            ("Heart Attack", "hospital", 0.8),
        )
        report = validate_plt_coverage(_make_cohort(), plt)
        self.assertFalse(report.has_errors())
        self.assertFalse(report.has_warnings())
        self.assertTrue(any(i.level == "INFO" for i in report.issues))

    def test_missing_event_gives_warning(self) -> None:
        plt = _make_plt(("Hypertension", "bp_cuff", 0.9))
        report = validate_plt_coverage(_make_cohort(), plt)
        self.assertTrue(report.has_warnings())
        msgs = [i.message for i in report.warnings()]
        self.assertTrue(any("Heart Attack" in m for m in msgs))

    def test_empty_cohort_gives_info(self) -> None:
        plt = _make_plt(("Hypertension", "bp_cuff", 0.9))
        report = validate_plt_coverage({}, plt)
        self.assertFalse(report.has_warnings())


# ---------------------------------------------------------------------------
# validate_cpt_coverage
# ---------------------------------------------------------------------------

class TestValidateCptCoverage(unittest.TestCase):
    def test_all_transitions_covered_gives_info(self) -> None:
        cpt = _make_cpt()
        report = validate_cpt_coverage(_make_cohort(), cpt)
        self.assertFalse(report.has_errors())
        self.assertFalse(report.has_warnings())

    def test_missing_transition_gives_warning(self) -> None:
        # CPT with no entries — all transitions are uncovered
        cpt = ConditionalProbabilityTable.from_counts({
            ("SomeOtherEvent", "AnotherEvent", "0-7 days"): 1,
        })
        report = validate_cpt_coverage(_make_cohort(), cpt)
        self.assertTrue(report.has_warnings())

    def test_max_lag_days_limits_checked_pairs(self) -> None:
        cohort = {"P001": [
            _ev("A", "2023-01-01"),
            _ev("B", "2023-12-31"),  # ~364 days apart
        ]}
        cpt = ConditionalProbabilityTable.from_counts({
            ("SomeOther", "Event", "0-7 days"): 1
        })
        # With max_lag_days=30, the pair (A, B) is ignored → no warning
        report = validate_cpt_coverage(cohort, cpt, max_lag_days=30.0)
        self.assertFalse(report.has_warnings())


# ---------------------------------------------------------------------------
# validate_tphg_cache
# ---------------------------------------------------------------------------

class TestValidateTphgCache(unittest.TestCase):
    def test_all_files_present_gives_info(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tphg_dir = Path(td)
            for pid in ["P001", "P002"]:
                (tphg_dir / f"{pid}_tphg.pkl").touch()
            report = validate_tphg_cache(_make_cohort(), tphg_dir)
        self.assertFalse(report.has_errors())

    def test_missing_file_gives_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tphg_dir = Path(td)
            (tphg_dir / "P001_tphg.pkl").touch()
            # P002 missing
            report = validate_tphg_cache(_make_cohort(), tphg_dir)
        self.assertTrue(report.has_errors())
        self.assertTrue(any("P002" in i.message for i in report.errors()))

    def test_nonexistent_directory_gives_error(self) -> None:
        report = validate_tphg_cache(_make_cohort(), "/nonexistent/path/")
        self.assertTrue(report.has_errors())
        self.assertTrue(any("does not exist" in i.message for i in report.errors()))


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------

class TestValidateAll(unittest.TestCase):
    def test_clean_setup_passes(self) -> None:
        plt = _make_plt(
            ("Hypertension", "bp_cuff", 0.9),
            ("Heart Attack", "hospital", 0.8),
        )
        report = validate_all(_make_cohort(), _make_cpt(), plt)
        self.assertFalse(report.has_errors())

    def test_combines_issues_from_all_checks(self) -> None:
        # PLT missing Heart Attack → warning from plt_coverage
        plt = _make_plt(("Hypertension", "bp_cuff", 0.9))
        report = validate_all(_make_cohort(), _make_cpt(), plt)
        checks = {i.check for i in report.issues}
        self.assertIn("plt_coverage", checks)
        self.assertIn("cpt_coverage", checks)

    def test_tphg_dir_included_when_provided(self) -> None:
        plt = _make_plt(
            ("Hypertension", "bp_cuff", 0.9),
            ("Heart Attack", "hospital", 0.8),
        )
        report = validate_all(
            _make_cohort(), _make_cpt(), plt,
            tphg_dir="/nonexistent/",
        )
        checks = {i.check for i in report.issues}
        self.assertIn("tphg_cache", checks)

    def test_tphg_dir_omitted_when_none(self) -> None:
        plt = _make_plt(
            ("Hypertension", "bp_cuff", 0.9),
            ("Heart Attack", "hospital", 0.8),
        )
        report = validate_all(_make_cohort(), _make_cpt(), plt)
        checks = {i.check for i in report.issues}
        self.assertNotIn("tphg_cache", checks)


if __name__ == "__main__":
    unittest.main()
