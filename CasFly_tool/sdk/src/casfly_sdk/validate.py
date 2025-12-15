"""Input validation utilities for CasFly.

Use these before running a trace to catch configuration issues early —
missing PLT routes, uncovered transitions, absent TPHG cache files, etc.

Example::

    from casfly_sdk.validate import validate_all

    report = validate_all(
        events_by_patient=cohort,
        cpt=cpt,
        plt=plt,
        tphg_dir="tphg_cache/smartwatch/",   # optional
    )
    print(report)
    if report.has_errors():
        raise SystemExit("Fix errors before tracing.")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lookup import ProbabilisticLookupTable
    from .models import TimedEvent
    from .tphg import ConditionalProbabilityTable


# ---------------------------------------------------------------------------
# ValidationIssue + ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    level: str    # "ERROR" | "WARNING" | "INFO"
    check: str    # which check raised this
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    def has_warnings(self) -> bool:
        return any(i.level == "WARNING" for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARNING"]

    def summary(self) -> str:
        n_err = len(self.errors())
        n_warn = len(self.warnings())
        status = "PASS" if not self.has_errors() else "FAIL"
        return f"Validation {status} — {n_err} error(s), {n_warn} warning(s)"

    def __str__(self) -> str:
        if not self.issues:
            return "Validation OK — no issues found."
        lines = [self.summary()]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)

    def _add(self, level: str, check: str, message: str) -> None:
        self.issues.append(ValidationIssue(level=level, check=check, message=message))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def validate_plt_coverage(
    events_by_patient: dict[str, list[TimedEvent]],
    plt: ProbabilisticLookupTable,
) -> ValidationReport:
    """Check that every observed event type has at least one PLT routing entry.

    Events with no PLT route will cause the chain to terminate early at that
    hop — the tracer cannot determine which device to query next.
    """
    report = ValidationReport()
    check = "plt_coverage"

    all_event_names: set[str] = set()
    for events in events_by_patient.values():
        for e in events:
            all_event_names.add(e.name)

    unrouted = sorted(all_event_names - plt.known_events)
    if unrouted:
        for name in unrouted:
            report._add(
                "WARNING", check,
                f"Event '{name}' has no PLT routing entry — "
                "chain will stop if this event is a predecessor.",
            )
    else:
        report._add("INFO", check, "All observed event types have PLT routing entries.")

    return report


def validate_cpt_coverage(
    events_by_patient: dict[str, list[TimedEvent]],
    cpt: ConditionalProbabilityTable,
    max_lag_days: float = 730.0,
) -> ValidationReport:
    """Check that observed event transitions have CPT entries.

    Transitions without a CPT entry are silently skipped during TPHG
    construction, which may produce sparse graphs and lower chain confidence.
    """
    from .lag import lag_to_bin

    report = ValidationReport()
    check = "cpt_coverage"
    missing: set[tuple[str, str, str]] = set()

    for events in events_by_patient.values():
        ordered = sorted(events, key=lambda e: e.timestamp)
        for i, ea in enumerate(ordered):
            for eb in ordered[i + 1:]:
                lag_days = (eb.timestamp - ea.timestamp).total_seconds() / 86400.0
                if lag_days > max_lag_days:
                    break
                lag_bin = lag_to_bin(lag_days)
                if cpt.get(ea.name, eb.name, lag_bin) is None:
                    missing.add((ea.name, eb.name, lag_bin))

    if missing:
        for cause, effect, lag_bin in sorted(missing):
            report._add(
                "WARNING", check,
                f"Transition '{cause}' -> '{effect}' [{lag_bin}] "
                "has no CPT entry — edge will be skipped in TPHG.",
            )
    else:
        report._add("INFO", check, "All observed transitions have CPT entries.")

    return report


def validate_tphg_cache(
    events_by_patient: dict[str, list[TimedEvent]],
    tphg_dir: str | Path,
) -> ValidationReport:
    """Check that a ``.pkl`` cache file exists for every patient.

    Missing files will cause ``CasFlyDevice.load_tphg()`` to return ``None``
    and the device to skip chain expansion for that patient.
    """
    report = ValidationReport()
    check = "tphg_cache"
    tphg_dir = Path(tphg_dir)

    if not tphg_dir.exists():
        report._add("ERROR", check, f"TPHG directory '{tphg_dir}' does not exist.")
        return report

    missing = []
    for patient_id in events_by_patient:
        pkl = tphg_dir / f"{patient_id}_tphg.pkl"
        if not pkl.exists():
            missing.append(patient_id)

    if missing:
        for pid in sorted(missing):
            report._add(
                "ERROR", check,
                f"Missing TPHG cache file for patient '{pid}': "
                f"{tphg_dir / (pid + '_tphg.pkl')}",
            )
    else:
        report._add(
            "INFO", check,
            f"All {len(events_by_patient)} patient TPHG cache files present.",
        )

    return report


def validate_all(
    events_by_patient: dict[str, list[TimedEvent]],
    cpt: ConditionalProbabilityTable,
    plt: ProbabilisticLookupTable,
    tphg_dir: str | Path | None = None,
    max_lag_days: float = 730.0,
) -> ValidationReport:
    """Run all validation checks and return a combined report.

    Parameters
    ----------
    events_by_patient:
        Per-patient event logs (from ``load_events_by_patient``).
    cpt:
        Fitted ``ConditionalProbabilityTable``.
    plt:
        ``ProbabilisticLookupTable`` for routing.
    tphg_dir:
        Optional path to a pre-built TPHG cache directory. If given,
        ``validate_tphg_cache`` is also run.
    max_lag_days:
        Maximum lag window used in ``validate_cpt_coverage``.

    Example::

        report = validate_all(cohort, cpt, plt, tphg_dir="tphg_cache/BPCuff/")
        print(report)
        if report.has_errors():
            raise SystemExit(1)
    """
    combined = ValidationReport()

    for check_report in [
        validate_plt_coverage(events_by_patient, plt),
        validate_cpt_coverage(events_by_patient, cpt, max_lag_days),
    ]:
        combined.issues.extend(check_report.issues)

    if tphg_dir is not None:
        combined.issues.extend(
            validate_tphg_cache(events_by_patient, tphg_dir).issues
        )

    return combined
