"""Data preparation utilities for CasFly.

These helpers bridge the gap between raw wearable/EHR data files and the
inputs that CasFly's core SDK expects.

Supported input formats
-----------------------
+----------+------------+-------------------------------------------+
| Format   | Extension  | Extra deps                                |
+==========+============+===========================================+
| CSV      | .csv       | none (stdlib)                             |
+----------+------------+-------------------------------------------+
| JSON     | .json      | none (stdlib)  — flat array or nested     |
+----------+------------+-------------------------------------------+
| FHIR R4  | .fhir.json | none (stdlib)  — FHIR R4 Bundle           |
+----------+------------+-------------------------------------------+
| Parquet  | .parquet   | pip install casfly-sdk[data]              |
+----------+------------+-------------------------------------------+
| Excel    | .xlsx .xls | pip install casfly-sdk[data]              |
+----------+------------+-------------------------------------------+

Typical workflow::

    from casfly_sdk.utils import load_events_by_patient, count_transitions, build_tphg_cache
    from casfly_sdk import ConditionalProbabilityTable

    # 1. Load per-patient events from each device's data file (any format).
    smartwatch = load_events_by_patient("smartwatch_events.csv")
    bp_cuff    = load_events_by_patient("bp_cuff_events.parquet")  # auto-detected
    ecg_patch  = load_events_by_patient("ecg_bundle.fhir.json")    # FHIR R4 bundle

    # 2. Merge all device events into a single cohort dict.
    cohort: dict[str, list] = {}
    for source in [smartwatch, bp_cuff, ecg_patch]:
        for patient_id, events in source.items():
            cohort.setdefault(patient_id, []).extend(events)

    # 3. Compute transition counts and fit the CPT.
    counts = count_transitions(cohort)
    cpt = ConditionalProbabilityTable.from_counts(counts)

    # 4. (Optional) Pre-build and save per-patient TPHG .pkl files.
    build_tphg_cache(smartwatch, cpt, output_dir="tphg_cache/smartwatch/")
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .lag import lag_to_bin
from .models import TimedEvent

if TYPE_CHECKING:
    from .tphg import ConditionalProbabilityTable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ts(raw: str, fmt: str | None) -> datetime:
    return datetime.strptime(raw, fmt) if fmt else datetime.fromisoformat(raw)


def _detect_fmt(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fhir.json"):
        return "fhir"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith((".xlsx", ".xls")):
        return "excel"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".csv"):
        return "csv"
    raise ValueError(
        f"Cannot auto-detect format for '{path.name}'. "
        "Pass fmt='csv'|'json'|'fhir'|'parquet'|'excel' explicitly."
    )


# ---------------------------------------------------------------------------
# Format-specific loaders (all return dict[patient_id, list[TimedEvent]])
# ---------------------------------------------------------------------------

def _load_csv(
    path: Path,
    patient_col: str,
    name_col: str,
    timestamp_col: str,
    timestamp_fmt: str | None,
) -> dict[str, list[TimedEvent]]:
    result: dict[str, list[TimedEvent]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row[patient_col].strip()].append(
                TimedEvent(
                    name=row[name_col].strip(),
                    timestamp=_parse_ts(row[timestamp_col].strip(), timestamp_fmt),
                )
            )
    return dict(result)


def _load_json(
    path: Path,
    patient_col: str,
    name_col: str,
    timestamp_col: str,
    timestamp_fmt: str | None,
) -> dict[str, list[TimedEvent]]:
    """Load JSON — two shapes are accepted:

    **Flat array** (one record per event)::

        [
          {"PATIENT": "P001", "DATE": "2023-01-10", "DESCRIPTION": "Hypertension"},
          ...
        ]

    **Nested by patient** (grouped)::

        {
          "P001": [{"DATE": "2023-01-10", "DESCRIPTION": "Hypertension"}, ...],
          "P002": [...]
        }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, list[TimedEvent]] = defaultdict(list)

    if isinstance(data, list):
        # Flat array of records
        for row in data:
            result[str(row[patient_col]).strip()].append(
                TimedEvent(
                    name=str(row[name_col]).strip(),
                    timestamp=_parse_ts(str(row[timestamp_col]).strip(), timestamp_fmt),
                )
            )
    elif isinstance(data, dict):
        # Nested: {patient_id: [{name_col: ..., timestamp_col: ...}, ...]}
        for patient_id, records in data.items():
            for row in records:
                result[str(patient_id).strip()].append(
                    TimedEvent(
                        name=str(row[name_col]).strip(),
                        timestamp=_parse_ts(str(row[timestamp_col]).strip(), timestamp_fmt),
                    )
                )
    else:
        raise ValueError("JSON file must contain a list of records or a dict keyed by patient ID.")

    return dict(result)


def _load_fhir(path: Path, timestamp_fmt: str | None) -> dict[str, list[TimedEvent]]:
    """Load a FHIR R4 Bundle JSON file.

    Extracts ``Condition`` and ``Observation`` resources. For each resource the
    patient ID is taken from ``subject.reference`` (e.g. ``"Patient/P001"``),
    the event label from ``code.text`` (or ``code.coding[0].display``), and the
    timestamp from ``recordedDate`` / ``effectiveDateTime`` / ``effectivePeriod.start``.

    Example FHIR Bundle::

        {
          "resourceType": "Bundle",
          "entry": [
            {
              "resource": {
                "resourceType": "Condition",
                "subject":      {"reference": "Patient/P001"},
                "code":         {"text": "Hypertension"},
                "recordedDate": "2023-01-06"
              }
            }
          ]
        }
    """
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)

    if bundle.get("resourceType") != "Bundle":
        raise ValueError("FHIR file must have resourceType 'Bundle'.")

    result: dict[str, list[TimedEvent]] = defaultdict(list)
    supported = {"Condition", "Observation"}

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") not in supported:
            continue

        # Patient ID
        ref = resource.get("subject", {}).get("reference", "")
        patient_id = ref.split("/")[-1] if ref else None
        if not patient_id:
            continue

        # Event label
        code = resource.get("code", {})
        label = code.get("text") or (
            code.get("coding", [{}])[0].get("display") if code.get("coding") else None
        )
        if not label:
            continue

        # Timestamp
        ts_raw = (
            resource.get("recordedDate")
            or resource.get("effectiveDateTime")
            or (resource.get("effectivePeriod") or {}).get("start")
        )
        if not ts_raw:
            continue

        result[patient_id].append(
            TimedEvent(
                name=label.strip(),
                timestamp=_parse_ts(ts_raw.strip(), timestamp_fmt),
            )
        )

    return dict(result)


def _load_tabular(
    path: Path,
    patient_col: str,
    name_col: str,
    timestamp_col: str,
    timestamp_fmt: str | None,
    sheet_name: str | int = 0,
    fmt: str = "parquet",
) -> dict[str, list[TimedEvent]]:
    """Load Parquet or Excel via pandas."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            f"Loading {fmt} files requires pandas: pip install casfly-sdk[data]"
        ) from exc

    if fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    result: dict[str, list[TimedEvent]] = defaultdict(list)
    for _, row in df.iterrows():
        patient_id = str(row[patient_col]).strip()
        name = str(row[name_col]).strip()
        ts_raw = row[timestamp_col]
        # pandas may give us a Timestamp directly
        if hasattr(ts_raw, "to_pydatetime"):
            ts = ts_raw.to_pydatetime()
        else:
            ts = _parse_ts(str(ts_raw).strip(), timestamp_fmt)
        result[patient_id].append(TimedEvent(name=name, timestamp=ts))

    return dict(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_events_by_patient(
    path: str | Path,
    fmt: str = "auto",
    patient_col: str = "PATIENT",
    name_col: str = "DESCRIPTION",
    timestamp_col: str = "DATE",
    timestamp_fmt: str | None = None,
    sheet_name: str | int = 0,
) -> dict[str, list[TimedEvent]]:
    """Load a data file and group events by patient ID.

    The format is auto-detected from the file extension unless *fmt* is given.

    Parameters
    ----------
    path:
        Path to the data file.
    fmt:
        One of ``"auto"`` (default), ``"csv"``, ``"json"``, ``"fhir"``,
        ``"parquet"``, or ``"excel"``.
    patient_col:
        Column / key name for the patient identifier. Default: ``"PATIENT"``.
        Not used for FHIR (patient ID is read from ``subject.reference``).
    name_col:
        Column / key name for the event label. Default: ``"DESCRIPTION"``.
        Not used for FHIR (label is read from ``code.text``).
    timestamp_col:
        Column / key name for the timestamp. Default: ``"DATE"``.
        Not used for FHIR.
    timestamp_fmt:
        Optional ``strptime`` format string (e.g. ``"%Y-%m-%d %H:%M:%S"``).
        If omitted, ``datetime.fromisoformat()`` is used, which handles
        ISO-8601 dates and datetimes.
    sheet_name:
        Sheet index or name for Excel files (default: first sheet).

    Returns
    -------
    dict[patient_id, list[TimedEvent]]

    Examples
    --------
    ::

        # CSV
        events = load_events_by_patient("smartwatch_events.csv")

        # JSON flat array
        events = load_events_by_patient("ecg_events.json")

        # JSON nested by patient
        events = load_events_by_patient("ecg_events.json")

        # FHIR R4 Bundle
        events = load_events_by_patient("conditions.fhir.json")

        # Parquet  (requires: pip install casfly-sdk[data])
        events = load_events_by_patient("bp_events.parquet")

        # Excel   (requires: pip install casfly-sdk[data])
        events = load_events_by_patient("lab_results.xlsx")

        # Custom column names
        events = load_events_by_patient(
            "data.csv",
            patient_col="patient_id",
            name_col="event_type",
            timestamp_col="event_time",
            timestamp_fmt="%d/%m/%Y",
        )
    """
    path = Path(path)
    resolved_fmt = _detect_fmt(path) if fmt == "auto" else fmt.lower()

    if resolved_fmt == "csv":
        return _load_csv(path, patient_col, name_col, timestamp_col, timestamp_fmt)
    if resolved_fmt == "json":
        return _load_json(path, patient_col, name_col, timestamp_col, timestamp_fmt)
    if resolved_fmt == "fhir":
        return _load_fhir(path, timestamp_fmt)
    if resolved_fmt == "parquet":
        return _load_tabular(path, patient_col, name_col, timestamp_col, timestamp_fmt, fmt="parquet")
    if resolved_fmt == "excel":
        return _load_tabular(path, patient_col, name_col, timestamp_col, timestamp_fmt, sheet_name, fmt="excel")

    raise ValueError(
        f"Unknown format '{resolved_fmt}'. "
        "Use 'csv', 'json', 'fhir', 'parquet', or 'excel'."
    )


def count_transitions(
    events_by_patient: dict[str, list[TimedEvent]],
    max_lag_days: float = 730.0,
) -> dict[tuple[str, str, str], int]:
    """Compute ``(cause, effect, lag_bin) -> count`` from per-patient event logs.

    For each patient, every ordered event pair ``(ea, eb)`` where
    ``ea.timestamp < eb.timestamp`` and the lag is within *max_lag_days* is
    counted as a transition. The result is a cohort-level count dict suitable
    for ``ConditionalProbabilityTable.from_counts()``.

    Parameters
    ----------
    events_by_patient:
        Mapping of ``patient_id -> list[TimedEvent]`` (unsorted is fine).
    max_lag_days:
        Maximum lag in days between two events to count as a causal pair.
        Default: 730 (2 years), matching the last finite lag bin.

    Returns
    -------
    dict suitable for ``ConditionalProbabilityTable.from_counts()``.

    Example::

        counts = count_transitions({
            "P001": [TimedEvent("Hypertension", t0), TimedEvent("Heart Attack", t1)],
        })
        cpt = ConditionalProbabilityTable.from_counts(counts)
    """
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for events in events_by_patient.values():
        ordered = sorted(events, key=lambda e: e.timestamp)
        for i, ea in enumerate(ordered):
            for eb in ordered[i + 1:]:
                lag_days = (eb.timestamp - ea.timestamp).total_seconds() / 86400.0
                if lag_days > max_lag_days:
                    break
                lag_bin = lag_to_bin(lag_days)
                counts[(ea.name, eb.name, lag_bin)] += 1

    return dict(counts)


def build_tphg_cache(
    events_by_patient: dict[str, list[TimedEvent]],
    cpt: ConditionalProbabilityTable,
    output_dir: str | Path,
) -> None:
    """Build and save one TPHG ``.pkl`` file per patient.

    Files are written as ``{output_dir}/{patient_id}_tphg.pkl``.
    The output directory is created if it does not exist.

    Requires ``joblib`` (``pip install joblib``).

    Parameters
    ----------
    events_by_patient:
        Mapping of ``patient_id -> list[TimedEvent]``.
    cpt:
        A fitted ``ConditionalProbabilityTable`` (built from cohort counts).
    output_dir:
        Directory to write ``.pkl`` files into.

    Example::

        build_tphg_cache(smartwatch_events, cpt, "tphg_cache/smartwatch/")
        # → tphg_cache/smartwatch/P001_tphg.pkl
        # → tphg_cache/smartwatch/P002_tphg.pkl
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "build_tphg_cache requires joblib: pip install joblib"
        ) from exc

    from .tphg import TPHGBuilder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = TPHGBuilder(cpt)
    for patient_id, events in events_by_patient.items():
        tphg = builder.build(events)
        joblib.dump(tphg, output_dir / f"{patient_id}_tphg.pkl")
