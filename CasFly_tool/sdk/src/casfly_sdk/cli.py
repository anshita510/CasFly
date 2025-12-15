"""CasFly command-line interface.

Subcommands
-----------
casfly trace         Run a causal chain trace from event files.
casfly build-cache   Pre-build per-patient TPHG .pkl files.
casfly validate      Validate PLT/CPT/TPHG coverage before tracing.

Run ``casfly <subcommand> --help`` for full usage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_devices(device_args: list[list[str]]) -> dict[str, list]:
    """Convert [[device_id, file_path], ...] pairs into events_by_patient per device."""
    from casfly_sdk.utils import load_events_by_patient

    device_events: dict[str, dict[str, list]] = {}
    for device_id, file_path in device_args:
        try:
            device_events[device_id] = load_events_by_patient(file_path)
        except Exception as exc:
            _die(f"Could not load events for device '{device_id}' from '{file_path}': {exc}")
    return device_events


def _die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def _merge_cohort(device_events: dict[str, dict[str, list]]) -> dict[str, list]:
    """Merge per-device event dicts into a single cohort dict."""
    cohort: dict[str, list] = {}
    for events_by_patient in device_events.values():
        for patient_id, events in events_by_patient.items():
            cohort.setdefault(patient_id, []).extend(events)
    return cohort


# ---------------------------------------------------------------------------
# casfly trace
# ---------------------------------------------------------------------------

def _cmd_trace(args: argparse.Namespace) -> None:
    from casfly_sdk import (
        CasFlyNode, CasFlyOrchestrator,
        ConditionalProbabilityTable, ProbabilisticLookupTable,
        TPHGBuilder, LagAwareViterbi, unity_lag_weight,
    )
    from casfly_sdk.utils import count_transitions

    if not args.device:
        _die("Provide at least one --device DEVICE_ID FILE pair.")

    # Load events per device
    device_events = _load_devices(args.device)

    # Build cohort-level CPT
    cohort = _merge_cohort(device_events)
    if args.patient not in cohort:
        _die(
            f"Patient '{args.patient}' not found in any event file. "
            f"Available: {sorted(cohort.keys())}"
        )

    counts = count_transitions(cohort, max_lag_days=args.max_lag_days)
    if not counts:
        _die("No transitions found in the cohort data. Check your event files.")

    cpt = ConditionalProbabilityTable.from_counts(counts)

    # Load PLT
    try:
        plt = ProbabilisticLookupTable.from_csv(args.routing)
    except Exception as exc:
        _die(f"Could not load routing table from '{args.routing}': {exc}")

    # Build per-device TPHG for the target patient
    builder = TPHGBuilder(cpt)
    lave = LagAwareViterbi(
        max_depth=args.max_depth,
        fallback_depth=args.fallback_depth,
        lag_weight_fn=unity_lag_weight,
    )
    orchestrator = CasFlyOrchestrator(max_hops=args.max_hops)

    for device_id, events_by_patient in device_events.items():
        patient_events = events_by_patient.get(args.patient, [])
        tphg = builder.build(patient_events)
        orchestrator.register(CasFlyNode(device_id, tphg, plt, lave=lave))

    if args.start_device not in device_events:
        _die(
            f"Start device '{args.start_device}' not found. "
            f"Available: {list(device_events.keys())}"
        )

    # Run trace
    print(f"Tracing '{args.event}' for patient '{args.patient}' "
          f"from device '{args.start_device}'...\n")
    result = orchestrator.trace(
        start_device=args.start_device,
        start_event=args.event,
    )

    # Output
    if args.output:
        result.export(args.output)
        print(f"Result written to: {args.output}")
    else:
        fmt = args.format
        if fmt == "json":
            print(result.to_json())
        elif fmt == "csv":
            print(result.to_csv())
        elif fmt == "dot":
            print(result.to_dot())
        else:  # text (default)
            print(f"Visited devices : {result.visited_devices}")
            print(f"Chain confidence: {result.chain_confidence:.4f}\n")
            for i, hop in enumerate(result.hops, 1):
                path_str = " -> ".join(hop.best_path)
                print(f"  Hop {i}: [{hop.source_device}]  {path_str}  "
                      f"(p={hop.path_probability:.4f})")
            print()
            audit = result.confidence_audit()
            print(f"  Weighted confidence : {audit['weighted_confidence']:.4f}")
            print(f"  Raw confidence      : {audit['raw_confidence']:.4f}")
            print(f"  Theorem lower bound : {audit['theorem_lower_bound']:.4f}")


# ---------------------------------------------------------------------------
# casfly build-cache
# ---------------------------------------------------------------------------

def _cmd_build_cache(args: argparse.Namespace) -> None:
    from casfly_sdk import ConditionalProbabilityTable
    from casfly_sdk.utils import build_tphg_cache, count_transitions

    if not args.device:
        _die("Provide at least one --device DEVICE_ID FILE pair.")

    device_events = _load_devices(args.device)
    cohort = _merge_cohort(device_events)
    counts = count_transitions(cohort, max_lag_days=args.max_lag_days)

    if not counts:
        _die("No transitions found in the cohort data. Check your event files.")

    cpt = ConditionalProbabilityTable.from_counts(counts)
    output_root = Path(args.output)

    for device_id, events_by_patient in device_events.items():
        out_dir = output_root / device_id
        print(f"Building TPHG cache for '{device_id}' "
              f"({len(events_by_patient)} patients) -> {out_dir}")
        try:
            build_tphg_cache(events_by_patient, cpt, out_dir)
        except ImportError:
            _die("build-cache requires joblib: pip install joblib")

    print(f"\nDone. Cache written under: {output_root}")


# ---------------------------------------------------------------------------
# casfly validate
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> None:
    from casfly_sdk import ConditionalProbabilityTable, ProbabilisticLookupTable
    from casfly_sdk.utils import count_transitions
    from casfly_sdk.validate import validate_all

    if not args.device:
        _die("Provide at least one --device DEVICE_ID FILE pair.")

    device_events = _load_devices(args.device)
    cohort = _merge_cohort(device_events)
    counts = count_transitions(cohort, max_lag_days=args.max_lag_days)
    cpt = ConditionalProbabilityTable.from_counts(counts)

    try:
        plt = ProbabilisticLookupTable.from_csv(args.routing)
    except Exception as exc:
        _die(f"Could not load routing table from '{args.routing}': {exc}")

    report = validate_all(
        events_by_patient=cohort,
        cpt=cpt,
        plt=plt,
        tphg_dir=args.tphg_dir,
        max_lag_days=args.max_lag_days,
    )

    print(report)
    if report.has_errors():
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", metavar=("DEVICE_ID", "FILE"), nargs=2, action="append",
        help=(
            "Device ID and path to its event file. "
            "Repeat for each device: "
            "--device smartwatch events.json --device bp_cuff readings.fhir.json"
        ),
    )


def _common_args(parser: argparse.ArgumentParser) -> None:
    _device_arg(parser)
    parser.add_argument(
        "--max-lag-days", type=float, default=730.0,
        help="Maximum lag in days between two events to count as a transition (default: 730).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="casfly",
        description="CasFly — decentralized causal chain tracing for IoT health networks.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- trace ----
    p_trace = sub.add_parser("trace", help="Run a causal chain trace.")
    _common_args(p_trace)
    p_trace.add_argument(
        "--routing", required=True, metavar="FILE",
        help="Path to the routing table CSV (event, device, probability).",
    )
    p_trace.add_argument("--patient", required=True, help="Patient ID to trace.")
    p_trace.add_argument("--event", required=True, help="Trigger event name.")
    p_trace.add_argument("--start-device", required=True, help="Device to start the trace from.")
    p_trace.add_argument("--max-hops", type=int, default=8, help="Max chain hops (default: 8).")
    p_trace.add_argument("--max-depth", type=int, default=10, help="Max Viterbi depth (default: 10).")
    p_trace.add_argument("--fallback-depth", type=int, default=3, help="Fallback depth (default: 3).")
    p_trace.add_argument(
        "--format", choices=["text", "json", "csv", "dot"], default="text",
        help="Output format when printing to stdout (default: text).",
    )
    p_trace.add_argument(
        "--output", metavar="FILE",
        help="Write result to file (.json / .csv / .dot). Overrides --format.",
    )
    p_trace.set_defaults(func=_cmd_trace)

    # ---- build-cache ----
    p_cache = sub.add_parser(
        "build-cache",
        help="Pre-build per-patient TPHG .pkl files for distributed device mode.",
    )
    _common_args(p_cache)
    p_cache.add_argument(
        "--output", required=True, metavar="DIR",
        help="Root directory for cache output. One subdirectory per device is created.",
    )
    p_cache.set_defaults(func=_cmd_build_cache)

    # ---- validate ----
    p_val = sub.add_parser(
        "validate",
        help="Validate PLT/CPT/TPHG coverage before running a trace.",
    )
    _common_args(p_val)
    p_val.add_argument(
        "--routing", required=True, metavar="FILE",
        help="Path to the routing table CSV.",
    )
    p_val.add_argument(
        "--tphg-dir", metavar="DIR", default=None,
        help="Optional: TPHG cache directory to check for missing .pkl files.",
    )
    p_val.set_defaults(func=_cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
