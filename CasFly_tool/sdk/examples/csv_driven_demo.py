"""Multi-format demo — realistic end-to-end workflow.

This example shows the full journey from raw wearable data files to a
CasFly causal chain trace. It deliberately uses different input formats
for each device to show that CasFly is format-agnostic:

    smartwatch  →  JSON flat array      (examples/data/smartwatch_events.json)
    bp_cuff     →  FHIR R4 Bundle JSON  (examples/data/bp_cuff_events.fhir.json)
    routing     →  CSV                  (examples/data/routing_table.csv)

Parquet and Excel work the same way — just change the file path.
For those formats install the optional extra first:

    pip install casfly-sdk[data]

Run from the sdk/ directory:

    PYTHONPATH=src python3 examples/csv_driven_demo.py
"""
from __future__ import annotations

from pathlib import Path

from casfly_sdk import (
    CasFlyNode,
    CasFlyOrchestrator,
    ConditionalProbabilityTable,
    ProbabilisticLookupTable,
    TPHGBuilder,
    unity_lag_weight,
    LagAwareViterbi,
)
from casfly_sdk.utils import count_transitions, load_events_by_patient

DATA_DIR = Path(__file__).parent / "data"
TRACE_PATIENT = "P001"
TRIGGER_EVENT = "Arrhythmia Alert"


def main() -> None:
    # ------------------------------------------------------------------ #
    # Step 1 — Load per-patient event logs from each device's data file.  #
    #           Format is auto-detected from the file extension.           #
    # ------------------------------------------------------------------ #
    print("Loading events (format auto-detected from extension)...")

    # JSON flat array — {"PATIENT": ..., "DATE": ..., "DESCRIPTION": ...}
    smartwatch_events = load_events_by_patient(DATA_DIR / "smartwatch_events.json")

    # FHIR R4 Bundle — Condition + Observation resources
    bp_cuff_events = load_events_by_patient(DATA_DIR / "bp_cuff_events.fhir.json")

    # Swap either line above with .csv / .parquet / .xlsx and it just works.

    print(f"  smartwatch (JSON):  {sum(len(v) for v in smartwatch_events.values())} events "
          f"across {len(smartwatch_events)} patients")
    print(f"  bp_cuff    (FHIR):  {sum(len(v) for v in bp_cuff_events.values())} events "
          f"across {len(bp_cuff_events)} patients")

    # ------------------------------------------------------------------ #
    # Step 2 — Merge all device events into a single cohort dict and      #
    #           compute cohort-level transition counts.                    #
    # ------------------------------------------------------------------ #
    cohort: dict[str, list] = {}
    for source in [smartwatch_events, bp_cuff_events]:
        for patient_id, events in source.items():
            cohort.setdefault(patient_id, []).extend(events)

    counts = count_transitions(cohort, max_lag_days=30.0)
    print(f"\nTransition counts from cohort ({len(cohort)} patients):")
    for (cause, effect, lag_bin), count in sorted(counts.items()):
        print(f"  {cause!r:35} -> {effect!r:30} [{lag_bin}]  n={count}")

    # ------------------------------------------------------------------ #
    # Step 3 — Fit the Conditional Probability Table.                     #
    # ------------------------------------------------------------------ #
    cpt = ConditionalProbabilityTable.from_counts(counts)

    # ------------------------------------------------------------------ #
    # Step 4 — Build one TPHG per device for the patient to trace.        #
    # ------------------------------------------------------------------ #
    builder = TPHGBuilder(cpt)

    patient_smartwatch = smartwatch_events.get(TRACE_PATIENT, [])
    patient_bp_cuff    = bp_cuff_events.get(TRACE_PATIENT, [])

    tphg_smartwatch = builder.build(patient_smartwatch)
    tphg_bp_cuff    = builder.build(patient_bp_cuff)

    print(f"\nTPHG for {TRACE_PATIENT} — smartwatch: "
          f"{len(tphg_smartwatch.edges)} edges, "
          f"{len(tphg_smartwatch.vertices)} vertices")
    print(f"TPHG for {TRACE_PATIENT} — bp_cuff:    "
          f"{len(tphg_bp_cuff.edges)} edges, "
          f"{len(tphg_bp_cuff.vertices)} vertices")

    # ------------------------------------------------------------------ #
    # Step 5 — Load the routing table from CSV.                           #
    # ------------------------------------------------------------------ #
    plt = ProbabilisticLookupTable.from_csv(DATA_DIR / "routing_table.csv")

    # ------------------------------------------------------------------ #
    # Step 6 — Register devices and run the trace.                        #
    # ------------------------------------------------------------------ #
    lave = LagAwareViterbi(max_depth=8, fallback_depth=2, lag_weight_fn=unity_lag_weight)
    orch = CasFlyOrchestrator(max_hops=6)
    orch.register(CasFlyNode("smartwatch", tphg_smartwatch, plt, lave=lave))
    orch.register(CasFlyNode("bp_cuff",    tphg_bp_cuff,    plt, lave=lave))

    print(f"\nTracing precursor chain for event: {TRIGGER_EVENT!r} (patient {TRACE_PATIENT})")
    result = orch.trace(start_device="smartwatch", start_event=TRIGGER_EVENT)

    print(f"\nVisited devices : {result.visited_devices}")
    for i, hop in enumerate(result.hops, start=1):
        print(
            f"Hop {i}: [{hop.source_device}] "
            f"path={hop.best_path}  "
            f"weighted_p={hop.path_probability:.4f}"
        )

    audit = result.confidence_audit()
    print(f"\nChain confidence : {audit['weighted_confidence']:.4f}")
    print(f"Raw confidence   : {audit['raw_confidence']:.4f}")
    print(f"Lag weight prod  : {audit['lag_weight_product']:.4f}")


if __name__ == "__main__":
    main()
