from __future__ import annotations

from datetime import datetime, timedelta, timezone

from casfly_sdk import (
    CasFlyNode,
    CasFlyOrchestrator,
    ConditionalProbabilityTable,
    LagAwareViterbi,
    ProbabilisticLookupTable,
    TPHGBuilder,
    TimedEvent,
    unity_lag_weight,
)


def main() -> None:
    # Step 1) Build lag-binned transition table from historical counts.
    counts = {
        ("Hypertension", "Elevated Heart Rate", "0-7 days"): 40,
        ("Elevated Heart Rate", "Arrhythmia Alert", "0-7 days"): 32,
    }
    cpt = ConditionalProbabilityTable.from_counts(counts)
    builder = TPHGBuilder(cpt)

    # Step 2) Prepare local event logs for two devices.
    now = datetime.now(timezone.utc)
    smartwatch_events = [
        TimedEvent("Elevated Heart Rate", now - timedelta(minutes=40)),
        TimedEvent("Arrhythmia Alert", now),
    ]
    bp_cuff_events = [
        TimedEvent("Hypertension", now - timedelta(hours=2)),
        TimedEvent("Elevated Heart Rate", now - timedelta(minutes=45)),
    ]

    smartwatch_tphg = builder.build(smartwatch_events)
    bp_cuff_tphg = builder.build(bp_cuff_events)

    # Step 3) Define event -> next-device routing.
    plt = ProbabilisticLookupTable(
        [
            ("Elevated Heart Rate", "bp_cuff", 0.95),
        ]
    )

    # Step 4) Register nodes and run tracing.
    lave = LagAwareViterbi(max_depth=8, fallback_depth=2, lag_weight_fn=unity_lag_weight)
    orchestrator = CasFlyOrchestrator(max_hops=6)
    orchestrator.register(CasFlyNode("smartwatch", smartwatch_tphg, plt, lave=lave))
    orchestrator.register(CasFlyNode("bp_cuff", bp_cuff_tphg, plt, lave=lave))

    result = orchestrator.trace(start_device="smartwatch", start_event="Arrhythmia Alert")

    print("Visited devices:", result.visited_devices)
    for i, hop in enumerate(result.hops, start=1):
        print(
            f"Hop {i}: {hop.source_device} -> {hop.target_device} | "
            f"path={hop.best_path} | weighted_p={hop.path_probability:.4f}"
        )

    print("Confidence audit:", result.confidence_audit())


if __name__ == "__main__":
    main()
