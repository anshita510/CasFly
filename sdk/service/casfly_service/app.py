from __future__ import annotations

from fastapi import FastAPI, HTTPException

from casfly_sdk import (
    CasFlyNode,
    CasFlyOrchestrator,
    ConditionalProbabilityTable,
    LagAwareViterbi,
    ProbabilisticLookupTable,
    TPHGBuilder,
    TimedEvent,
)

from .schemas import HopOut, TraceRequest, TraceResponse

app = FastAPI(
    title="CasFly Service",
    version="0.1.0",
    description="Production HTTP layer for CasFly SDK",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/trace", response_model=TraceResponse)
def trace_chain(payload: TraceRequest) -> TraceResponse:
    if not payload.cpt_counts:
        raise HTTPException(status_code=400, detail="cpt_counts must not be empty")

    counts = {
        (item.cause, item.effect, item.lag_bin): item.count
        for item in payload.cpt_counts
    }
    cpt = ConditionalProbabilityTable.from_counts(counts)
    builder = TPHGBuilder(cpt)

    routes = [(r.event_name, r.device_id, r.probability) for r in payload.routing_table]
    lookup_table = ProbabilisticLookupTable(routes)

    orchestrator = CasFlyOrchestrator(max_hops=payload.max_hops)

    known_devices: set[str] = set()
    for node in payload.nodes:
        known_devices.add(node.device_id)
        events = [TimedEvent(name=e.name, timestamp=e.timestamp) for e in node.events]
        tphg = builder.build(events)
        lave = LagAwareViterbi(max_depth=payload.max_depth, fallback_depth=payload.fallback_depth)
        orchestrator.register(
            CasFlyNode(
                device_id=node.device_id,
                tphg=tphg,
                lookup_table=lookup_table,
                lave=lave,
            )
        )

    if payload.start_device not in known_devices:
        raise HTTPException(status_code=400, detail=f"unknown start_device: {payload.start_device}")

    result = orchestrator.trace(start_device=payload.start_device, start_event=payload.start_event)

    hops = [
        HopOut(
            source_device=h.source_device,
            target_device=h.target_device,
            trigger_event=h.trigger_event,
            best_path=list(h.best_path),
            path_probability=h.path_probability,
            cumulative_lag_days=h.cumulative_lag_days,
        )
        for h in result.hops
    ]

    return TraceResponse(
        visited_devices=result.visited_devices,
        visited_edges=sorted(result.visited_edges),
        chain_confidence=result.chain_confidence,
        chain_raw_confidence=result.chain_raw_confidence,
        chain_lag_weight_product=result.chain_lag_weight_product,
        theorem_lower_bound=result.confidence_audit()["theorem_lower_bound"],
        hops=hops,
    )
