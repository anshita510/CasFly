# CasFly SDK

This tool builds lag-aware causal graphs from timestamped event logs, traces the most probable multi-device precursor chain for a target event using LaVE, routes expansion across devices via a probabilistic lookup table, and exposes the workflow through a Python SDK and FastAPI service:
- `TPHG` (Temporal Probabilistic Health Graph) construction
- `LaVE` (Lag-aware Viterbi Expansion) for most-probable dependency paths
- `PLT` (Probabilistic Lookup Table) for next-node routing
- Multi-device orchestration for decentralized causal chain tracing
- Configurable lag-weight function `f(L)` (unity, exponential decay, or per-bin mapping)
- Bounded fallback expansion when direct predecessor expansion is exhausted
- Confidence audit outputs (weighted confidence, raw confidence, lag-weight product)

## Install (editable)

```bash
cd sdk
python3 -m pip install -e .
```

## How to Use CasFly with Two Devices (Smartwatch + Blood Pressure Cuff)

Use this flow when you have fragmented logs from multiple devices and want a probable precursor chain.

1. Prepare historical transition counts as `(cause, effect, lag_bin) -> count`.
2. Convert each device's local timestamped records into `TimedEvent` lists.
3. Build one local `TPHG` per device using the same CPT.
4. Define a `ProbabilisticLookupTable` for routing (`event_name -> next device`).
5. Register all devices in `CasFlyOrchestrator`.
6. Call `trace(start_device, start_event)` and inspect `hops` + `confidence_audit()`.

Runnable example:

```bash
cd sdk
PYTHONPATH=src python3 examples/smartwatch_bp_cuff_demo.py
```

Example file:
- `examples/smartwatch_bp_cuff_demo.py`

How to map your own data:
- Smartwatch stream rows -> `TimedEvent(name=<event_label>, timestamp=<utc_datetime>)`
- Blood pressure cuff rows -> same `TimedEvent` format
- Historical cohort counts -> `ConditionalProbabilityTable.from_counts(...)`
- Routing knowledge -> `ProbabilisticLookupTable([(event, device_id, prob), ...])`

## Quick Integration Example

```python
from datetime import datetime, timedelta

from casfly_sdk import (
    CasFlyNode,
    CasFlyOrchestrator,
    ConditionalProbabilityTable,
    ProbabilisticLookupTable,
    TPHGBuilder,
    TimedEvent,
)

# 1) Build lag-conditioned transition table from counts.
counts = {
    ("Medication Non-Compliance", "Missed Beta-Blockers", "0-7 days"): 30,
    ("Missed Beta-Blockers", "Hypertension", "0-7 days"): 25,
    ("Hypertension", "Elevated Blood Pressure", "0-7 days"): 20,
    ("Elevated Blood Pressure", "Elevated Heart Rate", "0-7 days"): 16,
    ("Elevated Heart Rate", "Heart Attack", "0-7 days"): 14,
}
cpt = ConditionalProbabilityTable.from_counts(counts)

# 2) Create local event logs and per-device TPHGs.
now = datetime.utcnow()
node_a_events = [
    TimedEvent("Elevated Heart Rate", now - timedelta(hours=1.5)),
    TimedEvent("Heart Attack", now),
]
node_b_events = [
    TimedEvent("Hypertension", now - timedelta(hours=3)),
    TimedEvent("Elevated Blood Pressure", now - timedelta(hours=2)),
]
node_c_events = [
    TimedEvent("Medication Non-Compliance", now - timedelta(hours=6)),
    TimedEvent("Missed Beta-Blockers", now - timedelta(hours=2.5)),
]

builder = TPHGBuilder(cpt)
tphg_a = builder.build(node_a_events)
tphg_b = builder.build(node_b_events)
tphg_c = builder.build(node_c_events)

# 3) Define app-level routing table (event -> next device probability).
plt = ProbabilisticLookupTable([
    ("Elevated Heart Rate", "smartwatch", 0.8),
    ("Hypertension", "bp_cuff", 0.9),
    ("Missed Beta-Blockers", "med_dispenser", 0.95),
])

# 4) Register nodes and run chain tracing.
orchestrator = CasFlyOrchestrator(max_hops=8)
orchestrator.register(CasFlyNode("hospital", tphg_a, plt))
orchestrator.register(CasFlyNode("bp_cuff", tphg_b, plt))
orchestrator.register(CasFlyNode("med_dispenser", tphg_c, plt))
orchestrator.register(CasFlyNode("smartwatch", tphg_b, plt))

result = orchestrator.trace(start_device="hospital", start_event="Heart Attack")

print(result.visited_devices)
print(result.chain_confidence)
for hop in result.hops:
    print(hop)
```

## Integration Model for Any Application

- Keep the SDK core unchanged.
- Adapt only these boundaries in your app:
  - Local event source -> `list[TimedEvent]`
  - Historical transitions -> `ConditionalProbabilityTable`
  - Service routing metadata -> `ProbabilisticLookupTable`
  - Transport layer (HTTP, gRPC, Kafka, MQTT) around `CasFlyNode.handle`

## Test

```bash
cd sdk
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Production HTTP Layer (FastAPI)

Service code lives in `service/casfly_service`.

### Run locally

```bash
cd sdk
python3 -m pip install -e .
python3 -m pip install -r service/requirements.txt
uvicorn casfly_service.app:app --app-dir service --host 0.0.0.0 --port 8000
```

### API

- `GET /health`
- `POST /v1/trace`

Example request payload:
- `service/sample_trace_request.json`

Example curl:

```bash
cd sdk
curl -sS -X POST "http://127.0.0.1:8000/v1/trace" \
  -H "Content-Type: application/json" \
  --data @service/sample_trace_request.json
```

### Docker

```bash
cd sdk
docker build -f service/Dockerfile -t casfly-service:0.1.0 .
docker run --rm -p 8000:8000 casfly-service:0.1.0
```

## Citation

Please cite both the software repository and the CasFly paper.

```bibtex
@software{gupta_misra_casfly_repo_2026,
  title   = {CasFly: SDK, Service, and Reproducibility Pipeline},
  author  = {Gupta, Anshita and Misra, Sudip},
  year    = {2026},
  url     = {https://github.com/anshita510/CasFly},
  version = {v0.1.0}
}
```

```bibtex
@ARTICLE{11250683,
  author={Gupta, Anshita and Misra, Sudip},
  journal={IEEE Internet of Things Journal},
  title={CasFly: Causal Chain Tracing Across Fragmented Edge Data for IoT Healthcare},
  year={2026},
  volume={13},
  number={3},
  pages={4578-4586},
  keywords={Medical services;Internet of Things;Probabilistic logic;Viterbi algorithm;Heuristic algorithms;Protocols;Heart rate;Real-time systems;Wearable Health Monitoring Systems;Object recognition;Causal chain tracing;decentralized protocols;edge computing;fragmented data;IoT healthcare;probabilistic graphs},
  doi={10.1109/JIOT.2025.3633292}
}
```

IEEE Xplore: `https://ieeexplore.ieee.org/document/11250683/`
