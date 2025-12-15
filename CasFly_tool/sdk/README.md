# CasFly SDK

This tool builds lag-aware causal graphs from timestamped event logs, traces the most probable multi-device precursor chain for a target event using LaVE, routes expansion across devices via a probabilistic lookup table, and exposes the workflow through a Python SDK and FastAPI service:
- `TPHG` (Temporal Probabilistic Health Graph) construction
- `LaVE` (Lag-aware Viterbi Expansion) for most-probable dependency paths
- `PLT` (Probabilistic Lookup Table) for next-node routing
- Multi-device orchestration for decentralized causal chain tracing
- Configurable lag-weight function `f(L)` (unity, exponential decay, or per-bin mapping)
- Bounded fallback expansion when direct predecessor expansion is exhausted
- Confidence audit outputs (weighted confidence, raw confidence, lag-weight product)

## Install

From PyPI (recommended):

```bash
# Core SDK only (pure stdlib — no extra dependencies)
pip install casfly-sdk

# With the HTTP service layer (FastAPI + uvicorn)
pip install casfly-sdk[service]
```

From source (editable install for development):

```bash
cd sdk
pip install -e .
pip install -e ".[service]"   # also install service deps
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

## Command-Line Interface

After installing, a `casfly` command is available with three subcommands.

### `casfly trace` — run a chain trace from data files

```bash
casfly trace \
  --device smartwatch  examples/data/smartwatch_events.json \
  --device bp_cuff     examples/data/bp_cuff_events.fhir.json \
  --routing            examples/data/routing_table.csv \
  --patient P001 \
  --event   "Arrhythmia Alert" \
  --start-device smartwatch
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--device ID FILE` | required | Device ID and event file (repeat for each device) |
| `--routing FILE` | required | Routing table CSV |
| `--patient` | required | Patient ID to trace |
| `--event` | required | Trigger event name |
| `--start-device` | required | Device to start from |
| `--max-hops` | 8 | Maximum chain hops |
| `--max-lag-days` | 730 | Max transition lag window (days) |
| `--format` | text | Output format: `text` \| `json` \| `csv` \| `dot` |
| `--output FILE` | — | Write result to file (`.json` / `.csv` / `.dot`) |

### `casfly build-cache` — pre-build TPHG `.pkl` files

Required for distributed device mode. One `.pkl` file per patient per device.

```bash
casfly build-cache \
  --device smartwatch examples/data/smartwatch_events.json \
  --device bp_cuff    examples/data/bp_cuff_events.fhir.json \
  --output ./tphg_cache/
# → tphg_cache/smartwatch/P001_tphg.pkl, P002_tphg.pkl, ...
# → tphg_cache/bp_cuff/P001_tphg.pkl,    P002_tphg.pkl, ...
```

Requires `joblib`: `pip install joblib`.

### `casfly validate` — check coverage before tracing

Catches misconfiguration early: missing PLT routes, uncovered transitions,
absent TPHG cache files.

```bash
casfly validate \
  --device smartwatch examples/data/smartwatch_events.json \
  --routing           examples/data/routing_table.csv \
  --tphg-dir          ./tphg_cache/smartwatch/
```

Exits with code `1` if any errors are found.

### Export chain results

From Python, `ChainResult` can be exported to multiple formats:

```python
result = orch.trace(start_device="smartwatch", start_event="Arrhythmia Alert")

result.export("chain.json")   # JSON
result.export("chain.csv")    # CSV — one row per hop
result.export("chain.dot")    # Graphviz DOT — render with: dot -Tpng chain.dot -o chain.png

# Or get the string directly:
print(result.to_json())
print(result.to_csv())
print(result.to_dot())
```

### Validate from Python

```python
from casfly_sdk.validate import validate_all

report = validate_all(cohort, cpt, plt, tphg_dir="./tphg_cache/smartwatch/")
print(report)
if report.has_errors():
    raise SystemExit(1)
```

## Production HTTP Layer (FastAPI)

The HTTP service is a **per-device management wrapper** — raw patient data
never leaves the device. Each device runs its own container/process; the
causal chain flows between devices over UDP.

```
Smartwatch  :8001     BPCuff    :8002     CasFlyHub   :8003
────────────────────────────────────────────────────────────
CasFlyDevice          CasFlyDevice         CasFlyDevice
     │  <──── UDP chain packets (no raw data) ────>  │
```

Service code lives in `service/casfly_service`.

### API

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| GET    | `/health`   | Liveness check |
| POST   | `/start`    | Initialise this device (bind UDP port, load TPHG) |
| POST   | `/initiate` | Start causal chain tracing from this device |
| GET    | `/status`   | Return device identity and port |

### Run locally (one process per device)

```bash
cd sdk
pip install -e ".[service]"

# Start device 1
DEVICE_ID=BPCuff uvicorn casfly_service.app:app \
  --app-dir service --host 0.0.0.0 --port 8001 &

# Start device 2
DEVICE_ID=Smartwatch uvicorn casfly_service.app:app \
  --app-dir service --host 0.0.0.0 --port 8002 &

# Initialise each device (call POST /start on each)
curl -X POST http://localhost:8001/start \
  -H "Content-Type: application/json" \
  -d '{"device_id":"BPCuff","lookup_csv":"data/lookup.csv","tphg_dir":"data/tphg/BPCuff","filtered_patients_csv":"data/patients.csv"}'

# Trigger chain tracing from the initiator device
curl -X POST http://localhost:8002/initiate \
  -H "Content-Type: application/json" \
  -d '{"event":"Heart Attack","patient_id":"P001"}'
```

### Docker (one container per device)

```bash
cd sdk
docker build -f service/Dockerfile -t casfly-service:0.1.0 .

# Each device is a separate container with its own data volume
docker run --rm -p 8001:8000 \
  -e DEVICE_ID=BPCuff \
  -v /your/data:/data \
  casfly-service:0.1.0

docker run --rm -p 8002:8000 \
  -e DEVICE_ID=Smartwatch \
  -v /your/data:/data \
  casfly-service:0.1.0
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
