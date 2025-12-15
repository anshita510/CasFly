# CasFly

We are releasing CasFly as an open-source tool for researchers and developers building real-time health monitoring systems on IoT edge devices. CasFly lets you trace the most probable chain of precursor events leading up to a clinical trigger — across multiple wearables and bedside monitors — without ever centralizing the raw patient data.

The core idea: each physical device holds only its own slice of the causal graph. When a trigger event (e.g., a sudden heart rate spike) occurs, CasFly expands backward through the local graph, then hops to the next relevant device and continues — building a full precursor chain collaboratively, device by device, with no single point holding all the data.

This tool is designed to be data- and device-agnostic. You bring your own wearable streams and historical transition counts; CasFly handles the graph construction, probabilistic path expansion, and multi-device routing.

## How It Works

1. **TPHG** (Temporal Probabilistic Health Graph) — a directed weighted graph built from your historical event transition counts, conditioned on lag bins (e.g., "0-7 days"). One TPHG slice lives on each device.
2. **LaVE** (Lag-aware Viterbi Expansion) — backward Viterbi search that finds the most probable precursor path through a device's local TPHG.
3. **PLT** (Probabilistic Lookup Table) — routing table that maps an event type to the next device most likely to hold its predecessor, with associated probabilities.
4. **CasFlyOrchestrator** — coordinates multi-hop chain expansion across all registered devices.

## Install

```bash
pip install casfly-sdk            # core SDK (pure stdlib — no extra dependencies)
pip install casfly-sdk[service]   # + FastAPI HTTP service
```

## Quick Start

```python
from datetime import datetime, timedelta
from casfly_sdk import (
    CasFlyNode, CasFlyOrchestrator,
    ConditionalProbabilityTable, ProbabilisticLookupTable,
    TPHGBuilder, TimedEvent,
)

# 1. Build a lag-conditioned transition table from historical counts.
counts = {
    ("Medication Non-Compliance", "Missed Beta-Blockers",      "0-7 days"): 30,
    ("Missed Beta-Blockers",      "Hypertension",              "0-7 days"): 25,
    ("Hypertension",              "Elevated Blood Pressure",   "0-7 days"): 20,
    ("Elevated Blood Pressure",   "Elevated Heart Rate",       "0-7 days"): 16,
    ("Elevated Heart Rate",       "Heart Attack",              "0-7 days"): 14,
}
cpt = ConditionalProbabilityTable.from_counts(counts)

# 2. Create local event logs and build one TPHG per device.
now = datetime.utcnow()
tphg_hospital = TPHGBuilder(cpt).build([
    TimedEvent("Elevated Heart Rate", now - timedelta(hours=1.5)),
    TimedEvent("Heart Attack",        now),
])
tphg_bp = TPHGBuilder(cpt).build([
    TimedEvent("Hypertension",            now - timedelta(hours=3)),
    TimedEvent("Elevated Blood Pressure", now - timedelta(hours=2)),
])
tphg_med = TPHGBuilder(cpt).build([
    TimedEvent("Medication Non-Compliance", now - timedelta(hours=6)),
    TimedEvent("Missed Beta-Blockers",      now - timedelta(hours=2.5)),
])

# 3. Define routing: which event routes to which device.
plt = ProbabilisticLookupTable([
    ("Elevated Heart Rate",  "hospital",      0.8),
    ("Hypertension",         "bp_cuff",       0.9),
    ("Missed Beta-Blockers", "med_dispenser", 0.95),
])

# 4. Register devices and trace.
orch = CasFlyOrchestrator(max_hops=8)
orch.register(CasFlyNode("hospital",      tphg_hospital, plt))
orch.register(CasFlyNode("bp_cuff",       tphg_bp,       plt))
orch.register(CasFlyNode("med_dispenser", tphg_med,      plt))

result = orch.trace(start_device="hospital", start_event="Heart Attack")
print(result.visited_devices)   # ['hospital', 'bp_cuff', 'med_dispenser']
print(result.chain_confidence)  # overall chain probability
for hop in result.hops:
    print(hop)
```

## What You Provide

| Input | Description |
|---|---|
| Historical transition counts | `(cause, effect, lag_bin) → count` from your cohort data |
| Per-device event logs | `TimedEvent(name, timestamp)` lists, one per device |
| Routing table | Which event types live on which device, with routing probabilities |

CasFly never centralizes raw data. Each device processes only its own TPHG slice.

## Input Formats

`load_events_by_patient()` accepts CSV, JSON, FHIR R4 Bundle, Parquet, and Excel — auto-detected from file extension:

```python
from casfly_sdk.utils import load_events_by_patient, count_transitions

events = load_events_by_patient("smartwatch.csv")
events = load_events_by_patient("ecg.json")
events = load_events_by_patient("conditions.fhir.json")   # FHIR R4

counts = count_transitions(events)
cpt    = ConditionalProbabilityTable.from_counts(counts)
```

## Command-Line Interface

```bash
casfly trace \
  --device smartwatch smartwatch.csv \
  --device bp_cuff bp.fhir.json \
  --routing routing.csv \
  --patient P001 --event "Heart Attack" --start-device smartwatch

casfly validate --device smartwatch smartwatch.csv --routing routing.csv
casfly build-cache --device smartwatch smartwatch.csv --output ./cache/
```

## Distributed IoT Mode

For deployments on real hardware (Raspberry Pi, edge nodes), each physical device runs one `CasFlyDevice` process. It binds a UDP port and participates in the multi-hop chain expansion protocol, exchanging causal chain packets directly with peer devices — no central server involved.

See [CasFly_tool/sdk/README.md](CasFly_tool/sdk/README.md) for full API details.

## Repository Structure

```
CasFly_tool/
  sdk/
    src/casfly_sdk/    core SDK — TPHG, LaVE, PLT, orchestrator, device protocol
    service/           FastAPI HTTP service layer (per-device)
    examples/          runnable demo scripts
    tests/             118 unit tests
casflyIoTJ/            paper evaluation scripts (IEEE IoT-J)
```

## Future Work

- **FHIR-native integration.** CasFly already supports FHIR R4 Bundle ingestion for loading event data, but deeper integration with EHR infrastructure — such as subscribing to FHIR Subscription resources, resolving `Patient` references, and emitting results as FHIR `DiagnosticReport` or `RiskAssessment` resources — would allow CasFly to slot directly into hospital workflows without a custom data pipeline.

- **PLT auto-discovery for large deployments.** The current routing model requires a hand-authored PLT mapping event types to devices. In large-scale deployments with heterogeneous, dynamically joined device ecosystems, maintaining this manually becomes impractical. A device advertisement protocol — where each node broadcasts the event types it holds, and the PLT is assembled automatically — would make CasFly self-configuring and more robust to device churn.

## Citation

If you use CasFly in your work, please cite both the software and the paper:

```bibtex
@software{gupta_misra_casfly_repo_2026,
  title   = {CasFly: Causal Chain Tracing SDK for IoT Health Networks},
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

IEEE Xplore: https://ieeexplore.ieee.org/document/11250683/
