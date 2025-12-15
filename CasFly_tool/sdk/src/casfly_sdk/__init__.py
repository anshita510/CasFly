"""CasFly SDK — decentralized causal chain tracing for IoT health networks.

CasFly traces the most probable chain of precursor events leading up to a
clinical trigger (e.g. a sudden heart rate spike) across multiple wearable
devices and bedside monitors — without centralizing raw patient data.

Each device holds only its own slice of a Temporal Probabilistic Health Graph
(TPHG). When a trigger event occurs, CasFly expands the chain backward
locally, then hops to the next relevant device via a Probabilistic Lookup
Table (PLT), building the full precursor chain collaboratively.

Quick start
-----------
::

    from casfly_sdk import (
        CasFlyNode, CasFlyOrchestrator,
        ConditionalProbabilityTable, ProbabilisticLookupTable,
        TPHGBuilder, TimedEvent,
    )

    counts = {("Hypertension", "Heart Attack", "0-7 days"): 42}
    cpt  = ConditionalProbabilityTable.from_counts(counts)
    tphg = TPHGBuilder(cpt).build([TimedEvent("Heart Attack", datetime.utcnow())])
    plt  = ProbabilisticLookupTable([("Hypertension", "bp_device", 0.9)])

    orch = CasFlyOrchestrator(max_hops=8)
    orch.register(CasFlyNode("hospital", tphg, plt))
    result = orch.trace(start_device="hospital", start_event="Heart Attack")

    print(result.chain_confidence)
    result.export("chain.json")

Load data from files
--------------------
::

    from casfly_sdk.utils import load_events_by_patient, count_transitions

    events = load_events_by_patient("smartwatch.csv")          # CSV
    events = load_events_by_patient("ecg.json")                # JSON
    events = load_events_by_patient("conditions.fhir.json")    # FHIR R4
    events = load_events_by_patient("readings.parquet")        # Parquet *

    counts = count_transitions(events)
    cpt    = ConditionalProbabilityTable.from_counts(counts)

    * requires: pip install casfly-sdk[data]

Validate before tracing
-----------------------
::

    from casfly_sdk.validate import validate_all

    report = validate_all(cohort, cpt, plt)
    print(report)

Command-line interface
----------------------
::

    casfly trace \\
      --device smartwatch smartwatch.csv \\
      --device bp_cuff bp.fhir.json \\
      --routing routing.csv \\
      --patient P001 --event "Heart Attack" --start-device smartwatch

    casfly build-cache --device smartwatch smartwatch.csv --output ./cache/
    casfly validate   --device smartwatch smartwatch.csv --routing routing.csv
"""
__version__ = "0.1.0"

# CasFlyDevice is imported lazily to avoid pulling in heavy optional
# dependencies (networkx, joblib, pandas, psutil) on every `import casfly_sdk`.
# `from casfly_sdk import CasFlyDevice` still works — the import is deferred
# until the name is actually accessed.
from .lag import (
    DEFAULT_LAG_BINS,
    LagBin,
    exponential_decay_lag_weight,
    lag_to_bin,
    map_lag_bin_weight,
    unity_lag_weight,
)
from .lave import LAVEResult, LagAwareViterbi
from .lookup import ProbabilisticLookupTable
from .models import ChainHop, ChainResult, Edge, TimedEvent
from .node import CasFlyNode, QueryState
from .orchestrator import CasFlyOrchestrator
from .protocol import CustomProtocol
from .tphg import ConditionalProbabilityTable, TPHG, TPHGBuilder
from .utils import build_tphg_cache, count_transitions, load_events_by_patient
from .validate import ValidationIssue, ValidationReport, validate_all

_LAZY = {"CasFlyDevice", "CustomProtocol"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import device as _dev
        from . import protocol as _proto
        globals()["CasFlyDevice"] = _dev.CasFlyDevice
        globals()["CustomProtocol"] = _proto.CustomProtocol
        return globals()[name]
    raise AttributeError(f"module 'casfly_sdk' has no attribute {name!r}")


__all__ = [
    # Real distributed device (paper implementation — lazily imported)
    "CasFlyDevice",
    "CustomProtocol",
    # In-memory simulation (for local testing / unit tests)
    "CasFlyNode",
    "CasFlyOrchestrator",
    "QueryState",
    # Algorithms
    "LagAwareViterbi",
    "LAVEResult",
    # Graph
    "ConditionalProbabilityTable",
    "TPHG",
    "TPHGBuilder",
    # Routing
    "ProbabilisticLookupTable",
    # Models
    "ChainHop",
    "ChainResult",
    "Edge",
    "TimedEvent",
    # Lag utilities
    "DEFAULT_LAG_BINS",
    "LagBin",
    "exponential_decay_lag_weight",
    "lag_to_bin",
    "map_lag_bin_weight",
    "unity_lag_weight",
    # Data preparation utilities
    "count_transitions",
    "load_events_by_patient",
    "build_tphg_cache",
    # Validation
    "validate_all",
    "ValidationReport",
    "ValidationIssue",
    # Package metadata
    "__version__",
]
