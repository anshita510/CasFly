from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TimedEvent:
    """Observed event at a timestamp, usually local to one node/device."""

    name: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    cause: str
    effect: str
    probability: float
    lag_days: float
    lag_bin: str


@dataclass
class PathState:
    events: tuple[str, ...]
    probability: float
    lag_days: float


@dataclass
class ChainHop:
    source_device: str
    target_device: str | None
    trigger_event: str
    best_path: tuple[str, ...]
    path_probability: float  # weighted hop confidence
    cumulative_lag_days: float
    raw_path_probability: float = 1.0
    lag_weight_product: float = 1.0


@dataclass
class ChainResult:
    hops: list[ChainHop]
    visited_devices: list[str]
    visited_edges: set[tuple[str, str]]

    @property
    def chain_confidence(self) -> float:
        confidence = 1.0
        for hop in self.hops:
            confidence *= hop.path_probability
        return confidence

    @property
    def chain_raw_confidence(self) -> float:
        confidence = 1.0
        for hop in self.hops:
            confidence *= hop.raw_path_probability
        return confidence

    @property
    def chain_lag_weight_product(self) -> float:
        weight = 1.0
        for hop in self.hops:
            weight *= hop.lag_weight_product
        return weight

    def confidence_audit(self) -> dict[str, float]:
        # C = product_hops(raw_prob_hop * lag_weight_hop)
        theorem_lower_bound = self.chain_raw_confidence * self.chain_lag_weight_product
        return {
            "weighted_confidence": self.chain_confidence,
            "raw_confidence": self.chain_raw_confidence,
            "lag_weight_product": self.chain_lag_weight_product,
            "theorem_lower_bound": theorem_lower_bound,
        }
