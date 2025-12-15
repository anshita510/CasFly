from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable


@dataclass(frozen=True)
class LagBin:
    label: str
    lower_days: float
    upper_days: float | None = None

    def contains(self, lag_days: float) -> bool:
        if self.upper_days is None:
            return lag_days >= self.lower_days
        return self.lower_days <= lag_days < self.upper_days


DEFAULT_LAG_BINS = (
    LagBin("0-7 days", 0, 7),
    LagBin("7-14 days", 7, 14),
    LagBin("14-30 days", 14, 30),
    LagBin("30-90 days", 30, 90),
    LagBin("90-180 days", 90, 180),
    LagBin("180-365 days", 180, 365),
    LagBin("365-730 days", 365, 730),
    LagBin("730-Inf days", 730, None),
)


def lag_to_bin(lag_days: float, bins: tuple[LagBin, ...] = DEFAULT_LAG_BINS) -> str:
    if lag_days < 0:
        raise ValueError(f"lag_days must be non-negative, got {lag_days}")

    for lag_bin in bins:
        if lag_bin.contains(lag_days):
            return lag_bin.label

    # Defensive fallback, should never happen with open-ended final bin.
    return bins[-1].label


def unity_lag_weight(_: str, __: float) -> float:
    """Paper runtime setting f(L(tau)) = 1 to avoid double-weighting."""
    return 1.0


def exponential_decay_lag_weight(half_life_days: float) -> Callable[[str, float], float]:
    """Build f(L) = exp(-ln(2) * lag_days / half_life_days)."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")

    decay = math.log(2.0) / half_life_days

    def _weight(_: str, lag_days: float) -> float:
        if lag_days < 0:
            raise ValueError("lag_days must be non-negative")
        return math.exp(-decay * lag_days)

    return _weight


def map_lag_bin_weight(weights: dict[str, float]) -> Callable[[str, float], float]:
    """Build piecewise f(L) from lag-bin weights in (0,1]."""
    for lag_bin, weight in weights.items():
        if not (0.0 < weight <= 1.0):
            raise ValueError(f"invalid weight for {lag_bin}: {weight}")

    def _weight(lag_bin: str, _: float) -> float:
        return weights.get(lag_bin, 1.0)

    return _weight
