from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .lag import lag_to_bin
from .models import Edge, TimedEvent


class TPHG:
    """Temporal Probabilistic Health Graph: compact lag-indexed transition graph."""

    def __init__(self, edges: list[Edge]):
        self._edges = edges
        self._incoming: dict[str, list[Edge]] = defaultdict(list)
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._vertices: set[str] = set()

        for edge in edges:
            self._incoming[edge.effect].append(edge)
            self._outgoing[edge.cause].append(edge)
            self._vertices.add(edge.cause)
            self._vertices.add(edge.effect)

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    @property
    def vertices(self) -> set[str]:
        return set(self._vertices)

    def predecessors(self, event: str) -> list[Edge]:
        return list(self._incoming.get(event, []))

    def successors(self, event: str) -> list[Edge]:
        return list(self._outgoing.get(event, []))

    def has_event(self, event: str) -> bool:
        return event in self._vertices


class ConditionalProbabilityTable:
    """Bin-indexed conditional probabilities C[cause, effect, lag_bin] -> P(effect|cause, lag_bin)."""

    def __init__(self, probabilities: dict[tuple[str, str, str], float]):
        self._probabilities = probabilities

    @property
    def known_transitions(self) -> set[tuple[str, str, str]]:
        """All ``(cause, effect, lag_bin)`` tuples with a fitted probability."""
        return set(self._probabilities.keys())

    def get(self, cause: str, effect: str, lag_bin: str) -> float | None:
        return self._probabilities.get((cause, effect, lag_bin))

    @classmethod
    def from_counts(
        cls,
        counts: dict[tuple[str, str, str], int],
        alpha: float = 1.0,
    ) -> "ConditionalProbabilityTable":
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        by_cause_bin: dict[tuple[str, str], set[str]] = defaultdict(set)
        totals: dict[tuple[str, str], int] = defaultdict(int)

        for (cause, effect, lag_bin), count in counts.items():
            if count < 0:
                raise ValueError("counts must be non-negative")
            by_cause_bin[(cause, lag_bin)].add(effect)
            totals[(cause, lag_bin)] += count

        probabilities: dict[tuple[str, str, str], float] = {}
        for (cause, effect, lag_bin), count in counts.items():
            succ_count = len(by_cause_bin[(cause, lag_bin)])
            denominator = totals[(cause, lag_bin)] + alpha * succ_count
            probabilities[(cause, effect, lag_bin)] = (count + alpha) / denominator

        return cls(probabilities)


class TPHGBuilder:
    """Algorithm 1 style graph construction over local event logs."""

    def __init__(self, cpt: ConditionalProbabilityTable):
        self._cpt = cpt

    def build(self, events: list[TimedEvent]) -> TPHG:
        if not events:
            return TPHG(edges=[])

        ordered = sorted(events, key=lambda e: e.timestamp)
        edges: list[Edge] = []

        for i in range(len(ordered)):
            ea = ordered[i]
            for j in range(i + 1, len(ordered)):
                eb = ordered[j]
                lag_days = _days_between(ea.timestamp, eb.timestamp)
                lag_bin = lag_to_bin(lag_days)
                probability = self._cpt.get(ea.name, eb.name, lag_bin)

                if probability is None:
                    continue

                edges.append(
                    Edge(
                        cause=ea.name,
                        effect=eb.name,
                        probability=probability,
                        lag_days=lag_days,
                        lag_bin=lag_bin,
                    )
                )

        return TPHG(edges=edges)


def _days_between(t1: datetime, t2: datetime) -> float:
    delta = t2 - t1
    return delta.total_seconds() / 86400.0
