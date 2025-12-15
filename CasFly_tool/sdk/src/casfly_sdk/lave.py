from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .lag import unity_lag_weight
from .models import PathState
from .tphg import TPHG


@dataclass(frozen=True)
class LAVEResult:
    best_path: tuple[str, ...]
    probability: float  # weighted probability (includes lag weighting factor)
    raw_probability: float  # product of edge transition probabilities only
    lag_weight_product: float  # product of f(L(tau)) terms
    lag_days: float
    explored_paths: tuple[PathState, ...]


class LagAwareViterbi:
    """Algorithm 2 style lag-aware backward expansion with priority queue."""

    def __init__(
        self,
        max_depth: int = 10,
        fallback_depth: int = 3,
        lag_weight_fn: Callable[[str, float], float] | None = None,
    ):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if fallback_depth < 0:
            raise ValueError("fallback_depth must be >= 0")

        self.max_depth = max_depth
        self.fallback_depth = fallback_depth
        self.lag_weight_fn = lag_weight_fn or unity_lag_weight

    def expand(self, tphg: TPHG, start_event: str) -> LAVEResult:
        queue: list[tuple[float, float, tuple[str, ...], float, float]] = []
        # heap tuple: (-weighted_prob, lag_days, path, raw_prob, lag_weight_product)
        heapq.heappush(queue, (-1.0, 0.0, (start_event,), 1.0, 1.0))

        completed: list[PathState] = []
        terminal_paths: list[tuple[PathState, float, float]] = []
        best_seen: dict[str, float] = {}

        while queue:
            neg_prob, curr_lag, curr_path, curr_raw_prob, curr_lag_weight_prod = heapq.heappop(queue)
            curr_prob = -neg_prob  # weighted probability
            frontier_event = curr_path[0]

            if len(curr_path) > self.max_depth:
                continue

            if best_seen.get(frontier_event, -1.0) >= curr_prob:
                continue

            best_seen[frontier_event] = curr_prob
            completed.append(PathState(events=curr_path, probability=curr_prob, lag_days=curr_lag))

            predecessors = tphg.predecessors(frontier_event)
            if not predecessors and self.fallback_depth > 0 and len(curr_path) == 1:
                fallback_events = self._fallback_events(tphg, frontier_event)
                for fallback_event in fallback_events:
                    if fallback_event in curr_path:
                        continue
                    fallback_path = (fallback_event,) + curr_path
                    heapq.heappush(
                        queue,
                        (-curr_prob, curr_lag, fallback_path, curr_raw_prob, curr_lag_weight_prod),
                    )
                if not fallback_events:
                    terminal_paths.append((
                        PathState(events=curr_path, probability=curr_prob, lag_days=curr_lag),
                        curr_raw_prob,
                        curr_lag_weight_prod,
                    )
                    )
                continue
            if not predecessors:
                terminal_paths.append((
                    PathState(events=curr_path, probability=curr_prob, lag_days=curr_lag),
                    curr_raw_prob,
                    curr_lag_weight_prod,
                ))
                continue

            for edge in predecessors:
                if edge.cause in curr_path:
                    continue

                lag_weight = self.lag_weight_fn(edge.lag_bin, edge.lag_days)
                if not (0.0 <= lag_weight <= 1.0):
                    raise ValueError(f"lag_weight_fn must return in [0,1], got {lag_weight}")

                new_raw_prob = curr_raw_prob * edge.probability
                new_lag_weight_prod = curr_lag_weight_prod * lag_weight
                new_prob = curr_prob * edge.probability * lag_weight
                new_lag = curr_lag + edge.lag_days
                new_path = (edge.cause,) + curr_path
                heapq.heappush(queue, (-new_prob, new_lag, new_path, new_raw_prob, new_lag_weight_prod))

        if not completed:
            # Defensive fallback.
            terminal = (start_event,)
            return LAVEResult(terminal, 1.0, 1.0, 1.0, 0.0, (PathState(terminal, 1.0, 0.0),))

        ranked = terminal_paths
        if not ranked:
            ranked = [(path_state, 1.0, 1.0) for path_state in completed]
        ranked.sort(
            key=lambda item: (
                len(item[0].events) > 1,
                item[0].probability,
                -item[0].lag_days,
                len(item[0].events),
            ),
            reverse=True,
        )
        best_state, best_raw_prob, best_lag_weight_prod = ranked[0]
        return LAVEResult(
            best_state.events,
            best_state.probability,
            best_raw_prob,
            best_lag_weight_prod,
            best_state.lag_days,
            tuple(completed),
        )

    def _fallback_events(self, tphg: TPHG, from_event: str) -> list[str]:
        # Bounded neighborhood BFS over predecessor/successor links to keep continuity
        # when direct predecessor expansion is exhausted.
        discovered: list[str] = []
        seen = {from_event}
        q = deque([(from_event, 0)])

        while q:
            event, depth = q.popleft()
            if depth >= self.fallback_depth:
                continue

            neighbors = [e.effect for e in tphg.successors(event)] + [e.cause for e in tphg.predecessors(event)]
            for next_event in neighbors:
                if next_event in seen or next_event == event:
                    continue
                seen.add(next_event)
                discovered.append(next_event)
                q.append((next_event, depth + 1))

        return discovered
