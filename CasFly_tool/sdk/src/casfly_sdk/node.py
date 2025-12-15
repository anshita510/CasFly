"""CasFlyNode — in-memory simulation node (for local testing / unit tests only).

.. note::
   This module implements an **in-memory** version of the CasFly protocol used
   for unit tests and quick local demos.  For the actual distributed, UDP-based
   implementation that matches the paper experiments, use :class:`CasFlyDevice`
   from ``casfly_sdk.device``.

   In the paper each Device*.py script runs as a standalone process communicating
   over UDP sockets.  ``CasFlyNode`` / ``CasFlyOrchestrator`` simulate that
   multi-device interaction in-process without any network I/O, which is useful
   for rapid algorithm validation but does not reproduce the real distributed
   behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lave import LagAwareViterbi
from .lookup import ProbabilisticLookupTable
from .models import ChainHop
from .tphg import TPHG


@dataclass
class QueryState:
    current_event: str
    visited_devices: set[str] = field(default_factory=set)
    visited_edges: set[tuple[str, str]] = field(default_factory=set)


class CasFlyNode:
    """One CasFly participant: local TPHG + LaVE + PLT routing."""

    def __init__(
        self,
        device_id: str,
        tphg: TPHG,
        lookup_table: ProbabilisticLookupTable,
        lave: LagAwareViterbi | None = None,
    ):
        self.device_id = device_id
        self.tphg = tphg
        self.lookup_table = lookup_table
        self.lave = lave or LagAwareViterbi()

    def handle(self, query: QueryState) -> tuple[ChainHop, str | None, QueryState]:
        result = self.lave.expand(self.tphg, query.current_event)

        best_path = result.best_path
        new_edges = _path_to_edges(best_path)
        query.visited_edges.update(new_edges)
        query.visited_devices.add(self.device_id)

        next_event = best_path[0] if len(best_path) > 1 else query.current_event
        next_device = self.lookup_table.next_device(next_event, exclude=query.visited_devices)

        hop = ChainHop(
            source_device=self.device_id,
            target_device=next_device,
            trigger_event=query.current_event,
            best_path=best_path,
            path_probability=result.probability,
            raw_path_probability=result.raw_probability,
            lag_weight_product=result.lag_weight_product,
            cumulative_lag_days=result.lag_days,
        )
        query.current_event = next_event

        return hop, next_device, query


def _path_to_edges(path: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}
