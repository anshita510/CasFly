"""CasFlyOrchestrator — in-memory simulation coordinator (for local testing only).

.. note::
   This is a **simulation-only** orchestrator that drives multiple
   :class:`CasFlyNode` instances in-process with no UDP communication.
   It is useful for unit tests and quick algorithm validation.

   For the actual distributed deployment described in the paper — where each
   device runs as an independent OS process and communicates over real UDP
   sockets — use :class:`~casfly_sdk.device.CasFlyDevice` and launch one
   process per device (matching Device*.py in the paper).
"""
from __future__ import annotations

from .models import ChainResult
from .node import CasFlyNode, QueryState


class CasFlyOrchestrator:
    """In-memory coordinator for simulating multi-device chain expansion.

    Drives a set of registered :class:`CasFlyNode` instances without any
    network I/O.  For real distributed deployments, use :class:`CasFlyDevice`.
    """

    def __init__(self, max_hops: int = 16):
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self.max_hops = max_hops
        self._nodes: dict[str, CasFlyNode] = {}

    def register(self, node: CasFlyNode) -> None:
        self._nodes[node.device_id] = node

    def trace(self, start_device: str, start_event: str) -> ChainResult:
        if start_device not in self._nodes:
            raise KeyError(f"unknown start_device: {start_device}")

        query = QueryState(current_event=start_event)
        hops = []
        current = start_device

        for _ in range(self.max_hops):
            node = self._nodes[current]
            hop, next_device, query = node.handle(query)
            hops.append(hop)

            if next_device is None:
                break
            if next_device not in self._nodes:
                break

            current = next_device

        return ChainResult(
            hops=hops,
            visited_devices=sorted(query.visited_devices),
            visited_edges=query.visited_edges,
        )
