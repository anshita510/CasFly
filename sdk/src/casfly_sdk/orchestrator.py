from __future__ import annotations

from .models import ChainResult
from .node import CasFlyNode, QueryState


class CasFlyOrchestrator:
    """In-memory coordinator for decentralized multi-device chain expansion."""

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
