from __future__ import annotations

from collections import defaultdict


class ProbabilisticLookupTable:
    """PLT for routing event queries to the most relevant next device."""

    def __init__(self, records: list[tuple[str, str, float]]):
        self._routes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for event_name, device_id, probability in records:
            if not (0.0 <= probability <= 1.0):
                raise ValueError("probability must be between 0 and 1")
            self._routes[event_name].append((device_id, probability))

        for event_name in self._routes:
            self._routes[event_name].sort(key=lambda x: x[1], reverse=True)

    def next_device(self, event_name: str, exclude: set[str] | None = None) -> str | None:
        excluded = exclude or set()
        for device_id, _ in self._routes.get(event_name, []):
            if device_id not in excluded:
                return device_id
        return None
