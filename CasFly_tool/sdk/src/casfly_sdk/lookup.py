from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


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

    @property
    def known_events(self) -> set[str]:
        """All event names that have at least one routing entry."""
        return set(self._routes.keys())

    def next_device(self, event_name: str, exclude: set[str] | None = None) -> str | None:
        excluded = exclude or set()
        for device_id, _ in self._routes.get(event_name, []):
            if device_id not in excluded:
                return device_id
        return None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        event_col: str = "event",
        device_col: str = "device",
        probability_col: str = "probability",
    ) -> ProbabilisticLookupTable:
        """Load a PLT from a CSV file.

        The CSV must have one row per (event, device) routing rule.

        Parameters
        ----------
        path:
            Path to the CSV file.
        event_col:
            Column name for the event label. Default: ``"event"``.
        device_col:
            Column name for the target device. Default: ``"device"``.
        probability_col:
            Column name for the routing probability. Default: ``"probability"``.

        Example CSV::

            event,device,probability
            Elevated Heart Rate,smartwatch,0.9
            Hypertension,bp_cuff,0.85
        """
        records: list[tuple[str, str, float]] = []
        with open(Path(path), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.append((
                    row[event_col].strip(),
                    row[device_col].strip(),
                    float(row[probability_col]),
                ))
        return cls(records)
