from __future__ import annotations

import csv as _csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimedEvent:
    """Observed event at a timestamp, usually local to one node/device."""

    name: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        name_col: str = "DESCRIPTION",
        timestamp_col: str = "DATE",
        timestamp_fmt: str | None = None,
    ) -> list[TimedEvent]:
        """Load a flat list of TimedEvent from a CSV file.

        Parameters
        ----------
        path:
            Path to the CSV file.
        name_col:
            Column name for the event label. Default: ``"DESCRIPTION"``.
        timestamp_col:
            Column name for the timestamp. Default: ``"DATE"``.
        timestamp_fmt:
            strptime format string (e.g. ``"%Y-%m-%d"``). If omitted,
            ``datetime.fromisoformat()`` is used, which handles ISO-8601
            dates and datetimes (``"2023-01-15"`` or ``"2023-01-15T10:30:00"``).
        """
        events: list[TimedEvent] = []
        with open(Path(path), newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = row[name_col].strip()
                ts_raw = row[timestamp_col].strip()
                ts = (
                    datetime.strptime(ts_raw, timestamp_fmt)
                    if timestamp_fmt
                    else datetime.fromisoformat(ts_raw)
                )
                events.append(cls(name=name, timestamp=ts))
        return events

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        name_key: str = "DESCRIPTION",
        timestamp_key: str = "DATE",
        timestamp_fmt: str | None = None,
    ) -> list[TimedEvent]:
        """Load a flat list of TimedEvent from a JSON array file.

        Parameters
        ----------
        path:
            Path to the JSON file. Must contain a JSON array of objects.
        name_key:
            Key for the event label. Default: ``"DESCRIPTION"``.
        timestamp_key:
            Key for the timestamp string. Default: ``"DATE"``.
        timestamp_fmt:
            Optional strptime format. If omitted, ``datetime.fromisoformat()`` is used.

        Example JSON::

            [
              {"DATE": "2023-01-10", "DESCRIPTION": "Elevated Heart Rate"},
              {"DATE": "2023-01-11", "DESCRIPTION": "Arrhythmia Alert"}
            ]
        """
        with open(Path(path), encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("JSON file must contain an array of event objects.")
        events: list[TimedEvent] = []
        for row in records:
            ts_raw = str(row[timestamp_key]).strip()
            ts = (
                datetime.strptime(ts_raw, timestamp_fmt)
                if timestamp_fmt
                else datetime.fromisoformat(ts_raw)
            )
            events.append(cls(name=str(row[name_key]).strip(), timestamp=ts))
        return events


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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the chain result to a JSON string."""
        data = {
            "chain_confidence": self.chain_confidence,
            "visited_devices": self.visited_devices,
            "visited_edges": list(self.visited_edges),
            "audit": self.confidence_audit(),
            "hops": [
                {
                    "source_device": h.source_device,
                    "target_device": h.target_device,
                    "trigger_event": h.trigger_event,
                    "best_path": list(h.best_path),
                    "path_probability": h.path_probability,
                    "raw_path_probability": h.raw_path_probability,
                    "lag_weight_product": h.lag_weight_product,
                    "cumulative_lag_days": h.cumulative_lag_days,
                }
                for h in self.hops
            ],
        }
        return json.dumps(data, indent=indent)

    def to_csv(self) -> str:
        """Serialise the chain result to CSV (one row per hop)."""
        buf = io.StringIO()
        writer = _csv.writer(buf)
        writer.writerow([
            "hop", "source_device", "target_device", "trigger_event",
            "best_path", "path_probability", "raw_path_probability",
            "lag_weight_product", "cumulative_lag_days",
        ])
        for i, h in enumerate(self.hops, start=1):
            writer.writerow([
                i,
                h.source_device,
                h.target_device or "",
                h.trigger_event,
                " -> ".join(h.best_path),
                f"{h.path_probability:.6f}",
                f"{h.raw_path_probability:.6f}",
                f"{h.lag_weight_product:.6f}",
                f"{h.cumulative_lag_days:.2f}",
            ])
        return buf.getvalue()

    def to_dot(self, title: str = "CasFly Causal Chain") -> str:
        """Serialise the chain result to a Graphviz DOT string.

        Each hop gets its own device-labelled subgraph. Node IDs are
        hop-qualified (``hop0_EventName``) so the same event name can
        appear on different devices without colliding.

        Render with::

            dot -Tpng chain.dot -o chain.png
        """
        def _nid(hop_idx: int, event: str) -> str:
            return f'hop{hop_idx}_{event.replace(chr(34), "")}'

        def _esc(s: str) -> str:
            return s.replace('"', '\\"')

        lines = [
            "digraph CasFlyChain {",
            f'  label="{_esc(title)}  |  confidence={self.chain_confidence:.4f}";',
            "  rankdir=LR;",
            "  node [shape=ellipse, style=filled, fillcolor=lightblue];",
            "",
        ]

        # One subgraph per hop, labelled with its device
        for i, hop in enumerate(self.hops):
            lines.append(f'  subgraph "cluster_hop{i}" {{')
            lines.append(f'    label="{_esc(hop.source_device)}";')
            lines.append('    style=filled; fillcolor=lightyellow;')
            for event in hop.best_path:
                nid = _nid(i, event)
                lines.append(f'    "{nid}" [label="{_esc(event)}"];')
            lines.append("  }")
            lines.append("")

        # Intra-hop edges (causal path within device)
        for i, hop in enumerate(self.hops):
            for j in range(len(hop.best_path) - 1):
                src = _nid(i, hop.best_path[j])
                dst = _nid(i, hop.best_path[j + 1])
                lines.append(
                    f'  "{src}" -> "{dst}" [label="p={hop.path_probability:.3f}"];'
                )

        # Inter-hop edges (dashed arrow between consecutive device hops)
        for i in range(len(self.hops) - 1):
            hop_a, hop_b = self.hops[i], self.hops[i + 1]
            if hop_a.best_path and hop_b.best_path:
                src = _nid(i, hop_a.best_path[-1])
                dst = _nid(i + 1, hop_b.best_path[0])
                lines.append(
                    f'  "{src}" -> "{dst}" '
                    f'[style=dashed, label="→ {_esc(hop_b.source_device)}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def export(self, path: str | Path, fmt: str = "auto") -> None:
        """Write the chain result to a file.

        Parameters
        ----------
        path:
            Output file path.
        fmt:
            ``"json"``, ``"csv"``, ``"dot"``/``"gv"``, or ``"auto"``
            (detected from extension).
        """
        path = Path(path)
        resolved = fmt.lower() if fmt != "auto" else path.suffix.lstrip(".").lower()
        if resolved == "json":
            path.write_text(self.to_json(), encoding="utf-8")
        elif resolved == "csv":
            path.write_text(self.to_csv(), encoding="utf-8")
        elif resolved in ("dot", "gv"):
            path.write_text(self.to_dot(), encoding="utf-8")
        else:
            raise ValueError(
                f"Unknown format '{resolved}'. Use 'json', 'csv', or 'dot'."
            )
