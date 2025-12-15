"""CasFlyDevice — reusable class mirroring each Device*.py script in the paper.

Each physical IoT device (Raspberry Pi, etc.) instantiates *one* CasFlyDevice.
The class faithfully implements the distributed protocol from the paper:

  1. Binds a UDP port via CustomProtocol.
  2. Loads the per-patient TPHG partition from a ``.pkl`` file
     (joblib-serialised ``networkx.MultiDiGraph``).
  3. Runs backward Viterbi expansion (LaVE — Algorithm 2).
  4. Routes the growing causal chain to the next device using
     ``All_Device_Lookup_with_Probabilities.csv``.
  5. Logs per-device metrics: t1 (TPHG load), t2 (Viterbi),
     t_fallback, t_dash, memory/CPU/RAM/energy.

Packet structure (matches Device*.py in the paper)::

    {
        "effects":      str | list[str],   # current trigger event(s)
        "patient_id":   str,
        "chain":        list[list],        # [[pred, effect, prob, lag, device_id], ...]
        "initiator":    str,               # device that started the trace
        "final_packet": bool,              # True ⟹ chain complete, log and stop
    }

Quick start::

    from casfly_sdk import CasFlyDevice

    device = CasFlyDevice(
        device_id="Device20",
        lookup_csv="data/All_Device_Lookup_with_Probabilities.csv",
        tphg_dir="data/device_partitions_patientwise1_rasp/Device20",
        filtered_patients_csv="data/filtered_patients.csv",
        ip="127.0.0.1",
        port=5020,
        metrics_dir="logs/Device20",
    )
    device.start()  # binds UDP, starts background listener

    # On the initiator device only:
    device.initiate_chain(event="Stroke", patient_id="abc-123")
"""
from __future__ import annotations

import ast
import csv
import logging
import os
import socket
import subprocess
import threading
import time
from contextlib import closing
from datetime import datetime
from heapq import heappop, heappush
from typing import Any

try:
    import joblib
    import networkx as nx
    import pandas as pd
    import psutil
    _DEVICE_DEPS_OK = True
    _DEVICE_DEPS_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    _DEVICE_DEPS_OK = False
    _DEVICE_DEPS_ERROR = exc
    joblib = nx = pd = psutil = None  # type: ignore[assignment]

from .lag import DEFAULT_LAG_BINS
from .protocol import CustomProtocol


def _lag_bin_to_days(label: str) -> float:
    """Convert a lag-bin label (e.g. '0-7 days') to its lower-bound in days for heap ordering."""
    for lb in DEFAULT_LAG_BINS:
        if lb.label == label:
            return lb.lower_days
    return float("inf")

# Type aliases
Packet = dict[str, Any]
Chain = list[list]  # each entry: [pred, effect, prob, lag, device_id]


class CasFlyDevice:
    """One CasFly IoT device participant.

    Parameters
    ----------
    device_id:
        Name matching the ``device`` column in the lookup CSV (e.g. ``"Device20"``).
    lookup_csv:
        Path to ``All_Device_Lookup_with_Probabilities.csv``.
    tphg_dir:
        Directory containing per-patient ``.pkl`` files
        (``<patient_id>_tphg.pkl``), each a ``networkx.MultiDiGraph``.
    filtered_patients_csv:
        Path to ``filtered_patients.csv`` with ``PATIENT`` and
        ``RELEVANT_DEVICES`` columns.
    ip:
        IP address to bind.  Defaults to ``"127.0.0.1"``.
    port:
        UDP port to bind.  ``0`` → read from lookup CSV.
    metrics_dir:
        Directory where per-device metric CSV logs are written.
    max_depth:
        Maximum path depth for backward Viterbi expansion.
    """

    def __init__(
        self,
        device_id: str,
        lookup_csv: str,
        tphg_dir: str,
        filtered_patients_csv: str,
        ip: str = "127.0.0.1",
        port: int = 0,
        metrics_dir: str = "./logs",
        max_depth: int = 10,
    ) -> None:
        if not _DEVICE_DEPS_OK:
            raise ImportError(
                "CasFlyDevice requires joblib, networkx, pandas, and psutil.\n"
                "Install them with:  pip install casfly-sdk[device]"
            ) from _DEVICE_DEPS_ERROR
        self.device_id = device_id
        self.tphg_dir = tphg_dir
        self.max_depth = max_depth
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        # Load routing lookup table
        self.lookup_table = pd.read_csv(lookup_csv)
        if "ip" not in self.lookup_table.columns:
            self.lookup_table["ip"] = "127.0.0.1"
            self.lookup_table.to_csv(lookup_csv, index=False)
        self.lookup_table["device"] = self.lookup_table["device"].astype(str).str.strip()
        self.lookup_csv_path = lookup_csv

        # Load patient → relevant devices map
        self.filtered_patients = pd.read_csv(filtered_patients_csv)
        self.filtered_patients["PATIENT"] = self.filtered_patients["PATIENT"].astype(str)

        # Resolve IP and port from lookup CSV
        row = self.lookup_table[self.lookup_table["device"] == device_id]
        if not row.empty:
            self.ip = ip if ip != "127.0.0.1" else str(row["ip"].iloc[0])
            self.port = port if port != 0 else int(row["port"].iloc[0])
        else:
            self.ip = ip
            self.port = port if port != 0 else self._find_free_port()

        # Runtime state (set at chain start / packet receipt)
        self._protocol: CustomProtocol | None = None
        self._initiator: str | None = None
        self._experiment_id: str | None = None
        self._disease_name: str | None = None
        self._start_time: float = 0.0
        self._startdevicetimer: float = 0.0
        self._t1 = self._t2 = self._t_fallback = self._t_dash = 0.0
        self._visited_devices: set[str] = set()
        self._chain: Chain = []

        self._log = logging.getLogger(f"casfly.device.{device_id}.{self.port}")
        if not self._log.handlers:
            _handler = logging.FileHandler(
                os.path.join(metrics_dir, f"{device_id}_{self.port}.log")
            )
            _handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self._log.addHandler(_handler)
            self._log.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the UDP port and start the packet-receive loop in a background thread."""
        self._protocol = CustomProtocol(self.ip, self.port, log_dir=self.metrics_dir)
        self._start_time = time.time()
        t = threading.Thread(target=self._handle_request, daemon=False)
        t.start()
        self._log.info("[%s] Listening on %s:%s", self.device_id, self.ip, self.port)

    def initiate_chain(self, event: str, patient_id: str) -> None:
        """Kick off causal chain tracing from this device (initiator role).

        Typically called only on Device20 (or whichever device starts the trace).
        Requires :meth:`start` to have been called first.
        """
        if self._protocol is None:
            raise RuntimeError("Call start() before initiate_chain()")
        self._initiator = self.device_id
        self._disease_name = event
        chain: Chain = []
        self._expand_causal_chain(event, chain, set(), patient_id)
        if chain:
            self._forward_chain(chain, patient_id, self._initiator)

    # ------------------------------------------------------------------
    # TPHG loading
    # ------------------------------------------------------------------

    def load_tphg(self, patient_id: str) -> "nx.MultiDiGraph | None":
        """Load the per-patient TPHG from its ``.pkl`` cache file.

        Returns ``None`` if the file is absent.
        """
        cache_file = os.path.join(self.tphg_dir, f"{patient_id}_tphg.pkl")
        if not os.path.exists(cache_file):
            self._log.warning("[%s] No TPHG found for patient %s", self.device_id, patient_id)
            return None
        loaded = joblib.load(cache_file)
        if isinstance(loaded, (nx.Graph, nx.MultiDiGraph)):
            return loaded
        raise ValueError(f"Unexpected TPHG format in {cache_file}")

    # ------------------------------------------------------------------
    # Backward Viterbi  (LaVE — Algorithm 2)
    # ------------------------------------------------------------------

    def viterbi_expand_backward(
        self, event: str, tphg: "nx.MultiDiGraph"
    ) -> list[tuple]:
        """Lag-aware backward Viterbi expansion (Algorithm 2 from the paper).

        Returns causal path tuples ranked by probability (highest first).
        Each path is a tuple of event names from oldest cause → trigger event.
        """
        pq: list = []
        completed_paths: dict[str, tuple] = {}
        prob_cache: dict[str, float] = {}

        if tphg.has_node(event):
            heappush(pq, (0, 1.0, (event,)))
        else:
            t_fb = time.time()
            fallback = self._fallback_path_expansion(tphg)
            self._t_fallback = time.time() - t_fb
            for path, lag in fallback:
                heappush(pq, (lag, 1.0, tuple(path)))

        while pq:
            current_lag, current_prob, current_path = heappop(pq)
            last_node = current_path[0]

            if len(current_path) > self.max_depth:
                continue
            if prob_cache.get(last_node, -1.0) >= current_prob:
                continue

            completed_paths[last_node] = current_path
            prob_cache[last_node] = current_prob

            if not tphg.has_node(last_node):
                continue

            for pred in tphg.predecessors(last_node):
                if tphg.has_edge(pred, last_node):
                    data = tphg.edges[pred, last_node, 0]
                    lag_days = _lag_bin_to_days(data.get("lag_bin", ""))
                    edge_prob = data.get("probability", 1.0)
                    new_path = (pred,) + current_path
                    heappush(
                        pq,
                        (current_lag + lag_days, current_prob * edge_prob, new_path),
                    )

        if not completed_paths:
            return [(event,)]
        return sorted(
            completed_paths.values(),
            key=lambda x: prob_cache.get(x[0], 0),
            reverse=True,
        )

    @staticmethod
    def _fallback_path_expansion(
        tphg: "nx.MultiDiGraph", max_depth: int = 10
    ) -> list[tuple[list, float]]:
        """Shortest-path fallback when the trigger event is unknown to this TPHG."""
        paths = []
        for node in tphg.nodes:
            for target, path in nx.single_source_shortest_path(
                tphg, node, cutoff=max_depth
            ).items():
                if len(path) > 1:
                    lag = sum(
                        _lag_bin_to_days(tphg[path[i]][path[i + 1]][0].get("lag_bin", ""))
                        for i in range(len(path) - 1)
                    )
                    paths.append((path, lag))
            if not paths:
                paths.append(([node], 0))
        return paths

    # ------------------------------------------------------------------
    # Chain expansion
    # ------------------------------------------------------------------

    def _expand_causal_chain(
        self, event: str, chain: Chain, visited: set, patient_id: str
    ) -> list:
        t0 = time.time()
        tphg = self.load_tphg(patient_id)
        self._t1 = time.time() - t0

        if not tphg:
            return []

        viterbi_start = time.time()
        viterbi_path = self.viterbi_expand_backward(event, tphg)
        self._t2 = time.time() - viterbi_start

        local_paths = []
        edge_set: set[tuple[str, str]] = set()

        for path_data in viterbi_path:
            if isinstance(path_data, tuple) and len(path_data) > 1:
                for i in range(len(path_data) - 1):
                    pred = path_data[i]
                    effect = path_data[i + 1]
                    if pred == effect or not tphg.has_edge(pred, effect):
                        continue
                    edge = (pred, effect)
                    if edge not in edge_set:
                        edge_set.add(edge)
                        for _, data in tphg[pred][effect].items():
                            prob = data.get("probability", 1.0)
                            lag = data.get("lag_bin", "")
                            chain.append([pred, effect, prob, lag, self.device_id])
                            local_paths.append((pred, effect, prob, lag, self.device_id))
                            visited.add(edge)
            elif isinstance(path_data, tuple) and len(path_data) == 1:
                node = path_data[0]
                chain.append([node, node, 1.0, 0, self.device_id])

        return local_paths

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _get_relevant_devices(self, patient_id: str) -> list[str]:
        rows = self.filtered_patients[
            self.filtered_patients["PATIENT"] == str(patient_id)
        ]
        if rows.empty:
            return []
        raw = rows["RELEVANT_DEVICES"].iloc[0]
        return [d.strip() for d in str(raw).split(",")]

    def _find_next_device(
        self, effect: str, patient_id: str, visited_devices: set[str]
    ) -> dict[str, dict]:
        """Return a ``{device_id: {port, ip, effects, probability}}`` mapping.

        Implements the two-stage routing from the paper:
        1. Direct match: effect in the device's ``effect_type`` column.
        2. Fallback:     overlap between this device's ``cause_type`` and the
                         candidate's ``effect_type``.
        """
        escaped = effect.strip().lower()
        device_effect_map: dict[str, dict] = {}

        rel = self._get_relevant_devices(patient_id)
        filtered = (
            self.lookup_table[self.lookup_table["device"].isin(rel)]
            if rel
            else self.lookup_table
        )

        best_device = None
        max_prob = -1.0
        fallback_candidates = []
        effect_found = False

        for _, row in filtered.iterrows():
            try:
                effect_list = ast.literal_eval(row["effect_type"])
            except (ValueError, SyntaxError):
                effect_list = str(row["effect_type"]).strip("[]'").split(", ")
            effect_list = [str(e).strip().lower() for e in effect_list if str(e).strip()]

            next_device = str(row["device"]).strip()
            next_port = int(row["port"])
            next_ip = str(row["ip"])

            if next_device in visited_devices or next_device == self.device_id:
                continue

            if escaped in effect_list:
                effect_found = True
                try:
                    prob_list = ast.literal_eval(row.get("probability_list", "[]"))
                    prob_list = [float(p) for p in prob_list]
                except Exception:
                    prob_list = []
                local_best = max(prob_list) if prob_list else 0.0
                if local_best > max_prob:
                    max_prob = local_best
                    best_device = (next_device, next_port, next_ip, local_best)

        if best_device:
            nd, np_, nip, prob = best_device
            device_effect_map[nd] = {
                "port": np_,
                "ip": nip,
                "effects": {effect},
                "probability": prob,
            }
            return device_effect_map

        if not effect_found:
            my_row = self.lookup_table[self.lookup_table["device"] == self.device_id]
            if not my_row.empty:
                try:
                    cause_list = ast.literal_eval(my_row["cause_type"].iloc[0])
                except Exception:
                    cause_list = (
                        str(my_row["cause_type"].iloc[0]).strip("[]'").split(", ")
                    )
                cause_list = [c.strip().lower() for c in cause_list if str(c).strip()]

                for _, row in filtered.iterrows():
                    try:
                        eff_list = ast.literal_eval(row["effect_type"])
                    except Exception:
                        eff_list = str(row["effect_type"]).strip("[]'").split(", ")
                    eff_list = [str(e).strip().lower() for e in eff_list if str(e).strip()]

                    nd = str(row["device"]).strip()
                    if nd in visited_devices or nd == self.device_id:
                        continue
                    if set(cause_list).intersection(eff_list):
                        try:
                            pl = ast.literal_eval(row.get("probability_list", "[]"))
                            hp = max([float(p) for p in pl]) if pl else 0.0
                        except Exception:
                            hp = 0.0
                        fallback_candidates.append(
                            (nd, int(row["port"]), str(row["ip"]), hp)
                        )

        if fallback_candidates:
            nd, np_, nip, hp = max(fallback_candidates, key=lambda x: x[3])
            device_effect_map[nd] = {
                "port": np_,
                "ip": nip,
                "effects": {effect},
                "probability": hp,
            }

        return device_effect_map

    # ------------------------------------------------------------------
    # Chain forwarding
    # ------------------------------------------------------------------

    def _forward_chain(
        self, chain: Chain, patient_id: str, sender_device: str | None = None
    ) -> None:
        visited = {e for _, _, _, _, e in chain} if chain else set()
        self._visited_devices = visited

        if not chain:
            return

        last_effect = chain[-1][1]
        device_effect_map = self._find_next_device(last_effect, patient_id, visited)

        if not device_effect_map:
            self._return_to_initiator(
                chain, sender_device or self.device_id, patient_id
            )
            return

        for next_device, info in device_effect_map.items():
            self._t_dash = time.time() - self._startdevicetimer
            self._log_metrics(
                patient_id=patient_id,
                disease_name=self._disease_name or "",
                chain=chain,
                final_device=None,
            )
            effects = list(info["effects"])
            assert self._protocol is not None
            self._protocol.send_packet(
                {
                    "effects": effects[0] if len(effects) == 1 else effects,
                    "patient_id": patient_id,
                    "chain": chain,
                    "initiator": self._initiator,
                    "final_packet": False,
                },
                info["ip"],
                info["port"],
            )

    def _return_to_initiator(
        self, chain: Chain, initiator: str, patient_id: str
    ) -> None:
        row = self.lookup_table[self.lookup_table["device"] == initiator]
        if not row.empty:
            tgt_ip = str(row["ip"].iloc[0])
            tgt_port = int(row["port"].iloc[0])
        else:
            tgt_ip, tgt_port = self.ip, self.port

        final_effects = list({effect for _, effect, _, _, _ in chain})
        assert self._protocol is not None
        self._protocol.send_packet(
            {
                "effects": final_effects,
                "patient_id": patient_id,
                "chain": chain,
                "initiator": initiator,
                "final_packet": True,
            },
            tgt_ip,
            tgt_port,
        )

    # ------------------------------------------------------------------
    # Packet receive loop
    # ------------------------------------------------------------------

    def _handle_request(self) -> None:
        while True:
            try:
                assert self._protocol is not None
                packet, addr = self._protocol.receive_packet()
                if packet:
                    self._dispatch(packet)
            except Exception as exc:
                self._log.error("[%s] Error handling packet: %s", self.device_id, exc)

    def _dispatch(self, packet: Packet) -> None:
        event = packet.get("effects")
        patient_id = packet.get("patient_id")
        self._initiator = packet.get("initiator")
        chain: Chain = packet.get("chain", [])
        final_packet: bool = packet.get("final_packet", False)

        self._experiment_id = (
            f"Exp_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        if final_packet:
            total_time = time.time() - self._start_time
            self._log_metrics(
                patient_id=patient_id,
                disease_name=self._disease_name or str(event),
                chain=chain,
                final_device=self.device_id,
                total_time=total_time,
            )
            return

        if event:
            self._startdevicetimer = time.time()
            tphg = self.load_tphg(patient_id)
            if tphg:
                ev = event if isinstance(event, str) else str(event[0])
                self._expand_causal_chain(ev, chain, set(), patient_id)
                if chain:
                    self._forward_chain(chain, patient_id, self._initiator)
            else:
                self._return_to_initiator(
                    chain, self._initiator or self.device_id, patient_id
                )

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        patient_id: str,
        disease_name: str,
        chain: Chain,
        final_device: str | None,
        total_time: float = 0.0,
    ) -> None:
        csv_path = os.path.join(
            self.metrics_dir,
            f"{self.device_id}_{disease_name}_{patient_id}_metrics.csv",
        )
        write_header = not os.path.exists(csv_path)
        process = self._find_process_by_port(self.port)
        mem = cpu = ram = energy = 0.0
        if process:
            mem = process.memory_info().rss / 1024**2
            cpu = process.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().used / 1024**2
            energy = cpu * 0.1

        log_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5],
            disease_name,
            patient_id,
            self._initiator,
            final_device,
            self._experiment_id,
            len(self._visited_devices),
            str(chain),
            total_time,
            self._t1,
            self._t2,
            self._t_fallback,
            self._t_dash,
            mem,
            cpu,
            ram,
            energy,
            self.device_id,
        ]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Timestamp",
                        "Disease Name",
                        "Patient ID",
                        "Initiator Device",
                        "Final Device",
                        "Experiment ID",
                        "No. of Devices Accessed",
                        "Chain Data",
                        "Total Time (T)",
                        "Cached TPHG Load Time (t1)",
                        "Backward Viterbi Time (t2)",
                        "Fallback Path Time (t_fallback)",
                        "Time per Device (t_dash)",
                        "Memory Usage (MB)",
                        "CPU Usage (%)",
                        "RAM Usage (MB)",
                        "Energy Consumption (J)",
                        "current_device",
                    ]
                )
            writer.writerow(log_row)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_process_by_port(port: int) -> "psutil.Process | None":
        try:
            result = subprocess.run(
                ["lsof", "-nP", "-i", f"UDP:{port}"],
                stdout=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.splitlines()
                if len(lines) > 1:
                    pid = int(lines[1].split()[1])
                    return psutil.Process(pid)
        except Exception:
            pass
        return None

    @staticmethod
    def _find_free_port() -> int:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
