"""Distributed CasFly demo — two real UDP devices on localhost.

This demo mirrors what happens in the paper experiments:

  * Device1 (port 5001) — holds TPHG partitions for Hypertension → Elevated HR
  * Device2 (port 5002) — holds TPHG partitions for Elevated HR → Arrhythmia Alert
  * Device1 is the *initiator*; it expands its local TPHG and forwards the
    growing chain to Device2 via UDP.
  * Device2 expands further and, finding no next device, returns the final
    packet back to Device1.

Prerequisites
-------------
    pip install casfly-sdk[device]      # installs joblib, networkx, pandas, psutil

Run
---
    python distributed_demo.py

The demo builds synthetic .pkl TPHG files, writes a minimal lookup CSV and
filtered_patients CSV, then starts both devices as threads and initiates the
chain.  It waits up to 10 s for the final packet, then prints the chain.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import joblib
import networkx as nx
import pandas as pd

from casfly_sdk import CasFlyDevice

# ---------------------------------------------------------------------------
# Build synthetic data files for the demo
# ---------------------------------------------------------------------------

PATIENT_ID = "demo_patient_001"


def _make_tphg(edges: list[tuple[str, str, float, float]]) -> nx.MultiDiGraph:
    """Build a minimal MultiDiGraph with probability and lag_bin edge attrs."""
    g = nx.MultiDiGraph()
    for cause, effect, prob, lag_bin in edges:
        g.add_edge(cause, effect, probability=prob, lag_bin=lag_bin)
    return g


def setup_demo_data(tmpdir: str) -> tuple[str, str, str]:
    """Write synthetic pkl, lookup CSV, and filtered_patients CSV.

    Returns
    -------
    lookup_csv, filtered_patients_csv, tphg_base_dir
    """
    # TPHG for Device1: Hypertension → Elevated Heart Rate
    tphg1 = _make_tphg([("Hypertension", "Elevated Heart Rate", 0.85, 7.0)])
    d1_dir = os.path.join(tmpdir, "Device1")
    os.makedirs(d1_dir, exist_ok=True)
    joblib.dump(tphg1, os.path.join(d1_dir, f"{PATIENT_ID}_tphg.pkl"))

    # TPHG for Device2: Elevated HR → Arrhythmia Alert
    tphg2 = _make_tphg([("Elevated Heart Rate", "Arrhythmia Alert", 0.90, 3.0)])
    d2_dir = os.path.join(tmpdir, "Device2")
    os.makedirs(d2_dir, exist_ok=True)
    joblib.dump(tphg2, os.path.join(d2_dir, f"{PATIENT_ID}_tphg.pkl"))

    # Lookup CSV — two rows, one per device
    lookup_csv = os.path.join(tmpdir, "All_Device_Lookup_with_Probabilities.csv")
    lookup_df = pd.DataFrame(
        [
            {
                "device": "Device1",
                "ip": "127.0.0.1",
                "port": 5001,
                "cause_type": "['Hypertension']",
                "effect_type": "['Hypertension', 'Elevated Heart Rate']",
                "probability_list": "[0.85]",
            },
            {
                "device": "Device2",
                "ip": "127.0.0.1",
                "port": 5002,
                "cause_type": "['Elevated Heart Rate']",
                "effect_type": "['Elevated Heart Rate', 'Arrhythmia Alert']",
                "probability_list": "[0.90]",
            },
        ]
    )
    lookup_df.to_csv(lookup_csv, index=False)

    # Filtered patients CSV
    fp_csv = os.path.join(tmpdir, "filtered_patients.csv")
    pd.DataFrame(
        [{"PATIENT": PATIENT_ID, "RELEVANT_DEVICES": "Device1,Device2"}]
    ).to_csv(fp_csv, index=False)

    return lookup_csv, fp_csv, tmpdir


# ---------------------------------------------------------------------------
# Run the demo
# ---------------------------------------------------------------------------


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup_csv, fp_csv, tphg_base = setup_demo_data(tmpdir)
        metrics_dir = os.path.join(tmpdir, "logs")

        # --- Device 1 (initiator) ---
        dev1 = CasFlyDevice(
            device_id="Device1",
            lookup_csv=lookup_csv,
            tphg_dir=os.path.join(tphg_base, "Device1"),
            filtered_patients_csv=fp_csv,
            ip="127.0.0.1",
            port=5001,
            metrics_dir=metrics_dir,
        )
        dev1.start()

        # --- Device 2 ---
        dev2 = CasFlyDevice(
            device_id="Device2",
            lookup_csv=lookup_csv,
            tphg_dir=os.path.join(tphg_base, "Device2"),
            filtered_patients_csv=fp_csv,
            ip="127.0.0.1",
            port=5002,
            metrics_dir=metrics_dir,
        )
        dev2.start()

        # Give the listeners a moment to bind
        time.sleep(0.2)

        print(f"\n[demo] Initiating chain from Device1 for patient {PATIENT_ID}")
        threading.Thread(
            target=dev1.initiate_chain,
            args=("Hypertension", PATIENT_ID),
            daemon=True,
        ).start()

        # Wait for the chain to complete (final packet returns to Device1)
        timeout = 10.0
        deadline = time.time() + timeout
        done_flag = os.path.join(tmpdir, "done.flag")

        # Patch Device1 to write a done flag on final packet
        _orig_dispatch = dev1._dispatch

        def _patched_dispatch(packet):
            _orig_dispatch(packet)
            if packet.get("final_packet"):
                with open(done_flag, "w") as f:
                    f.write(str(packet.get("chain", [])))

        dev1._dispatch = _patched_dispatch  # type: ignore[method-assign]

        while time.time() < deadline:
            if os.path.exists(done_flag):
                with open(done_flag) as f:
                    chain_str = f.read()
                print("\n[demo] Chain complete!")
                print(f"  Chain: {chain_str}")
                break
            time.sleep(0.1)
        else:
            print("[demo] Timed out waiting for final packet.")

        dev1._protocol.close()  # type: ignore[union-attr]
        dev2._protocol.close()  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
