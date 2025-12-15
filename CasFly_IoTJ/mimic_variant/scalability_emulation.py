#!/usr/bin/env python3
"""
scalability_emulation.py — MIMIC Scalability Emulation for CasFly

Emulates CasFly chain formation across varying device federation sizes and
network latency conditions using the MIMIC-IV patient cohort.

Pipeline:
  For each (latency_ms, device_count, trial):
    1. Sample 1 patient from filtered_patients_updated.csv.
    2. Randomly pick `device_count` devices from the lookup table.
    3. Write a temporary patient CSV restricted to those devices.
    4. Run _one_initiate_wrapper.sh with SLEEP_EXTRA=latency_ms to emulate load.
    5. Read the resulting Total Time from the metrics logs.

Inputs (via config.yaml):
  - output_dir              : where scalability_results.csv is written
  - logs_root               : directory with per-device *_metrics_log.csv files
  - lookup_csv              : All_Device_Lookup_with_Probabilities.csv
  - filtered_patients_csv   : filtered_patients_updated.csv
  - scalability.device_counts       : list of N values, e.g. [5, 10, 20]
  - scalability.per_count_trials    : repetitions per (N, latency) pair
  - scalability.emulate_latency_ms  : list of latency values, e.g. [0, 50, 100]

Output:
  - <output_dir>/scalability_results.csv
      columns: device_count, latency_ms, trial, T_total
"""

import os
import ast
import time
import yaml
import random
import pandas as pd
from pathlib import Path

# Load experiment configuration
CFG = yaml.safe_load(open("config.yaml"))
OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
LOGS = Path(CFG["logs_root"])

# Load device lookup table (device name → port/probability)
lookup = pd.read_csv(CFG["lookup_csv"]).copy()


def pick_devices(n):
    """Randomly sample n device names from the lookup table."""
    ds = lookup["device"].astype(str).unique().tolist()
    random.shuffle(ds)
    return ds[:n]


def read_total_time():
    """
    Scan all per-device metrics logs and return the mean of the last
    non-zero Total Time (T) entry found across all logs.
    Returns None if no valid entries exist.
    """
    rows = []
    for csv_path in LOGS.glob("*/*_metrics_log.csv"):
        df = pd.read_csv(csv_path)
        if "Total Time (T)" in df and (df["Total Time (T)"] > 0).any():
            r = df[df["Total Time (T)"] > 0].iloc[-1]
            rows.append(float(r["Total Time (T)"]))
    return sum(rows) / len(rows) if rows else None


# Read sweep parameters from config
device_counts = CFG["scalability"]["device_counts"]
trials        = CFG["scalability"]["per_count_trials"]
latencies     = CFG["scalability"]["emulate_latency_ms"]

results = []

for L in latencies:
    for n in device_counts:
        for t in range(trials):
            # Build a single-patient CSV restricted to n randomly chosen devices
            patients = pd.read_csv(CFG["filtered_patients_csv"]).head(1).copy()
            devices_n = pick_devices(n)
            patients["RELEVANT_DEVICES"] = [",".join(devices_n)] * len(patients)
            temp_csv = OUT / f"filtered_tmp_{n}_{L}_{t}.csv"
            patients.to_csv(temp_csv, index=False)

            # Invoke one initiation run; SLEEP_EXTRA injects simulated latency (ms)
            cmd = (
                f"bash -lc 'cd {os.path.dirname(__file__)} && "
                f"FILTERED={temp_csv} SLEEP_EXTRA={L} ./_one_initiate_wrapper.sh'"
            )
            os.system(cmd)

            # Wait for the run to finish plus the emulated latency
            time.sleep(2 + L / 1000.0)

            T = read_total_time()
            results.append({"device_count": n, "latency_ms": L, "trial": t, "T_total": T})

# Save all trial results
pd.DataFrame(results).to_csv(OUT / "scalability_results.csv", index=False)
print("Saved:", OUT / "scalability_results.csv")
