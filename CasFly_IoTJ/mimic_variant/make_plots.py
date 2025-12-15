#!/usr/bin/env python3
"""
make_plots.py — Generate Summary Plots and Tables for MIMIC Experiments

Reads results produced by the earlier pipeline steps and generates:
  1. scalability_plot.png  — mean total chain time vs device count
  2. methods_table.csv     — consolidated baseline comparison table
                             (written only if baselines_summary.csv exists)

Inputs (via config.yaml):
  - output_dir : directory containing scalability_results.csv,
                 mimic_validation_summary.csv, and baselines_summary.csv

Output:
  - <output_dir>/scalability_plot.png
  - <output_dir>/methods_table.csv  (if baselines_summary.csv exists)
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load experiment configuration
CFG = yaml.safe_load(open("config.yaml"))
OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)

# --- Plot: mean total chain time vs number of devices ---
sc = pd.read_csv(OUT / "scalability_results.csv")
fig = plt.figure()
sc.groupby("device_count")["T_total"].mean().plot(marker="o")
plt.xlabel("Device count")
plt.ylabel("Total time (s)")
plt.title("Scalability of chain formation")
plt.grid(True, alpha=0.3)
fig.savefig(OUT / "scalability_plot.png", bbox_inches="tight")

# --- Table: merge validation summary with baselines if both exist ---
val  = pd.read_csv(OUT / "mimic_validation_summary.csv") if (OUT / "mimic_validation_summary.csv").exists() else None
base = pd.read_csv(OUT / "baselines_summary.csv")        if (OUT / "baselines_summary.csv").exists()        else None

if base is not None:
    base.to_csv(OUT / "methods_table.csv", index=False)

print("Plots/tables saved in:", OUT)
