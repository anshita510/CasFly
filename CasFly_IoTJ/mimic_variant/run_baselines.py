#!/usr/bin/env python3
"""
run_baselines.py — Run SOTA baseline causal-graph methods for comparison.

Description:
    Computes causal edge sets using simple baseline methods:
      - CPT@threshold: retain edges from the conditional probability table
        where conditional_probability >= threshold (two thresholds: 0.02, 0.05)
      - Greedy@K: take the top-K highest-probability edges

    Optionally runs PCMCI+ (from the tigramite package) if:
      - tigramite is installed
      - baselines.pcmci_plus is True in config.yaml
      - a pre-prepared time-series panel file (timeseries_panel.npy) exists
        in the output directory

    Outputs:
      - baselines_summary.csv  — method name and edge count for each baseline

Usage:
    cd mimic_variant/
    python3 run_baselines.py

Configuration:
    Reads config.yaml in the current working directory.
    Key fields used: output_dir, prob_file, baselines
"""

import os
import yaml
import warnings

import pandas as pd
from pathlib import Path

CFG = yaml.safe_load(open("config.yaml"))
OUT  = Path(CFG["output_dir"]); OUT.mkdir(parents=True, exist_ok=True)
PROB = pd.read_csv(CFG["prob_file"])


def cpt_edges(th=0.02):
    """Return edge set from conditional probability table at given threshold."""
    g = PROB[PROB["conditional_probability"] >= th]
    return set(zip(g["cause"].astype(str), g["effect"].astype(str)))


def greedy_edges(k=100):
    """Return top-K edges sorted by conditional probability (descending)."""
    g = PROB.sort_values("conditional_probability", ascending=False).head(k)
    return set(zip(g["cause"].astype(str), g["effect"].astype(str)))


rows = []
rows.append({"method": "CPT@0.02",    "edges": len(cpt_edges(0.02))})
rows.append({"method": "CPT@0.05",    "edges": len(cpt_edges(0.05))})
rows.append({"method": "Greedy@100",  "edges": len(greedy_edges(100))})

# ----- Optional: PCMCI+ (if tigramite is available and you have time-series) -----
try:
    import tigramite  # type: ignore
    HAS_TIG = True
except Exception:
    HAS_TIG = False
    warnings.warn("tigramite not installed; skipping PCMCI+ baseline")

# If you have a prepared panel time-series matrix X (T x N) per patient,
# drop it as 'timeseries_panel.npy' (or implement your loader here).
if HAS_TIG and CFG["baselines"].get("pcmci_plus", False):
    # Placeholder: run PCMCI+ on a toy ts if provided
    import numpy as np
    ts_path = OUT / "timeseries_panel.npy"
    if ts_path.exists():
        from tigramite.data_processing import DataFrame
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests import ParCorr
        X = np.load(ts_path)      # shape (T, N)
        var_names = [f"V{i}" for i in range(X.shape[1])]
        dataframe = DataFrame(X, var_names=var_names)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
        results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
        qmatrix = results["q_matrix"]
        edges = set()
        for j in range(qmatrix.shape[0]):
            for i in range(qmatrix.shape[1]):
                if i == j:
                    continue
                if (qmatrix[j, i, 1] < 0.05):  # any lag significant
                    edges.add((var_names[i], var_names[j]))
        rows.append({"method": "PCMCI+", "edges": len(edges)})

pd.DataFrame(rows).to_csv(OUT / "baselines_summary.csv", index=False)
print("Saved:", OUT / "baselines_summary.csv")
