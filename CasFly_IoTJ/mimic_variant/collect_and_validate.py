#!/usr/bin/env python3
"""
collect_and_validate.py — Collect CasFly chain results and validate against ground truth.

Description:
    Scans all device metrics logs produced by the MIMIC experiment run, extracts
    the most recent chain prediction per patient/device, and computes precision,
    recall, and F1 against two sources of ground truth:
      1. Per-patient TPHG cache (device-local .pkl files)
      2. Global conditional probability table (weak plausibility check)

    Results are saved to:
      - mimic_validation.csv     — per-row metrics (one row per device/patient)
      - mimic_validation_summary.csv — mean precision/recall/F1/total-time

Usage:
    cd mimic_variant/
    python3 collect_and_validate.py

Configuration:
    Reads config.yaml in the current working directory.
    Key fields used: logs_root, output_dir, prob_file
"""

import os
import ast
import glob
import json

import yaml
import joblib
import pandas as pd
from collections import defaultdict

CFG = yaml.safe_load(open("config.yaml"))
LOGS = CFG["logs_root"]
OUT  = CFG["output_dir"]
PROB = CFG["prob_file"]
os.makedirs(OUT, exist_ok=True)

# Load global conditional probabilities for reference
prob_df = pd.read_csv(PROB)


def chain_edges_from_row(row_chain):
    """Parse chain data string into a set of directed (cause, effect) edges.

    row_chain is a str(list of [pred, effect, p, lag, device]) or already a list.
    """
    if isinstance(row_chain, list):
        L = row_chain
    else:
        try:
            L = ast.literal_eval(row_chain)
        except Exception:
            return set()
    edges = {(a, b) for a, b, _, _, _ in L if isinstance(a, str) and isinstance(b, str)}
    return edges


def tphg_edges(cache_path):
    """Load a TPHG graph from a joblib cache file and return its edge set."""
    if not os.path.exists(cache_path):
        return set()
    try:
        g = joblib.load(cache_path)
    except Exception:
        return set()
    E = set()
    for u, v, k in g.edges(keys=True):
        E.add((str(u), str(v)))
    return E


# Scan metrics logs and compute validation metrics
rows = []
for csv_path in glob.glob(os.path.join(LOGS, "*", "*_metrics_log.csv")):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    # Pick rows with non-empty chain (most recent)
    df = df[df["Chain Data"].astype(str).str.len() > 4]
    if df.empty:
        continue
    r = df.iloc[-1]
    edges_pred = chain_edges_from_row(r["Chain Data"])
    patient = str(r["Patient ID"])
    disease = str(r["Disease Name"])
    dev     = os.path.basename(os.path.dirname(csv_path))

    # TPHG cache convention
    cache = os.path.join(LOGS, dev, f"{patient}_tphg.pkl")
    edges_true = tphg_edges(cache)

    # Reference "ground truth" from global prob table (weak check):
    # treat any pair seen in PROB as plausible true
    global_true = set(zip(prob_df["cause"].astype(str), prob_df["effect"].astype(str)))

    tp = len(edges_pred & edges_true) or len(edges_pred & global_true)
    fp = len(edges_pred - edges_true) if edges_true else len(edges_pred - global_true)
    fn = len((edges_true - edges_pred)) if edges_true else 0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    rows.append({
        "device": dev,
        "patient": patient,
        "disease": disease,
        "edges_pred": len(edges_pred),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "T_total": r.get("Total Time (T)", 0),
        "t1_cache": r.get("Cached TPHG Load Time (t1)", 0),
        "t2_viterbi": r.get("Backward Viterbi Time (t2)", 0),
        "t_fallback": r.get("Fallback Path Time (t_fallback)", 0),
        "t_dash": r.get("Time per Device (t_dash)", 0),
        "n_devices_accessed": r.get("No. of Devices Accessed", 0),
    })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "mimic_validation.csv"), index=False)
print("Saved:", os.path.join(OUT, "mimic_validation.csv"))

if not res.empty:
    agg = res[["precision", "recall", "f1", "T_total"]].mean().to_frame("mean").T
    agg.to_csv(os.path.join(OUT, "mimic_validation_summary.csv"), index=False)
    print(agg)
