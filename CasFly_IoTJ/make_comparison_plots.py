#!/usr/bin/env python3
"""
make_comparison_plots.py

Generates three comparison figures between LaVE and PCMCI+:

  (a) F1 vs number of PCMCI+ edges per device (complexity scatter plot).
  (b) Edge-overlap bar chart: LaVE-only / Both / PCMCI-only edges per device,
      matched by (cause_bin, effect_bin, lag_bin).
  (c) Per-device precision / recall / F1 bar chart from the LaVE metrics log.

Inputs:
    --lave_metrics : CSV with per-device LaVE metrics
                     expected columns: device, precision, recall, f1
                     (case-insensitive; close variants are accepted)
    --lave_edges   : LaVE edges CSV; needs at least device, cause, effect.
                     If cause_bin / effect_bin columns are absent they are
                     inferred from bracket notation or pure range strings.
    --pcmci_root   : directory containing DeviceX/edges.csv files
                     (cause_bin, effect_bin, lag_bin required)

Outputs (under --out_dir, default figs_cmp/):
    edge_overlap_counts.csv
    fig_complexity_scatter.png
    fig_overlap_bars.png
    fig_per_device_bars.png
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_cols(df):
    """Lowercase-strip all column names."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_pcmci_edges(root: Path) -> pd.DataFrame:
    """Load and concatenate all Device*/edges.csv files under root."""
    rows = []
    for d in sorted(root.glob("Device*/edges.csv")):
        dev = d.parent.name
        try:
            x = pd.read_csv(d)
        except Exception:
            continue
        if x.empty:
            continue
        x["device"] = dev
        rows.append(x)
    if not rows:
        return pd.DataFrame(columns=["device","cause_bin","effect_bin","lag_bin"])
    df = pd.concat(rows, ignore_index=True)
    return normalize_cols(df)

def load_lave_edges(path: Path) -> pd.DataFrame:
    """Load LaVE edges and infer cause_bin/effect_bin if not explicitly present."""
    df = pd.read_csv(path)
    df = normalize_cols(df)

    def extract_bin(s):
        """Extract the bin range from 'Glucose[70-99]' or a bare '70-99' string."""
        s = str(s)
        m = re.search(r'\[(.*?)\]', s)
        if m: return m.group(1)
        if re.fullmatch(r'\d+(\.\d+)?-\d+(\.\d+)?', s): return s
        return None

    if "cause_bin" not in df.columns:
        df["cause_bin"] = df["cause"].apply(extract_bin)
    if "effect_bin" not in df.columns:
        df["effect_bin"] = df["effect"].apply(extract_bin)
    return df

def jaccard_like(a, b):
    """Return (only_a, intersection, only_b) counts for two sets."""
    inter  = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    return only_a, inter, only_b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lave_metrics", required=True)
    ap.add_argument("--lave_edges",   required=True)
    ap.add_argument("--pcmci_root",   required=True)
    ap.add_argument("--out_dir",      default="figs_cmp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load and standardise the LaVE metrics table
    lave_metrics_df = normalize_cols(pd.read_csv(args.lave_metrics))
    colmap = {c:c for c in lave_metrics_df.columns}
    for k in ["device","precision","recall","f1"]:
        if k not in lave_metrics_df.columns:
            # Accept close variants (e.g. 'f1_score' -> 'f1')
            hit = [c for c in lave_metrics_df.columns if k in c]
            if hit: colmap[hit[0]] = k
    lave_metrics_df = lave_metrics_df.rename(columns=colmap)
    lave_metrics_df = lave_metrics_df[["device","precision","recall","f1"]].dropna(subset=["device"]).copy()

    lave_edges_df = load_lave_edges(Path(args.lave_edges))
    pcmci_edges_df = load_pcmci_edges(Path(args.pcmci_root))

    # Edge overlap: match by (cause_bin, effect_bin, lag_bin)
    overlap_rows = []
    devices = sorted(set(lave_edges_df["device"].unique()) | set(pcmci_edges_df["device"].unique()))
    for dev in devices:
        lave_dev  = lave_edges_df[lave_edges_df["device"]==dev]
        pcmci_dev = pcmci_edges_df[pcmci_edges_df["device"]==dev]
        lave_edge_set = set(
            (str(a), str(b), str(lb))
            for a, b, lb in zip(lave_dev["cause_bin"].fillna(""), lave_dev["effect_bin"].fillna(""),
                                lave_dev.get("lag_bin", pd.Series([""]*len(lave_dev))))
            if a and b
        )
        pcmci_edge_set = set(
            (str(a), str(b), str(lb))
            for a, b, lb in zip(pcmci_dev["cause_bin"].fillna(""), pcmci_dev["effect_bin"].fillna(""),
                                pcmci_dev.get("lag_bin", pd.Series([""]*len(pcmci_dev))))
            if a and b
        )
        only_l, both, only_p = jaccard_like(lave_edge_set, pcmci_edge_set)
        overlap_rows.append({"device":dev, "lave_only":only_l, "both":both,
                             "pcmci_only":only_p, "pcmci_edges":len(pcmci_edge_set)})

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir/"edge_overlap_counts.csv", index=False)

    # Merge per-device metrics with PCMCI edge counts for the scatter plot
    metrics_with_edge_counts = lave_metrics_df.merge(
        overlap_df[["device","pcmci_edges"]], on="device", how="left"
    ).fillna({"pcmci_edges": 0})

    # Plot A: F1 vs number of PCMCI+ edges (complexity scatter)
    plt.figure(figsize=(10,6))
    plt.scatter(metrics_with_edge_counts["pcmci_edges"], metrics_with_edge_counts["f1"])
    for _, r in metrics_with_edge_counts.iterrows():
        plt.annotate(r["device"], (r["pcmci_edges"], r["f1"]),
                     fontsize=9, xytext=(3,3), textcoords="offset points")
    plt.xlabel("# PCMCI edges (per device)")
    plt.ylabel("F1 (LaVE vs PCMCI+)")
    plt.title("F1 vs PCMCI model complexity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"fig_complexity_scatter.png", dpi=200)

    # Plot B: Edge-overlap grouped bar chart per device
    overlap_indexed = overlap_df.set_index("device").loc[sorted(overlap_df["device"])]
    x = np.arange(len(overlap_indexed))
    w = 0.28
    plt.figure(figsize=(14,6))
    plt.bar(x - w, overlap_indexed["lave_only"],  width=w, label="LaVE only")
    plt.bar(x,      overlap_indexed["both"],       width=w, label="Both")
    plt.bar(x + w,  overlap_indexed["pcmci_only"], width=w, label="PCMCI only")
    plt.xticks(x, overlap_indexed.index, rotation=45, ha="right")
    plt.ylabel("# Edges")
    plt.title("Edge overlap per device (bin+lag matched)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"fig_overlap_bars.png", dpi=200)

    # Plot C: Per-device precision / recall / F1 bars
    metrics_indexed = lave_metrics_df.set_index("device").loc[sorted(lave_metrics_df["device"])]
    plt.figure(figsize=(12,6))
    x = np.arange(len(metrics_indexed))
    plt.bar(x-0.2, metrics_indexed["precision"], width=0.2, label="precision")
    plt.bar(x,      metrics_indexed["recall"],    width=0.2, label="recall")
    plt.bar(x+0.2,  metrics_indexed["f1"],        width=0.2, label="f1")
    plt.xticks(x, metrics_indexed.index, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Per-device metrics (LaVE vs PCMCI+)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"fig_per_device_bars.png", dpi=200)

    print("Wrote:")
    print(" ", (out_dir/"edge_overlap_counts.csv").resolve())
    print(" ", (out_dir/"fig_complexity_scatter.png").resolve())
    print(" ", (out_dir/"fig_overlap_bars.png").resolve())
    print(" ", (out_dir/"fig_per_device_bars.png").resolve())

if __name__ == "__main__":
    main()
