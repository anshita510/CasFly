#!/usr/bin/env python3
"""
eval_compare_chains.py

Offline evaluation of PCMCI+ vs LaVE for next-device prediction.

For each hop recorded by LaVE (device, cause_event, chosen_next_device),
this script looks up PCMCI+ causal edges on the same device, ranks the
candidate next devices by effect strength (|val| descending), and checks
whether the LaVE-chosen device appears in the top-1 or top-3 predictions.

Inputs:
    1) artifacts/pcmci/DeviceX/edges.csv  -- from an offline PCMCI+ training
       step; columns: cause, effect, lag, val, pval [, lag_bin]
    2) device_mapping.csv                 -- maps Event/Condition strings to
       device names; columns: Event/Condition, Assigned_Device
    3) artifacts/chains/lave_hops.csv     -- LaVE hop log; required columns:
         device, cause_event, chosen_next_device
       optional: lag_days  OR  timestamp_cause + timestamp_effect

Outputs:
    artifacts/pcmci_eval/pairwise_results.csv
    artifacts/pcmci_eval/compare_summary_by_device.csv
    artifacts/pcmci_eval/compare_summary_overall.csv

Notes:
    - Hops with no PCMCI+ edges for (device, cause_event) are skipped.
    - When lag_days is available, PCMCI+ edges are pre-filtered to the same
      lag_bin before ranking; otherwise all lags are used.
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

PCMCI_ROOT = Path("artifacts/pcmci")
EVAL_OUT   = Path("artifacts/pcmci_eval")
LAVE_HOPS  = Path("artifacts/chains/lave_hops.csv")
DEV_MAP    = Path("device_mapping.csv")

LAG_BINS   = [0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
LAG_LABELS = [f"{LAG_BINS[i]}-{LAG_BINS[i+1]}" for i in range(len(LAG_BINS)-1)]


def norm(s):
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", str(s or "").strip())

def assign_lag_bin(lag_days):
    """Map a numeric lag in days to the corresponding bin label."""
    try:
        x = float(lag_days)
    except Exception:
        return None
    if not np.isfinite(x) or x < 0:
        return None
    idx = int(np.digitize([x], LAG_BINS, right=False)[0] - 1)
    idx = max(0, min(idx, len(LAG_BINS)-2))
    return LAG_LABELS[idx]

def load_device_mapping(dev_map_csv: Path) -> dict:
    """Return a dict mapping lowercase event/condition string -> device name."""
    m = pd.read_csv(dev_map_csv)
    m.columns = ["Event/Condition", "Assigned_Device"]
    m["k"] = m["Event/Condition"].astype(str).str.strip().str.lower()
    return dict(zip(m["k"], m["Assigned_Device"].astype(str)))

def load_pcmci_edges(root: Path) -> dict:
    """
    Load per-device PCMCI+ edge files.
    Returns dict: device_name -> DataFrame sorted by cause, then |val| desc.
    """
    out = {}
    for dev_dir in sorted(root.glob("Device*")):
        fp = dev_dir / "edges.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            out[dev_dir.name] = df
            continue
        df["cause"]  = df["cause"].astype(str).str.strip()
        df["effect"] = df["effect"].astype(str).str.strip()
        # Add lag_bin if not already present
        if "lag_bin" not in df.columns:
            df["lag_bin"] = df["lag"].apply(assign_lag_bin)
        df["_abs"] = df["val"].abs()
        df = df.sort_values(["cause","_abs"], ascending=[True, False]).drop(columns=["_abs"])
        out[dev_dir.name] = df
    return out

def load_lave_hops(csv_path: Path) -> pd.DataFrame:
    """Load the LaVE hop log and derive lag_bin from timestamps if lag_days is absent."""
    df = pd.read_csv(csv_path)
    needed  = {"device","cause_event","chosen_next_device"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    # Derive lag_days from timestamps if not directly provided
    if "lag_days" not in df.columns and {"timestamp_cause","timestamp_effect"}.issubset(df.columns):
        t1 = pd.to_datetime(df["timestamp_cause"],  errors="coerce")
        t2 = pd.to_datetime(df["timestamp_effect"], errors="coerce")
        df["lag_days"] = (t2 - t1).dt.days

    df["device"]             = df["device"].astype(str)
    df["cause_event"]        = df["cause_event"].astype(str).str.strip()
    df["chosen_next_device"] = df["chosen_next_device"].astype(str)
    if "lag_days" in df.columns:
        df["lag_bin"] = df["lag_days"].apply(assign_lag_bin)
    else:
        df["lag_bin"] = None
    return df

def rank_next_devices_for(device_name: str,
                          cause_event: str,
                          lag_bin: str | None,
                          edges_df: pd.DataFrame,
                          event2device: dict,
                          topk: int = 3) -> list[str]:
    """
    Predict the next devices for a given (device, cause_event, optional lag_bin).
    PCMCI+ effect labels are mapped to device names via event2device and
    deduplicated while preserving |val| order.
    """
    if edges_df is None or edges_df.empty:
        return []

    sub = edges_df[edges_df["cause"].str.lower() == cause_event.lower()]
    if sub.empty:
        return []

    # Narrow to the same lag bin if available; fall back to all lags
    if lag_bin:
        sub2 = sub[sub["lag_bin"] == lag_bin]
        if not sub2.empty:
            sub = sub2

    devs = []
    seen = set()
    for _, r in sub.iterrows():
        eff = str(r["effect"]).strip().lower()
        dev = event2device.get(eff)
        if not dev:
            continue
        if dev not in seen:
            seen.add(dev)
            devs.append(dev)
            if len(devs) >= topk:
                break
    return devs

def main():
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    print("-> Loading device mapping ...")
    event2device = load_device_mapping(DEV_MAP)
    print("-> Loading PCMCI+ edges ...")
    pcmci_edges = load_pcmci_edges(PCMCI_ROOT)
    print("-> Loading LaVE hops ...")
    hops = load_lave_hops(LAVE_HOPS)

    rows = []
    for _, r in hops.iterrows():
        dev             = r["device"]
        cause           = r["cause_event"]
        chosen_next_dev = r["chosen_next_device"]
        lag_bin         = r.get("lag_bin", None)

        edges_df = pcmci_edges.get(dev)
        if edges_df is None:
            continue

        preds_top3 = rank_next_devices_for(dev, cause, lag_bin, edges_df, event2device, topk=3)
        if not preds_top3:
            # Skip hops where PCMCI+ has no knowledge for this (device, cause)
            continue

        top1     = preds_top3[0] if len(preds_top3) >= 1 else None
        top3_hit = chosen_next_dev in preds_top3
        top1_hit = (chosen_next_dev == top1)

        rows.append({
            "device":              dev,
            "cause_event":         cause,
            "chosen_next_device":  chosen_next_dev,
            "pcmci_top1":          top1,
            "pcmci_top3_list":     "|".join(preds_top3),
            "top1_hit":            int(top1_hit),
            "top3_hit":            int(top3_hit),
            "lag_bin":             lag_bin if pd.notna(lag_bin) else None
        })

    if not rows:
        print("No comparable hops found. Check your LaVE CSV columns and PCMCI edges.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(EVAL_OUT / "pairwise_results.csv", index=False)

    # Aggregate accuracy metrics
    overall = {
        "N":        len(out_df),
        "Top1_acc": out_df["top1_hit"].mean(),
        "Top3_acc": out_df["top3_hit"].mean()
    }
    by_device      = (out_df.groupby("device")[["top1_hit","top3_hit"]]
                      .mean().rename(columns={"top1_hit":"Top1_acc","top3_hit":"Top3_acc"}))
    by_device["N"] = out_df.groupby("device").size()

    print("\n=== Overall ===")
    print(pd.Series(overall))

    print("\n=== By Device ===")
    print(by_device.sort_values("N", ascending=False))

    by_device.reset_index().to_csv(EVAL_OUT / "compare_summary_by_device.csv", index=False)
    pd.DataFrame([overall]).to_csv(EVAL_OUT / "compare_summary_overall.csv", index=False)

    print(f"\nSaved results in {EVAL_OUT.resolve()}")

if __name__ == "__main__":
    main()
