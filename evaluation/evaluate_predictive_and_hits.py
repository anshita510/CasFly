#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_predictive_and_hits.py

Adds three paper-grade analyses on top of your existing artifacts:
  (1) Downstream predictive utility (LaVE vs PCMCI+) using per-effect models,
      concatenating predictions to compute AUROC/AUPRC/LogLoss/Brier.
  (2) Reverse Hits@K (is PCMCI covered by LaVE's top-K?)
  (3) Paired statistics across devices (Wilcoxon + Cliff's delta)

Inputs
------
- devices_root/: Device*/Patient_*.csv (DESCRIPTION, VALUE, DATE)
- lave_csv:      .../lave_edges_normalized.csv  (device,cause_bin,effect_bin,lag_bin)
- pcmci_root:    .../pcmci_tau_{tau}/Device*/edges.csv (cause_bin,effect_bin,lag_bin)

Outputs
-------
- out_root/predictive/per_device_metrics.csv
- out_root/predictive/summary_by_tau.csv
- out_root/predictive/paired_tests.csv
- out_root/reverse_hits/reverse_hits_by_tau.csv
"""

import os, re, warnings, argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon

# ---------- shared config ----------
LAG_BINS    = [0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
LAG_LABELS  = [f"{LAG_BINS[i]}-{LAG_BINS[i+1]}" for i in range(len(LAG_BINS)-1)]
_NUM_RE     = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
DEV_RE      = re.compile(r"^device\s*(\d+)$", re.I)
BIN_RE      = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")  # e.g. 7-14, 70-99

BIN_RULES: Dict[str, List[float]] = {
    'ALT (Elevated)': [0, 30, 60, 120, 180, np.inf],
    'AST (Elevated)': [0, 40, 80, 150, np.inf],
    'Alanine aminotransferase': [0, 30, 60, 120, 180, np.inf],
    'Albumin': [0, 3.5, 5.0, np.inf],
    'Alkaline Phosphatase': [0, 100, 200, 300, np.inf],
    'Anion Gap': [0, 8, 16, np.inf],
    'Bicarbonate': [0, 22, 26, np.inf],
    'Bilirubin (Total)': [0, 1.2, 2.0, np.inf],
    'Body Height': [0, 150, 170, 190, np.inf],
    'Body Temperature': [0, 36.0, 37.5, np.inf],
    'Body Weight': [0, 50, 70, 90, np.inf],
    'BMI (Body Mass Index)': [0, 18.5, 24.9, 29.9, 34.9, np.inf],
    'Calcium': [0, 8.5, 10.2, np.inf],
    'Carbon Dioxide': [0, 35, 45, np.inf],
    'Chloride': [0, 98, 107, np.inf],
    'Creatinine': [0, 1.0, 1.5, np.inf],
    'Diastolic Blood Pressure': [0, 60, 80, 90, np.inf],
    'Fibrin D-dimer': [0, 250, 500, np.inf],
    'Ferritin': [0, 30, 300, np.inf],
    'Glucose': [0, 70, 99, 125, np.inf],
    'Hemoglobin': [0, 12.0, 16.0, np.inf],
    'Heart Rate': [0, 60, 100, np.inf],
    'Oxygen Saturation': [0, 90, 95, np.inf],
    'Platelet Count': [0, 150, 450, np.inf],
    'Potassium': [0, 3.5, 5.0, np.inf],
    'Prothrombin Time (PT)': [0, 10, 14, np.inf],
    'Respiratory Rate': [0, 12, 20, np.inf],
    'Sodium': [0, 135, 145, np.inf],
    'Systolic Blood Pressure': [0, 90, 120, 140, np.inf]
}

# ---------- utilities ----------
def norm_device(s: str) -> str:
    s = str(s or "").strip().replace("_"," ").replace("-"," ")
    m = DEV_RE.match(s)
    return f"Device{m.group(1)}" if m else s.replace(" ", "")

def to_day(ts) -> pd.Timestamp:
    try: return pd.to_datetime(ts).normalize()
    except Exception: return pd.NaT

def extract_numeric(v) -> Optional[float]:
    if pd.isna(v): return None
    s = str(v); m = _NUM_RE.search(s)
    if not m: return None
    try: return float(m.group(0))
    except Exception: return None

def bin_value(event: str, value) -> Optional[str]:
    if event not in BIN_RULES: return None
    x = extract_numeric(value)
    if x is None: return None
    bins = BIN_RULES[event]
    idx = int(np.digitize([x], bins, right=False)[0] - 1)
    idx = max(0, min(idx, len(bins)-2))
    return f"{event}[{bins[idx]}-{bins[idx+1]}]"

def lag_center(lag_bin: str) -> Optional[int]:
    """Turn '7-14' into the integer midpoint 10; '0-7' -> 3, etc."""
    if not isinstance(lag_bin, str) or not BIN_RE.match(lag_bin): return None
    a, b = lag_bin.split("-")
    try:
        a = float(a); b = float(b)
    except Exception:
        return None
    if not np.isfinite(a) or not np.isfinite(b): return None
    c = int(round((a + b) / 2.0))
    return max(0, c)

# ---------- data loading ----------
def build_daily_matrix(device_dir: Path, min_days_present: int=1) -> pd.DataFrame:
    rows = []
    for pf in device_dir.glob("Patient_*.csv"):
        try:
            df = pd.read_csv(pf)
        except Exception:
            continue
        if not {"DESCRIPTION","VALUE","DATE"}.issubset(df.columns): 
            continue
        df = df.dropna(subset=["DESCRIPTION","DATE"]).copy()
        if df.empty: continue
        df["DATE"] = df["DATE"].apply(to_day)
        df = df.dropna(subset=["DATE"])
        if df.empty: continue
        df["evt_bin"] = [bin_value(str(a), b) for a,b in zip(df["DESCRIPTION"], df["VALUE"])]
        df = df.dropna(subset=["evt_bin"])
        if df.empty: continue
        tmp = df[["DATE","evt_bin"]].drop_duplicates()
        tmp["flag"] = 1
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    all_ev = pd.concat(rows, ignore_index=True)
    pivot = (all_ev.pivot_table(index="DATE", columns="evt_bin", values="flag",
                                aggfunc="max", fill_value=0).sort_index())
    col_counts = pivot.sum(axis=0)
    keep_cols = col_counts[col_counts >= max(1, min_days_present)].index
    pivot = pivot.loc[:, keep_cols]
    pivot = pivot.loc[:, (pivot.sum(axis=0) > 0)]
    return pivot

def load_lave_edges(lave_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lave_csv)
    need = {"device","cause_bin","effect_bin","lag_bin"}
    if not need.issubset(df.columns):
        raise ValueError(f"{lave_csv} missing columns {need}")
    df["device"] = df["device"].astype(str).apply(norm_device)
    df = df.dropna(subset=["cause_bin","effect_bin","lag_bin"]).drop_duplicates()
    return df

def load_pcmci_edges_for_tau(pcmci_root: Path) -> pd.DataFrame:
    rows = []
    for devdir in sorted(pcmci_root.glob("Device*")):
        f = devdir / "edges.csv"
        if not f.exists(): 
            continue
        df = pd.read_csv(f)
        if {"cause_bin","effect_bin","lag_bin"}.issubset(df.columns):
            use = df[["cause_bin","effect_bin","lag_bin"]].copy()
            use["device"] = devdir.name
            rows.append(use.dropna())
    if not rows:
        return pd.DataFrame(columns=["device","cause_bin","effect_bin","lag_bin"])
    return pd.concat(rows, ignore_index=True).drop_duplicates()

# ---------- reverse hits@K (PCMCI in LaVE's top-K) ----------
def reverse_hits_at_k(lave: pd.DataFrame, pcmci: pd.DataFrame, ks=(1,3,5)) -> pd.DataFrame:
    if pcmci.empty or lave.empty:
        return pd.DataFrame({"tau":[None], "K":[None], "hits":[0], "total":[0], "hit_rate":[0.0]})[:0]
    hits = {k:0 for k in ks}
    tot  = len(pcmci)
    grouped = pcmci.groupby(["device","cause_bin"])["effect_bin"].value_counts().rename("cnt").reset_index()
    for _, row in pcmci.iterrows():
        dev, c, e = row["device"], row["cause_bin"], row["effect_bin"]
        cand = grouped[(grouped["device"]==dev) & (grouped["cause_bin"]==c)].sort_values("cnt", ascending=False)["effect_bin"].tolist()
        for k in ks:
            lave_cands = lave[(lave["device"]==dev) & (lave["cause_bin"]==c)]["effect_bin"].value_counts().index.tolist()
            if e in set(lave_cands[:k]):
                hits[k] += 1
    out = pd.DataFrame({"K":list(ks), "hits":[hits[k] for k in ks], "total":[tot]*len(ks)})
    out["hit_rate"] = out["hits"] / out["total"].replace({0:np.nan})
    return out

# ---------- bin→series aggregation ----------
def columns_for_bin(daily: pd.DataFrame, bin_str: str) -> List[str]:
    suffix = f"[{bin_str}]"
    return [c for c in daily.columns if isinstance(c, str) and c.endswith(suffix)]

def agg_bin_series(daily: pd.DataFrame, bin_str: str) -> Optional[pd.Series]:
    cols = columns_for_bin(daily, bin_str)
    if not cols:
        return None
    s = daily[cols].max(axis=1)  # OR over events with that bin
    return (s > 0).astype(int)

# ---------- per-effect dataset builder ----------
def build_Xy_for_effect(daily: pd.DataFrame,
                        effect_bin: str,
                        edges_for_effect: pd.DataFrame,
                        min_positives: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build X,y for a single effect_bin by OR-aggregating all columns sharing that bin.
    Causes are shifted by lag_center individually and concatenated as features.
    """
    y = agg_bin_series(daily, effect_bin)
    if y is None:
        return None
    if y.sum() < min_positives or y.sum() == len(y):
        return None

    feat_cols = []
    max_lag = 0
    for _, r in edges_for_effect.iterrows():
        cbin = str(r["cause_bin"])
        lbin = str(r["lag_bin"])
        lc   = lag_center(lbin)
        if lc is None:
            continue
        x = agg_bin_series(daily, cbin)
        if x is None:
            continue
        max_lag = max(max_lag, lc)
        feat_cols.append(x.shift(lc).values.reshape(-1, 1))

    if not feat_cols:
        return None

    X = np.hstack(feat_cols)
    yv = y.values.astype(int)

    # drop first max_lag rows (NaNs from shifting)
    X = X[max_lag:]
    yv = yv[max_lag:]

    if yv.sum() == 0 or yv.sum() == len(yv):
        return None

    return X, yv

# ---------- downstream predictive utility (per-effect models) ----------
def evaluate_edges_predictive(daily: pd.DataFrame,
                              edges: pd.DataFrame,
                              random_state: int = 7) -> Dict[str,float]:
    """
    Train one logistic model per effect_bin (its parents = rows in edges),
    collect predictions across effects, and compute global metrics.
    """
    if daily.empty or edges.empty:
        return dict(auroc=np.nan, auprc=np.nan, logloss=np.nan, brier=np.nan, n=0)

    y_all, p_all = [], []

    for effect_bin, sub in edges.groupby("effect_bin"):
        ds = build_Xy_for_effect(daily, effect_bin, sub)
        if ds is None:
            continue
        X, y = ds
        # time-aware split
        n = X.shape[0]
        split = int(n*0.7)
        if split <= 1 or split >= n-1:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
        else:
            Xtr, ytr = X[:split], y[:split]
            Xte, yte = X[split:], y[split:]

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            # degenerate split; skip this effect
            continue

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        try:
            clf.fit(Xtr, ytr)
            pr = clf.predict_proba(Xte)[:,1]
        except Exception:
            continue

        y_all.append(yte)
        p_all.append(pr)

    if not y_all:
        return dict(auroc=np.nan, auprc=np.nan, logloss=np.nan, brier=np.nan, n=0)

    y_cat = np.concatenate(y_all)
    p_cat = np.concatenate(p_all)

    try:
        auroc = roc_auc_score(y_cat, p_cat)
        auprc = average_precision_score(y_cat, p_cat)
        ll    = log_loss(y_cat, p_cat, labels=[0,1])
        br    = brier_score_loss(y_cat, p_cat)
    except Exception:
        auroc, auprc, ll, br = np.nan, np.nan, np.nan, np.nan

    return dict(auroc=auroc, auprc=auprc, logloss=ll, brier=br, n=len(y_cat))

# ---------- effect size ----------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta (x vs y): in [-1,1]; >0 means x larger."""
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x)==0 or len(y)==0: return np.nan
    greater = 0; less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less    += np.sum(xi < y)
    m, n = len(x), len(y)
    return (greater - less) / float(m*n)

# ---------- main orchestration ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices_root", required=True)
    ap.add_argument("--lave_csv", required=True)
    ap.add_argument("--pcmci_root_base", required=True, help="prefix like artifacts/.../pcmci_tau_ (we append <tau>)")
    ap.add_argument("--taus", required=True, help="comma-separated list, e.g. 7,14,30,60,90")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--min_days_present", type=int, default=1)
    args = ap.parse_args()

    devices_root = Path(args.devices_root)
    lave_all     = load_lave_edges(Path(args.lave_csv))
    taus         = [int(x) for x in str(args.taus).split(",") if x.strip()]
    out_root     = Path(args.out_root)
    out_pred     = out_root / "predictive"; out_pred.mkdir(parents=True, exist_ok=True)
    out_hits     = out_root / "reverse_hits"; out_hits.mkdir(parents=True, exist_ok=True)

    # prebuild daily matrices once per device
    daily_by_device: Dict[str,pd.DataFrame] = {}
    for devdir in sorted(devices_root.glob("Device*")):
        mat = build_daily_matrix(devdir, min_days_present=args.min_days_present)
        daily_by_device[devdir.name] = mat

    per_device_rows = []
    paired_out = []
    reverse_hits_all = []

    for tau in taus:
        pcmci_root = Path(f"{args.pcmci_root_base}{tau}")
        pcmci_tau  = load_pcmci_edges_for_tau(pcmci_root)

        # per-device predictive comparison
        for dev, daily in daily_by_device.items():
            if daily.empty or daily.shape[1] < 2:
                continue
            lave_edges_d  = lave_all [lave_all ["device"]==dev][["cause_bin","effect_bin","lag_bin"]]
            pcmci_edges_d = pcmci_tau[pcmci_tau["device"]==dev][["cause_bin","effect_bin","lag_bin"]]

            m_lave  = evaluate_edges_predictive(daily, lave_edges_d)
            m_pcmci = evaluate_edges_predictive(daily, pcmci_edges_d)

            per_device_rows.append(dict(tau=tau, device=dev, method="LaVE",   **m_lave))
            per_device_rows.append(dict(tau=tau, device=dev, method="PCMCI+", **m_pcmci))

        # reverse hits@K at this τ
        rh = reverse_hits_at_k(lave_all, pcmci_tau, ks=(1,3,5))
        rh["tau"] = tau
        reverse_hits_all.append(rh)

    # ---- write predictive metrics
    per_device_df = pd.DataFrame(per_device_rows)
    per_device_df.to_csv(out_pred / "per_device_metrics.csv", index=False)

    # macro by tau & method
    macro = (per_device_df
             .groupby(["tau","method"])[["auroc","auprc","logloss","brier"]]
             .mean().reset_index())
    macro.to_csv(out_pred / "summary_by_tau.csv", index=False)

    # paired tests per tau (LaVE vs PCMCI+): AUROC & AUPRC
    for tau in taus:
        wide = (per_device_df[per_device_df["tau"]==tau]
                .pivot_table(index="device", columns="method", values=["auroc","auprc"]))
        wide = wide.dropna()
        if wide.empty: 
            continue
        for metric in ["auroc","auprc"]:
            x = wide[(metric,"LaVE")].values
            y = wide[(metric,"PCMCI+")].values
            try:
                stat, p = wilcoxon(x, y, alternative="greater")  # LaVE > PCMCI+
            except Exception:
                stat, p = np.nan, np.nan
            delta = cliffs_delta(x, y)  # >0 favors LaVE
            paired_out.append(dict(tau=tau, metric=metric, N=len(x), wilcoxon_stat=stat, p_value=p, cliffs_delta=delta))
    pd.DataFrame(paired_out).to_csv(out_pred / "paired_tests.csv", index=False)

    # reverse hits
    rev_hits_df = pd.concat(reverse_hits_all, ignore_index=True)
    rev_hits_df.to_csv(out_hits / "reverse_hits_by_tau.csv", index=False)

    # console summary
    print("\n=== Predictive utility (macro by τ) ===")
    print(macro.to_string(index=False))
    if paired_out:
        print("\n=== Paired tests (LaVE > PCMCI+) ===")
        print(pd.DataFrame(paired_out).to_string(index=False))
    print(f"\nSaved:\n  {out_pred/'per_device_metrics.csv'}\n  {out_pred/'summary_by_tau.csv'}\n  {out_pred/'paired_tests.csv'}\n  {out_hits/'reverse_hits_by_tau.csv'}")

if __name__ == "__main__":
    main()
