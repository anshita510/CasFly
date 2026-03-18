#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCMCI+ vs LaVE — τ sweep with tables & plots (resume-aware)

- Trains PCMCI+ per device for each τ (skips if artifacts already present, unless forced)
- Extracts LaVE edges once from all_merged_metrics_log.csv (skips if present, unless forced)
- Computes overlap/metrics/hits per τ (skips if cached CSVs exist, unless forced)
- Produces consolidated tables + multiple paper-ready plots

Example:
python3.11 compare_pcmci_lave_tau.py \
  --devices_root device_partitions_patientwise \
  --log_csv all_merged_metrics_log.csv \
  --out_root artifacts/compare_pcmci_lave_tau_paper \
  --taus 7,14,30,60,90 \
  --alpha 0.05 \
  --min_days_present 1 \
  --hits_k 1,3,5 \
  --top_n_devices 12
"""

import os, re, ast, warnings, argparse
from pathlib import Path
from typing import Optional, Tuple, Iterable, List, Dict
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# ---------------------------- Global styling ----------------------------
# Make hatches visible everywhere (keep colors + patterns)
mpl.rcParams['hatch.linewidth'] = 1.2
mpl.rcParams['hatch.color'] = 'black'

# Font (Times if available) + sizes
_font_lbl = {'size': 22}
if any('Times New Roman' in f.name for f in _fm.fontManager.ttflist):
    _font_lbl['family'] = 'Times New Roman'
_TICK_SIZE = 18
_LEG_SIZE  = 20

def _apply_axes_style(ax, x_label=None, y_label=None, show_grid=True):
    if x_label: ax.set_xlabel(x_label, **_font_lbl)
    if y_label: ax.set_ylabel(y_label, **_font_lbl)
    ax.tick_params(axis='x', labelsize=_TICK_SIZE)
    ax.tick_params(axis='y', labelsize=_TICK_SIZE)
    if show_grid:
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)

def _legend(ax, title=None, loc='best', ncol=None):
    kw = dict(fontsize=_LEG_SIZE, loc=loc, frameon=True, framealpha=0.9)
    if title: kw['title'] = title
    if ncol is not None: kw['ncol'] = ncol
    return ax.legend(**kw)

# Reusable hatch sequence for B/W friendly bars (includes *, etc.)
_HATCH_SEQ = ['///', '\\\\\\', 'xxx', '---', '+++', '...', 'oo', '***', '**', '///***']
def _hatch(i: int) -> str:
    return _HATCH_SEQ[i % len(_HATCH_SEQ)]

# Color cycles (keep colors)
try:
    _COLOR_SEQ = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
except Exception:
    _COLOR_SEQ = []
if not _COLOR_SEQ:
    # Fallback palette
    _COLOR_SEQ = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
def _color(i: int) -> str:
    return _COLOR_SEQ[i % len(_COLOR_SEQ)]

# Default bar edge style (keep facecolor from color, only set edges)
_BAR_KW = dict(edgecolor='black', linewidth=1.0)

# ----------------------------- CONFIG -----------------------------
LAG_BINS    = [0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
LAG_LABELS  = [f"{LAG_BINS[i]}-{LAG_BINS[i+1]}" for i in range(len(LAG_BINS)-1)]
BIN_RE      = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")
DEV_RE      = re.compile(r"^device\s*(\d+)$", re.I)
_NUM_RE     = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

# Align with your LaVE binning
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

# ----------------------------- UTILS -----------------------------
def norm_device(s: str) -> str:
    s = str(s or "").strip()
    m = DEV_RE.match(s.replace("_", " ").replace("-", " ").strip())
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

def lag_to_bin(days: float) -> Optional[str]:
    if days is None or not np.isfinite(days) or days < 0: return None
    idx = int(np.digitize([days], LAG_BINS, right=False)[0] - 1)
    idx = max(0, min(idx, len(LAG_BINS)-2))
    return LAG_LABELS[idx]

# ----------------------- Tigramite import ------------------------
def import_tigramite():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tigramite.data_processing import DataFrame as TigaDataFrame
        try:
            from tigramite.independence_tests.parcorr import ParCorr
        except Exception:
            from tigramite.independence_tests import ParCorr
        from tigramite.pcmci import PCMCI
    return TigaDataFrame, ParCorr, PCMCI

# ----------------------- PCMCI+ helpers --------------------------
def build_daily_matrix(device_dir: Path, min_days_present: int) -> pd.DataFrame:
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

    # prune very-rare signals
    col_counts = pivot.sum(axis=0)
    keep_cols = col_counts[col_counts >= max(1, min_days_present)].index
    pivot = pivot.loc[:, keep_cols]

    # EXTRA: drop columns that are all zeros (safety)
    pivot = pivot.loc[:, (pivot.sum(axis=0) > 0)]

    return pivot

def choose_effective_tau(T: int, requested_tau: int) -> int:
    margin = 5
    tau_eff = min(requested_tau, max(1, T - margin))
    return tau_eff

def matrix_is_usable(df: pd.DataFrame, tau_eff: int) -> bool:
    if df is None or df.empty: return False
    T, N = df.shape
    if N < 2: return False
    if T < (tau_eff + 5): return False
    pos = (df.sum(axis=0) >= 3).sum()
    return pos >= 2

def run_pcmci_on_matrix(df: pd.DataFrame, tau_max: int, alpha: float):
    if df.empty or df.shape[1] < 2: 
        return []
    TigaDataFrame, ParCorr, PCMCI = import_tigramite()
    data = df.values.astype(float)
    var_names = list(df.columns)
    tiga = TigaDataFrame(data=data, var_names=var_names)
    itest = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=tiga, cond_ind_test=itest)
    try:
        res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    except Exception as e:
        print(f"[WARN] PCMCI failed (tau={tau_max}, N={df.shape[1]}, T={df.shape[0]}): {e}")
        return []
    val = res.get("val_matrix"); p = res.get("p_matrix")
    if val is None or p is None: 
        return []
    edges = []
    N = len(var_names)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            for tau in range(1, tau_max + 1):
                vij = val[i, j, tau]; pij = p[i, j, tau]
                if np.isfinite(vij) and np.isfinite(pij) and pij <= alpha:
                    edges.append((var_names[i], var_names[j], tau, float(vij), float(pij)))
    edges.sort(key=lambda x: abs(x[3]), reverse=True)
    return edges

def split_evt_bin(lbl: str) -> Tuple[str, Optional[str]]:
    m = re.match(r'^(.*)\[(.*)\]$', str(lbl))
    if m: return m.group(1), m.group(2)
    return str(lbl), None

def save_pcmci_edges(device: str, edges, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not edges:
        pd.DataFrame(columns=["device","cause","effect","cause_bin","effect_bin","lag","lag_bin","val","pval"]).to_csv(out_dir/"edges.csv", index=False)
        return
    rows = []
    for cause, effect, lag, val, pval in edges:
        c_evt, c_bin = split_evt_bin(cause)
        e_evt, e_bin = split_evt_bin(effect)
        rows.append({
            "device": device,
            "cause": c_evt, "effect": e_evt,
            "cause_bin": c_bin or "", "effect_bin": e_bin or "",
            "lag": lag, "lag_bin": lag_to_bin(lag) or "",
            "val": val, "pval": pval
        })
    pd.DataFrame(rows).to_csv(out_dir/"edges.csv", index=False)

def pcmci_artifacts_exist(pcmci_dir: Path) -> bool:
    if not pcmci_dir.exists(): return False
    some = list(pcmci_dir.glob("Device*/edges.csv"))
    return len(some) > 0

def train_pcmci_per_device(devices_root: Path, out_root: Path,
                           tau_max=30, alpha=0.05, min_days_present=1,
                           resume=True) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    if resume and pcmci_artifacts_exist(out_root):
        print(f"[SKIP] PCMCI artifacts already exist in {out_root}. Use --force_pcmci to retrain.")
        return out_root
    dev_dirs = sorted([p for p in devices_root.glob("Device*") if p.is_dir()])
    for dev_dir in dev_dirs:
        dev = dev_dir.name.replace(" ", "")
        print(f"[PCMCI] Building series for {dev} ...")
        mat = build_daily_matrix(dev_dir, min_days_present=min_days_present)
        dev_out = out_root / dev
        if mat.empty or mat.shape[1] < 2:
            print(f"[PCMCI] {dev}: not enough signals; writing empty edges.csv")
            save_pcmci_edges(dev, [], dev_out); continue
        T = mat.shape[0]
        tau_eff = choose_effective_tau(T, tau_max)
        if not matrix_is_usable(mat, tau_eff):
            print(f"[PCMCI] {dev}: matrix too sparse/short for τ={tau_max} (T={T}, N={mat.shape[1]}). Writing empty edges.csv")
            save_pcmci_edges(dev, [], dev_out); continue
        print(f"[PCMCI] {dev}: T={T} days, N={mat.shape[1]} events -> running PCMCI+ (τ_eff={tau_eff}, α={alpha}) ...")
        edges = run_pcmci_on_matrix(mat, tau_eff, alpha)
        print(f"[PCMCI] {dev}: significant edges = {len(edges)}")
        save_pcmci_edges(dev, edges, dev_out)
    return out_root

# ----------------------- LaVE EDGE EXTRACTION --------------------
def find_chain_col(df: pd.DataFrame) -> str:
    if "Chain Data" in df.columns: return "Chain Data"
    norm = {c: re.sub(r"\s+", "", c.strip().lower()) for c in df.columns}
    for c, n in norm.items():
        if "chain" in n and "data" in n: return c
    for c in ["ChainData", "chain", "chain_data"]:
        if c in df.columns: return c
    raise ValueError("Chain Data column not found in merged log CSV.")

def assign_lag_bin(x):
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0 or not np.isfinite(v): return None
    idx = int(np.digitize([v], LAG_BINS, right=False)[0] - 1)
    idx = max(0, min(idx, len(LAG_BINS)-2))
    return LAG_LABELS[idx]

def parse_chain(chain_str) -> Iterable[dict]:
    if not isinstance(chain_str, str) or not chain_str.strip(): return []
    s = (chain_str.replace("np.float64", "float")
                  .replace("nan", "None")
                  .replace("NaN", "None"))
    try:
        chain = ast.literal_eval(s)
    except Exception:
        return []
    out = []
    for step in chain:
        # [cause_label, effect_label, prob, lag, 'DeviceX']
        if not isinstance(step, (list, tuple)) or len(step) < 5: continue
        cause, effect, _, lag, dev = step[0], step[1], step[2], step[3], step[4]
        cause, effect = str(cause), str(effect)
        if not (BIN_RE.match(cause) and BIN_RE.match(effect)): continue
        lag_bin = assign_lag_bin(lag)
        if not lag_bin: continue
        out.append(dict(device=norm_device(dev), cause_bin=cause, effect_bin=effect, lag_bin=lag_bin))
    return out

def extract_lave_edges_from_log(log_csv: Path, out_csv: Path, resume=True) -> Path:
    if resume and out_csv.exists():
        print(f"[SKIP] LaVE edges already extracted at {out_csv}. Use --force_lave to re-extract.")
        return out_csv
    df = pd.read_csv(log_csv, dtype=str, keep_default_na=False, na_values=[], engine="python", on_bad_lines="skip")
    chain_col = find_chain_col(df)
    rows: List[dict] = []
    for s in df[chain_col].tolist():
        rows.extend(parse_chain(s))
    out = pd.DataFrame(rows).drop_duplicates()
    if not out.empty:
        out["device"] = out["device"].astype(str).str.replace(r"\s+", "", regex=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ LaVE edges extracted: {len(out)} -> {out_csv}")
    return out_csv

# ----------------------- EVALUATION + METRICS --------------------
def load_pcmci_edges(root: Path) -> pd.DataFrame:
    rows = []
    for devdir in sorted(root.glob("Device*")):
        f = devdir / "edges.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        if not {"cause_bin","effect_bin"}.issubset(df.columns): continue
        use = df[["cause_bin","effect_bin"]].copy()
        use["lag_bin"] = df["lag_bin"].fillna("").astype(str) if "lag_bin" in df.columns else ""
        use["device"] = devdir.name.replace(" ", "")
        rows.append(use.dropna())
    if rows:
        return pd.concat(rows, ignore_index=True).drop_duplicates()
    return pd.DataFrame(columns=["device","cause_bin","effect_bin","lag_bin"])

def load_lave_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"device","cause_bin","effect_bin","lag_bin"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {need}")
    df["device"] = df["device"].astype(str).str.replace(r"\s+", "", regex=True)
    return df[["device","cause_bin","effect_bin","lag_bin"]].dropna().drop_duplicates()

def overlap_counts(lave: pd.DataFrame, pcmci: pd.DataFrame) -> pd.DataFrame:
    stats = []
    devices = sorted(set(lave["device"]).union(pcmci["device"]))
    for dev in devices:
        L = set(map(tuple, lave[lave["device"]==dev][["cause_bin","effect_bin"]].values))
        P = set(map(tuple, pcmci[pcmci["device"]==dev][["cause_bin","effect_bin"]].values))
        both   = len(L & P)
        l_only = len(L - P)
        p_only = len(P - L)
        stats.append(dict(device=dev, lave_only=l_only, both=both, pcmci_only=p_only,
                          pcmci_edges=len(P), lave_edges=len(L)))
    return pd.DataFrame(stats)

def metrics_from_counts(df_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_counts.iterrows():
        tp = r["both"]; fp = r["lave_only"]; fn = r["pcmci_only"]
        prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        rows.append(dict(device=r["device"], precision=prec, recall=rec, f1=f1, pcmci_edges=r["pcmci_edges"]))
    return pd.DataFrame(rows)

def hits_at_k(lave: pd.DataFrame, pcmci: pd.DataFrame, ks=(1,3,5)) -> pd.DataFrame:
    if pcmci.empty or lave.empty:
        return pd.DataFrame({"K": list(ks), "hits": [0]*len(ks), "total": [len(lave)]*len(ks), "hit_rate": [0.0]*len(ks)})
    hits = {k:0 for k in ks}
    grouped = pcmci.groupby(["device","cause_bin"])["effect_bin"].value_counts().rename("cnt").reset_index()
    for _, row in lave.iterrows():
        dev, c, e = row["device"], row["cause_bin"], row["effect_bin"]
        cand = grouped[(grouped["device"]==dev) & (grouped["cause_bin"]==c)].sort_values("cnt", ascending=False)["effect_bin"].tolist()
        for k in ks:
            if e in set(cand[:k]):
                hits[k] += 1
    tot = len(lave)
    out = pd.DataFrame({"K": list(ks), "hits": [hits[k] for k in ks], "total": [tot]*len(ks)})
    out["hit_rate"] = out["hits"] / out["total"].replace({0: np.nan})
    return out

# ------------------------------ PLOTS ----------------------------
# 95% CI helper for macro line plot
def _mean_ci_95(a: np.ndarray) -> Tuple[float, float]:
    """Return (mean, 95% CI half-width). Safe on NaNs and small N."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return 0.0, 0.0
    m  = float(np.mean(a))
    if n == 1:
        return m, 0.0
    se = float(np.std(a, ddof=1)) / np.sqrt(n)
    return m, 1.96 * se

def plot_macro_metrics_vs_tau(macro_df: pd.DataFrame, per_dev_df: pd.DataFrame, out_dir: Path):
    """Mean±95% CI across devices for each τ (keeps colors)."""
    g = (per_dev_df
         .assign(tau=pd.to_numeric(per_dev_df["tau"], errors='coerce'))
         .dropna(subset=["tau"]))
    rows = []
    for t, df in g.groupby("tau"):
        mp, cp = _mean_ci_95(df["precision"].values) if "precision" in df else (np.nan, 0.0)
        mr, cr = _mean_ci_95(df["recall"].values)    if "recall"    in df else (np.nan, 0.0)
        mf, cf = _mean_ci_95(df["f1"].values)        if "f1"        in df else (np.nan, 0.0)
        rows.append(dict(tau=float(t), p=mp, p_ci=cp, r=mr, r_ci=cr, f=mf, f_ci=cf))
    if not rows:
        print("[WARN] No rows for CI plot — check per_dev_df content."); return
    mci = pd.DataFrame(rows).sort_values("tau")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    col_p, col_r, col_f = _color(0), _color(1), _color(2)

    ax.plot(mci["tau"], mci["p"], marker="o", label="Precision", color=col_p)
    ax.fill_between(mci["tau"], mci["p"]-mci["p_ci"], mci["p"]+mci["p_ci"], alpha=0.15, color=col_p)

    ax.plot(mci["tau"], mci["r"], marker="o", label="Recall", color=col_r)
    ax.fill_between(mci["tau"], mci["r"]-mci["r_ci"], mci["r"]+mci["r_ci"], alpha=0.15, color=col_r)

    ax.plot(mci["tau"], mci["f"], marker="o", label="F1", color=col_f)
    ax.fill_between(mci["tau"], mci["f"]-mci["f_ci"], mci["f"]+mci["f_ci"], alpha=0.15, color=col_f)

    ax.set_ylim(0, 1.0)
    _apply_axes_style(ax, x_label="τ (days)", y_label="Score")
    _legend(ax, loc='upper left', ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_metrics_vs_tau.png", dpi=300)
    plt.close(fig)

def _multi_bar(df_wide: pd.DataFrame, taus: List[int], out_path: Path, y_label="Score"):
    if df_wide.empty: return
    devices = list(df_wide.index)
    x = np.arange(len(devices))
    w = 0.8 / max(1, len(taus))
    fig, ax = plt.subplots(figsize=(max(10, len(devices)*0.6), 4.8))
    for i, tau in enumerate(taus):
        y = df_wide.get(tau, pd.Series(0, index=devices)).values
        ax.bar(
            x + (i - len(taus)/2)*w + w/2,
            y, width=w, label=f"τ={tau}",
            color=_color(i), hatch=_hatch(i), **_BAR_KW
        )
    ax.set_xticks(x); ax.set_xticklabels(devices, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    _apply_axes_style(ax, x_label="Device (top-N by mean F1)", y_label=y_label)
    _legend(ax, loc='upper right', ncol=min(6, len(taus)))
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)

def plot_per_device_multi_bars(per_dev_df: pd.DataFrame, taus: List[int], out_dir: Path, top_n_devices: int):
    if per_dev_df.empty: return
    tops = (per_dev_df.groupby("device")["f1"].mean()
            .sort_values(ascending=False).head(top_n_devices).index.tolist())
    sel = per_dev_df[per_dev_df["device"].isin(tops)]
    def pivot(metric):
        return sel.pivot_table(index="device", columns="tau", values=metric, aggfunc="mean").reindex(tops)
    _multi_bar(pivot("precision"), taus, out_dir / "per_device_precision_multi_bar.png", y_label="Precision")
    _multi_bar(pivot("recall"),    taus, out_dir / "per_device_recall_multi_bar.png",    y_label="Recall")
    _multi_bar(pivot("f1"),        taus, out_dir / "per_device_f1_multi_bar.png",        y_label="F1")

def _heatmap(matrix: pd.DataFrame, out_path: Path, y_label="Device", x_label="τ (days)", vmin=0.0, vmax=1.0):
    if matrix.empty: return
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1]*1.1), max(6, matrix.shape[0]*0.45)))
    im = ax.imshow(matrix.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1])); ax.set_xticklabels(matrix.columns.astype(str), fontsize=_TICK_SIZE)
    ax.set_yticks(np.arange(matrix.shape[0])); ax.set_yticklabels(matrix.index, fontsize=_TICK_SIZE)
    ax.set_xlabel(x_label, **_font_lbl); ax.set_ylabel(y_label, **_font_lbl)
    cbar = fig.colorbar(im, ax=ax); cbar.ax.set_ylabel("Score", rotation=270, labelpad=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)

def plot_heatmaps(per_dev_df: pd.DataFrame, taus: List[int], out_dir: Path):
    if per_dev_df.empty: return
    f1m = per_dev_df.pivot_table(index="device", columns="tau", values="f1",        aggfunc="mean").fillna(0)
    prm = per_dev_df.pivot_table(index="device", columns="tau", values="precision", aggfunc="mean").fillna(0)
    rcm = per_dev_df.pivot_table(index="device", columns="tau", values="recall",    aggfunc="mean").fillna(0)
    _heatmap(f1m, out_dir / "per_device_f1_heatmap.png")
    _heatmap(prm, out_dir / "per_device_precision_heatmap.png")
    _heatmap(rcm, out_dir / "per_device_recall_heatmap.png")

def plot_overlap_stacked_bars(overlap_by_tau: pd.DataFrame, out_dir: Path):
    if overlap_by_tau.empty: return
    df = overlap_by_tau.sort_values("tau")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = (_color(0), _color(1), _color(2))
    ax.bar(x, df["both"],        label="Edges in both", color=colors[0], hatch=_hatch(0), **_BAR_KW, zorder=2)
    ax.bar(x, df["lave_only"],   bottom=df["both"], label="LaVE only",  color=colors[1], hatch=_hatch(1), **_BAR_KW, zorder=2)
    ax.bar(x, df["pcmci_only"],  bottom=df["both"] + df["lave_only"], label="PCMCI only",
           color=colors[2], hatch=_hatch(2), **_BAR_KW, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(df["tau"].astype(str))
    _apply_axes_style(ax, x_label="τ (days)", y_label="# edges")
    _legend(ax, loc='upper left', ncol=1)
    fig.tight_layout(); fig.savefig(out_dir / "edge_overlap_stacked_bars_by_tau.png", dpi=300); plt.close(fig)

def plot_f1_boxplot(per_dev_df: pd.DataFrame, taus: List[int], out_dir: Path):
    if per_dev_df.empty:
        return

    # Collect per-τ arrays
    data = [per_dev_df[per_dev_df["tau"]==t]["f1"].dropna().values for t in taus]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # patch_artist=True lets us color + hatch the boxes
    try:
        bp = ax.boxplot(data, tick_labels=[str(t) for t in taus], patch_artist=True)  # mpl ≥3.9
    except TypeError:
        bp = ax.boxplot(data, labels=[str(t) for t in taus], patch_artist=True)       # older mpl

    # Style each box with a color + hatch (patterns)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(_color(i))
        box.set_hatch(_hatch(i))         # add visible pattern like *, /, x, etc.
        box.set_edgecolor('black')
        box.set_linewidth(1.2)

    # Whiskers, caps, medians: keep readable with dark lines
    for whisk in bp['whiskers']:
        whisk.set_color('black'); whisk.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('black'); cap.set_linewidth(1.0)
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.6)
    for flier in bp.get('fliers', []):
        flier.set_markeredgecolor('black')

    _apply_axes_style(ax, x_label="τ (days)", y_label="F1")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_distribution_boxplot_by_tau.png", dpi=300)
    plt.close(fig)


def plot_complexity_scatter(per_dev_df: pd.DataFrame, out_dir: Path):
    if per_dev_df.empty or "pcmci_edges" not in per_dev_df.columns: return
    df = per_dev_df.dropna(subset=["pcmci_edges"]).copy()
    if df.empty: return
    taus = sorted(df["tau"].unique().tolist())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, t in enumerate(taus):
        dft = df[df["tau"]==t]
        ax.scatter(dft["pcmci_edges"], dft["f1"], label=f"τ={t}", color=_color(i), edgecolor='black', linewidth=0.6)
    _apply_axes_style(ax, x_label="# PCMCI edges (per device)", y_label="F1 (LaVE vs PCMCI)")
    _legend(ax, loc='lower right', ncol=min(6, len(taus)))
    fig.tight_layout(); fig.savefig(out_dir / "complexity_scatter_by_tau.png", dpi=300); plt.close(fig)

def plot_hits_at_k_by_tau(hits_df: pd.DataFrame, out_dir: Path):
    if hits_df.empty: return
    piv = hits_df.pivot_table(index="tau", columns="K", values="hits", aggfunc="sum").fillna(0).sort_index()
    x = np.arange(len(piv))
    ks = list(piv.columns)
    w = 0.8 / max(1, len(ks))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, k in enumerate(ks):
        ax.bar(x + (i - len(ks)/2)*w + w/2,
               piv[k].values, width=w, label=f"K={k}",
               color=_color(i), hatch=_hatch(i), **_BAR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.astype(str))
    _apply_axes_style(ax, x_label="τ (days)", y_label="Hits")
    _legend(ax, loc='upper right', ncol=min(5, len(ks)))
    fig.tight_layout()
    fig.savefig(out_dir / "hits_at_k_by_tau.png", dpi=300)
    plt.close(fig)

# ------------------------------ PIPELINE -------------------------
def cached_tau_outputs_exist(tau_dir: Path) -> bool:
    need = ["edge_overlap_counts.csv", "computed_metrics_from_overlap.csv", "hits_at_k.csv"]
    return tau_dir.exists() and all((tau_dir / f).exists() for f in need)

def run_for_tau(devices_root: Path, log_csv: Path, base_out: Path,
                tau: int, alpha: float, min_days_present: int,
                hits_k: List[int], resume=True, force_pcmci=False, force_eval=False):
    tau_dir = base_out / f"tau_{tau}"
    plots_dir = tau_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pcmci_dir = base_out / f"pcmci_tau_{tau}"
    print(f"\n=== τ = {tau} ===")
    if not (resume and pcmci_artifacts_exist(pcmci_dir)) or force_pcmci:
        train_pcmci_per_device(devices_root, pcmci_dir,
                               tau_max=tau, alpha=alpha, min_days_present=min_days_present,
                               resume=not force_pcmci)
    else:
        print(f"[SKIP] Using existing PCMCI artifacts in {pcmci_dir}")

    lave_csv = base_out / "lave_edges_normalized.csv"
    if not lave_csv.exists():
        print("[INFO] LaVE edges not found; extracting once from log …")

    lave  = load_lave_edges(lave_csv) if lave_csv.exists() else pd.DataFrame()
    pcmci = load_pcmci_edges(pcmci_dir)

    if lave.empty:
        raise SystemExit("LaVE edges empty — check merged log & 'Chain Data' parsing.")
    if pcmci.empty:
        print("[WARN] PCMCI produced no edges for this τ.")

    if resume and cached_tau_outputs_exist(tau_dir) and not force_eval:
        print(f"[SKIP] Metrics already computed for τ={tau}. Using cached CSVs.")
        counts  = pd.read_csv(tau_dir / "edge_overlap_counts.csv")
        metrics = pd.read_csv(tau_dir / "computed_metrics_from_overlap.csv")
        hitsdf  = pd.read_csv(tau_dir / "hits_at_k.csv")
        for df in (counts, metrics, hitsdf):
            if "tau" not in df.columns: df["tau"] = tau
        macro = metrics[["precision","recall","f1"]].mean().to_dict()
        print(f"Macro-avg (τ={tau}): {macro}")
        return counts.assign(tau=tau), metrics.assign(tau=tau), hitsdf.assign(tau=tau)

    # Fresh evaluation
    counts  = overlap_counts(lave, pcmci)
    metrics = metrics_from_counts(counts)
    hitsdf  = hits_at_k(lave, pcmci, ks=tuple(hits_k))

    counts.to_csv(tau_dir / "edge_overlap_counts.csv", index=False)
    metrics.to_csv(tau_dir / "computed_metrics_from_overlap.csv", index=False)
    hitsdf.to_csv(tau_dir / "hits_at_k.csv", index=False)

    # Quick per-τ bar for record (colors + hatches)
    order = counts.sort_values("pcmci_edges", ascending=False)["device"].tolist()
    a = counts.set_index("device").loc[order] if len(order)>0 else counts.set_index("device")
    if not a.empty:
        x = np.arange(len(a)); w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(a)*0.5), 3.6))
        ax.bar(x - w, a["lave_only"], width=w, label="LaVE only",  color=_color(0), hatch=_hatch(0), **_BAR_KW)
        ax.bar(x,      a["both"],      width=w, label="Both",       color=_color(1), hatch=_hatch(1), **_BAR_KW)
        ax.bar(x + w,  a["pcmci_only"],width=w, label="PCMCI only", color=_color(2), hatch=_hatch(2), **_BAR_KW)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
        _apply_axes_style(ax, x_label="Device", y_label="# edges")
        _legend(ax, loc='upper right')
        fig.tight_layout(); fig.savefig(plots_dir / "fig_overlap_bars.png", dpi=180); plt.close(fig)

    macro = metrics[["precision","recall","f1"]].mean().to_dict()
    print(f"Macro-avg (τ={tau}): {macro}")
    return counts.assign(tau=tau), metrics.assign(tau=tau), hitsdf.assign(tau=tau)

def write_tables(out_root: Path,
                 macro_df: pd.DataFrame,
                 per_dev_df: pd.DataFrame,
                 overlap_by_tau: pd.DataFrame,
                 hits_by_tau: pd.DataFrame):
    tbl_dir = out_root / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    macro_df.sort_values("tau").to_csv(tbl_dir / "macro_by_tau.csv", index=False)
    per_dev_df = per_dev_df.sort_values(["device","tau"])
    per_dev_df.to_csv(tbl_dir / "per_device_metrics_by_tau.csv", index=False)

    def wide(metric, fname):
        w = per_dev_df.pivot_table(index="device", columns="tau", values=metric, aggfunc="mean")
        w = w.reindex(sorted(w.index), axis=0).reindex(sorted(w.columns), axis=1)
        w.to_csv(tbl_dir / fname); return w
    w_f1  = wide("f1",        "per_device_metrics_by_tau_wide_f1.csv")
    w_pre = wide("precision", "per_device_metrics_by_tau_wide_prec.csv")
    w_rec = wide("recall",    "per_device_metrics_by_tau_wide_rec.csv")

    overlap_by_tau.sort_values("tau").to_csv(tbl_dir / "overlap_by_tau.csv", index=False)
    hits_by_tau.sort_values(["tau","K"]).to_csv(tbl_dir / "hits_at_k_by_tau.csv", index=False)

    with open(tbl_dir / "README_tables.md", "w") as fh:
        fh.write("# Tables summary\n\n")
        fh.write("- macro_by_tau.csv: macro averages (precision, recall, F1) across devices for each τ.\n")
        fh.write("- per_device_metrics_by_tau.csv: per-device metrics per τ (long format).\n")
        fh.write("- per_device_metrics_by_tau_wide_*.csv: device × τ matrices for F1/precision/recall.\n")
        fh.write("- overlap_by_tau.csv: sum of overlaps across devices per τ.\n")
        fh.write("- hits_at_k_by_tau.csv: hits and hit_rate per τ and K.\n")

    return w_f1, w_pre, w_rec

def make_paper_plots(out_root: Path,
                     macro_df: pd.DataFrame,
                     per_dev_df: pd.DataFrame,
                     overlap_by_tau: pd.DataFrame,
                     hits_by_tau: pd.DataFrame,
                     taus: List[int],
                     top_n_devices: int):
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_macro_metrics_vs_tau(macro_df, per_dev_df, plots_dir)   # CI + colors
    plot_per_device_multi_bars(per_dev_df, taus, plots_dir, top_n_devices)
    plot_heatmaps(per_dev_df, taus, plots_dir)
    plot_f1_boxplot(per_dev_df, taus, plots_dir)
    plot_overlap_stacked_bars(overlap_by_tau, plots_dir)
    plot_hits_at_k_by_tau(hits_by_tau, plots_dir)
    plot_complexity_scatter(per_dev_df, plots_dir)

def main():
    ap = argparse.ArgumentParser(description="τ sweep with tables & plots (PCMCI+ vs LaVE) — resume-aware")
    ap.add_argument("--devices_root", required=True, help="root containing Device*/Patient_*.csv")
    ap.add_argument("--log_csv", required=True, help="LaVE merged log CSV with 'Chain Data' column")
    ap.add_argument("--out_root", required=True, help="output root folder")
    ap.add_argument("--taus", required=True, help="comma-separated, e.g., 7,14,30,60,90")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min_days_present", type=int, default=1)
    ap.add_argument("--hits_k", default="1,3,5")
    ap.add_argument("--top_n_devices", type=int, default=12, help="for multi-bar plots")
    # resume/force toggles
    ap.add_argument("--resume", action="store_true", default=True, help="skip steps already done (default)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force_pcmci", action="store_true", help="retrain PCMCI even if artifacts exist")
    ap.add_argument("--force_lave", action="store_true", help="re-extract LaVE edges even if CSV exists")
    ap.add_argument("--force_eval", action="store_true", help="recompute per-τ metrics even if cached")

    args = ap.parse_args()

    devices_root = Path(args.devices_root)
    log_csv      = Path(args.log_csv)
    out_root     = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    taus         = [int(x) for x in str(args.taus).split(",") if x.strip()]
    hits_k       = [int(x) for x in str(args.hits_k).split(",") if x.strip()]

    # Extract LaVE edges once (resume-aware)
    lave_csv = out_root / "lave_edges_normalized.csv"
    extract_lave_edges_from_log(log_csv, lave_csv, resume=(args.resume and not args.force_lave))

    # Run τ sweep
    macro_rows, perdev_rows, overlap_rows, hits_rows = [], [], [], []
    for tau in taus:
        counts, metrics, hits = run_for_tau(devices_root, log_csv, out_root,
                                            tau=tau,
                                            alpha=args.alpha,
                                            min_days_present=args.min_days_present,
                                            hits_k=hits_k,
                                            resume=args.resume,
                                            force_pcmci=args.force_pcmci,
                                            force_eval=args.force_eval)
        macro = metrics[["precision","recall","f1"]].mean().to_dict(); macro["tau"] = tau
        macro_rows.append(macro)

        c_small = counts[["device","pcmci_edges"]].copy().drop_duplicates()
        m = metrics.merge(c_small, on="device", how="left"); m["tau"] = tau
        perdev_rows.append(m)

        s = counts[["lave_only","both","pcmci_only"]].sum().to_dict(); s["tau"] = tau
        overlap_rows.append(s)

        hits_rows.append(hits)

    macro_df      = pd.DataFrame(macro_rows).sort_values("tau")
    per_dev_df    = pd.concat(perdev_rows, ignore_index=True)
    overlap_by_tau= pd.DataFrame(overlap_rows).sort_values("tau")
    hits_by_tau   = pd.concat(hits_rows, ignore_index=True).sort_values(["tau","K"])

    write_tables(out_root, macro_df, per_dev_df, overlap_by_tau, hits_by_tau)
    make_paper_plots(out_root, macro_df, per_dev_df, overlap_by_tau, hits_by_tau, taus, args.top_n_devices)

    print("\n=== SUMMARY (macro by τ) ===")
    print(macro_df.to_string(index=False))
    print("\nTables →", (out_root / "tables").resolve())
    print("Plots  →", (out_root / "plots").resolve())
    print("Done.")

if __name__ == "__main__":
    main()
