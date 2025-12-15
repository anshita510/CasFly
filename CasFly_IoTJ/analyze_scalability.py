#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_scalability.py

Reads the aggregated experiments.csv produced by scalability_sweep.py and
generates publication-quality plots and summary tables.

Analyses performed:
  - Mean total time with 95% bootstrap CI vs device count, grouped by
    network profile and by topology.
  - P95 latency, success rate, message count, and byte count vs device count.
  - Log-log scaling plot (for estimating the power-law exponent).
  - Log-log OLS regression slope table (scaling exponent per profile).

Input:
    experiments.csv  -- from scalability_sweep.py (or repeat_scalability_for_ci.py)
    Required columns (detected flexibly): device_count, profile, topology,
    and at least one of total_time_s / total_time_ms / wall_time_s.

Outputs (under --out/):
    tables/regression_slopes.csv
    plots/time_vs_devices_by_profile_ci.png
    plots/time_vs_devices_by_topology_ci.png
    plots/time_vs_devices_by_profile_topology_ci.png
    plots/p95latency_vs_devices_by_profile.png
    plots/success_vs_devices_by_profile.png
    plots/msgs_vs_devices_by_profile.png
    plots/bytes_vs_devices_by_profile.png
    plots/loglog_time_vs_devices_by_profile.png
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Optional, Dict, Tuple, List

# ---------------------------- Styling ----------------------------
font1 = {'family': 'Times New Roman', 'size': 28}  # axis labels
font2 = {'family': 'Times New Roman', 'size': 26}  # x tick labels
font3 = {'family': 'Times New Roman', 'size': 24}  # legend / y tick labels
LINE_WIDTH  = 2.6
MARKER_SIZE = 8
CI_ALPHA    = 0.18

def _apply_axes_style(ax, y_label, percent_y=False):
    """Apply consistent axis styling; no plot titles (paper convention)."""
    ax.set_ylabel(y_label, **font1)
    ax.tick_params(axis='y', labelsize=font3['size'])
    ax.tick_params(axis='x', labelsize=font2['size'])
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, min(1.05, max(0.05, ymax)))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# ---------------------------- Legend label formatting ----------------------------
ABBREV_UPPER = {"lan", "wan", "er", "nsclc"}  # tokens that should be ALL-CAPS

def _fmt_token(tok: str) -> str:
    t = str(tok).strip()
    low = t.lower()
    if low in ABBREV_UPPER:
        return low.upper()
    return t.capitalize()

def format_label(label: str) -> str:
    """
    Format a legend label: uppercase known abbreviations, Title-Case others.
    Handles separators: '/', '-', '_'.  E.g. 'lan/er' -> 'LAN/ER'.
    """
    s = str(label)
    for sep in ("/", "-", "_"):
        if sep in s:
            parts = s.split(sep)
            return sep.join(_fmt_token(p) for p in parts)
    return _fmt_token(s)

def _styled_legend(ax, title):
    ax.legend(title=format_label(title), fontsize=font3['size'],
              title_fontsize=font3['size'], loc="best")

# ---------------------------- Column detection ----------------------------
def canon(s: str) -> str:
    """Canonicalise a column name to lowercase alphanumeric + underscores."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

CANDIDATES = {
    "run_id":         ["run_id","run","exp_id","experiment_id"],
    "device_count":   ["device_count","n_devices","num_devices","devices"],
    "topology":       ["topology","graph","net_topo","topo"],
    "profile":        ["profile","net_profile","network_profile","profile_name"],
    "wall_time_s":    ["wall_time_s","wall","time_sec"],
    "median_T":       ["median_t","median_total_time_s","median_time_s"],
    "total_time_s":   ["total_time_s","total_time_sec","total_time","t_total","total_time_t",
                       "total_time__t","time_s","time_sec","wall_time_s","wall","t"],
    "total_time_ms":  ["total_time_ms","time_ms","wall_time_ms"],
    "p95_latency_ms": ["p95_latency_ms","latency_p95_ms","p95_ms","t_p95_ms","p95_latency"],
    "p99_latency_ms": ["p99_latency_ms","latency_p99_ms","p99_ms","t_p99_ms","p99_latency"],
    "success_rate":   ["success_rate","success","ok_rate","completed_rate"],
    "messages":       ["messages","message_count","msg_count","msgs"],
    "bytes":          ["bytes","bytes_sent","bytes_total","total_bytes","tx_bytes"],
}

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    """Map logical metric names to the actual column names present in df."""
    colmap   = {canon(c): c for c in df.columns}
    resolved = {}
    for key, cands in CANDIDATES.items():
        for c in cands:
            if canon(c) in colmap:
                resolved[key] = colmap[canon(c)]
                break
    # Explicit fallback for median_T
    if "median_T" not in resolved and "median_T" in df.columns:
        resolved["median_T"] = "median_T"
    # Extra fallback for total_time_s by scanning column names
    if "total_time_s" not in resolved:
        for c in df.columns:
            if canon(c) in ["total_time_t","total_time__t","t","total_time"]:
                resolved["total_time_s"] = c
                break
    return resolved

def get_series(df, colnames: Dict[str,str], key: str):
    if key in colnames: return df[colnames[key]]
    return None

def to_float_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None: return None
    def coerce(x):
        try: return float(str(x).strip().replace(",",""))
        except Exception: return np.nan
    return s.map(coerce)

def ensure_time_seconds(df, colnames: Dict[str,str]) -> pd.Series:
    """Return a float Series of total time in seconds, trying multiple columns."""
    s_sec = get_series(df, colnames, "total_time_s")
    if s_sec is not None: return to_float_series(s_sec)
    s_ms = get_series(df, colnames, "total_time_ms")
    if s_ms is not None: return to_float_series(s_ms) / 1000.0
    ws = get_series(df, colnames, "wall_time_s")
    if ws is not None: return to_float_series(ws)
    raise ValueError("Could not locate a total time metric (total_time_s / total_time_ms / wall_time_s).")

# ---------------------------- Bootstrap CI (mean) ----------------------------
def bootstrap_ci_mean(
    v: np.ndarray, B: int = 3000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float, int]:
    """Return (mean, CI_lo, CI_hi, n) using percentile bootstrap."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0: return np.nan, np.nan, np.nan, 0
    mu = float(np.mean(v))
    if n == 1: return mu, mu, mu, 1
    rng  = np.random.default_rng(seed)
    boots = [np.mean(v[rng.integers(0, n, n)]) for _ in range(B)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return mu, float(lo), float(hi), n

def agg_with_ci_mean(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Group df by group_cols, compute mean and 95% bootstrap CI for value_col."""
    rows = []
    for key, sub in df.groupby(group_cols):
        vals = sub[value_col].values
        mean, lo, hi, n = bootstrap_ci_mean(vals, B=3000, alpha=0.05, seed=123)
        rec = {c: k for c, k in zip(group_cols, (key if isinstance(key, tuple) else (key,)))}
        rec.update(dict(mean=mean, lo=lo, hi=hi, n=n))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

# ---------------------------- Plotting with CI ----------------------------
def _lineplot_ci(ax, df_ci, xcol, groupcol, ylabel, legend_title):
    """Line plot with shaded 95% CI ribbon for each group."""
    if df_ci.empty: return
    groups  = [g for g in sorted(df_ci[groupcol].unique().tolist()) if isinstance(g, str) or pd.notna(g)]
    markers = ["o","s","^","D","P","v","<",">","X"]
    for i, g in enumerate(groups):
        sub = df_ci[df_ci[groupcol] == g].sort_values(xcol)
        ax.plot(sub[xcol], sub["mean"],
                marker=markers[i % len(markers)],
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label=format_label(g))
        ax.fill_between(sub[xcol], sub["lo"], sub["hi"], alpha=CI_ALPHA)
    ax.set_xlabel("No. of Devices", **font1)
    _apply_axes_style(ax, ylabel, percent_y=False)
    _styled_legend(ax, legend_title)

# ---------------------------- Log-log slope (using means) ----------------------------
def fit_loglog_slope(n: np.ndarray, t: np.ndarray) -> Tuple[float,float,float]:
    """
    Fit log10(t) ~ intercept + slope*log10(n) by OLS.
    Returns (slope, intercept, R²) — slope is the power-law scaling exponent.
    """
    mask = (n > 0) & (t > 0) & np.isfinite(n) & np.isfinite(t)
    if mask.sum() < 2: return np.nan, np.nan, np.nan
    log_n = np.log10(n[mask])
    log_t = np.log10(t[mask])
    # Design matrix: [1, log_n] so beta = [intercept, slope]
    A = np.vstack([np.ones_like(log_n), log_n]).T
    beta, *_ = np.linalg.lstsq(A, log_t, rcond=None)
    intercept, slope = beta[0], beta[1]
    log_t_pred = A @ beta
    ss_res = np.sum((log_t - log_t_pred)**2)
    ss_tot = np.sum((log_t - np.mean(log_t))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)

# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="artifacts/newscalability/experiments.csv")
    ap.add_argument("--out", default="artifacts/newscalability/analysis")
    args = ap.parse_args()

    out_root = Path(args.out)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    df       = pd.read_csv(args.csv)
    colnames = detect_columns(df)

    # Resolve device_count column; infer from run_id if needed
    devc_col = colnames.get("device_count", None)
    if devc_col is None:
        rid = colnames.get("run_id", None)
        if rid:
            def infer_n(s):
                m = re.search(r"[nN](\d+)", str(s))
                return int(m.group(1)) if m else np.nan
            df["device_count_norm"] = df[rid].map(infer_n)
        else:
            raise ValueError("device_count column not found and cannot be inferred from run_id.")
    else:
        df["device_count_norm"] = pd.to_numeric(df[devc_col], errors="coerce")

    # Total time in seconds (primary analysis metric)
    df["time_s"] = ensure_time_seconds(df, colnames)

    # Optional extras: fall back to all-NaN series if the column is absent
    p95_latency_series  = to_float_series(get_series(df, colnames, "p95_latency_ms")) or pd.Series([np.nan]*len(df))
    success_rate_series = to_float_series(get_series(df, colnames, "success_rate"))   or pd.Series([np.nan]*len(df))
    messages_series     = to_float_series(get_series(df, colnames, "messages"))       or pd.Series([np.nan]*len(df))
    bytes_series        = to_float_series(get_series(df, colnames, "bytes"))          or pd.Series([np.nan]*len(df))
    df["p95_latency_ms"] = p95_latency_series.values
    df["success_rate"]   = success_rate_series.values
    df["messages"]       = messages_series.values
    df["bytes"]          = bytes_series.values

    # Normalise grouper columns
    topo_col = colnames.get("topology", None)
    prof_col = colnames.get("profile", None)
    df["profile_norm"]  = (df[prof_col].astype(str).str.strip().str.lower() if prof_col else "unknown")
    df["topology_norm"] = (df[topo_col].astype(str).str.strip().str.lower() if topo_col else "unknown")

    # ------------------- summaries with mean + CI -------------------
    overall_ci = agg_with_ci_mean(df, ["device_count_norm"], "time_s")
    by_prof_ci = agg_with_ci_mean(df, ["profile_norm","device_count_norm"], "time_s")
    by_topo_ci = agg_with_ci_mean(df, ["topology_norm","device_count_norm"], "time_s")
    by_pt_ci   = agg_with_ci_mean(df, ["profile_norm","topology_norm","device_count_norm"], "time_s")

    # ------------------- regression on scaling (log-log, using means) -------------------
    rows = []
    for key, sub in by_prof_ci.groupby("profile_norm"):
        n = sub["device_count_norm"].values.astype(float)
        t = sub["mean"].values.astype(float)
        b, a, r2 = fit_loglog_slope(n, t)
        rows.append(dict(group="profile", name=key, slope_b=b, intercept_a=a, r2=r2, n_points=len(sub)))
    reg = pd.DataFrame(rows).sort_values(["group","name"])
    reg.to_csv(out_root / "tables" / "regression_slopes.csv", index=False)

    # ------------------- plots -------------------
    plots = out_root / "plots"; plots.mkdir(parents=True, exist_ok=True)

    if not by_prof_ci.empty:
        fig, ax = plt.subplots(figsize=(12,8))
        _lineplot_ci(ax, by_prof_ci, "device_count_norm", "profile_norm",
                     ylabel="Total Time (s)", legend_title="Profile")
        fig.tight_layout(); fig.savefig(plots/"time_vs_devices_by_profile_ci.png", dpi=300); plt.close(fig)

    if not by_topo_ci.empty:
        fig, ax = plt.subplots(figsize=(12,8))
        _lineplot_ci(ax, by_topo_ci, "device_count_norm", "topology_norm",
                     ylabel="Total Time (s)", legend_title="Topology")
        fig.tight_layout(); fig.savefig(plots/"time_vs_devices_by_topology_ci.png", dpi=300); plt.close(fig)

    if not by_pt_ci.empty:
        fig, ax = plt.subplots(figsize=(12,8))
        temp = by_pt_ci.copy()
        # Combine profile and topology into a single pre-formatted label
        temp["label"] = (temp["profile_norm"].astype(str) + "/" +
                         temp["topology_norm"].astype(str)).map(format_label)
        temp = temp.rename(columns={"label":"combo"})
        _lineplot_ci(ax, temp, "device_count_norm", "combo",
                     ylabel="Total Time (s)", legend_title="Profile/Topology")
        fig.tight_layout(); fig.savefig(plots/"time_vs_devices_by_profile_topology_ci.png", dpi=300); plt.close(fig)

    # P95 latency (median across reps)
    latency_by_profile = (df.groupby(["profile_norm","device_count_norm"])
                            .agg(p95_latency_median_ms=("p95_latency_ms","median")).reset_index())
    if "p95_latency_median_ms" in latency_by_profile.columns and not latency_by_profile["p95_latency_median_ms"].isna().all():
        fig, ax = plt.subplots(figsize=(12,8))
        markers = ["o","s","^","D","P","v","<",">","X"]
        for i, (profile_name, sub) in enumerate(latency_by_profile.groupby("profile_norm")):
            sub = sub.sort_values("device_count_norm")
            ax.plot(sub["device_count_norm"], sub["p95_latency_median_ms"],
                    marker=markers[i % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=format_label(profile_name))
        ax.set_xlabel("No. of Devices", **font1)
        _apply_axes_style(ax, "Median P95 latency (ms)")
        _styled_legend(ax, "Profile")
        fig.tight_layout(); fig.savefig(plots/"p95latency_vs_devices_by_profile.png", dpi=300); plt.close(fig)

    # Success rate
    success_by_profile = (df.groupby(["profile_norm","device_count_norm"])
                            .agg(success_median=("success_rate","median")).reset_index())
    if "success_median" in success_by_profile.columns and not success_by_profile["success_median"].isna().all():
        fig, ax = plt.subplots(figsize=(12,8))
        markers = ["o","s","^","D","P","v","<",">","X"]
        for i, (profile_name, sub) in enumerate(success_by_profile.groupby("profile_norm")):
            sub = sub.sort_values("device_count_norm")
            ax.plot(sub["device_count_norm"], sub["success_median"],
                    marker=markers[i % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=format_label(profile_name))
        ax.set_xlabel("No. of Devices", **font1)
        _apply_axes_style(ax, "Median success rate", percent_y=True)
        ax.set_ylim(0, 1.05)
        _styled_legend(ax, "Profile")
        fig.tight_layout(); fig.savefig(plots/"success_vs_devices_by_profile.png", dpi=300); plt.close(fig)

    # Message count
    messages_by_profile = (df.groupby(["profile_norm","device_count_norm"])
                             .agg(msgs_median=("messages","median")).reset_index())
    if "msgs_median" in messages_by_profile.columns and not messages_by_profile["msgs_median"].isna().all():
        fig, ax = plt.subplots(figsize=(12,8))
        markers = ["o","s","^","D","P","v","<",">","X"]
        for i, (profile_name, sub) in enumerate(messages_by_profile.groupby("profile_norm")):
            sub = sub.sort_values("device_count_norm")
            ax.plot(sub["device_count_norm"], sub["msgs_median"],
                    marker=markers[i % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=format_label(profile_name))
        ax.set_xlabel("No. of Devices", **font1)
        _apply_axes_style(ax, "Median messages")
        _styled_legend(ax, "Profile")
        fig.tight_layout(); fig.savefig(plots/"msgs_vs_devices_by_profile.png", dpi=300); plt.close(fig)

    # Byte throughput
    bytes_by_profile = (df.groupby(["profile_norm","device_count_norm"])
                          .agg(bytes_median=("bytes","median")).reset_index())
    if "bytes_median" in bytes_by_profile.columns and not bytes_by_profile["bytes_median"].isna().all():
        fig, ax = plt.subplots(figsize=(12,8))
        markers = ["o","s","^","D","P","v","<",">","X"]
        for i, (profile_name, sub) in enumerate(bytes_by_profile.groupby("profile_norm")):
            sub = sub.sort_values("device_count_norm")
            ax.plot(sub["device_count_norm"], sub["bytes_median"],
                    marker=markers[i % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=format_label(profile_name))
        ax.set_xlabel("No. of Devices", **font1)
        _apply_axes_style(ax, "Median bytes")
        _styled_legend(ax, "Profile")
        fig.tight_layout(); fig.savefig(plots/"bytes_vs_devices_by_profile.png", dpi=300); plt.close(fig)

    # Log-log scaling plot using means
    if not by_prof_ci.empty:
        fig, ax = plt.subplots(figsize=(12,8))
        markers = ["o","s","^","D","P","v","<",">","X"]
        for i, (p, sub) in enumerate(by_prof_ci.groupby("profile_norm")):
            x   = sub["device_count_norm"].values.astype(float)
            y   = sub["mean"].values.astype(float)
            msk = (x>0) & (y>0) & np.isfinite(x) & np.isfinite(y)
            if msk.sum() < 2: continue
            ax.plot(np.log10(x[msk]), np.log10(y[msk]),
                    marker=markers[i % len(markers)],
                    linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=format_label(p))
        ax.set_xlabel("log10(Device count)", **font1)
        _apply_axes_style(ax, "log10(Mean total Time [s])")
        _styled_legend(ax, "Profile")
        fig.tight_layout(); fig.savefig(plots/"loglog_time_vs_devices_by_profile.png", dpi=300); plt.close(fig)

    print(f"Wrote tables -> {(out_root/'tables').resolve()}")
    print(f"Wrote plots  -> {(out_root/'plots').resolve()}")
    print("Note: time plots show MEAN total time with 95% bootstrap CI.")

if __name__ == "__main__":
    main()
