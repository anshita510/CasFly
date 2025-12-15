#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_scalability_ci.py

Builds 95% CI (shaded ribbon) scalability plots and summary tables from
experiments.csv produced by repeat_scalability_for_ci.py.

Inputs:
    experiments.csv  -- required columns: device_count, profile, topology,
                        seed, total_time_s
                        optional: n_hops, msgs_total, bytes_total,
                                  cpu_util, ram_mb, energy_j

Outputs (default under artifacts/scalability_multiple/plots_ci/):
    time_vs_devices_by_profile.png
    time_vs_devices_by_topology.png
    time_vs_devices_by_profile_topology.png
    loglog_time_vs_devices_by_profile.png
    loglog_time_vs_devices_by_topology.png
    summary_by_profile.csv
    summary_by_topology.csv
    summary_by_profile_topology.csv
    regression_slopes.csv
    summary_overall.csv

Usage:
    python3 plot_scalability_ci.py \\
      --experiments artifacts/scalability_multiple/experiments.csv \\
      --outdir artifacts/scalability_multiple/plots_ci
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ci95(n, std):
    """95% CI for mean using normal approximation: 1.96 * sd / sqrt(n)."""
    return 1.96 * (std / np.sqrt(np.maximum(n, 1)))

def summarize(df, group_cols, value_col):
    """Group by group_cols, compute count/mean/std/CI for value_col."""
    agg = (df.groupby(group_cols)[value_col]
             .agg(['count', 'mean', 'std'])
             .reset_index()
             .rename(columns={'count':'n', 'mean':'mean', 'std':'std'}))
    agg['ci'] = _ci95(agg['n'].values, agg['std'].values)
    agg['lo'] = agg['mean'] - agg['ci']
    agg['hi'] = agg['mean'] + agg['ci']
    return agg

def plot_ribbons(x, y_mean, y_lo, y_hi, labels, title, xlabel, ylabel, outpath):
    """Line + shaded-ribbon plot, one ribbon per group label."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lab, g in y_mean.groupby(labels):
        g = g.sort_values('device_count')
        # Select CI bounds matching this group (safe merge by equality)
        lo = y_lo[y_lo[labels[0]]==lab].sort_values('device_count') if len(labels)==1 else \
             y_lo[(y_lo[labels[0]]==lab[0]) & (y_lo[labels[1]]==lab[1])].sort_values('device_count')
        hi = y_hi[y_hi[labels[0]]==lab].sort_values('device_count') if len(labels)==1 else \
             y_hi[(y_hi[labels[0]]==lab[0]) & (y_hi[labels[1]]==lab[1])].sort_values('device_count')

        ax.plot(g['device_count'], g['mean'], marker='o', label=str(lab))
        ax.fill_between(g['device_count'], lo['lo'].values, hi['hi'].values, alpha=0.20)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_ribbons_loglog(y_mean, y_lo, y_hi, labels, title, outpath):
    """Same as plot_ribbons but with log-log axes."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lab, g in y_mean.groupby(labels):
        g = g.sort_values('device_count')
        lo = y_lo[y_lo[labels[0]]==lab].sort_values('device_count') if len(labels)==1 else \
             y_lo[(y_lo[labels[0]]==lab[0]) & (y_lo[labels[1]]==lab[1])].sort_values('device_count')
        hi = y_hi[y_hi[labels[0]]==lab].sort_values('device_count') if len(labels)==1 else \
             y_hi[(y_hi[labels[0]]==lab[0]) & (y_hi[labels[1]]==lab[1])].sort_values('device_count')

        ax.plot(g['device_count'], g['mean'], marker='o', label=str(lab))
        ax.fill_between(g['device_count'], lo['lo'].values, hi['hi'].values, alpha=0.20)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Devices (log)")
    ax.set_ylabel("Total time (s, log)")
    ax.set_title(title)
    ax.grid(True, which='both', ls=':', alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def fit_loglog_slopes(summary_pf, summary_top, df_pt):
    """Estimate power-law scaling exponent via log-log OLS for each grouping."""
    rows = []

    def slope_one(g, label_cols):
        """OLS log-log fit. Returns (slope, intercept, n_points) or None if < 2 points."""
        g = g.dropna(subset=['mean']).sort_values('device_count')
        if len(g) < 2:
            return None
        log_n = np.log(g['device_count'].values.astype(float))
        log_t = np.log(np.maximum(g['mean'].values.astype(float), 1e-9))
        A = np.vstack([log_n, np.ones_like(log_n)]).T
        slope, intercept = np.linalg.lstsq(A, log_t, rcond=None)[0]
        return slope, intercept, len(g)

    for pf, group_profile in summary_pf.groupby('profile'):
        res = slope_one(group_profile, ['profile'])
        if res:
            slope, intercept, n = res
            rows.append(dict(level='profile', profile=pf, topology='', slope=slope, intercept=intercept, points=n))

    for tp, group_topo in summary_top.groupby('topology'):
        res = slope_one(group_topo, ['topology'])
        if res:
            slope, intercept, n = res
            rows.append(dict(level='topology', profile='', topology=tp, slope=slope, intercept=intercept, points=n))

    for (pf, tp), group_pt in df_pt.groupby(['profile','topology']):
        res = slope_one(group_pt, ['profile','topology'])
        if res:
            slope, intercept, n = res
            rows.append(dict(level='profile+topology', profile=pf, topology=tp,
                             slope=slope, intercept=intercept, points=n))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True,
                    help="path to experiments.csv from the repeated sweep")
    ap.add_argument("--outdir", required=True,
                    help="output directory for plots and CSV tables")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.experiments)
    required = {"device_count","profile","topology","seed","total_time_s"}
    missing  = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.experiments}: {missing}")

    # Ensure correct types
    df['device_count'] = pd.to_numeric(df['device_count'], errors='coerce')
    df = df.dropna(subset=['device_count', 'total_time_s'])
    df['profile']  = df['profile'].astype(str)
    df['topology'] = df['topology'].astype(str)

    # Summaries with 95% CI
    by_pf  = summarize(df, ['profile', 'device_count'], 'total_time_s')
    by_top = summarize(df, ['topology', 'device_count'], 'total_time_s')
    by_pt  = summarize(df, ['profile','topology','device_count'], 'total_time_s')

    # Save summary tables
    by_pf.to_csv(outdir / "summary_by_profile.csv",          index=False)
    by_top.to_csv(outdir / "summary_by_topology.csv",        index=False)
    by_pt.to_csv(outdir / "summary_by_profile_topology.csv", index=False)

    # Linear plots
    plot_ribbons(
        x='device_count',
        y_mean=by_pf.rename(columns={'device_count':'device_count'}),
        y_lo=by_pf[['profile','device_count','lo']],
        y_hi=by_pf[['profile','device_count','hi']],
        labels=['profile'],
        title="Scalability by network profile (95% CI)",
        xlabel="Devices", ylabel="Total time (s)",
        outpath=outdir / "time_vs_devices_by_profile.png"
    )
    plot_ribbons(
        x='device_count',
        y_mean=by_top.rename(columns={'device_count':'device_count'}),
        y_lo=by_top[['topology','device_count','lo']],
        y_hi=by_top[['topology','device_count','hi']],
        labels=['topology'],
        title="Scalability by topology (95% CI)",
        xlabel="Devices", ylabel="Total time (s)",
        outpath=outdir / "time_vs_devices_by_topology.png"
    )
    plot_ribbons(
        x='device_count',
        y_mean=by_pt.rename(columns={'device_count':'device_count'}),
        y_lo=by_pt[['profile','topology','device_count','lo']],
        y_hi=by_pt[['profile','topology','device_count','hi']],
        labels=['profile','topology'],
        title="Scalability by (profile, topology) (95% CI)",
        xlabel="Devices", ylabel="Total time (s)",
        outpath=outdir / "time_vs_devices_by_profile_topology.png"
    )

    # Log-log plots for scaling exponent estimation
    plot_ribbons_loglog(
        y_mean=by_pf, y_lo=by_pf[['profile','device_count','lo']], y_hi=by_pf[['profile','device_count','hi']],
        labels=['profile'],
        title="Log-log: time vs devices by profile",
        outpath=outdir / "loglog_time_vs_devices_by_profile.png"
    )
    plot_ribbons_loglog(
        y_mean=by_top, y_lo=by_top[['topology','device_count','lo']], y_hi=by_top[['topology','device_count','hi']],
        labels=['topology'],
        title="Log-log: time vs devices by topology",
        outpath=outdir / "loglog_time_vs_devices_by_topology.png"
    )

    # Slope table (scaling exponent on log-log)
    slopes = fit_loglog_slopes(by_pf, by_top, by_pt)
    slopes.to_csv(outdir / "regression_slopes.csv", index=False)

    # Overall summary
    overall = summarize(df, ['device_count'], 'total_time_s')
    overall.to_csv(outdir / "summary_overall.csv", index=False)

    print("Wrote:")
    for f in [
        "time_vs_devices_by_profile.png",
        "time_vs_devices_by_topology.png",
        "time_vs_devices_by_profile_topology.png",
        "loglog_time_vs_devices_by_profile.png",
        "loglog_time_vs_devices_by_topology.png",
        "summary_by_profile.csv",
        "summary_by_topology.csv",
        "summary_by_profile_topology.csv",
        "regression_slopes.csv",
        "summary_overall.csv",
    ]:
        print(" -", outdir / f)

if __name__ == "__main__":
    main()
