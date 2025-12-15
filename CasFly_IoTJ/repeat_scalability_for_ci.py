#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repeat_scalability_for_ci.py

Runs any per-experiment command many times across combinations of
(device_count, network_profile, topology) to collect enough samples
for 95% confidence-interval estimation.

Each run is identified by a unique run_id and gets its own output
directory under <out_root>/runs/.  Wall-clock timing and the return
code are written to <out_root>/experiments.csv.

Format string tokens available in --cmd:
    {N}        -- device count (int)
    {profile}  -- network profile, e.g. "lan", "lossy", "wan"
    {topology} -- graph topology, e.g. "er", "star"
    {seed}     -- integer PRNG seed (increments with run_counter)
    {out_dir}  -- unique per-run output directory
    {run_id}   -- string key, e.g. N50_lossy_er_12345

Output columns in experiments.csv:
    run_id, device_count, profile, topology, seed,
    total_time_s, return_code, started_at, finished_at, cmd

Example:
    python3 repeat_scalability_for_ci.py \\
      --repeats 20 \\
      --Ns 10,20,30,50,100 \\
      --profiles lan,lossy,wan \\
      --topologies er,star \\
      --out_root artifacts/scalability \\
      --cmd "python run_scalability_once.py --device_count {N} --profile {profile} \\
             --topology {topology} --seed {seed} --out {out_dir}"
"""

import argparse, csv, os, shlex, subprocess, sys, time, random
from pathlib import Path
from datetime import datetime


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats",    type=int, default=20,
                    help="repetitions per (N, profile, topology)")
    ap.add_argument("--Ns",         default="10,20,30,50,100",
                    help="comma-separated device counts")
    ap.add_argument("--profiles",   default="lan,lossy,wan",
                    help="comma-separated network profiles")
    ap.add_argument("--topologies", default="er,star",
                    help="comma-separated graph topologies")
    ap.add_argument("--out_root",   default="artifacts/scalability",
                    help="root directory for aggregated table and per-run subdirs")
    ap.add_argument("--cmd",        required=True,
                    help="format string command; see header for available tokens")
    ap.add_argument("--seed0",      type=int, default=12345,
                    help="base seed; each successive run increments this by 1")
    args = ap.parse_args()

    Ns         = [int(x) for x in parse_list(args.Ns)]
    profiles   = parse_list(args.profiles)
    topologies = parse_list(args.topologies)

    out_root  = Path(args.out_root)
    runs_dir  = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    table_csv = out_root / "experiments.csv"

    # Write CSV header if file does not yet exist
    if not table_csv.exists():
        with open(table_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run_id","device_count","profile","topology","seed",
                        "total_time_s","return_code","started_at","finished_at","cmd"])

    run_counter = 0
    for N in Ns:
        for profile in profiles:
            for topo in topologies:
                for r in range(args.repeats):
                    seed    = args.seed0 + run_counter
                    run_id  = f"N{N}_{profile}_{topo}_{seed}"
                    out_dir = runs_dir / run_id
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Substitute all tokens into the command template
                    cmd_str = args.cmd.format(
                        N=N, profile=profile, topology=topo,
                        seed=seed, out_dir=str(out_dir), run_id=run_id
                    )
                    print(f"[RUN] {run_id}\n  -> {cmd_str}")
                    started = now_iso()
                    t0 = time.perf_counter()
                    try:
                        rc = subprocess.run(shlex.split(cmd_str), check=False).returncode
                    except FileNotFoundError as e:
                        print(f"[ERR] Command not found: {e}", file=sys.stderr)
                        rc = 127
                    dur      = time.perf_counter() - t0
                    finished = now_iso()

                    with open(table_csv, "a", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow([run_id, N, profile, topo, seed,
                                    f"{dur:.3f}", rc, started, finished, cmd_str])

                    status = "OK" if rc == 0 else f"RC={rc}"
                    print(f"[DONE] {run_id} | {status} | wall={dur:.2f}s | logdir={out_dir}\n")
                    run_counter += 1

    print(f"All runs complete. Aggregated: {table_csv.resolve()}")
    print("Next: use plot_scalability_ci.py to build CI ribbon plots.")

if __name__ == "__main__":
    main()
