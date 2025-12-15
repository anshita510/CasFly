#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scalability_sweep.py

Scalability sweep for CasFly (LaVE) with a separate simulated device set.

Overview:
  - Creates <out_root>/sim_devices/RenalWatch..DeviceK by sampling patient CSVs
    from the real device pool (read-only; originals are never modified).
  - Devices are balanced and reproducibly cloned with a fixed RNG seed.
  - For each combination of (device_count, network_profile, topology):
      - Optional warm-up runs (not recorded).
      - One or more usable repetitions (recorded to experiments.csv).
      - Simulated network parameters are injected via environment variables
        (USE_SIM_NET, SIMNET_*, DEVICE_PARTITIONS_ROOT).
  - Protocol traces (optional) are collected per run under runs/<run_id>/traces/.
  - Metrics are parsed from allmerged_metrics_log.csv after each run and
    appended to <out_root>/experiments.csv.

Network profiles (latency / bandwidth / loss):
    lan   -- 1.5 ms base, 100 Mbps, 0.05% loss
    wan   -- 15 ms base,  25 Mbps,  0.20% loss
    lossy -- 35 ms base,  10 Mbps,  1.00% loss

Inputs:
    devices_root/Device*/Patient_*.csv  -- real per-patient observation CSVs

Outputs:
    out_root/sim_devices/Device*/       -- simulated device folders (copies)
    out_root/experiments.csv            -- aggregated result table
    out_root/runs/<run_id>/             -- per-run artifacts and logs

Example:
    python3 scalability_sweep.py \\
      --devices_root data/synthea/device_scripts \\
      --out_root artifacts/scalability \\
      --run_cmd "python3 distribute.py" \\
      --device_counts 10,20,30,50,100 \\
      --topologies star,er \\
      --profiles lan,wan,lossy \\
      --enable_tracing 1 \\
      --per_device_patients 30 \\
      --warmup 1 \\
      --reps 20
"""

import os
import re
import csv
import json
import time
import shutil
import shlex
import random
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------- Helpers: listing & cloning (read-only originals) ----------------------

DEVICE_REGEX = re.compile(r"Device(\d+)$", re.I)

def list_device_dirs(root: Path) -> List[Path]:
    """Return sorted list of Device* subdirectories under root."""
    return sorted(
        [p for p in root.glob("Device*") if p.is_dir() and DEVICE_REGEX.search(p.name)],
        key=lambda p: int(DEVICE_REGEX.search(p.name).group(1))
    )

def collect_all_patient_files(sources: List[Path]) -> List[Path]:
    """Gather every Patient_*.csv file from all provided device directories."""
    files = []
    for d in sources:
        files.extend(sorted(d.glob("Patient_*.csv")))
    return files

def ensure_sim_devices(
    real_root: Path,
    sim_root: Path,
    target_n: int,
    per_device_patients: int = 30,
    seed: int = 7
) -> None:
    """
    Create simulated RenalWatch..DeviceN under sim_root by sampling from all real devices.
    - Never touches real_root.
    - If a simulated Devicei already has >= per_device_patients CSVs, it is kept as-is.
    - Otherwise it is (re)created with exactly per_device_patients files using a
      reproducible round-robin draw from the full patient pool.
    """
    sim_root.mkdir(parents=True, exist_ok=True)

    real_devices = list_device_dirs(real_root)
    if not real_devices:
        raise SystemExit(f"No real Device* folders found in {real_root}")

    pool = collect_all_patient_files(real_devices)
    if not pool:
        raise SystemExit(f"No Patient_*.csv files found under {real_root}/Device*/")

    rng = random.Random(seed)

    # Shuffle once for reproducibility
    pool_sorted = sorted(pool, key=lambda p: p.as_posix())
    rng.shuffle(pool_sorted)

    # Round-robin draw: device i gets files starting at offset (i * per_device_patients)
    def picks_for_device(i: int) -> List[Path]:
        start = (i * per_device_patients) % len(pool_sorted)
        out = []
        j = start
        while len(out) < per_device_patients:
            out.append(pool_sorted[j % len(pool_sorted)])
            j += 1
        return out

    for i in range(1, target_n + 1):
        dst = sim_root / f"Device{i}"
        existing = list(dst.glob("Patient_*.csv")) if dst.exists() else []
        if len(existing) >= per_device_patients:
            continue  # already complete, skip
        if not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            # Remove incomplete files before refilling
            for f in existing:
                try: f.unlink()
                except Exception: pass
        for f in picks_for_device(i):
            shutil.copy2(f, dst / f.name)

# ---------------------- Sim network profiles & env ----------------------

DEFAULT_PROFILES: Dict[str, Dict[str, float]] = {
    "lan":   dict(BASE_LAT_MS=1.5,  JITTER_MS=0.7,  BW_MBPS=100.0, LOSS_P=0.0005),
    "wan":   dict(BASE_LAT_MS=15.0, JITTER_MS=5.0,  BW_MBPS=25.0,  LOSS_P=0.0020),
    "lossy": dict(BASE_LAT_MS=35.0, JITTER_MS=12.0, BW_MBPS=10.0,  LOSS_P=0.0100),
}

def choose_star_hub(n: int, preferred: str = "CasFlyHub") -> str:
    """Clamp the preferred star-hub device number to the current device count."""
    m = re.search(r"(\d+)", preferred or "")
    want = int(m.group(1)) if m else 20
    hub_i = max(1, min(want, n))
    return f"Device{hub_i}"

def build_device_roster(n: int) -> str:
    return ",".join([f"Device{i}" for i in range(1, n+1)])

def set_simnet_env(
    n: int,
    profile: Dict[str, float],
    topology: str,
    star_hub_pref: str,
    tracing_dir: Optional[Path],
    enable_tracing: bool,
    sim_devices_root: Path,
    data_env_key: str = "DEVICE_PARTITIONS_ROOT"
):
    """Set all environment variables read by the device runner (USE_SIM_NET etc.)."""
    os.environ["USE_SIM_NET"]         = "1"
    os.environ["SIMNET_DEVICES"]      = build_device_roster(n)
    os.environ["SIMNET_TOPOLOGY"]     = topology
    os.environ["SIMNET_STAR_HUB"]     = choose_star_hub(n, star_hub_pref)
    os.environ["SIMNET_BASE_LAT_MS"]  = str(profile["BASE_LAT_MS"])
    os.environ["SIMNET_JITTER_MS"]    = str(profile["JITTER_MS"])
    os.environ["SIMNET_BW_MBPS"]      = str(profile["BW_MBPS"])
    os.environ["SIMNET_LOSS_P"]       = str(profile["LOSS_P"])
    # Point the pipeline at the simulated device set, not the originals
    os.environ[data_env_key]          = str(sim_devices_root.resolve())
    if enable_tracing:
        os.environ["PROTOCOL_TRACING"]  = "1"
        if tracing_dir:
            os.environ["PROTOCOL_TRACE_DIR"] = str(tracing_dir)
    else:
        os.environ.pop("PROTOCOL_TRACING", None)
        os.environ.pop("PROTOCOL_TRACE_DIR", None)

# ---------------------- Run a command ----------------------

def run_command(cmd: str, cwd: Optional[Path] = None, tee_log: Optional[Path] = None) -> Tuple[int, float]:
    """Run a shell command, stream its output, and return (return_code, wall_time_s)."""
    start = time.time()
    with subprocess.Popen(shlex.split(cmd), cwd=str(cwd) if cwd else None,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          bufsize=1, universal_newlines=True) as p:
        if tee_log:
            tee_log.parent.mkdir(parents=True, exist_ok=True)
            with tee_log.open("w", encoding="utf-8") as fh:
                for line in p.stdout:
                    print(line, end="")
                    fh.write(line)
        else:
            for line in p.stdout:
                print(line, end="")
        p.wait()
        rc = p.returncode
    return rc, time.time() - start

# ---------------------- Metrics extraction ----------------------

# Aliases mapping logical metric names to actual CSV column variants
COL_ALIASES = {
    "Total Time (T)": ["Total Time (T)", "TotalTime", "total_time", "T", "Total Time"],
    "Cached TPHG Load Time (t1)": ["Cached TPHG Load Time (t1)", "t1", "TPHG Load Time", "Cached TPHG Load Time"],
    "Backward Viterbi Time (t2)": ["Backward Viterbi Time (t2)", "t2", "Viterbi Time", "Backward Viterbi Time"],
    "Fallback Path Time (t_fallback)": ["Fallback Path Time (t_fallback)", "t_fallback", "Fallback Time"],
    "No. of Devices Accessed": ["No. of Devices Accessed", "Devices Accessed", "num_devices", "n_devices"],
    "Memory Usage (MB)": ["Memory Usage (MB)", "Memory(MB)", "Memory", "mem_mb"],
    "CPU Usage (%)": ["CPU Usage (%)", "CPU", "CPU (%)", "cpu_pct"],
    "RAM Usage (MB)": ["RAM Usage (MB)", "RAM", "RAM(MB)", "ram_mb"],
    "Experiment ID": ["Experiment ID", "ExpID", "exp_id"],
    "Initiator Device": ["Initiator Device", "Start Device", "start_device"],
    "Final Device": ["Final Device", "End Device", "end_device"],
    "Chain Data": ["Chain Data", "ChainData", "chain", "chain_data"],
}

def first_present(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    """Return the first alias of logical_name that exists as a DataFrame column."""
    for cand in COL_ALIASES.get(logical_name, [logical_name]):
        if cand in df.columns: return cand
    return None

def numeric_series(df: pd.DataFrame, logical_name: str) -> pd.Series:
    col = first_present(df, logical_name)
    if not col: return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def parse_allmerged_metrics(log_path: Path) -> Dict[str, float]:
    """Extract median values for key timing and resource metrics from the merged log."""
    d: Dict[str, float] = {}
    if not log_path or not log_path.exists():
        return d
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return d
    def med(logical):
        s = numeric_series(df, logical)
        return float(np.nanmedian(s)) if s.size else float("nan")
    d["median_T"]       = med("Total Time (T)")
    d["median_t1"]      = med("Cached TPHG Load Time (t1)")
    d["median_t2"]      = med("Backward Viterbi Time (t2)")
    d["median_t_fb"]    = med("Fallback Path Time (t_fallback)")
    d["median_mem_mb"]  = med("Memory Usage (MB)")
    d["median_cpu_pct"] = med("CPU Usage (%)")
    d["median_ram_mb"]  = med("RAM Usage (MB)")
    nd_col = first_present(df, "No. of Devices Accessed")
    d["median_devices_accessed"] = float(np.nanmedian(pd.to_numeric(df[nd_col], errors="coerce"))) if nd_col else float("nan")
    d["n_experiments"] = int(len(df))
    return d

def summarize_protocol_traces(trace_dir: Path) -> Dict[str, float]:
    """Aggregate network trace files (CSV or JSONL) into summary statistics."""
    if not trace_dir or not trace_dir.exists():
        return {}
    csvs   = list(trace_dir.glob("*.csv"))
    jsonls = list(trace_dir.glob("*.jsonl"))
    if csvs:
        total_msgs = 0; total_bytes = 0; rtts = []; retries = 0; oks = 0
        for f in csvs:
            try: df = pd.read_csv(f)
            except Exception: continue
            if "bytes"   in df.columns: total_bytes += pd.to_numeric(df["bytes"],   errors="coerce").fillna(0).sum()
            if "rtt_ms"  in df.columns: rtts.extend(pd.to_numeric(df["rtt_ms"],  errors="coerce").dropna().tolist())
            if "retries" in df.columns: retries     += pd.to_numeric(df["retries"], errors="coerce").fillna(0).sum()
            if "ok"      in df.columns: oks         += (df["ok"].astype(str).str.lower().isin(["1","true","yes","ok"])).sum()
            total_msgs += len(df)
        return dict(net_total_msgs=float(total_msgs), net_total_bytes=float(total_bytes),
                    net_rtt_p50_ms=float(np.nanpercentile(rtts,50)) if rtts else float("nan"),
                    net_rtt_p95_ms=float(np.nanpercentile(rtts,95)) if rtts else float("nan"),
                    net_retry_events=float(retries), net_ok_msgs=float(oks))
    if jsonls:
        total_msgs = 0; total_bytes = 0; rtts = []; retries = 0; oks = 0
        for f in jsonls:
            try:
                for line in f.open():
                    obj = json.loads(line.strip())
                    total_msgs  += 1
                    total_bytes += float(obj.get("bytes", 0))
                    if (r := obj.get("rtt_ms")) is not None: rtts.append(float(r))
                    retries += int(obj.get("retries", 0))
                    oks     += 1 if obj.get("ok", True) else 0
            except Exception: continue
        return dict(net_total_msgs=float(total_msgs), net_total_bytes=float(total_bytes),
                    net_rtt_p50_ms=float(np.nanpercentile(rtts,50)) if rtts else float("nan"),
                    net_rtt_p95_ms=float(np.nanpercentile(rtts,95)) if rtts else float("nan"),
                    net_retry_events=float(retries), net_ok_msgs=float(oks))
    return {}

def find_merged_log(log_glob: str, workdir: Path) -> Optional[Path]:
    """Locate the most recently modified file matching log_glob under workdir."""
    p = Path(log_glob)
    if p.is_absolute():
        return p if p.exists() else None
    matches = list((workdir).glob(log_glob))
    if matches:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]
    return None

# ---------------------- Main sweep ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices_root", required=True, help="Real folders: Device*/Patient_*.csv (read-only)")
    ap.add_argument("--out_root", required=True, help="Where results + simulated devices are written")
    ap.add_argument("--run_cmd", required=True, help='Command to run the experiment, e.g. "python3 distribute.py"')
    ap.add_argument("--device_counts", default="10,20,30,50,100")
    ap.add_argument("--profiles", default="lan,wan,lossy")
    ap.add_argument("--topologies", default="star,er")
    ap.add_argument("--star_hub", default="CasFlyHub")
    ap.add_argument("--log_glob", default="allmerged_metrics_log.csv")
    ap.add_argument("--enable_tracing", type=int, default=1)
    ap.add_argument("--per_device_patients", type=int, default=30,
                    help="Exactly this many CSVs per simulated device")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=0,
                    help="Warm-up runs per (N,profile,topology) -- not recorded")
    ap.add_argument("--reps", type=int, default=1,
                    help="Usable repetitions per (N,profile,topology) -- recorded to CSV")
    ap.add_argument("--sleep_between_reps", type=float, default=0.0,
                    help="Seconds to sleep between reps")
    ap.add_argument("--data_env_key", default="DEVICE_PARTITIONS_ROOT",
                    help="Env var your runner reads to locate Device* root (points to simulated set)")
    args = ap.parse_args()

    real_root = Path(args.devices_root)
    out_root  = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    sim_root  = out_root / "sim_devices"
    sim_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "experiments.csv"

    device_counts = [int(x) for x in str(args.device_counts).split(",") if x.strip()]
    profiles_keys = [x.strip().lower() for x in str(args.profiles).split(",") if x.strip()]
    topologies    = [x.strip().lower() for x in str(args.topologies).split(",") if x.strip()]

    for k in profiles_keys:
        if k not in DEFAULT_PROFILES:
            raise SystemExit(f"Unknown profile '{k}'. Valid: {list(DEFAULT_PROFILES.keys())}")

    # Build the simulated device pool up to max N (never modifies real_root)
    ensure_sim_devices(
        real_root=real_root,
        sim_root=sim_root,
        target_n=max(device_counts),
        per_device_patients=args.per_device_patients,
        seed=args.seed
    )

    # Prepare results CSV with header (append-safe; only written once)
    if not results_csv.exists():
        with results_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "run_id","device_count","profile","topology",
                "return_code","wall_time_s",
                "n_experiments","median_T","median_t1","median_t2","median_t_fb",
                "median_devices_accessed","median_mem_mb","median_cpu_pct","median_ram_mb",
                "net_total_msgs","net_total_bytes","net_rtt_p50_ms","net_rtt_p95_ms",
                "net_retry_events","net_ok_msgs",
                "run_dir","log_path","trace_dir"
            ])

    # Main sweep: iterate over (N, profile, topology) combinations
    for N in device_counts:
        for prof_key in profiles_keys:
            profile = DEFAULT_PROFILES[prof_key]
            for topo in topologies:

                # Warm-up runs (not recorded)
                for w in range(max(0, args.warmup)):
                    set_simnet_env(
                        n=N, profile=profile, topology=topo, star_hub_pref=args.star_hub,
                        tracing_dir=None, enable_tracing=False,
                        sim_devices_root=sim_root, data_env_key=args.data_env_key
                    )
                    print(f"[Warmup] N={N} profile={prof_key} topo={topo} (warmup {w+1}/{args.warmup})")
                    run_command(args.run_cmd)

                # Usable repetitions (recorded)
                for rep in range(1, int(args.reps) + 1):
                    run_stamp = int(time.time())
                    run_id    = f"N{N}_{prof_key}_{topo}_rep{rep}_{run_stamp}"
                    run_dir   = out_root / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    trace_dir = run_dir / "traces" if args.enable_tracing else None
                    tee_log   = run_dir / "runner_stdout.log"

                    set_simnet_env(
                        n=N, profile=profile, topology=topo, star_hub_pref=args.star_hub,
                        tracing_dir=trace_dir, enable_tracing=bool(args.enable_tracing),
                        sim_devices_root=sim_root, data_env_key=args.data_env_key
                    )

                    # Additional env hints for the runner script
                    os.environ["RUN_REP"]          = str(rep)
                    os.environ["RUN_PROFILE"]      = prof_key
                    os.environ["RUN_TOPOLOGY"]     = topo
                    os.environ["RUN_DEVICE_COUNT"] = str(N)

                    print(f"\n=== RUN {run_id} ===")
                    print(f"SIMNET: N={N} profile={prof_key} topo={topo} "
                          f"STAR={os.environ['SIMNET_STAR_HUB']} "
                          f"DEV_ROOT={os.environ[args.data_env_key]}")
                    rc, wall = run_command(args.run_cmd, cwd=None, tee_log=tee_log)

                    # Find the merged metrics log (current dir first, then run_dir)
                    log_path = find_merged_log(args.log_glob, Path.cwd()) or find_merged_log(args.log_glob, run_dir)

                    merged_metrics = parse_allmerged_metrics(log_path) if log_path else {}
                    proto_metrics  = summarize_protocol_traces(trace_dir) if trace_dir else {}

                    saved_log = ""
                    if log_path and log_path.exists():
                        dest_log = run_dir / log_path.name
                        if str(log_path.resolve()) != str(dest_log.resolve()):
                            try: shutil.copy2(log_path, dest_log)
                            except Exception: pass
                        saved_log = str(dest_log)

                    row = {
                        "run_id": run_id,
                        "device_count": N,
                        "profile": prof_key,
                        "topology": topo,
                        "return_code": rc,
                        "wall_time_s": round(wall, 3),
                        "run_dir": str(run_dir),
                        "log_path": saved_log,
                        "trace_dir": str(trace_dir) if trace_dir else "",
                    }
                    for k in ["n_experiments","median_T","median_t1","median_t2","median_t_fb",
                              "median_devices_accessed","median_mem_mb","median_cpu_pct","median_ram_mb"]:
                        row[k] = merged_metrics.get(k, float("nan"))
                    for k in ["net_total_msgs","net_total_bytes","net_rtt_p50_ms","net_rtt_p95_ms",
                              "net_retry_events","net_ok_msgs"]:
                        row[k] = proto_metrics.get(k, float("nan"))

                    with results_csv.open("a", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh)
                        writer.writerow([row.get(h,"") for h in [
                            "run_id","device_count","profile","topology",
                            "return_code","wall_time_s",
                            "n_experiments","median_T","median_t1","median_t2","median_t_fb",
                            "median_devices_accessed","median_mem_mb","median_cpu_pct","median_ram_mb",
                            "net_total_msgs","net_total_bytes","net_rtt_p50_ms","net_rtt_p95_ms",
                            "net_retry_events","net_ok_msgs",
                            "run_dir","log_path","trace_dir"
                        ]])

                    print(f"[Done] {run_id} rc={rc} wall={wall:.1f}s | log={saved_log or 'N/A'}")

                    if args.sleep_between_reps > 0:
                        time.sleep(args.sleep_between_reps)

    print(f"\nSimulated devices: {(out_root/'sim_devices').resolve()} (originals untouched)")
    print(f"Aggregated table:  {(out_root/'experiments.csv').resolve()}")
    print("Tip: ensure your runner reads env var DEVICE_PARTITIONS_ROOT (or pass --data_env_key).")

if __name__ == "__main__":
    main()
