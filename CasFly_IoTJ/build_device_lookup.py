"""
build_device_lookup.py

Assigns each (cause, effect) pair from conditional_probabilities_all_events.csv
to one of the 26 simulated IoT devices (RenalWatch–BPCuff), then writes the full
set of lookup tables consumed by the Device*.py federation scripts.

How assignment works:
  1. If the cause_type or effect_type label matches a known observation/device
     keyword mapping, route to the corresponding device.
  2. If an old lookup file (All_Device_Lookup_merged_with_ports.csv) already
     exists, preserve the device names, ports, and IP addresses from it so that
     re-runs don't change device identities.
  3. For unmatched pairs, use a deterministic hash so the same pair always lands
     on the same device across runs.

Inputs:
    conditional_probabilities_all_events.csv  -- output of compute_conditional_probs.py
    All_Device_Lookup_merged_with_ports.csv   -- (optional) prior lookup to preserve
    All_Device_Lookup_causes_with_ports.csv   -- (optional) prior per-device cause lists
    All_Device_Lookup_effects_with_ports.csv  -- (optional) prior per-device effect lists

Outputs (all in the current working directory):
    All_Device_Lookup_with_Probabilities.csv  -- primary lookup (used by Device*.py)
    All_Device_Lookup_merged_with_ports.csv   -- device → cause_type, effect_type, port, ip
    All_Device_Lookup_merged.csv              -- lightweight version (no port/ip)
    All_Device_Lookup_causes_with_ports.csv   -- device → cause_type, port, ip
    All_Device_Lookup_effects_with_ports.csv  -- device → effect_type, port, ip
    All_Device_Lookup_causes.csv              -- device → cause_type
    All_Device_Lookup_effect.csv              -- device → effect_type
    device_partitions/Device*_causal_pairs.csv -- per-device probability tables
"""

import os
import ast
import re
import hashlib
from collections import defaultdict

import pandas as pd


# ──────────────────────────── config ────────────────────────────
PROB_FILE        = "../data/synthea/processed/conditional_probabilities_all_events.csv"
OLD_LOOKUP       = "../data/synthea/processed/All_Device_Lookup_merged_with_ports.csv"
CAUSES_WITH_PORTS  = "../data/synthea/processed/All_Device_Lookup_causes_with_ports.csv"
EFFECTS_WITH_PORTS = "../data/synthea/processed/All_Device_Lookup_effects_with_ports.csv"
OUT_DIR          = "../data/synthea/causal_pairs"
PROCESSED_DIR    = "../data/synthea/processed"

# Keyword mapping: observation label → device routing category
# (must match the labels in conditional_probabilities_all_events.csv)
CONDITION_KEYWORDS = {
    "Blood Pressure":      ["hypertension", "hypotension", "bp"],
    "Glucose":             ["diabetes", "hyperglycemia", "hypoglycemia"],
    "Heart Rate":          ["tachycardia", "bradycardia", "arrhythmia"],
    "Body Weight":         ["obesity", "underweight"],
    "Creatinine":          ["kidney", "renal", "creatinine"],
    "ALT (Elevated)":      ["liver", "alt", "hepatitis"],
    "AST (Elevated)":      ["ast"],
    "Sodium":              ["sodium", "natremia"],
    "Respiratory Rate":    ["asthma", "pneumonia", "copd", "wheezing"],
    "Calcium":             ["hypocalcemia", "hypercalcemia"],
    "Hemoglobin":          ["anemia", "polycythemia"],
    "Platelet Count":      ["thrombocytopenia", "thrombocytosis"],
    "Oxygen Saturation":   ["hypoxemia", "respiratory failure"],
}
# ────────────────────────────────────────────────────────────────


# ─── text helpers ───────────────────────────────────────────────
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def norm_key(s: str) -> str:
    return norm_text(s).lower()

def norm_device_name(s: str) -> str:
    s = str(s).strip()
    m = re.match(r"(?i)\s*device[_\s-]?(\d+)\s*$", s)
    return f"Device{m.group(1)}" if m else s

def parse_list(txt) -> list:
    if pd.isna(txt):
        return []
    s = str(txt)
    try:
        v = ast.literal_eval(s)
        return list(v) if isinstance(v, (list, tuple, set)) else [v]
    except Exception:
        try:
            v = ast.literal_eval(s.replace('""', '"'))
            return list(v) if isinstance(v, (list, tuple, set)) else [v]
        except Exception:
            inner = s.strip().strip("[]")
            if not inner:
                return []
            return [p.strip(" '\"") for p in inner.split(",") if p.strip()]

def stable_index(s: str, n: int) -> int:
    """Deterministic hash-based partition index in [0, n)."""
    h = hashlib.blake2b(norm_key(s).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % max(n, 1)
# ────────────────────────────────────────────────────────────────


# ─── 1. load probability table ──────────────────────────────────
prob_df = pd.read_csv(PROB_FILE)
required = {"cause", "effect", "cause_type", "effect_type", "conditional_probability"}
missing = required - set(prob_df.columns)
if missing:
    raise ValueError(f"Missing columns in {PROB_FILE}: {missing}")
prob_df["conditional_probability"] = pd.to_numeric(
    prob_df["conditional_probability"], errors="coerce"
)
prob_df = prob_df.dropna(subset=["conditional_probability"])


# ─── 2. restore prior device names / ports / IPs if available ───
device_names   = []
device_ports   = {}
device_ips     = {}
old_causes_raw = defaultdict(set)
old_effects_raw = defaultdict(set)

if os.path.exists(OLD_LOOKUP):
    base = pd.read_csv(OLD_LOOKUP)
    if "device" in base.columns:
        devs = [norm_device_name(d) for d in base["device"].dropna() if str(d).strip()]
        seen = set()
        device_names = [d for d in devs if not (d in seen or seen.add(d))]

        if "port" in base.columns:
            device_ports = {norm_device_name(d): p
                            for d, p in zip(base["device"], base["port"])}
        if "ip" in base.columns:
            device_ips = {norm_device_name(d): ip
                          for d, ip in zip(base["device"], base["ip"])}

        if os.path.exists(CAUSES_WITH_PORTS):
            tmp = pd.read_csv(CAUSES_WITH_PORTS)
            for r in tmp.itertuples(index=False):
                dev = norm_device_name(getattr(r, "device", None))
                for item in parse_list(getattr(r, "cause_type", "[]")):
                    t = norm_text(item)
                    if t:
                        old_causes_raw[dev].add(t)

        if os.path.exists(EFFECTS_WITH_PORTS):
            tmp = pd.read_csv(EFFECTS_WITH_PORTS)
            for r in tmp.itertuples(index=False):
                dev = norm_device_name(getattr(r, "device", None))
                for item in parse_list(getattr(r, "effect_type", "[]")):
                    t = norm_text(item)
                    if t:
                        old_effects_raw[dev].add(t)

# Defaults when no prior lookup exists
if not device_names:
    device_names = [f"Device{i}" for i in range(1, 27)]
if not device_ports:
    device_ports = {d: 6000 + i for i, d in enumerate(device_names)}

default_ip = os.environ.get("DEFAULT_IP", "127.0.0.1")
for d in device_names:
    k = norm_device_name(d)
    device_ips.setdefault(k, os.environ.get(f"{k.upper()}_IP", default_ip))

old_causes  = {dev: {norm_key(x) for x in xs} for dev, xs in old_causes_raw.items()}
old_effects = {dev: {norm_key(x) for x in xs} for dev, xs in old_effects_raw.items()}


# ─── 3. build observation-label → device mapping from prior lookup ───
obs_to_device = {}
for dev in device_names:
    for lab in (old_effects.get(dev, set()) | old_causes.get(dev, set())):
        obs_to_device[lab] = dev

# Ensure blood pressure maps to a single stable device
bp_device = None
for dev in device_names:
    if any(x in old_effects.get(dev, set())
           for x in ("systolic blood pressure", "diastolic blood pressure", "blood pressure")):
        bp_device = dev
        break
bp_device = bp_device or device_names[0]
for lab in ("Blood Pressure", "Systolic Blood Pressure", "Diastolic Blood Pressure"):
    obs_to_device[norm_key(lab)] = bp_device

for dev in device_names:
    if ("immunization" in old_effects.get(dev, set())
            or "immunization" in old_causes.get(dev, set())):
        obs_to_device[norm_key("immunization")] = dev
        break


# ─── 4. routing function ─────────────────────────────────────────
def route_condition(condition_text: str) -> str:
    """Route a 'condition' type event to a device via keyword match or hash."""
    ck = norm_key(condition_text)
    if ck in obs_to_device:
        return obs_to_device[ck]
    for obs_label, keys in CONDITION_KEYWORDS.items():
        if any(k in ck for k in keys):
            d = obs_to_device.get(norm_key(obs_label))
            if d:
                return d
    return device_names[stable_index(ck, len(device_names))]


def pick_device(row) -> str:
    ct = norm_text(row["cause_type"])
    et = norm_text(row["effect_type"])
    c  = norm_text(row["cause"])
    e  = norm_text(row["effect"])
    for token in (ct, et, c, e):
        if norm_key(token) in obs_to_device:
            return obs_to_device[norm_key(token)]
    if ct.lower() == "condition":
        return route_condition(c)
    if ct.lower() == "immunization" or et.lower() == "immunization":
        return obs_to_device.get(norm_key("immunization"),
                                 device_names[min(1, len(device_names) - 1)])
    return device_names[stable_index(ct + "→" + et, len(device_names))]


# ─── 5. assign devices and write per-device partition files ──────
prob_df = prob_df.copy()
prob_df["device"] = prob_df.apply(pick_device, axis=1)

os.makedirs(OUT_DIR, exist_ok=True)
for dev, sub in prob_df.groupby("device"):
    sub.to_csv(os.path.join(OUT_DIR, f"{dev}_causal_pairs.csv"), index=False)


# ─── 6. build per-device probability tables (concrete type labels) ───
device_probabilities = {}
for dev, sub in prob_df.groupby("device"):
    sub = sub.copy()
    for col in ("cause_type", "effect_type", "cause", "effect"):
        sub[col] = sub[col].astype(str)
    sub.loc[sub["cause_type"].str.lower()  == "condition",    "cause_type"]  = sub["cause"]
    sub.loc[sub["effect_type"].str.lower() == "condition",    "effect_type"] = sub["effect"]
    sub.loc[sub["cause_type"].str.lower()  == "immunization", "cause_type"]  = sub["cause"]
    sub.loc[sub["effect_type"].str.lower() == "immunization", "effect_type"] = sub["effect"]
    grouped = (sub.groupby(["cause_type", "effect_type"])["conditional_probability"]
               .apply(list).reset_index())
    device_probabilities[dev] = grouped


# ─── 7. rebuild lookup tables with preserved names/ports/IPs ─────
rows = []
for dev in device_names:
    prob_df_dev = device_probabilities.get(
        dev, pd.DataFrame(columns=["cause_type", "effect_type", "conditional_probability"])
    )
    causes  = sorted({norm_text(x) for x in prob_df_dev.get("cause_type",  pd.Series([], dtype=str))}, key=str.lower)
    effects = sorted({norm_text(x) for x in prob_df_dev.get("effect_type", pd.Series([], dtype=str))}, key=str.lower)
    rows.append({
        "device":      dev,
        "cause_type":  str(causes),
        "effect_type": str(effects),
        "port":        device_ports.get(dev),
        "ip":          device_ips.get(dev, default_ip),
    })
merged_ports = pd.DataFrame(rows)


# ─── 8. attach probability_list to the main lookup ───────────────
def attach_probability_list(row) -> list:
    dev = norm_device_name(row["device"])
    prob_df_dev = device_probabilities.get(dev)
    if prob_df_dev is None or prob_df_dev.empty:
        return []
    pdn = prob_df_dev.copy()
    pdn["_cause_n"]  = pdn["cause_type"].astype(str).str.strip().str.lower()
    pdn["_effect_n"] = pdn["effect_type"].astype(str).str.strip().str.lower()
    c_set = {norm_key(x) for x in parse_list(row.get("cause_type", "[]"))}
    e_set = {norm_key(x) for x in parse_list(row.get("effect_type", "[]"))}
    matched = (pdn[pdn["_effect_n"].isin(e_set) & pdn["_cause_n"].isin(c_set)]
               if c_set else pdn[pdn["_effect_n"].isin(e_set)])
    if matched.empty:
        return []
    flat = []
    for lst in matched["conditional_probability"]:
        flat.extend(lst if isinstance(lst, (list, tuple)) else [lst])
    return [float(v) for v in flat if _is_float(v)]

def _is_float(v) -> bool:
    try:
        float(v); return True
    except Exception:
        return False

with_probs = merged_ports.copy()
with_probs["probability_list"] = with_probs.apply(attach_probability_list, axis=1)


# ─── 9. write all output files ───────────────────────────────────
import os as _os
_os.makedirs(PROCESSED_DIR, exist_ok=True)

with_probs.to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_with_Probabilities.csv", index=False)
merged_ports.to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_merged_with_ports.csv", index=False)
merged_ports[["device", "cause_type", "effect_type"]].to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_merged.csv", index=False)
merged_ports[["device", "cause_type", "port", "ip"]].to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_causes_with_ports.csv", index=False)
merged_ports[["device", "effect_type", "port", "ip"]].to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_effects_with_ports.csv", index=False)
merged_ports[["device", "cause_type"]].to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_causes.csv", index=False)
merged_ports[["device", "effect_type"]].to_csv(f"{PROCESSED_DIR}/All_Device_Lookup_effect.csv", index=False)

empty_rows = (with_probs["probability_list"].apply(len) == 0).sum()
print(f"Device partitions written to {OUT_DIR}/")
print(f"Lookup tables written to current directory.")
print(f"Rows with empty probability_list: {empty_rows} of {len(with_probs)}")
