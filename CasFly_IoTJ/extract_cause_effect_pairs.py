"""
extract_cause_effect_pairs.py

Extracts temporal cause-effect event pairs from Synthea CSV exports.
For each patient, all clinical events (conditions, observations, immunizations)
are collected and every ordered pair (A → B) where A occurs before B is
emitted as a potential causal relationship.

This produces cause_effect_pairs.csv, which is the input to
compute_conditional_probs.py.

Inputs (Synthea output CSV files):
    conditions.csv    -- columns: START, PATIENT, CODE, DESCRIPTION
    observations.csv  -- columns: DATE, PATIENT, CODE, DESCRIPTION, VALUE
    immunizations.csv -- columns: DATE, PATIENT, CODE, DESCRIPTION

Output:
    cause_effect_pairs.csv -- columns:
        cause, cause_date, cause_type, effect, effect_date, effect_type
"""

import pandas as pd
from itertools import combinations


# --- Input/output paths (relative to casflyIoTJ/) ---
CONDITIONS_CSV    = "../data/synthea/raw_data/conditions.csv"
OBSERVATIONS_CSV  = "../data/synthea/raw_data/observations.csv"
IMMUNIZATIONS_CSV = "../data/synthea/raw_data/immunizations.csv"
OUTPUT_CSV        = "../data/synthea/processed/cause_effect_pairs.csv"

# Observation descriptions to include (the clinically relevant metrics used
# in the CasFly TPHG model; matches the keys in distribute.py condition_keywords)
OBSERVATION_METRICS = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "Body Weight",
    "Body Height",
    "Body Mass Index",
    "Glucose",
    "Creatinine",
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "Sodium",
    "Hemoglobin [Mass/volume] in Blood",
    "Platelet mean volume [Entitic volume] in Blood by Automated count",
    "Oxygen saturation in Arterial blood",
    "Calcium [Mass/volume] in Blood",
]


def load_conditions(path: str) -> pd.DataFrame:
    """Load conditions and normalise to (patient, event_name, event_date, event_type)."""
    df = pd.read_csv(path, usecols=["PATIENT", "START", "DESCRIPTION"])
    df = df.rename(columns={"PATIENT": "patient", "START": "date", "DESCRIPTION": "event"})
    df["date"] = pd.to_datetime(df["date"].str[:10], format="%Y-%m-%d", errors="coerce")
    df["event_type"] = "condition"
    return df.dropna(subset=["date"])


def load_observations(path: str) -> pd.DataFrame:
    """Load observations, keeping only the clinically relevant metrics."""
    try:
        df = pd.read_csv(path, usecols=["PATIENT", "DATE", "DESCRIPTION", "VALUE"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["patient", "date", "event", "event_type"])
    df = df[df["DESCRIPTION"].isin(OBSERVATION_METRICS)].copy()
    df = df.rename(columns={"PATIENT": "patient", "DATE": "date", "DESCRIPTION": "event"})
    df["date"] = pd.to_datetime(df["date"].str[:10], format="%Y-%m-%d", errors="coerce")
    df["event_type"] = "observation"
    return df[["patient", "date", "event", "event_type"]].dropna(subset=["date"])


def load_immunizations(path: str) -> pd.DataFrame:
    """Load immunizations and normalise to common schema."""
    try:
        df = pd.read_csv(path, usecols=["PATIENT", "DATE", "DESCRIPTION"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["patient", "date", "event", "event_type"])
    df = df.rename(columns={"PATIENT": "patient", "DATE": "date", "DESCRIPTION": "event"})
    df["date"] = pd.to_datetime(df["date"].str[:10], format="%Y-%m-%d", errors="coerce")
    df["event_type"] = "immunization"
    return df.dropna(subset=["date"])


# Load and combine all event types
events = pd.concat(
    [load_conditions(CONDITIONS_CSV),
     load_observations(OBSERVATIONS_CSV),
     load_immunizations(IMMUNIZATIONS_CSV)],
    ignore_index=True,
)

# For each patient, generate all ordered (earlier, later) event pairs
pair_rows = []
for patient_id, group in events.groupby("patient"):
    group = group.sort_values("date").reset_index(drop=True)
    for i, j in combinations(range(len(group)), 2):
        a, b = group.iloc[i], group.iloc[j]
        if a["date"] <= b["date"]:
            pair_rows.append({
                "cause":       a["event"],
                "cause_date":  a["date"].strftime("%Y-%m-%d"),
                "cause_type":  a["event_type"],
                "effect":      b["event"],
                "effect_date": b["date"].strftime("%Y-%m-%d"),
                "effect_type": b["event_type"],
            })

pairs_df = pd.DataFrame(pair_rows)
pairs_df.to_csv(OUTPUT_CSV, index=False)
print(f"Extracted {len(pairs_df)} cause-effect pairs → {OUTPUT_CSV}")
