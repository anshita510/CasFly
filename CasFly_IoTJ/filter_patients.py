"""
filter_patients.py

Filters the Synthea conditions.csv to retain only patients with at least one
of the four diseases studied in the CasFly paper:
  - Stroke
  - Heart Attack (Myocardial infarction)
  - Lung Cancer
  - Seizure

Input:
    conditions.csv  -- Synthea output; columns include PATIENT, DESCRIPTION, CODE

Output:
    filtered_patients.csv -- columns: PATIENT, DESCRIPTION, CODE, CONDITION
"""

import pandas as pd


# --- Input/output paths (relative to casflyIoTJ/) ---
CONDITIONS_CSV = "../data/synthea/raw_data/conditions.csv"
OUTPUT_CSV     = "../data/synthea/processed/filtered_patients.csv"

# --- Disease search terms per condition label ---
# 'Heart Attack' uses exact match (case-insensitive) to avoid partial hits.
# Others use substring search (case-insensitive).
CONDITION_TERMS = {
    "Stroke":       ["stroke"],
    "Heart Attack": ["Myocardial infarction (disorder)"],
    "Lung Cancer":  ["lung cancer"],
    "Seizure":      ["Seizure disorder"],
}

conditions = pd.read_csv(CONDITIONS_CSV)

buckets = []
for condition_label, terms in CONDITION_TERMS.items():
    if condition_label == "Heart Attack":
        # Exact match to avoid catching unrelated mentions of "infarction"
        mask = conditions["DESCRIPTION"].str.strip().str.lower().isin(
            [t.lower() for t in terms]
        )
    else:
        mask = conditions["DESCRIPTION"].str.contains(
            "|".join(terms), case=False, na=False
        )

    subset = conditions[mask].copy()
    subset["CONDITION"] = condition_label
    buckets.append(subset)

filtered_patients = pd.concat(buckets, ignore_index=True)
filtered_patients[["PATIENT", "DESCRIPTION", "CODE", "CONDITION"]].to_csv(
    OUTPUT_CSV, index=False
)
print(f"Filtered patients saved to {OUTPUT_CSV} ({len(filtered_patients)} rows)")
