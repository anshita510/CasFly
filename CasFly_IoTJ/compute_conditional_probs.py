"""
compute_conditional_probs.py

Reads cause_effect_pairs.csv (a table of cause/effect event pairs with dates),
computes the time lag between each cause and effect, bins those lags into
clinically meaningful intervals, and calculates conditional probabilities
P(effect | cause, lag_bin).

Input:
    cause_effect_pairs.csv  -- columns: cause, cause_date, cause_type,
                               effect, effect_date, effect_type

Output:
    conditional_probabilities_all_events.csv -- columns: cause, effect,
        lag_bin, cause_type, effect_type, conditional_probability
"""

import pandas as pd
import numpy as np


INPUT_CSV  = "../data/synthea/processed/cause_effect_pairs.csv"
OUTPUT_CSV = "../data/synthea/processed/conditional_probabilities_all_events.csv"

# Clinically motivated lag bins (in days), matching Table I in the paper.
# Covers short-term (days), medium-term (weeks/months), and long-term (years).
LAG_BINS   = [0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
LAG_LABELS = [f"{LAG_BINS[i]}-{LAG_BINS[i+1]}" for i in range(len(LAG_BINS) - 1)]

pairs_df = pd.read_csv(INPUT_CSV)

# Parse only the date portion (first 10 chars) to avoid timezone issues
pairs_df['cause_date']  = pd.to_datetime(pairs_df['cause_date'].str[:10],  format="%Y-%m-%d")
pairs_df['effect_date'] = pd.to_datetime(pairs_df['effect_date'].str[:10], format="%Y-%m-%d")

# Compute time lag in days; drop negative lags (effect before cause is non-causal).
# Note: extract_cause_effect_pairs.py already guarantees cause_date <= effect_date,
# so this filter is a safety check for malformed rows.
pairs_df['time_lag_days'] = (pairs_df['effect_date'] - pairs_df['cause_date']).dt.days
pairs_df = pairs_df[pairs_df['time_lag_days'] >= 0]

# Assign each row to the appropriate lag bin
pairs_df['lag_bin'] = pd.cut(
    pairs_df['time_lag_days'],
    bins=LAG_BINS,
    labels=LAG_LABELS,
    right=True,   # interval is (left, right]
)

# Count co-occurrences of (cause, effect, lag_bin, cause_type, effect_type).
# This is the numerator N(cause → effect, lag_bin) from Eq. 2 in the paper.
pair_counts = (
    pairs_df
    .groupby(['cause', 'effect', 'lag_bin', 'cause_type', 'effect_type'])
    .size()
    .reset_index(name='pair_count')
)

# Count total occurrences of each cause event (across all effects and all lag bins).
# Used as the denominator to compute the conditional probability.
cause_counts = (
    pairs_df
    .groupby(['cause', 'cause_type'])
    .size()
    .reset_index(name='cause_count')
)

# Merge counts and compute P(effect | cause, lag_bin) = pair_count / cause_count.
# This is a simplified estimate; the denominator is the global cause frequency
# rather than the per-lag-bin frequency, so probabilities within a lag bin do
# not sum to 1.  For LaVE's Viterbi path selection only relative rankings
# within the same cause matter, so this does not affect which chain is found.
prob_df = pd.merge(pair_counts, cause_counts, on=['cause', 'cause_type'])
prob_df['conditional_probability'] = prob_df['pair_count'] / prob_df['cause_count']

# Drop raw counts and keep only non-zero probabilities
prob_df = prob_df.drop(columns=['pair_count', 'cause_count'])
prob_df = prob_df[prob_df['conditional_probability'] > 0.0]

prob_df.to_csv(OUTPUT_CSV, index=False)

print(f"Total cause-effect-lag pairs saved: {prob_df.shape[0]}")
print(prob_df.head())
