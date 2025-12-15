#!/usr/bin/env bash
# run_all.sh — End-to-end MIMIC experiment pipeline for CasFly
#
# Steps:
#   0. Launch all device UDP listeners (Device1-Device26)
#   1. Wait for Device20 (initiator) to be ready, then fire MIMIC patient runs
#   2. Collect metrics logs and validate chains against ground-truth edges
#   3. Run SOTA baselines (CPT, Greedy, PCMCI+)
#   4. Emulate scalability across device counts and latency profiles
#   5. Generate summary plots and comparison tables
#
# Prerequisites:
#   - config.yaml is present in the same directory
#   - All_Device_Lookup_with_Probabilities.csv and filtered_patients_updated.csv
#     are accessible at the paths set in LOOKUP_FILE / PATIENTS_CSV
#
# Usage:
#   cd mimic_variant/
#   bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

# --- env defaults (can be overridden before calling this script) ---
export LOOKUP_FILE="${LOOKUP_FILE:-$PWD/All_Device_Lookup_with_Probabilities.csv}"
export PATIENTS_CSV="${PATIENTS_CSV:-$PWD/filtered_patients_updated.csv}"

# Sanity-check required files
[[ -f "$LOOKUP_FILE" ]]  || { echo "ERROR: LOOKUP_FILE not found: $LOOKUP_FILE"; exit 1; }
[[ -f "$PATIENTS_CSV" ]] || { echo "ERROR: PATIENTS_CSV not found: $PATIENTS_CSV"; exit 1; }

mkdir -p ../experiments_out ./logs

echo "Step 0: launch device listeners"
bash ./launch_devices.sh

# Wait until Device20's UDP port is open before sending the first initiation
D20_PORT="$(python3 - <<'PY'
import os, pandas as pd
lf = os.environ.get("LOOKUP_FILE")
df = pd.read_csv(lf)
r  = df.loc[df['device'].astype(str).str.strip().str.lower() == 'device20']
print(int(r['port'].iloc[0]) if not r.empty else 6019)
PY
)"
echo "Waiting for Device20 on UDP port $D20_PORT ..."
for i in {1..60}; do
  if lsof -nP -iUDP:$D20_PORT >/dev/null 2>&1; then
    echo "Device20 is listening."
    break
  fi
  sleep 1
  if [[ $i -eq 60 ]]; then
    echo "ERROR: Device20 not detected on UDP $D20_PORT after 60s. Aborting."
    exit 1
  fi
done

echo "Step 1: initiate MIMIC runs"
bash ./initiate_mimic.sh

echo "Step 2: collect and validate"
python3 collect_and_validate.py

echo "Step 3: run baselines"
python3 run_baselines.py || true   # non-fatal: PCMCI+ may be unavailable

echo "Step 4: scalability emulation"
chmod +x _one_initiate_wrapper.sh
python3 scalability_emulation.py

echo "Step 5: generate plots and tables"
python3 make_plots.py

echo "Done. All outputs are under experiments_out/"
