#!/usr/bin/env bash
# ============================================================
# launch_devices.sh — Start all CasFly device listener processes
#
# Description:
#   Reads the device lookup CSV to discover all device names,
#   validates the CSV for required columns and port collisions,
#   then launches each device's Python listener as a background
#   process with logs written to ./logs/devices/<DeviceX>.log.
#
#   Already-running devices are skipped. Devices without a
#   matching script file are warned and skipped.
#
# Usage:
#   bash launch_devices.sh
#
#   Optional environment variables:
#     LOOKUP_FILE  — path to All_Device_Lookup_with_Probabilities.csv
#                    (default: auto-discovered in ./ or ../)
#     EXCLUDE      — regex of device names to skip, e.g. "Device6|Device10"
#
# Outputs:
#   ./logs/devices/<DeviceX>.log  — stdout/stderr for each device
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# -------------------------
# 1) Locate lookup CSV
# -------------------------
LOOKUP_FILE="${LOOKUP_FILE:-./All_Device_Lookup_with_Probabilities.csv}"
if [[ ! -f "$LOOKUP_FILE" ]]; then
  if   [[ -f "./newAll_Device_Lookup_with_Probabilities.csv" ]]; then
    LOOKUP_FILE="./newAll_Device_Lookup_with_Probabilities.csv"
  elif [[ -f "../All_Device_Lookup_with_Probabilities.csv" ]]; then
    LOOKUP_FILE="../All_Device_Lookup_with_Probabilities.csv"
  elif [[ -f "../newAll_Device_Lookup_with_Probabilities.csv" ]]; then
    LOOKUP_FILE="../newAll_Device_Lookup_with_Probabilities.csv"
  else
    echo "ERROR: Could not find lookup CSV.
Tried:
  ./All_Device_Lookup_with_Probabilities.csv
  ./newAll_Device_Lookup_with_Probabilities.csv
  ../All_Device_Lookup_with_Probabilities.csv
  ../newAll_Device_Lookup_with_Probabilities.csv
Or set:  export LOOKUP_FILE=/absolute/path/to/your.csv"
    exit 1
  fi
fi
echo "Using lookup: $LOOKUP_FILE"

# -------------------------
# 2) Validate CSV (cols & port collisions)
# -------------------------
CHECK_OUT="$(python3 - <<'PY'
import os, sys, pandas as pd
lf = os.environ.get("LOOKUP_FILE")
try:
    df = pd.read_csv(lf)
except Exception as e:
    print("READ_CSV_ERROR:", e)
    sys.exit(10)

need = {"device","ip","port"}
missing = need - set(df.columns)
if missing:
    print("MISSING_COLS:" + ",".join(sorted(missing)))
    sys.exit(11)

dups = df[df.duplicated(["ip","port"], keep=False)].copy()
if not dups.empty:
    print("DUP_PORTS")
    print(dups[["device","ip","port"]].to_csv(index=False))
    sys.exit(12)

for d in df["device"].dropna().unique():
    s = str(d).strip()
    if s:
        print("DEV="+s)
PY
)" || true

if [[ "$CHECK_OUT" == READ_CSV_ERROR* ]]; then
  echo "$CHECK_OUT"; exit 1
fi
if [[ "$CHECK_OUT" == MISSING_COLS:* ]]; then
  echo "ERROR: $CHECK_OUT"
  echo "Tip: regenerate the lookup with an 'ip' column (default 127.0.0.1 is fine)."
  exit 1
fi
if echo "$CHECK_OUT" | grep -q "^DUP_PORTS$"; then
  echo "ERROR: (ip,port) collisions detected in $LOOKUP_FILE"
  echo "$CHECK_OUT" | sed -n '2,$p'
  exit 1
fi

DEVICES="$(echo "$CHECK_OUT" | sed -n 's/^DEV=//p')"
if [[ -z "$DEVICES" ]]; then
  echo "No devices listed in: $LOOKUP_FILE"
  exit 1
fi

# Optional exclude pattern: e.g. EXCLUDE="Device6|Device10"
EXCLUDE="${EXCLUDE:-}"

# -------------------------
# 3) Where device code lives
# -------------------------
BASES=( "./data/synthea/device_scripts" "../data/synthea/device_scripts" )
find_script_path () {
  local dev="$1"
  local path=""
  for base in "${BASES[@]}"; do
    if [[ -f "${base}/${dev}/${dev}.py" ]]; then
      path="${base}/${dev}/${dev}.py"
      break
    fi
  done
  echo "$path"
}

is_running () {
  # returns 0 if a python proc already running this script path
  local script_path="$1"
  pgrep -f "python[0-9]* .*${script_path}" >/dev/null 2>&1
}

# -------------------------
# 4) Launch devices with visible logs
# -------------------------
echo "Launching device listeners..."
mkdir -p ./logs/devices
started=0; skipped_missing=0; skipped_running=0; skipped_excluded=0

for DEV in $DEVICES; do
  if [[ -n "$EXCLUDE" ]] && [[ "$DEV" =~ $EXCLUDE ]]; then
    echo "SKIP (excluded): $DEV"
    ((skipped_excluded++)) || true
    continue
  fi

  SCRIPT="$(find_script_path "$DEV")"
  if [[ -z "$SCRIPT" ]]; then
    echo "WARN: Could not find script for $DEV in any known BASE."
    ((skipped_missing++)) || true
    continue
  fi

  if is_running "$SCRIPT"; then
    echo "SKIP (already running): $DEV ($SCRIPT)"
    ((skipped_running++)) || true
    continue
  fi

  LOG="./logs/devices/${DEV}.log"
  echo "Starting $DEV via $SCRIPT -> $LOG"
  # Unbuffered logs so tail -f shows things immediately
  env PYTHONUNBUFFERED=1 LOOKUP_FILE="$LOOKUP_FILE" \
    nohup python3 -u "$SCRIPT" >"$LOG" 2>&1 &
  ((started++)) || true
  sleep 0.2
done

echo "All device listeners launched."
echo "Summary: started=$started, already_running=$skipped_running, missing_script=$skipped_missing, excluded=$skipped_excluded"
echo "Tip: tail -f logs/devices/Device20.log   # or any DeviceX.log to watch chain expansion."
