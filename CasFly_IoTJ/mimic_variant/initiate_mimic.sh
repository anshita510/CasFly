#!/usr/bin/env bash
# ============================================================
# initiate_mimic.sh — Sequentially initiate CasFly chain runs
#                     for all patients in filtered_patients_updated.csv
#
# Description:
#   For each patient row in the patients CSV, sends a UDP
#   initiation packet to Device20 triggering a full CasFly
#   chain expansion. Waits for a per-patient .done flag file
#   (written by the device code) before moving to the next
#   patient, with a 20-minute timeout per patient.
#
# Usage:
#   bash initiate_mimic.sh
#
#   Optional environment variables:
#     LOOKUP_FILE    — path to lookup CSV (default: $PWD/All_Device_Lookup_with_Probabilities.csv)
#     PATIENTS_CSV   — path to patients CSV (default: $PWD/filtered_patients_updated.csv)
#     EXP_OUT        — output directory for done-flags (default: $PWD/../experiments_out)
#
# Prerequisites:
#   launch_devices.sh must be run first so Device20 is listening.
#
# Outputs:
#   $EXP_OUT/done/<patient_id>.done  — completion flag per patient
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Inputs / defaults ---
LOOKUP_FILE="${LOOKUP_FILE:-$PWD/All_Device_Lookup_with_Probabilities.csv}"
PATIENTS_CSV="${PATIENTS_CSV:-$PWD/filtered_patients_updated.csv}"
EXP_OUT="${EXP_OUT:-$PWD/../experiments_out}"
DONE_DIR="$EXP_OUT/done"

[[ -f "$LOOKUP_FILE" ]]  || { echo "ERROR: LOOKUP_FILE not found: $LOOKUP_FILE"; exit 1; }
[[ -f "$PATIENTS_CSV" ]] || { echo "ERROR: PATIENTS_CSV not found: $PATIENTS_CSV"; exit 1; }
mkdir -p "$DONE_DIR"

# --- Resolve Device20 port from the lookup ---
D20_PORT="$(python3 - <<'PY'
import os, pandas as pd
lf=os.environ.get("LOOKUP_FILE")
df=pd.read_csv(lf)
r=df.loc[df['device'].astype(str).str.strip().str.lower()=='device20']
print(int(r['port'].iloc[0]) if not r.empty else 6019)
PY
)"

# --- Wait for Device20 UDP listener to be up once ---
echo "Waiting for Device20 on UDP port $D20_PORT ..."
for i in {1..30}; do
  if lsof -nP -iUDP:$D20_PORT >/dev/null 2>&1; then
    echo "Device20 is listening."
    break
  fi
  sleep 1
  [[ $i -eq 30 ]] && echo "WARN: Device20 not detected; continuing anyway."
done

# --- helper: send one initiation to Device20 ---
send_one() {
  local pid="$1" disease="$2"
  python3 - "$pid" "$disease" <<'PY'
import os, sys, json, socket, pandas as pd

patient_id = sys.argv[1]
event      = sys.argv[2]

lf = os.environ.get("LOOKUP_FILE")
df = pd.read_csv(lf)
r = df.loc[df['device'].astype(str).str.strip().str.lower()=='device20']
if r.empty:
    raise SystemExit("Device20 is not in lookup")

tgt_ip   = str(r['ip'].iloc[0])
tgt_port = int(r['port'].iloc[0])

pkt = {
    'effects': event,
    'patient_id': patient_id,
    'chain': [],
    'initiator': 'Device20',
    'final_packet': False
}
buf = json.dumps(pkt).encode('utf-8')
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.sendto(buf, (tgt_ip, tgt_port))
print(f"Sent to {tgt_ip}:{tgt_port} -> {pkt}")
PY
}

# --- sequential loop: one patient at a time, wait for done flag ---
total=$(($(wc -l < "$PATIENTS_CSV") - 1))
i=0

# clear any stale dones
rm -f "$DONE_DIR"/*.done 2>/dev/null || true

while IFS=',' read -r PATIENT DESCRIPTION CODE CONDITION RELEVANT_DEVICES; do
  # skip header if present
  [[ "${PATIENT^^}" == "PATIENT" ]] && continue
  ((i++)) || true

  disease="$(echo "$DESCRIPTION" | tr -d '[:space:]')"
  done_flag="$DONE_DIR/${PATIENT}.done"
  rm -f "$done_flag"

  echo "[$i/$total] Initiating $disease for patient $PATIENT"
  send_one "$PATIENT" "$disease"

  # wait up to 20 minutes; tweak if needed
  waited=0
  until [[ -f "$done_flag" ]]; do
    sleep 1
    ((waited++)) || true
    if (( waited % 60 == 0 )); then
      echo "  ...still waiting for $PATIENT ($waited s)"
    fi
    if (( waited >= 1200 )); then
      echo "TIMEOUT: No completion for $PATIENT ($disease). Moving on."
      break
    fi
  done

  [[ -f "$done_flag" ]] && echo "Completed patient $PATIENT ($disease)."
done < <(tail -n +2 "$PATIENTS_CSV")
