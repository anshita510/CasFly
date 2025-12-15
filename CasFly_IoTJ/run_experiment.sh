#!/usr/bin/env bash
# run_experiment.sh
#
# Reads filtered_patients_updated.csv (columns: PATIENT, DESCRIPTION, CODE,
# CONDITION, RELEVANT_DEVICES) and, for each patient/disease pair, spawns
# CasFlyHub as the chain initiator on its designated UDP port.
#
# CasFlyHub acts as the entry point into the CasFly causal-chain network.
# A 20-second sleep between pairs prevents port conflicts from overlapping runs.
#
# Usage:
#   bash run_experiment.sh

csv_file="../data/synthea/processed/filtered_patients_updated.csv"

device20_script="../data/synthea/device_scripts/CasFlyHub/CasFlyHub.py"

python_interpreter="python3"

device20_port=6019

# Free a UDP port if it is already bound (kills the occupying process)
cleanup_port() {
    local port=$1
    echo "Checking if port $port is in use..."
    while lsof -i udp:$port > /dev/null; do
        echo "Port $port is in use. Terminating the process..."
        pid=$(lsof -i udp:$port | awk 'NR>1 {print $2}' | head -n 1)
        if [ -n "$pid" ]; then
            kill -9 "$pid" && echo "Terminated process using port $port (PID: $pid)."
        fi
        sleep 2
    done
    echo "Port $port is now free."
}

# Launch CasFlyHub for a single patient/disease pair
process_patient_disease() {
    local patient=$1
    local disease_name=$2

    echo "Starting initiator (CasFlyHub) for Patient: $patient, Disease: $disease_name..."

    cleanup_port "$device20_port"

    $python_interpreter "$device20_script" --initiate "$disease_name" --patient_id "$patient" &
    device20_pid=$!

    echo "CasFlyHub started (PID: $device20_pid) on port $device20_port."
    wait $device20_pid
    echo "CasFlyHub (PID: $device20_pid) finished."
}

# Iterate over every patient/disease row; skip the CSV header
while IFS=',' read -r PATIENT DESCRIPTION CODE CONDITION RELEVANT_DEVICES; do

    if [[ "$PATIENT" == "PATIENT" ]]; then
        continue
    fi

    # Strip whitespace that may be present in the CSV
    disease_name=$(echo "$DESCRIPTION" | tr -d '[:space:]')
    patient=$(echo "$PATIENT" | tr -d '[:space:]')

    process_patient_disease "$patient" "$disease_name"

    echo "Waiting for 20 seconds before starting the next pair..."
    sleep 20
done < "$csv_file"

echo "All initiator processes triggered."
