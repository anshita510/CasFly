#!/usr/bin/env bash
# reproduce_casfly_results.sh
#
# Two-step pipeline to reproduce the CasFly paper results.
#
# Step 1 (compare_pcmci_lave.py):
#   Runs a tau sweep comparing PCMCI+ against LaVE causal edges.
#   For each tau value it trains PCMCI+ per device, extracts LaVE edges
#   from the merged run log, computes overlap/precision/recall/F1/Hits@K,
#   and writes tables + plots under OUT_ROOT.
#
# Step 2 (evaluate_predictive_metrics.py):
#   Adds predictive-utility analysis (per-effect logistic models),
#   reverse Hits@K, and paired statistical tests (Wilcoxon + Cliff's delta)
#   comparing LaVE vs PCMCI+.
#
# Usage:
#   bash reproduce_casfly_results.sh [out_root_dir]
#
# Override defaults via environment variables, e.g.:
#   TAUS=7,30,90 bash reproduce_casfly_results.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_DIR="${ROOT_DIR}/../data/synthea"

OUT_ROOT="${1:-${CSV_DIR}/artifacts/repro_paper}"
TAUS="${TAUS:-7,14,30,60,90}"
ALPHA="${ALPHA:-0.05}"
MIN_DAYS_PRESENT="${MIN_DAYS_PRESENT:-1}"
HITS_K="${HITS_K:-1,3,5}"
TOP_N_DEVICES="${TOP_N_DEVICES:-12}"

DEVICES_ROOT="${CSV_DIR}/device_scripts"
LOG_CSV="${CSV_DIR}/metrics/allmerged_metrics_log.csv"

check_file() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] Missing required path: $p" >&2
    exit 1
  fi
}

check_file "${DEVICES_ROOT}"
check_file "${LOG_CSV}"
check_file "${ROOT_DIR}/compare_pcmci_lave.py"
check_file "${ROOT_DIR}/evaluate_predictive_metrics.py"

mkdir -p "${OUT_ROOT}"

echo "[1/2] Running tau-sweep pipeline (PCMCI+ vs LaVE)"
python3 "${ROOT_DIR}/compare_pcmci_lave.py" \
  --devices_root "${DEVICES_ROOT}" \
  --log_csv "${LOG_CSV}" \
  --out_root "${OUT_ROOT}" \
  --taus "${TAUS}" \
  --alpha "${ALPHA}" \
  --min_days_present "${MIN_DAYS_PRESENT}" \
  --hits_k "${HITS_K}" \
  --top_n_devices "${TOP_N_DEVICES}" \
  --resume

echo "[2/2] Running paper addon metrics"
python3 "${ROOT_DIR}/evaluate_predictive_metrics.py" \
  --devices_root "${DEVICES_ROOT}" \
  --lave_csv "${OUT_ROOT}/lave_edges_normalized.csv" \
  --pcmci_root_base "${OUT_ROOT}/pcmci_tau_" \
  --taus "${TAUS}" \
  --out_root "${OUT_ROOT}/addons" \
  --min_days_present "${MIN_DAYS_PRESENT}"

echo
echo "Repro complete. Outputs:"
echo "  - ${OUT_ROOT}/plots"
echo "  - ${OUT_ROOT}/tables"
echo "  - ${OUT_ROOT}/tau_*/{edge_overlap_counts.csv,computed_metrics_from_overlap.csv,hits_at_k.csv}"
echo "  - ${OUT_ROOT}/addons/predictive/{per_device_metrics.csv,summary_by_tau.csv,paired_tests.csv}"
echo "  - ${OUT_ROOT}/addons/reverse_hits/reverse_hits_by_tau.csv"
