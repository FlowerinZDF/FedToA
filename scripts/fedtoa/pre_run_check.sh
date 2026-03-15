#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

script_name="${SCRIPT_NAME:-manual}"
dataset="${DATASET_NAME:-unknown}"
algorithm="${ALGORITHM:-fedtoa}"
out_dir="${FEDTOA_OUT_DIR:-outputs/fedtoa/manual}"
log_file="${FEDTOA_LOG_FILE:-logs/fedtoa/precheck_$(fedtoa_ts).log}"

beta_topo="${BETA_TOPO:-1.0}"
gamma_spec="${GAMMA_SPEC:-1.0}"
eta_lip="${ETA_LIP:-1.0}"
warmup_rounds="${WARMUP_ROUNDS:-5}"
warmup_start_beta="${WARMUP_START_BETA:-0.05}"
warmup_mode="${WARMUP_MODE:-linear}"
prompt_only="${FEDTOA_PROMPT_ONLY:-true}"
freeze_backbone="${FREEZE_BACKBONE:-true}"
topk_edges="${TOPK_EDGES:-2048}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-0.5}"

fedtoa_prepare_paths "$out_dir" "$log_file"

fedtoa_print_run_config \
  "$script_name" "$dataset" "$algorithm" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "$prompt_only" "$freeze_backbone" "$topk_edges" "$var_threshold" \
  "$out_dir" "$log_file"

echo "[PRECHECK] output_dir_ok=true"
echo "[PRECHECK] log_path=${log_file}"
echo "[PRECHECK] comm_estimation_header=teacher_payload_bytes + student_upload_bytes + server_broadcast_bytes"
