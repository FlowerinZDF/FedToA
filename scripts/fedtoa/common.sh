#!/usr/bin/env bash
# Shared FedToA script helpers.
set -euo pipefail

fedtoa_ts() {
  date +%Y%m%d_%H%M%S
}

fedtoa_bool() {
  if [[ "${1:-0}" == "1" || "${1:-false}" == "true" ]]; then
    echo true
  else
    echo false
  fi
}

fedtoa_prepare_paths() {
  local out_dir="$1"
  local log_file="$2"
  mkdir -p "$out_dir" "$(dirname "$log_file")"
}

fedtoa_print_run_config() {
  cat <<CFG
[RUN_CONFIG] script=$1
[RUN_CONFIG] dataset=$2
[RUN_CONFIG] algorithm=$3
[RUN_CONFIG] beta_topo=$4
[RUN_CONFIG] gamma_spec=$5
[RUN_CONFIG] eta_lip=$6
[RUN_CONFIG] warmup_rounds=$7
[RUN_CONFIG] warmup_start_beta=$8
[RUN_CONFIG] warmup_mode=$9
[RUN_CONFIG] fedtoa_prompt_only=${10}
[RUN_CONFIG] freeze_backbone=${11}
[RUN_CONFIG] topk_edges=${12}
[RUN_CONFIG] fedtoa_var_threshold=${13}
[RUN_CONFIG] output_dir=${14}
[RUN_CONFIG] log_file=${15}
[RUN_CONFIG] comm_estimation=teacher_payload_bytes + student_upload_bytes + server_broadcast_bytes
CFG
}
