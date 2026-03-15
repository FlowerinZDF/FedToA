#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

goal="${GOAL:-FedToA-Smoke}"
root="${DATA_ROOT_PREFIX:-}"
out_dir="${FEDTOA_OUT_DIR:-outputs/fedtoa/smoke}"
log_file="${FEDTOA_LOG_FILE:-logs/fedtoa/flickr_smoke_$(fedtoa_ts).log}"

ic="${IC:-1}"
tc="${TC:-1}"
mc="${MC:-2}"
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-1.0}"
nt="${NUM_THREAD:-2}"
b="${BATCH_SIZE:-8}"
rounds="${ROUNDS:-2}"
local_epochs="${LOCAL_EPOCHS:-1}"
reduce_samples="${REDUCE_SAMPLES:-64}"
reduce_test_samples="${REDUCE_TEST_SAMPLES:-32}"

beta_topo="${BETA_TOPO:-1.0}"
gamma_spec="${GAMMA_SPEC:-1.0}"
eta_lip="${ETA_LIP:-1.0}"
warmup_rounds="${WARMUP_ROUNDS:-2}"
warmup_start_beta="${WARMUP_START_BETA:-0.05}"
warmup_mode="${WARMUP_MODE:-linear}"
topk_edges="${TOPK_EDGES:-64}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-none}"

fedtoa_prepare_paths "$out_dir" "$log_file"

fedtoa_print_run_config \
  "flickr_smoke.sh" "Flickr30k" "fedtoa" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "true" "true" "$topk_edges" "$var_threshold" \
  "$out_dir" "$log_file"

echo "[PRECHECK] output directory ready: ${out_dir}"
echo "[PRECHECK] estimated communication header will be emitted by FedToA server before training rounds."

python main.py \
  --exp_name FedToA_Flickr_Smoke \
  --output_path "$out_dir" \
  --shared_param none \
  --share_scope dataset \
  --colearn_param none \
  --algorithm fedtoa \
  --seed "${SEED:-1}" \
  --multi-task \
  --datasets Flickr30k Flickr30k Flickr30k Flickr30k \
  --modalities img txt img+txt img+txt \
  --data_paths "${root}data/flickr30k" "${root}data/flickr30k" "${root}data/flickr30k" "${root}data/flickr30k" \
  --Ks "$ic" "$tc" "$mc" \
  --Cs "$c" \
  --test_size -1 \
  --split_type diri \
  --cncntrtn "$cncntrtn" \
  --model_name mome_small_patch16 \
  --resize 224 \
  --imnorm \
  --eval_type global \
  --eval_every 1 \
  --eval_metrics acc1 \
  --R "$rounds" \
  --E "$local_epochs" \
  --B "$b" \
  --beta1 0 \
  --optimizer AdamW \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --lr_decay 1.0 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread "$nt" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --reduce_samples "$reduce_samples" \
  --reduce_test_samples "$reduce_test_samples" \
  --eval_batch_size 64 \
  --fedavg_eval \
  --fedtoa_prompt_only \
  --freeze_backbone \
  --use_topo \
  --use_spec \
  --use_lip \
  --topk_edges "$topk_edges" \
  --beta_topo "$beta_topo" \
  --gamma_spec "$gamma_spec" \
  --eta_lip "$eta_lip" \
  --fedtoa_topo_warmup_rounds "$warmup_rounds" \
  --fedtoa_topo_warmup_start_beta "$warmup_start_beta" \
  --fedtoa_topo_warmup_mode "$warmup_mode" \
  2>&1 | tee "$log_file"
