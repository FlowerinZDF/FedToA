#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

goal="${GOAL:-YourGoal}"
root="${DATA_ROOT_PREFIX:-}"
out_dir="${FEDTOA_OUT_DIR:-outputs/fedtoa/flickr_short}"
log_file="${FEDTOA_LOG_FILE:-logs/fedtoa/flickr_short_$(fedtoa_ts).log}"

ic="${IC:-12}"
tc="${TC:-12}"
mc="${MC:-8}"
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-0.25}"
nt="${NUM_THREAD:-8}"
b="${BATCH_SIZE:-128}"
rounds="${ROUNDS:-3}"
local_epochs="${LOCAL_EPOCHS:-1}"

beta_topo="${BETA_TOPO:-0.2}"
gamma_spec="${GAMMA_SPEC:-0.0}"
eta_lip="${ETA_LIP:-0.0}"
warmup_rounds="${WARMUP_ROUNDS:-5}"
warmup_start_beta="${WARMUP_START_BETA:-0.0}"
warmup_mode="${WARMUP_MODE:-linear}"
topk_edges="${TOPK_EDGES:-2048}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-0.5}"

fedtoa_prepare_paths "$out_dir" "$log_file"

fedtoa_print_run_config \
  "flickr_short.sh" "Flickr30k" "fedtoa" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "true" "true" "$topk_edges" "$var_threshold" \
  "$out_dir" "$log_file"

echo "[PRECHECK] output directory ready: ${out_dir}"
echo "[PRECHECK] dataset/algorithm/script identity: Flickr30k / fedtoa / flickr_short.sh"

python main.py \
  --exp_name FedToA \
  --result_path "$out_dir" \
  --log_path "$(dirname "$log_file")" \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedtoa \
  --seed "${SEED:-1}" \
  --multi-task \
  --datasets CIFAR100 AG_NEWS Flickr30k Coco \
  --modalities img txt img+txt img+txt \
  --data_paths "${root}data/cifar100" "${root}data/agnews" "${root}data/flickr30k" "${root}data/coco" \
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
  --lr 1e-4 \
  --lr_decay 0.99 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread "$nt" \
  --loader_num_workers "${LOADER_NUM_WORKERS:-6}" \
  --loader_pin_memory \
  --loader_persistent_workers \
  --loader_prefetch_factor "${LOADER_PREFETCH_FACTOR:-4}" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --eval_batch_size 512 \
  --fedtoa_prompt_only \
  --freeze_backbone \
  --use_topo \
  --no-use_spec \
  --no-use_lip \
  --tau 0.2 \
  --eig_k 4 \
  --topk_edges "$topk_edges" \
  --fedtoa_var_threshold "$var_threshold" \
  --beta_topo "$beta_topo" \
  --fedtoa_topo_warmup_rounds "$warmup_rounds" \
  --fedtoa_topo_warmup_start_beta "$warmup_start_beta" \
  --fedtoa_topo_warmup_mode "$warmup_mode" \
  --gamma_spec "$gamma_spec" \
  --eta_lip "$eta_lip" \
  --fedtoa_retrieval_task_weight "${FEDTOA_RETRIEVAL_TASK_WEIGHT:-1.0}" \
  --fedtoa_aux_task_weight "${FEDTOA_AUX_TASK_WEIGHT:-0.2}" \
  --prompt_len 10 \
  --diagonal_eps 1e-4 \
  2>&1 | tee "$log_file"
