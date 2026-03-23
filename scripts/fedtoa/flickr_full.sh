#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

goal="${GOAL:-FedToA-Flickr-Full}"
root="${DATA_ROOT_PREFIX:-}"
out_dir="${FEDTOA_OUT_DIR:-outputs/fedtoa/flickr_full}"
log_file="${FEDTOA_LOG_FILE:-logs/fedtoa/flickr_full_$(fedtoa_ts).log}"

# client split aligned with FedCola
ic="${IC:-12}"
tc="${TC:-12}"
mc="${MC:-8}"
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-0.25}"
nt="${NUM_THREAD:-4}"
seed="${SEED:-1}"

# full run aligned with original FedCola Flickr script
rounds="${ROUNDS:-30}"
local_epochs="${LOCAL_EPOCHS:-5}"
b="${BATCH_SIZE:-32}"
eval_b="${EVAL_BATCH_SIZE:-64}"
eval_every="${EVAL_EVERY:-5}"

# dataloader
loader_workers="${LOADER_NUM_WORKERS:-2}"
loader_prefetch="${LOADER_PREFETCH_FACTOR:-2}"
loader_persistent="${LOADER_PERSISTENT_WORKERS:-false}"

# current practical FedToA defaults
beta_topo="${BETA_TOPO:-0.05}"
gamma_spec="${GAMMA_SPEC:-0.0}"
eta_lip="${ETA_LIP:-0.0}"
warmup_rounds="${WARMUP_ROUNDS:-5}"
warmup_start_beta="${WARMUP_START_BETA:-0.0}"
warmup_mode="${WARMUP_MODE:-linear}"
topk_edges="${TOPK_EDGES:-1024}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-0.5}"

retrieval_w="${FEDTOA_RETRIEVAL_TASK_WEIGHT:-1.0}"
aux_w="${FEDTOA_AUX_TASK_WEIGHT:-0.1}"
topo_min_edges="${FEDTOA_TOPO_MIN_ACTIVE_EDGES:-8}"
topo_loss_cap="${FEDTOA_TOPO_LOSS_CAP:-0.5}"
topo_task_ratio_cap="${FEDTOA_TOPO_TASK_RATIO_CAP:-0.10}"

fedtoa_prepare_paths "$out_dir" "$log_file"

fedtoa_print_run_config \
  "flickr_full.sh" "Flickr30k" "fedtoa" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "true" "true" "$topk_edges" "$var_threshold" \
  "$out_dir" "$log_file"

echo "[RUN_CONFIG] script=flickr_full.sh"
echo "[RUN_CONFIG] dataset=Flickr30k"
echo "[RUN_CONFIG] algorithm=fedtoa"
echo "[RUN_CONFIG] rounds=${rounds}"
echo "[RUN_CONFIG] local_epochs=${local_epochs}"
echo "[RUN_CONFIG] batch_size=${b}"
echo "[RUN_CONFIG] eval_batch_size=${eval_b}"
echo "[RUN_CONFIG] eval_every=${eval_every}"
echo "[RUN_CONFIG] retrieval_task_weight=${retrieval_w}"
echo "[RUN_CONFIG] aux_task_weight=${aux_w}"
echo "[RUN_CONFIG] topo_min_active_edges=${topo_min_edges}"
echo "[RUN_CONFIG] topo_loss_cap=${topo_loss_cap}"
echo "[RUN_CONFIG] topo_task_ratio_cap=${topo_task_ratio_cap}"
echo "[RUN_CONFIG] loader_num_workers=${loader_workers}"
echo "[RUN_CONFIG] loader_prefetch_factor=${loader_prefetch}"
echo "[RUN_CONFIG] loader_persistent_workers=${loader_persistent}"
echo "[RUN_CONFIG] output_dir=${out_dir}"
echo "[RUN_CONFIG] log_file=${log_file}"

echo "[PRECHECK] output directory ready: ${out_dir}"
echo "[PRECHECK] dataset/algorithm/script identity: Flickr30k / fedtoa / flickr_full.sh"

loader_persistent_flag="--no-loader_persistent_workers"
if [[ "${loader_persistent}" == "true" ]]; then
  loader_persistent_flag="--loader_persistent_workers"
fi

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
  --seed "$seed" \
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
  --eval_every "$eval_every" \
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
  --loader_num_workers "$loader_workers" \
  --loader_pin_memory \
  ${loader_persistent_flag} \
  --loader_prefetch_factor "$loader_prefetch" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --eval_batch_size "$eval_b" \
  --no-detect_anomaly \
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
  --fedtoa_student_objective "${FEDTOA_STUDENT_OBJECTIVE:-retrieval_plus_aux}" \
  --fedtoa_retrieval_task_weight "$retrieval_w" \
  --fedtoa_aux_task_weight "$aux_w" \
  --fedtoa_topo_min_active_edges "$topo_min_edges" \
  --fedtoa_topo_loss_cap "$topo_loss_cap" \
  --fedtoa_topo_task_ratio_cap "$topo_task_ratio_cap" \
  --prompt_len 10 \
  --diagonal_eps 1e-4 \
  2>&1 | tee "$log_file"

echo "[DONE] log: ${log_file}"
echo "[DONE] output: ${out_dir}"
