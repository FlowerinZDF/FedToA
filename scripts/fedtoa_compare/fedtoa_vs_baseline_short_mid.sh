#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../fedtoa/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

run_level="${RUN_LEVEL:-short}"  # short|mid
dataset="${DATASET:-flickr}"      # flickr|coco

goal="${GOAL:-FedToA-vs-FedCola}"
root="${DATA_ROOT_PREFIX:-}"
out_root="${COMPARE_OUT_DIR:-outputs/fedtoa_compare}"
log_root="${COMPARE_LOG_DIR:-logs/fedtoa_compare}"

ic="${IC:-12}"
tc="${TC:-12}"
mc="${MC:-8}"
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-0.25}"
nt="${NUM_THREAD:-8}"
seed="${SEED:-1}"

case "$run_level" in
  short)
    rounds="${ROUNDS:-3}"
    local_epochs="${LOCAL_EPOCHS:-1}"
    ;;
  mid)
    rounds="${ROUNDS:-10}"
    local_epochs="${LOCAL_EPOCHS:-2}"
    ;;
  *)
    echo "Unsupported RUN_LEVEL=${run_level}. Use short or mid." >&2
    exit 1
    ;;
esac

case "$dataset" in
  flickr)
    dataset_name="Flickr30k"
    datasets=(CIFAR100 AG_NEWS Flickr30k Coco)
    data_paths=("${root}data/cifar100" "${root}data/agnews" "${root}data/flickr30k" "${root}data/coco")
    batch_size="${BATCH_SIZE:-112}"
    ;;
  coco)
    dataset_name="MS-COCO"
    datasets=(CIFAR100 AG_NEWS Coco Coco)
    data_paths=("${root}data/cifar100" "${root}data/agnews" "${root}data/coco" "${root}data/coco")
    batch_size="${BATCH_SIZE:-96}"
    ;;
  *)
    echo "Unsupported DATASET=${dataset}. Use flickr or coco." >&2
    exit 1
    ;;
esac

beta_topo="${BETA_TOPO:-1.0}"
gamma_spec="${GAMMA_SPEC:-1.0}"
eta_lip="${ETA_LIP:-1.0}"
warmup_rounds="${WARMUP_ROUNDS:-5}"
warmup_start_beta="${WARMUP_START_BETA:-0.05}"
warmup_mode="${WARMUP_MODE:-linear}"
topk_edges="${TOPK_EDGES:-2048}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-0.5}"

mkdir -p "$out_root" "$log_root"
ts="$(fedtoa_ts)"

fedtoa_log="${log_root}/fedtoa_${dataset}_${run_level}_${ts}.log"
base_log="${log_root}/fedcola_${dataset}_${run_level}_${ts}.log"

fedtoa_print_run_config \
  "fedtoa_vs_baseline_short_mid.sh" "$dataset_name" "fedtoa" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "true" "true" "$topk_edges" "$var_threshold" \
  "$out_root" "$fedtoa_log"

echo "[COMPARE] launching baseline=fedavg then fedtoa on dataset=${dataset_name} level=${run_level}"

echo "[RUN_CONFIG] script=fedtoa_vs_baseline_short_mid.sh" | tee "$base_log"
echo "[RUN_CONFIG] dataset=${dataset_name}" | tee -a "$base_log"
echo "[RUN_CONFIG] algorithm=fedavg" | tee -a "$base_log"
echo "[RUN_CONFIG] output_dir=${out_root}" | tee -a "$base_log"
echo "[RUN_CONFIG] log_file=${base_log}" | tee -a "$base_log"
python main.py \
  --exp_name FedCola \
  --result_path "$out_root" \
  --log_path "$log_root" \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedavg \
  --seed "$seed" \
  --multi-task \
  --datasets "${datasets[@]}" \
  --modalities img txt img+txt img+txt \
  --data_paths "${data_paths[@]}" \
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
  --B "$batch_size" \
  --beta1 0 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr_decay 0.99 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread "$nt" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --eval_batch_size 512 \
  2>&1 | tee -a "$base_log"

python main.py \
  --exp_name FedToA \
  --result_path "$out_root" \
  --log_path "$log_root" \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedtoa \
  --seed "$seed" \
  --multi-task \
  --datasets "${datasets[@]}" \
  --modalities img txt img+txt img+txt \
  --data_paths "${data_paths[@]}" \
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
  --B "$batch_size" \
  --beta1 0 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr_decay 0.99 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread "$nt" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --eval_batch_size 512 \
  --fedtoa_prompt_only \
  --freeze_backbone \
  --use_topo \
  --use_spec \
  --use_lip \
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
  --prompt_len 10 \
  --diagonal_eps 1e-4 \
  2>&1 | tee "$fedtoa_log"

echo "[COMPARE] logs: baseline=${base_log}, fedtoa=${fedtoa_log}"
