#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../fedtoa/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

goal="${GOAL:-YourGoal}"
root="${DATA_ROOT_PREFIX:-}"
out_dir="${FEDCOLA_OUT_DIR:-outputs/fedtoa_compare/fedcola_flickr_short}"
log_file="${FEDCOLA_LOG_FILE:-logs/fedtoa_compare/fedcola_flickr_short_$(fedtoa_ts).log}"

ic="${IC:-12}"          # number of img clients
tc="${TC:-12}"          # number of txt clients
mc="${MC:-8}"           # number of img+txt clients
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-0.25}"
nt="${NUM_THREAD:-8}"

b="${BATCH_SIZE:-16}"
eval_b="${EVAL_BATCH_SIZE:-32}"
rounds="${ROUNDS:-3}"
local_epochs="${LOCAL_EPOCHS:-1}"

fedtoa_prepare_paths "$out_dir" "$log_file"

echo "[RUN_CONFIG] script=fedcola_flickr_short.sh"
echo "[RUN_CONFIG] dataset=Flickr30k"
echo "[RUN_CONFIG] algorithm=fedavg"
echo "[RUN_CONFIG] rounds=${rounds}"
echo "[RUN_CONFIG] local_epochs=${local_epochs}"
echo "[RUN_CONFIG] batch_size=${b}"
echo "[RUN_CONFIG] eval_batch_size=${eval_b}"
echo "[RUN_CONFIG] teachers=${mc}"
echo "[RUN_CONFIG] students=$((ic + tc))"
echo "[RUN_CONFIG] img_clients=${ic}"
echo "[RUN_CONFIG] txt_clients=${tc}"
echo "[RUN_CONFIG] output_dir=${out_dir}"
echo "[RUN_CONFIG] log_file=${log_file}"

echo "[PRECHECK] output directory ready: ${out_dir}"
echo "[PRECHECK] dataset/algorithm/script identity: Flickr30k / fedavg / fedcola_flickr_short.sh"

python main.py \
  --exp_name FedCola \
  --result_path "$out_dir" \
  --log_path "$(dirname "$log_file")" \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedavg \
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
  --eval_batch_size "$eval_b" \
  2>&1 | tee "$log_file"
