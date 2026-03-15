#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1

goal=YourGoal

ic=12 # number of img clients
tc=12 # number of txt clients
mc=8 # number of img+txt clients
cncntrtn=0.5 # concentration parameter for Dirichlet distribution
c=0.25 # sampling ratio for clients
nt=8 # number of threads for parallel training
b=112 # batch size
root='' # root path of the dataset

python main.py \
  --exp_name FedToA \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedtoa \
  --seed 1 \
  --multi-task \
  --datasets CIFAR100 AG_NEWS Flickr30k Coco \
  --modalities img txt img+txt img+txt \
  --data_paths ${root}data/cifar100 ${root}data/agnews ${root}data/flickr30k ${root}data/coco \
  --Ks $ic $tc $mc \
  --Cs $c \
  --test_size -1 \
  --split_type diri \
  --cncntrtn $cncntrtn \
  --model_name mome_small_patch16 \
  --resize 224 \
  --imnorm \
  --eval_type global \
  --eval_every 1 \
  --eval_metrics acc1 \
  --R 30 \
  --E 5 \
  --B $b \
  --beta1 0 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr_decay 0.99 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread $nt \
  --use_bert_tokenizer \
  --pretrained \
  --goal $goal \
  --equal_sampled \
  --eval_batch_size 512 \
  --use_topo \
  --use_spec \
  --use_lip \
  --tau 0.2 \
  --eig_k 4 \
  --beta_topo 1.0 \
  --gamma_spec 1.0 \
  --eta_lip 1.0 \
  --prompt_len 10 \
  --diagonal_eps 1e-4
