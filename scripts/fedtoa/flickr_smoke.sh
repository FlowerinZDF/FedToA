goal=FedToA-Smoke

# Small/fast smoke setup:
# - 4 total FL clients in this run: 1 img-only, 1 txt-only, 2 multimodal
# - short schedule to validate end-to-end FedToA runtime path
ic=1
tc=1
mc=2

cncntrtn=0.5
c=1.0
nt=2
b=8
root=''

python main.py \
  --exp_name FedToA_Flickr_Smoke \
  --shared_param none \
  --share_scope dataset \
  --colearn_param none \
  --algorithm fedtoa \
  --seed 1 \
  --multi-task \
  --datasets CIFAR100 AG_NEWS Flickr30k Flickr30k \
  --modalities img txt img+txt img+txt \
  --data_paths ${root}data/cifar100 ${root}data/agnews ${root}data/flickr30k ${root}data/flickr30k \
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
  --R 2 \
  --E 1 \
  --B $b \
  --beta1 0 \
  --optimizer AdamW \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --lr_decay 1.0 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread $nt \
  --use_bert_tokenizer \
  --pretrained \
  --goal $goal \
  --equal_sampled \
  --reduce_samples 64 \
  --reduce_test_samples 32 \
  --eval_batch_size 64 \
  --fedavg_eval
