#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
set -x

cap=1
pretrain=llama3.2-1B-pretrain
random_seed=42
split=hh_nce
DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/test.json"
micro_train_batch_size=2
train_batch_size=2
lambda=0.1
tau=0.2

# 只用一张卡
export CUDA_VISIBLE_DEVICES=0

export DEEPSPEED_USE_MPI=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501      # 随便一个空闲端口
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# VS Code 里先建一个 attach 配置，连到 5678 端口，然后再运行本脚本
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client \
  -m openrlhf.cli.train_rm \
  --save_path "./checkpoint/${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}" \
  --ckpt_path "./ckpt/checkpoints_rm_${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}" \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps 100 \
  --train_batch_size "${train_batch_size}" \
  --micro_train_batch_size "${micro_train_batch_size}" \
  --max_samples 1000 \
  --pretrain "/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08" \
  --bf16 \
  --max_epochs 1 \
  --max_len 1024 \
  --zero_stage 0 \
  --method nce \
  --seed "${random_seed}" \
  --learning_rate 9e-6 \
  --lambda_cmi "${lambda}" \
  --clap_cap "${cap}" \
  --tau "${tau}" \
  --estimation_var_sample_num 0 \
  --estimation_var_sample_rate 0.0 \
  --dataset "${DATASET}" \
  --eval_dataset "${EVAL_DATASET}" \
  --prompt_key prompt \
  --chosen_key chosen \
  --rejected_key rejected \
  --flash_attn \
  --packing_samples \
  --apply_chat_template
  # 调试期建议先关：
  
  # --gradient_checkpointing
