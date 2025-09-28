#!/usr/bin/env bash
set -x

# CUDA_VISIBLE_DEVICES=0,1,3,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

pretrain=llama3.1-8B-pretrain
# dataset=ShareGPT52k_clean
# dataset=ShareGPT52k
dataset=ShareGPT_V3_unfiltered_cleaned_split
random_seed=42

# 根据 pretrain 选择对应的快照路径（兼容两个常见写法）
if [[ "$pretrain" == "llama3.2-1B-pretrain" ]]; then
  pretrain_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
elif [[ "$pretrain" == "llama3.28B-pretrain" || "$pretrain" == "llama3.1-8B-pretrain" ]]; then
  pretrain_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
else
  echo "Unknown pretrain: $pretrain"
  exit 1
fi

# Jinja 模板作为一个变量（用双引号包裹，避免 brace expansion/分词问题）
# chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>
 
# ' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

# ' }}{% endif %}"
chat_template='{% if add_generation_prompt is not defined %}{% set add_generation_prompt=false %}{% endif %}{% for message in messages %}{% if loop.index0==0 %}{{ bos_token }}{% endif %}<|start_header_id|>{{ message["role"] }}<|end_header_id|>{{ "\n\n" }}{{ message["content"]|trim }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>{{ "\n\n" }}{% endif %}'


# 训练参数用 Bash 数组，保证每个参数是一个独立元素
training_args=(
  --max_len 2048
  --dataset "/mnt/data/user/liu_shaofan/OpenRLHF/data/${dataset}/train.json"
  --input_key conversations
  --train_batch_size 64
  --micro_train_batch_size 8
  --pretrain "${pretrain_path}"
  --save_path "./checkpoint/${pretrain}_${dataset}"
  --ckpt_path "./ckpt/checkpoints_sft_${pretrain}_${dataset}"
  --save_steps 200
  --logging_steps 1
  --eval_steps 200
  --zero_stage 3
  --max_epochs 1
  --bf16
  --flash_attn
  --learning_rate 2e-5
  --load_checkpoint
  --apply_chat_template
  --tokenizer_chat_template "$chat_template"
  --packing_samples
  --save_hf_ckpt
  --disable_ds_ckpt
  --use_wandb "f4964340b710e6450355ca2bd2b2f29de3d86312"
  --wandb_project "cmi_rm-llama3.18B_sft"
  --wandb_run_name "${pretrain}-sft-${dataset}-rs_${random_seed}"
  --gradient_checkpointing
  --seed "${random_seed}"
)

if [[ "${1:-}" != "slurm" ]]; then
  deepspeed --master_port 29502 --module openrlhf.cli.train_sft "${training_args[@]}"
fi
