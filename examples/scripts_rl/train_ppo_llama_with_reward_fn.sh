# /openrlhf/examples/scripts/reward_func.py
# import torch

# def reward_func(queries, prompts, labels):
#     # queries is prompts + responses
#     # labels is answers
#     print(queries)
#     return torch.randn(len(queries))

set -x 

export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
# export CUDA_VISIBLE_DEVICES=0,2

prompt_dataset=hh

sft_model="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain_ShareGPT_V3_unfiltered_cleaned_split"
# sft_model="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.1-8B-pretrain_ShareGPT_V3_unfiltered_cleaned_split"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
      "working_dir": "openrlhf",
      "excludes": ["datasets", "datasets/**"]
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain ${sft_model} \
   --remote_rm_url examples/python/reward_func.py \
   --save_path checkpoint/llama3.2-1b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 1.5e-6 \
   --init_kl_coef 0.01 \
   --prompt_data "data/${prompt_dataset}/train.json" \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing 
   # --use_wandb "${WANDB_API_KEY}" \
   # --wandb_project "cmi_rm_exp-llama3.1_8B-RL" \
   # --wandb_run_name "cmi_rm_exp-llama3.1_8B-RL-try" 
# --input_key conversations
# /mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain_ShareGPT_V3_unfiltered_cleaned_split

#                 --apply_chat_template 
#                 --tokenizer_chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" 
#                 --load_checkpoint


   # --prompt_data OpenRLHF/prompt-collection-v0.1 \
   # --input_key context_messages \