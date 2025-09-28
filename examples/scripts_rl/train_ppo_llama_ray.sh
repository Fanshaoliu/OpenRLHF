set -x 
# export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
# export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL_CUMEM_ENABLE=0
# NCCL_DEBUG=INFO

prompt_dataset=hh
# export RAY_TMPDIR=/mnt/data/user/zhang_guoqiang/ray_tmp

sft_model="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain_ShareGPT_V3_unfiltered_cleaned_split"
# sft_model="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.1-8B-pretrain_ShareGPT_V3_unfiltered_cleaned_split"
reward_model="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain-hh-rm-baseline-0.0-tau_0.0-rs_42-new_rm_test-hf_ckpt"

# '172.17.0.4:6379'

ray job submit --address='http://127.0.0.1:8265' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain ${sft_model} \
   --reward_pretrain ${reward_model} \
   --save_path /mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1b-rlhf \
   --micro_train_batch_size 4 \
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
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data "/mnt/data/user/liu_shaofan/OpenRLHF/data/${prompt_dataset}/train.json" \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint 
   # --use_wandb {wandb_token}



# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward
# ray start --head --dashboard-port=8266
# ray stop
# ray start --head --dashboard-port=8266