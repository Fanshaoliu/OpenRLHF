set -x

# CUDA_VISIBLE_DEVICES=0,1,3,5,6,7 
# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,7

pretrain=llama3.2-1B-pretrain
# dataset=ShareGPT52k_clean
dataset=ShareGPT52k
random_seed=42

# 根据 pretrain 选择对应的快照路径
if [[ "$pretrain" == "llama3.2-1B-pretrain" ]]; then
    pretrain_path=/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
elif [[ "$pretrain" == "llama3.2-8B-pretrain" ]]; then
    pretrain_path=/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
else
    echo "Unknown pretrain: $pretrain"
    exit 1
fi

read -r -d '' training_args <<EOF
--max_len 2048
--dataset /mnt/data/user/liu_shaofan/OpenRLHF/data/${dataset}/train.json
--input_key conversations
--train_batch_size 48
--micro_train_batch_size 8
--pretrain ${pretrain_path}
--save_path ./checkpoint/${pretrain}_${dataset}
--ckpt_path ./ckpt/checkpoints_sft_${pretrain}_${dataset}
--save_steps 3000
--logging_steps 1
--eval_steps -1
--zero_stage 3
--max_epochs 1
--bf16
--flash_attn
--learning_rate 2e-5
--load_checkpoint
--apply_chat_template
--packing_samples
--use_wandb f4964340b710e6450355ca2bd2b2f29de3d86312
--wandb_project cmi_rm-llama_sft
--wandb_run_name ${pretrain}-sft-${dataset}-rs_${random_seed}
--gradient_checkpointing
--seed ${random_seed}
EOF
  
if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29501 --module openrlhf.cli.train_sft ${training_args}
fi
