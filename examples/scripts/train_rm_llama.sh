export HUGGINGFACE_HUB_CACHE="/mnt/data/user/liu_shaofan/HF_CACHE"
export HF_DATASETS_CACHE="/mnt/data/user/liu_shaofan/HF_CACHE/datasets_cache"
export HOME="/mnt/data/user/liu_shaofan/HF_CACHE"

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain $HOME/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset $HOME/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb f4964340b710e6450355ca2bd2b2f29de3d86312 
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
