set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Llama-3.2-3B \
   --save_path ./checkpoint/llama3.2-3B-sft \
   --ckpt_path ./ckpt/checkpoints_sft_Llama-3.2-3B \
   --save_steps 2000 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --packing_samples \
   --use_wandb f4964340b710e6450355ca2bd2b2f29de3d86312 \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29501 --module $training_commands
fi

