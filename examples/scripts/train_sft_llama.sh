set -x

# CUDA_VISIBLE_DEVICES=0,1,3,5,6,7 
export CUDA_VISIBLE_DEVICES=0,2

pretrain=llama3.2-1B-pretrain
dataset=ShareGPT52k_clean
random_seed=42

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset /mnt/data/user/liu_shaofan/OpenRLHF/data/${dataset}/train.json \
   --input_key conversations \
   --train_batch_size 36 \
   --micro_train_batch_size 18 \
   --pretrain /mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
   --save_path ./checkpoint/${pretrain}_${dataset} \
   --ckpt_path ./ckpt/checkpoints_sft_${pretrain}_${dataset} \
   --save_steps 3000 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --load_checkpoint \
   --apply_chat_template \
   --packing_samples \
   --wandb_project ${pretrain}-sft \
   --wandb_run_name ${pretrain}-sft-${dataset}-rs_${random_seed} \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29501 --module $training_commands
fi

#    --use_wandb f4964340b710e6450355ca2bd2b2f29de3d86312 \
#    --output_key response \