export HF_ENDPOINT=https://hf-mirror.com
set -x

cap=1
random_seed=42
pretrain=llama3.2-1B-pretrain

for lambda in 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 1.0 ; do
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/${pretrain}-rm-1-grad-logit_cap_${cap}-${lambda} \
   --ckpt_path ./ckpt/checkpoints_rm_${pretrain}-rm-1-grad-logit_cap_${cap}-${lambda} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 32 \
   --pretrain /mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
   --bf16 \
   --max_epochs 1 \
   --max_len 8096 \
   --zero_stage 3 \
   --seed ${random_seed} \
   --learning_rate 9e-6 \
   --lambda_cmi ${lambda} \
   --clap_cap ${cap} \
   --dataset Anthropic/hh-rlhf \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --packing_samples \
   --use_wandb f4964340b710e6450355ca2bd2b2f29de3d86312 \
   --wandb_project ${pretrain}-rm-1-grad-logit_cap_${cap}-rs_${random_seed} \
   --wandb_run_name ${pretrain}-rm-1-grad-logit_cap_${cap}-${lambda} \
   --gradient_checkpointing 
EOF

    if [[ ${1} != "slurm" ]]; then
        deepspeed --module $training_commands
    fi
done
