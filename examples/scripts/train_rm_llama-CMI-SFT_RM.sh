#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
set -x

cap=1
pretrain=llama3.2-1B-pretrain
# method=2_grad
# exp_postfix="_grad_2_exact"
exp_postfix="_same_head-sft_model"
PRETRAIN_MODEL="/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-sft"
# 如果要用 wandb，推荐这样设置
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"

for method in 1_grad ; do
    if [ "${method}" = "1_grad" ]; then
        estimation_var_sample_num=0
        estimation_var_sample_rate=0.0
        echo ${method}
        echo ${estimation_var_sample_num}
        echo ${estimation_var_sample_rate}
    fi
    if [ "${method}" = "2_grad" ]; then
        estimation_var_sample_num=4
        estimation_var_sample_rate=0.2
        echo ${method}
        echo ${estimation_var_sample_num}
        echo ${estimation_var_sample_rate}
    fi
      
    for cap in 2; do
        for random_seed in 42 ; do
            for split in hh ; do
                DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
                EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/test.json"

                if [ "${split}" = "hh" ]; then
                    micro_train_batch_size=16
                    train_batch_size=256
                    echo ${split}
                    echo ${micro_train_batch_size}
                    echo ${train_batch_size}
                fi
                if [ "${split}" = "skywork" ]; then
                    micro_train_batch_size=8
                    train_batch_size=64
                    echo ${split}
                    echo ${micro_train_batch_size}
                    echo ${train_batch_size}
                fi

                for lambda in 0.0 0.05 ; do
                    # 用 bash 数组构造命令，避免空格/引号/换行问题
                    cmd=(
                    deepspeed
                    --module openrlhf.cli.train_rm
                    --save_path "./checkpoint/${pretrain}-${split}-rm-${method}-logit_cap_${cap}-${lambda}-rs_${random_seed}${exp_postfix}"
                    --ckpt_path "./ckpt/checkpoints_rm_${pretrain}-${split}-rm-${method}-logit_cap_${cap}-${lambda}-rs_${random_seed}${exp_postfix}"
                    --save_steps -1
                    --logging_steps 1
                    --eval_steps 100
                    --train_batch_size ${train_batch_size}
                    --micro_train_batch_size ${micro_train_batch_size}
                    --pretrain "${PRETRAIN_MODEL}"
                    --bf16
                    --max_epochs 1
                    --max_len 8096
                    --zero_stage 3
                    --seed "${random_seed}"
                    --learning_rate 9e-6
                    --lambda_cmi "${lambda}"
                    --clap_cap "${cap}"
                    --estimation_var_sample_num ${estimation_var_sample_num}
                    --estimation_var_sample_rate ${estimation_var_sample_rate}
                    --dataset "${DATASET}"
                    --eval_dataset "${EVAL_DATASET}"
                    --prompt_key prompt
                    --chosen_key chosen
                    --rejected_key rejected
                    --flash_attn
                    --apply_chat_template
                    --load_checkpoint
                    --use_wandb True
                    --wandb_project "${pretrain}-${split}-rm"
                    --wandb_run_name "${pretrain}-${split}-rm-${method}-logit_cap_${cap}-${lambda}-rs_${random_seed}${exp_postfix}"
                    --gradient_checkpointing
                    )
                    # 若在 slurm 里别直接跑 deepspeed，这里保留你的判定
                    # --packing_samples
                    if [[ ${1:-} != "slurm" ]]; then
                    "${cmd[@]}"
                    fi
                done
            done
        done
    done
done