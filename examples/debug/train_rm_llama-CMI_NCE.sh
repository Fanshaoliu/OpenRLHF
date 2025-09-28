#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
set -x


# 希尔伯特投影分解loss
cap=1
method=grad_1
estimation_var_sample_num=0
estimation_var_sample_rate=0.0
exp_postfix="debug_my_hhdata_my_own_template"
# exp_postfix="_grad_2_exact"
# exp_postfix="_hh-nce_new-para"

# NCE loss
# method=nce
method=baseline
# split=hh_nce
split=hh
tau=0.2

pretrain=llama3.2-1B-pretrain
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
# export CUDA_VISIBLE_DEVICES=2,3,4,5

# 注意，当前的二阶优化不支持packing sample
for random_seed in 42 ; do
    for split in hh ; do
        DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
        # EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/test.json"
        EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/RMB-harmless/test.json"

        micro_train_batch_size=8
        train_batch_size=64

        # if [ "${split}" = "hh" ]; then
        #     micro_train_batch_size=32
        #     train_batch_size=256
        #     echo ${split}
        #     echo ${micro_train_batch_size}
        #     echo ${train_batch_size}
        # fi
        # if [ "${split}" = "skywork" ]; then
        #     micro_train_batch_size=8
        #     train_batch_size=64
        #     echo ${split}
        #     echo ${micro_train_batch_size}
        #     echo ${train_batch_size}
        # fi
        for tau in 0.0 ; do
            for lambda in 0.0 ; do
        # for tau in 0.2 ; do
        #     for lambda in 0.05 ; do
                # 用 bash 数组构造命令，避免空格/引号/换行问题
                cmd=(
                deepspeed
                --module openrlhf.cli.train_rm
                --save_path "./checkpoint/${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                --ckpt_path "./ckpt/checkpoints_${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                --save_steps -1
                --logging_steps 1
                --eval_steps 100
                --train_batch_size ${train_batch_size}
                --micro_train_batch_size ${micro_train_batch_size}
                --pretrain "/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
                --bf16
                --max_epochs 1
                --max_len 4096
                --zero_stage 3
                --seed "${random_seed}"
                --learning_rate 5e-6
                --lambda_cmi "${lambda}"
                --tau "${tau}"
                --method "${method}"
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
                --packing_samples
                --wandb_project "${pretrain}-NCE_CIM-rm-new_para"
                --wandb_run_name "${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                --gradient_checkpointing
                )
                # 若在 slurm 里别直接跑 deepspeed，这里保留你的判定
                
                if [[ ${1:-} != "slurm" ]]; then
                "${cmd[@]}"
                fi
            done
        done
    done
done

# --wandb_project "${pretrain}-${split}-rm"
                    