#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
set -x

# 希尔伯特投影分解loss
cap=1
method=grad_1
estimation_var_sample_num=0
estimation_var_sample_rate=0.0

# NCE loss
method=nce
# method=baseline
split=hh
tau=0.0
exp_postfix="-rm_mechanism_in_avg5-hf_ckpt"

# pretrain=llama3.2-1B-pretrain
pretrain=llama3.2-1B-sft
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

nce_hard_level="hard"
nce_grad_type="rm_w_grad"

# 注意，当前的二阶优化不支持packing sample
for random_seed in 42 ; do
    for split in hh_cm_avg5_1-6 hh_cm_avg5_2-7 hh_cm_avg5_3-8 hh_cm_avg5_4-9 hh_cm_avg5_5-10 ; do
        DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
        EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/hh/test.json"
        # EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/RMB-harmless/test.json"

        micro_train_batch_size=64
        train_batch_size=256

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
        for tau in 1.0 ; do
            for lambda in 0.05 0.0 ; do
                method="nce"

                if [ "${lambda}" = 0.0 ]; then
                    method="baseline"
                fi
        # for tau in 0.2 ; do
        #     for lambda in 0.05 ; do
                # 用 bash 数组构造命令，避免空格/引号/换行问题
                cmd=(
                deepspeed
                --module openrlhf.cli.train_rm
                --save_path "./checkpoint/${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                --ckpt_path "./ckpt/checkpoints_${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                --save_steps 100
                --logging_steps 1
                --eval_steps 100
                --train_batch_size ${train_batch_size}
                --micro_train_batch_size ${micro_train_batch_size}
                --pretrain /mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain_ShareGPT_V3_unfiltered_cleaned_split
                --bf16
                --max_epochs 1
                --max_len 2048
                --zero_stage 3
                --nce_hard_level "${nce_hard_level}"
                --nce_grad_type "${nce_grad_type}"
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
                --tokenizer_chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" 
                --load_checkpoint
                # --use_wandb True
                --packing_samples
                --save_hf_ckpt
                --disable_ds_ckpt
                --max_ckpt_num 100
                # --wandb_project "cmi_rm_exp-llama3.2_1B"
                # --wandb_run_name "${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
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
                    



# 注意，当前的二阶优化不支持packing sample
for random_seed in 42 ; do
    for split in hh_cm_avg5_1-6 hh_cm_avg5_2-7 hh_cm_avg5_3-8 hh_cm_avg5_4-9 hh_cm_avg5_5-10 ; do
        DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
        EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/hh/test.json"
        # EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/RMB-harmless/test.json"

        micro_train_batch_size=32
        train_batch_size=256

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
        for tau in 1.0 ; do
            for lambda in 0.05 0.0 ; do
                method="nce"

                if [ "${lambda}" = 0.0 ]; then
                    method="baseline"
                fi

                # 用数组更稳：空格、引号都会被正确传递
                training_commands=(
                openrlhf.cli.test_rm_ckpt
                --pretrain /mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/llama3.2-1B-pretrain_ShareGPT_V3_unfiltered_cleaned_split
                --save_steps -1
                --logging_steps 1
                --eval_steps 100
                --train_batch_size 128
                --micro_train_batch_size 16
                --bf16
                --max_epochs 1
                --max_len 2048
                --zero_stage 3
                --learning_rate 9e-6
                --clap_cap 1
                --eval_dataset_list "RMB-harmless-dsh RMB-helpful-dsh webGPT RMB-harmless RMB-helpful SHP RB hh skywork"
                # --eval_ckpt_dir ""
                --eval_ckpt_dir "ckpt/checkpoints_${pretrain}-${split}-rm-${method}-${lambda}-tau_${tau}-rs_${random_seed}${exp_postfix}"
                # --eval_ckpt_dir "ckpt/checkpoints_llama3.2-1B-pretrain-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-two_head_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-easy-rm_wo_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-middle-rm_wo_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-rm_wo_grad ckpt/checkpoints_llama3.2-1B-pretrain-hh-rm-baseline-0.0-tau_0.0-rs_42-rm_performance-hf_ckpt"
                # ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-rm_w_grad
                --apply_chat_template
                --prompt_key prompt
                --chosen_key chosen
                --rejected_key rejected             # 你的本地文件用的是 reject
                --flash_attn
                --load_checkpoint
                --packing_samples
                --gradient_checkpointing
                )

                deepspeed --master_port 29522 --module "${training_commands[@]}"

            done
        done
    done
done


