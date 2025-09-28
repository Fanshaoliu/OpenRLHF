# export HF_ENDPOINT=https://hf-mirror.com
# set -x

# pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
# # pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

# # 用数组更稳：空格、引号都会被正确传递
# training_commands=(
#   openrlhf.cli.test_rm_ckpt
#   --pretrain ${pretrain_model_path}  # 不能删除，要用它提取tokenizer
#   --save_steps -1
#   --logging_steps 1
#   --eval_steps 100
#   --train_batch_size 128
#   --micro_train_batch_size 16
#   --bf16
#   --max_epochs 1
#   --max_len 2048
#   --zero_stage 3
#   --learning_rate 9e-6
#   --clap_cap 1
#   --eval_dataset_list "RB"  
#   --eval_ckpt_dir ""
#   # --eval_ckpt_dir "/mnt/data/user/zhang_guoqiang/hf_cache/global_step2400_hf"
#   # --eval_ckpt_dir "ckpt/checkpoints_llama3.2-1B-pretrain-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-two_head_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-easy-rm_wo_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-middle-rm_wo_grad ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-rm_wo_grad ckpt/checkpoints_llama3.2-1B-pretrain-hh-rm-baseline-0.0-tau_0.0-rs_42-rm_performance-hf_ckpt"
#   # ckpt/checkpoints_llama3.2-1B-sft-hh_nce-rm-nce-0.05-tau_1.0-rs_42-rm_performance-hf_ckpt-EXP-hard-rm_w_grad
#   --apply_chat_template
#   --prompt_key prompt
#   --chosen_key chosen
#   --rejected_key rejected             # 你的本地文件用的是 reject
#   --flash_attn
#   --load_checkpoint
#   --packing_samples
#   --gradient_checkpointing
# )


# deepspeed --master_port 29522 --module "${training_commands[@]}"





# 注意，当前的二阶优化不支持packing sample
for random_seed in 42 ; do
    for split in hh_cm_avg5_high_nce hh_cm_avg5_low_nce ; do
        DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
        EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/hh/test.json"
        # EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/RMB-harmless/test.json"

        micro_train_batch_size=16
        train_batch_size=128

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
                --nce_hard_level "${nce_hard_level}"
                --nce_grad_type "${nce_grad_type}"
                --learning_rate 9e-6
                --clap_cap 1
                --eval_dataset_list "RB"  
                --eval_ckpt_dir ""
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