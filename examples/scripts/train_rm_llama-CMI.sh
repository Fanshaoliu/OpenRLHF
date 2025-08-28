# #!/usr/bin/env bash
# set -euo pipefail
# export HF_ENDPOINT=https://hf-mirror.com
# set -x

# cap=1
# pretrain=llama3.2-1B-pretrain
# comment=test_HHRLHF

# # 如果要用 wandb，推荐这样设置
# export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"

# for random_seed in 44; do
#     for split in hh-rlhf; do
#     DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
#     for lambda in 0.0 0.02 0.05 ; do
#         # 用 bash 数组构造命令，避免空格/引号/换行问题
#         cmd=(
#         deepspeed
#         --module openrlhf.cli.train_rm
#         --save_path "./checkpoint/${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}-${comment}"
#         --ckpt_path "./ckpt/checkpoints_rm_${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}-${comment}"
#         --save_steps -1
#         --logging_steps 1
#         --eval_steps 100
#         --train_batch_size 256
#         --micro_train_batch_size 32
#         --pretrain "/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
#         --bf16
#         --max_epochs 1
#         --max_len 8096
#         --zero_stage 3
#         --seed "${random_seed}"
#         --learning_rate 9e-6
#         --lambda_cmi "${lambda}"
#         --clap_cap "${cap}"
#         --dataset Anthropic/hh-rlhf
#         --eval_dataset Anthropic/hh-rlhf
#         --chosen_key chosen
#         --rejected_key rejected
#         --flash_attn
#         --load_checkpoint
#         --packing_samples
#         --use_wandb True
#         --wandb_project "${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-rs_${random_seed}-${comment}"
#         --wandb_run_name "${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}-${comment}"
#         --gradient_checkpointing
#         )
#         # 若在 slurm 里别直接跑 deepspeed，这里保留你的判定
#         if [[ ${1:-} != "slurm" ]]; then
#         "${cmd[@]}"
#         fi
#     done
#     done
# done


#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
set -x

cap=1
pretrain=llama3.2-1B-pretrain

# 如果要用 wandb，推荐这样设置
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"

for random_seed in 44 45; do
    for split in skywork hh ; do
    DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
    EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/test.json"

    if [ "${split}" = "hh" ]; then
        micro_train_batch_size=32
        train_batch_size=256
        echo ${split}
        echo ${micro_train_batch_size}
        echo ${train_batch_size}
    fi
    if [ "${split}" = "skywork" ]; then
        micro_train_batch_size=16
        train_batch_size=128
        echo ${split}
        echo ${micro_train_batch_size}
        echo ${train_batch_size}
    fi
    

    for lambda in 0.0 0.02 0.04 0.06 0.08; do
        # 用 bash 数组构造命令，避免空格/引号/换行问题
        cmd=(
        deepspeed
        --module openrlhf.cli.train_rm
        --save_path "./checkpoint/${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}"
        --ckpt_path "./ckpt/checkpoints_rm_${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}"
        --save_steps -1
        --logging_steps 1
        --eval_steps 100
        --train_batch_size ${train_batch_size}
        --micro_train_batch_size ${micro_train_batch_size}
        --pretrain "/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
        --bf16
        --max_epochs 1
        --max_len 8096
        --zero_stage 3
        --seed "${random_seed}"
        --learning_rate 9e-6
        --lambda_cmi "${lambda}"
        --clap_cap "${cap}"
        --dataset "${DATASET}"
        --eval_dataset "${EVAL_DATASET}"
        --prompt_key prompt
        --chosen_key chosen
        --rejected_key rejected
        --flash_attn
        --apply_chat_template
        --load_checkpoint
        --packing_samples
        --use_wandb True
        --wandb_project "${pretrain}-${split}-rm-1_grad-logit_cap_${cap}"
        --wandb_run_name "${pretrain}-${split}-rm-1_grad-logit_cap_${cap}-${lambda}-rs_${random_seed}"
        --gradient_checkpointing
        )
        # 若在 slurm 里别直接跑 deepspeed，这里保留你的判定
        if [[ ${1:-} != "slurm" ]]; then
        "${cmd[@]}"
        fi
    done
    done
done