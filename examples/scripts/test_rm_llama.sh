export HF_ENDPOINT=https://hf-mirror.com
set -x


for split in RB RMB-harmless RMB-helpful SHP; do
  DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/${split}.jsonl"

  for lambda in 0.0 0.02 0.04 0.05 0.06 0.08 1.0 ; do
    # 用数组更稳：空格、引号都会被正确传递
    training_commands=(
      openrlhf.cli.test_rm
      --pretrain "./checkpoint/llama3.2-1B-rm-1-grad-logit_cap_1-new-${lambda}"
      --save_steps 300
      --logging_steps 1
      --eval_steps 100
      --train_batch_size 256
      --micro_train_batch_size 32
      --bf16
      --max_epochs 1
      --max_len 8096
      --zero_stage 3
      --learning_rate 9e-6
      --lambda_cmi "${lambda}"
      --clap_cap 1
      --dataset "${DATASET}"            # 只保留这一条 dataset
      --prompt_key prompt
      --chosen_key chosen
      --rejected_key reject             # 你的本地文件用的是 reject
      --flash_attn
      --apply_chat_template
      --load_checkpoint
      --packing_samples
      --gradient_checkpointing
    )

    if [[ ${1} != "slurm" ]]; then
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --module "${training_commands[@]}"
    fi
  done
done