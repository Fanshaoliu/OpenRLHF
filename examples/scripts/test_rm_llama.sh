export HF_ENDPOINT=https://hf-mirror.com
set -x


for split in RB RMB-harmless RMB-helpful SHP; do
  # MODEL_PATH="llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.0-rs_44"
  DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/${split}/test.json"
  # DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/hh/test.json"

  lambda=0.0
  for MODEL_PATH in llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.0-rs_44 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.0-rs_45 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.02-rs_44 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.02-rs_45 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.04-rs_44 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.04-rs_45 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.06-rs_44 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.06-rs_45 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.08-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.0-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.0-rs_45 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.02-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.02-rs_45 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.04-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.04-rs_45 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.06-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.06-rs_45 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.08-rs_44 llama3.2-1B-pretrain-skywork-rm-1_grad-logit_cap_1-0.08-rs_45 llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_1-0.08-rs_45; do
    # 用数组更稳：空格、引号都会被正确传递
    training_commands=(
      openrlhf.cli.test_rm
      --pretrain "/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/${MODEL_PATH}"
      --save_steps -1
      --logging_steps 1
      --eval_steps 100
      --train_batch_size 112
      --micro_train_batch_size 16
      --bf16
      --max_epochs 1
      --max_len 8096
      --zero_stage 3
      --learning_rate 9e-6
      --lambda_cmi "${lambda}"
      --clap_cap 1
      --dataset "${DATASET}"            # 只保留这一条 dataset
      --apply_chat_template
      --prompt_key prompt
      --chosen_key chosen
      --rejected_key rejected             # 你的本地文件用的是 reject
      --flash_attn
      --load_checkpoint
      --packing_samples
      --gradient_checkpointing
    )

    if [[ ${1} != "slurm" ]]; then
      CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 deepspeed --module "${training_commands[@]}"
    fi
  done
done