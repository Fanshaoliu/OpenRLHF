export HF_ENDPOINT=https://hf-mirror.com
set -x

# exp_postfix=_grad_2_exact
# exp_postfix=_same_head
# exp_postfix=_same_head
# exp_postfix=_y-cr_y-r
# exp_postfix=_y-cr_y-r_sky
exp_postfix="_hh-nce"

pretrain_type="llama3.1-8B"
# pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

# pretrain_type="llama3.2-1B-pretrain"
pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

# 用数组更稳：空格、引号都会被正确传递
training_commands=(
  openrlhf.cli.test_rm
  --pretrain ${pretrain_model_path}  # 不能删除，要用它提取tokenizer
  --save_steps -1
  --logging_steps 1
  --eval_steps 100
  --train_batch_size 128
  --micro_train_batch_size 32
  --bf16
  --max_epochs 1
  --max_len 8096
  --zero_stage 3
  --learning_rate 9e-6
  --clap_cap 1
  --eval_model_dir "/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint"
  --eval_model_pretrain_type ${pretrain_type}
  --eval_dataset_list "webGPT"  
  --eval_model_data_type "hh_nce"
  --eval_model_grad_type "1"
  --eval_model_cap_type "2"
  --eval_model_lambda_type "0.0 0.01 0.05 0.1 0.2"
  --eval_model_tau "0.0 0.05 0.1 0.2 0.4 0.5 1.0"
  --eval_model_rs_type "42"
  --exp_postfix ${exp_postfix}
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
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --module "${training_commands[@]}"
fi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --module "${training_commands[@]}"
# --eval_dataset_list "RB RMB-harmless RMB-helpful SHP hh skywork"
  # --eval_model_lambda_type "0.0 0.01 0.05 0.1 1.0"
    # --eval_model_data_type "hh_y-cr hh_y-r"
      # --eval_dataset_list "RB RMB-harmless RMB-helpful SHP hh skywork"  