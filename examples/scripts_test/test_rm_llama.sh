export HF_ENDPOINT=https://hf-mirror.com
set -x

# pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
pretrain_model_path="/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

# 用数组更稳：空格、引号都会被正确传递
training_commands=(
  openrlhf.cli.test_rm_ckpt
  --pretrain ${pretrain_model_path}  # 不能删除，要用它提取tokenizer
  --save_steps -1
  --logging_steps 1
  --eval_steps 100
  --train_batch_size 128
  --micro_train_batch_size 32
  --bf16
  --max_epochs 1
  --max_len 2048
  --zero_stage 3
  --learning_rate 9e-6
  --clap_cap 1
  --eval_dataset_list "RMB-harmless-dsh RMB-helpful-dsh webGPT RMB-harmless RMB-helpful SHP RB hh skywork"  
  --eval_ckpt_dir "checkpoint/llama3.2-1B-sft-hh_cm_avg5_1-6-rm-nce-0.05-tau_1.0-rs_42-rm_mechanism_in_avg5-hf_ckpt checkpoint/llama3.2-1B-sft-hh_cm_avg5_2-7-rm-nce-0.05-tau_1.0-rs_42-rm_mechanism_in_avg5-hf_ckpt checkpoint/llama3.2-1B-sft-hh_cm_avg5_3-8-rm-nce-0.05-tau_1.0-rs_42-rm_mechanism_in_avg5-hf_ckpt checkpoint/llama3.2-1B-sft-hh_cm_avg5_4-9-rm-nce-0.05-tau_1.0-rs_42-rm_mechanism_in_avg5-hf_ckpt checkpoint/llama3.2-1B-sft-hh_cm_avg5_5-10-rm-nce-0.05-tau_1.0-rs_42-rm_mechanism_in_avg5-hf_ckpt"
  # --eval_ckpt_dir "/mnt/data/user/zhang_guoqiang/hf_cache/checkpoints_rm_InfoRM_Llama-3.1-8B_hh_lr5e-6_bs64"
  # --eval_ckpt_dir "/mnt/data/user/zhang_guoqiang/hf_cache/global_step2400_hf"
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

if [[ ${1} != "slurm" ]]; then
  CUDA_VISIBLE_DEVICES=6,7 deepspeed --master_port 29511 --module "${training_commands[@]}"
fi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --module "${training_commands[@]}"
# --eval_dataset_list "RB RMB-harmless RMB-helpful SHP hh skywork"
  # --eval_model_lambda_type "0.0 0.01 0.05 0.1 1.0"
    # --eval_model_data_type "hh_y-cr hh_y-r"
      # --eval_dataset_list "RB RMB-harmless RMB-helpful SHP hh skywork"  

        # for nce_hard_level in middle easy hard ; do
        #     for nce_grad_type in rm_wo_grad two_head_grad rm_w_grad ; do

