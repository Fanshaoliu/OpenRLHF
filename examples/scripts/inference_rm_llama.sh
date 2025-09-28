export HF_ENDPOINT=https://hf-mirror.com
set -x

split=hh
DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/train.json"
EVAL_DATASET="/mnt/data/user/liu_shaofan/OpenRLHF/data/${split}/test.json"

for INF_DATASET in ${DATASET} ; do

  RM_OUTPUT='/mnt/data/user/liu_shaofan/OpenRLHF/output/infer_log/'
  # PRETRAIN_MODEL=/mnt/data/user/liu_shaofan/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
  PRETRAIN_MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M

  # 分离模块名和参数，避免解析错误
  MODULE_NAME="openrlhf.cli.batch_inference"
  training_args=(
      --eval_task rm \
      --pretrain ${PRETRAIN_MODEL} \
      --bf16 \
      --max_len 4096 \
      --dataset "${INF_DATASET}" \
      --input_key chosen \
      --apply_chat_template \
      --zero_stage 2 \
      --post_processor csft \
      --normalize_reward \
      --micro_batch_size 2 \
      --output_path "${RM_OUTPUT}"
  )

  if [[ ${1} != "slurm" ]]; then
    # 正确传递模块名和参数，确保DeepSpeed能识别
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --module "$MODULE_NAME" "${training_args[@]}"
  fi
done

    # --pretrain "/mnt/data/user/liu_shaofan/OpenRLHF/checkpoint/${PRETRAIN_MODEL}" \
    # Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \
    # --pretrain Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \