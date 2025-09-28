import argparse
import math
import os, json
import torch
import gc
from datetime import datetime
from transformers import AutoTokenizer

from transformers.trainer import get_scheduler
import torch.distributed as dist

from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer.rm_trainer import RewardModelTrainer
from openrlhf.utils import get_strategy, get_tokenizer

def _default_serializer(o):
    # 兼容 numpy / torch 标量
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
    except Exception:
        pass
    try:
        import torch
        if isinstance(o, torch.Tensor) and o.numel() == 1:
            return o.item()
    except Exception:
        pass
    # 其他不可序列化对象
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def append_jsonl(path: str, record: dict, ensure_dir: bool = True):
    """把一条字典记录追加到指定 jsonl，没有就创建。"""
    if ensure_dir:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=_default_serializer)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _rank_is():
    return int(dist.get_rank())

def eval(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    all_eval_ckpt_dir_list = args.eval_ckpt_dir
    all_eval_dataset_list = args.eval_dataset_list
    
    
    dataset_dir_list = []
    for dataset_name_split in all_eval_dataset_list.split():
        if dataset_name_split in ['hh', 'skywork']:
            dataset_dir = f"/mnt/data/user/liu_shaofan/OpenRLHF/data/{dataset_name_split}/test.json"
        else:
            dataset_dir = f"/mnt/data/user/liu_shaofan/OpenRLHF/data/ood/{dataset_name_split}/test.json"
        dataset_dir_list.append(dataset_dir)

    model_dir_list = []
    
    for root_dir in all_eval_ckpt_dir_list.split(' '):
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            print(f"错误: 目录 '{root_dir}' 不存在!")
            return []
        
        # 检查是否是一个目录
        if not os.path.isdir(root_dir):
            print(f"错误: '{root_dir}' 不是一个目录!")
            return []
        
        # global_dirs = []
        
        # 遍历目录下的所有条目
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            # 检查是否是目录且名称以"global"开头
            if os.path.isdir(entry_path) and entry.startswith("global_step") and (not entry.startswith("global_step200_")) and (not entry.startswith("global_step600_")) and (not entry.startswith("global_step1000_")) and (not entry.startswith("global_step1400_")) and (not entry.startswith("global_step1800_")) and (not entry.startswith("global_step2200_")):
                model_dir_list.append(entry_path)
    
    if model_dir_list == []:
        model_dir_list = all_eval_ckpt_dir_list.split(' ')

    if _rank_is() == 0:
        print("Test dataset:")
        for dataset_dir in dataset_dir_list:
            print(dataset_dir)
        print("Models will be evaluated:")
        for model_dir in model_dir_list:
            print(model_dir)

    # 然后是模型循环
    for model_dir in model_dir_list:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain,
            use_fast=not args.disable_fast_tokenizer,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id


        if _rank_is() == 0:
            # print(f"Testing dataset dir:\n {dataset_dir}")
            print(f"Testing model dir:\n {model_dir}")
        model = get_llm_for_sequence_regression(
            model_dir,
            "reward",
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            # ds_config=None,
            ds_config=strategy.get_ds_eval_config(),
            init_value_head=False,
            value_head_prefix=args.value_head_prefix,
            packing_samples=args.packing_samples,
        )

        strategy.print(model)

        # 2) 评测模式：禁梯度/禁ckpt
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if getattr(args, "gradient_checkpointing", False):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass

        # 3) 只准备模型（不要创建 optimizer/scheduler，也不要把它交给训练引擎）
        #    若你的 strategy 支持单参 prepare，就用它；否则直接 .to(device)
        try:
            model = strategy.prepare(model)
        except TypeError:
            device = getattr(strategy, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.to(device)

        # batch_size here is micro_batch_size * 2
        # we use merged chosen + rejected response forward
        trainer = RewardModelTrainer(
            model=model,
            strategy=strategy,
            optim=None,
            tokenizer=tokenizer,
            train_dataloader=None,
            eval_dataloader=None,
            scheduler=None,
            max_norm=None,
            max_epochs=args.max_epochs,
            lambda_cmi=args.lambda_cmi,
            clap_cap=args.clap_cap,
            loss=args.loss,
            disable_ds_ckpt=True,
            save_hf_ckpt=False,
        )


        # 最外层是模型循环
        for dataset_dir in dataset_dir_list:
            eval_data = blending_datasets(
                dataset_dir,
                args.dataset_probs,
                strategy,
                args.seed,
                max_count=args.max_samples,
                dataset_split=args.eval_split,
            )

            eval_dataset = RewardDataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                input_template=args.input_template,
            )
            eval_dataloader = strategy.setup_dataloader(
                eval_dataset,
                args.micro_train_batch_size,
                True,
                False,
                eval_dataset.collate_fn,
            )

            # 5) 真正评测：不建图，显存最省
            with torch.inference_mode():
                saved_bar_dict = trainer.evaluate(eval_dataloader, 0)

                model_name=f"{model_dir.split('/')[-2]}#{model_dir.split('/')[-1]}"
                data_name=dataset_dir.split('/')[-2]
                saved_bar_dict['model_name'] = model_name
                saved_bar_dict['data_name'] = data_name
                saved_bar_dict['rs'] = model_name.split('rs_')[-1]
                saved_bar_dict['lambda'] = model_name.split('-')[-1]
                saved_bar_dict['logit_cap'] = model_name.split('cap_')[-1].split('-')[0]
                saved_bar_dict['grad'] = model_name.split('_grad')[0].split('-')[-1]
                saved_bar_dict['ts'] = datetime.now().isoformat(timespec="seconds")

                append_jsonl(f"./output/eval_log/{data_name}.jsonl", saved_bar_dict)
                # return saved_bar_dict

        # 6) 彻底清理
        try:
            # DeepSpeed Engine 的干净退出（如果 prepare 返回的是 engine）
            if hasattr(model, "destroy") and callable(model.destroy):
                model.destroy()
        except Exception:
            pass
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    # CMI loss
    parser.add_argument("--lambda_cmi", type=float, default=0.0, help="Lambda for CMI loss")
    parser.add_argument("--clap_cap", type=float, default=1.0, help="Clipping value for CMI loss")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=512)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    # Eval only
    parser.add_argument("--eval_ckpt_dir", type=str, default=None, help="Path to the evaluation model directory")
    parser.add_argument("--eval_dataset_list", type=str, default=None, help="List of evaluation datasets")
    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    eval(args)
