from contextlib import redirect_stdout
import argparse
import os
from datetime import timedelta
import math
import jsonlines
import torch
from torch import distributed as dist

from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import PromptDataset, SFTDataset, RewardDataset, RewardDatasetRaw
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import get_processor, get_strategy, get_tokenizer

from pathlib import Path
import json

def append_jsonl(path: str, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    with p.open('a', encoding='utf-8') as f:     # 追加模式；不存在会新建
        json.dump(record, f, ensure_ascii=False) # 中文不转义
        f.write('\n')

def drop_random_prompt_slice(mask: torch.Tensor, no_prompt_mask: torch.Tensor, drop_indx: int = -1) -> torch.Tensor:
    """
    支持 “有prompt” 与 “无prompt” 样本在各自批次中独立补齐、长度不同的情形。
    - b: 该行 mask 第一个为1的位置
    - e: b + (len_with - len_without) - 1，其中 len_* 为该行 mask 中 1 的个数
    然后把 [b,e] 等分为5段，随机挑一段将 mask 置0。
    """
    new_mask = mask.clone()
    B, L = mask.shape

    for i in range(B):
        row     = mask[i]
        row_nop = no_prompt_mask[i]

        # 序列实际长度（1 的个数）；两者可不同
        len_with    = int(row.sum().item())
        len_without = int(row_nop.sum().item())

        # b：第一个1的索引（兼容左/右padding）
        one_idx = torch.nonzero(row, as_tuple=False)
        if one_idx.numel() == 0:
            continue
        b = int(one_idx[0].item())

        # prompt_len：两者长度差；若 <=0，说明本行无可删的prompt
        prompt_len = max(0, len_with - len_without)
        if prompt_len == 0:
            continue

        # e：prompt结束位置；对齐到当前张量边界，防御异常
        e = min(b + prompt_len - 1, L - 1)
        if e < b:
            continue

        # 将 [b, e] 等分成5段并随机挑一段置0
        plen = e - b + 1
        if plen < 5:
            start = int(torch.randint(b, e + 1, (1,), device=mask.device).item())
            end   = start
        else:
            base = plen // 5
            rem  = plen % 5
            starts, ends, cur = [], [], b
            for k in range(5):
                seg_len = base + (1 if k < rem else 0)
                s, t = cur, cur + seg_len - 1
                starts.append(s); ends.append(t)
                cur = t + 1
            if drop_indx == -1:
                seg_idx = int(torch.randint(0, 5, (1,), device=mask.device).item())
            else:
                seg_idx = drop_indx
            start, end = starts[seg_idx], ends[seg_idx]

        new_mask[i, start:end+1] = 0  # dtype 兼容（long/bool/int 都可）

    return new_mask

def _batch_decode_with_mask(tokenizer, ids: torch.Tensor, mask: torch.Tensor,
                            skip_special_tokens=False, keep_spaces=True):
    """
    ids:  [B, L]  (int)
    mask: [B, L]  (0/1 或 bool) 1 表示有效 token
    return: List[str] 长度为 B
    """
    # 放到 CPU，避免在 GPU 上做字符串操作
    
    try:
        ids_cpu  = ids.detach().to("cpu")
        # mask_cpu = mask.detach().to("cpu").bool()
    except:
        ids_cpu  = [ids]
        # mask_cpu = [mask]

    texts = []
    for seq, m in zip(ids_cpu, ids_cpu):
        # valid = seq[m]             # 只保留有效 token
        # valid = True
        # if valid.numel() == 0:
            # texts.append("")       # 空序列兜底
            # continue
        text = tokenizer.decode(
            seq.tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=not keep_spaces,  # True 会合并空格，False 原样保留
        )
        texts.append(text)
    return texts

def debug_preview_batch(tokenizer, chosen_ids, c_mask, reject_ids, r_mask,file='out.txt',
                        n_preview=2, show_special=True):
    """
    打印前 n_preview 个样本的 chosen / reject 文本，便于人工核对。
    """
    chosen_texts  = _batch_decode_with_mask(tokenizer, chosen_ids,  c_mask,
                                            skip_special_tokens=not show_special, keep_spaces=True)
    reject_texts  = _batch_decode_with_mask(tokenizer, reject_ids,  r_mask,
                                            skip_special_tokens=not show_special, keep_spaces=True)

    for i in range(min(n_preview, len(chosen_texts))):
        with open(f"{file}", "w", encoding="utf-8") as f, redirect_stdout(f):
            print("=" * 80) 
            print(f"[{i}] CHOSEN ↓\n{chosen_texts[i]}\n") 
            print(f"[{i}] REJECT ↓\n{reject_texts[i]}\n")
            print("=" * 80) 

def batch_generate_vllm(args):
    from vllm import LLM, SamplingParams

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = []
    for _, prompt, _ in prompts_dataset:
        prompts.append(prompt)

    # Conditional SFT inference
    if args.enable_csft:
        for i in range(len(prompts)):
            prompts[i] += args.csft_prompt.strip() + " "

    # best of n
    N = args.best_of_n
    output_dataset = []

    outputs = llm.generate(prompts * N, sampling_params)
    for output in outputs:
        prompt = output.prompt
        output = output.outputs[0].text
        output_dataset.append({"input": prompt, "output": output})

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(output_dataset)


def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []

    for _, prompts, _ in pbar:
        # Conditional SFT inference
        if args.enable_csft:
            for i in range(len(prompts)):
                prompts[i] += args.csft_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=False,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

        dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


def batch_rm_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = RewardDatasetRaw(
        dataset, tokenizer, args.max_len, strategy, input_template=args.input_template
    )

    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        desc="Rewarding",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    output_dataset = []

    save_json_file_name = args.output_path + f"{args.dataset.split('/')[-2]}-drop20p_prompt-{args.dataset.split('/')[-1].split('.')[0]}{args.pretrain.split('/')[-1]}.jsonl"
    if os.path.isfile(save_json_file_name): open(save_json_file_name, "w", encoding="utf-8").close()

    # 推理context脚本
    # 推理脚本中：PRETRAIN_MODEL=llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_5-0.0-rs_44_same_head
    with torch.no_grad():
        for data in pbar:
            raw_data, chosen_ids, c_mask, reject_ids, r_mask, chosen_no_prompt_ids, chosen_no_prompt_mask, reject_no_prompt_ids, reject_no_prompt_mask, margin = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_no_prompt_ids = chosen_no_prompt_ids.squeeze(1).to(torch.cuda.current_device())
            chosen_no_prompt_mask = chosen_no_prompt_mask.squeeze(1).to(torch.cuda.current_device())
            reject_no_prompt_ids = reject_no_prompt_ids.squeeze(1).to(torch.cuda.current_device())
            reject_no_prompt_mask = reject_no_prompt_mask.squeeze(1).to(torch.cuda.current_device())

            # debug_preview_batch(tokenizer, chosen_ids, c_mask, reject_ids, r_mask, file='output/data_log/inference_origin.txt', show_special=True)
            
            B, L = chosen_ids.size(0), chosen_ids.size(1)
            # 生成“随机抹去一段 prompt”的新一类数据（只改 mask）：

            chosen_reward, reject_reward, _ = model.concatenated_forward(tokenizer, chosen_ids, c_mask, reject_ids, r_mask)
            

            drop_reward_dict = {}
            for drop_ind in range(5):
                c_mask_drop = drop_random_prompt_slice(c_mask, chosen_no_prompt_mask, drop_indx=drop_ind)
                r_mask_drop = drop_random_prompt_slice(r_mask, reject_no_prompt_mask, drop_indx=drop_ind)

                # if int(dist.get_rank()) == 0:
                #     print(c_mask[0])
                #     print(c_mask_drop[0])
                #     print(c_mask[1])
                #     print(c_mask_drop[1])
                chosen_drop20p_reward, reject_drop20p_reward, _ = model.concatenated_forward(tokenizer, chosen_ids, c_mask_drop, reject_ids, r_mask_drop)
                drop_reward_dict[f'drop_{drop_ind}_c'] = chosen_drop20p_reward
                drop_reward_dict[f'drop_{drop_ind}_r'] = reject_drop20p_reward

            # chosen_logit_delta = torch.abs(chosen_reward - chosen_drop20p_reward)
            # reject_logit_delta = torch.abs(reject_reward - reject_drop20p_reward)
            
            # sum_logit_delta_abs = torch.abs(chosen_logit_delta + reject_logit_delta)

            # for raw_data, sum_logit_delta_abs_, chosen_logit_delta_, reject_logit_delta_ in zip(raw_data, sum_logit_delta_abs, chosen_logit_delta, reject_logit_delta):
            #     output_dataset.append({ "delta_abs_sum": sum_logit_delta_abs_.item(), "chosen_logit_delta" :chosen_logit_delta_.item(), "reject_logit_delta_":reject_logit_delta_.item(), "raw_data": raw_data})

            for raw_data_, chosen_reward_, reject_reward_, drop_0_c_, drop_0_r_, drop_1_c_, drop_1_r_, drop_2_c_, drop_2_r_, drop_3_c_, drop_3_r_, drop_4_c_, drop_4_r_ in zip(raw_data, chosen_reward, reject_reward, drop_reward_dict['drop_0_c'], drop_reward_dict['drop_0_r'], drop_reward_dict['drop_1_c'], drop_reward_dict['drop_1_r'], drop_reward_dict['drop_2_c'], drop_reward_dict['drop_2_r'], drop_reward_dict['drop_3_c'], drop_reward_dict['drop_3_r'], drop_reward_dict['drop_4_c'], drop_reward_dict['drop_4_r']):
                append_jsonl(save_json_file_name, { "chosen_reward": chosen_reward_.item(), "reject_reward": reject_reward_.item(), "drop_0_c": drop_0_c_.item(), "drop_0_r": drop_0_r_.item(), "drop_1_c": drop_1_c_.item(), "drop_1_r": drop_1_r_.item(), "drop_2_c": drop_2_c_.item(), "drop_2_r": drop_2_r_.item(), "drop_3_c": drop_3_c_.item(), "drop_3_r": drop_3_r_.item(), "drop_4_c": drop_4_c_.item(), "drop_4_r": drop_4_r_.item(), "raw_data": raw_data})

    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # with jsonlines.open( + str(strategy.get_rank()), mode="w") as writer:
    #     writer.write_all(output_dataset)
    # wait unitl all processes generate done
    dist.barrier()


    # 推理有无context数据，
    # 推理脚本中：PRETRAIN_MODEL=llama3.2-1B-pretrain-hh-rm-1_grad-logit_cap_5-0.0-rs_44_same_head
    # with torch.no_grad():
    #     for data in pbar:
    #         raw_data, chosen_ids, c_mask, reject_ids, r_mask, chosen_no_prompt_ids, chosen_no_prompt_mask, reject_no_prompt_ids, reject_no_prompt_mask, margin = data
    #         chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
    #         c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
    #         reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
    #         r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
            
    #         chosen_no_prompt_ids = chosen_no_prompt_ids.squeeze(1).to(torch.cuda.current_device())
    #         chosen_no_prompt_mask = chosen_no_prompt_mask.squeeze(1).to(torch.cuda.current_device())
    #         reject_no_prompt_ids = reject_no_prompt_ids.squeeze(1).to(torch.cuda.current_device())
    #         reject_no_prompt_mask = reject_no_prompt_mask.squeeze(1).to(torch.cuda.current_device())

    #         chosen_reward, reject_reward, _ = model.concatenated_forward(tokenizer, chosen_ids, c_mask, reject_ids, r_mask)
    #         chosen_np_reward, reject_np_reward, _ = model.concatenated_forward(tokenizer, chosen_no_prompt_ids, chosen_no_prompt_mask, reject_no_prompt_ids, reject_no_prompt_mask)

    #         logit_delta = chosen_reward - reject_reward
    #         np_logit_delta = chosen_np_reward - reject_np_reward
    #         delta_logit_delta = torch.abs(logit_delta - np_logit_delta)

    #         for raw_data, delta_logit_delta_, logit_delta_, np_logit_delta_ in zip(raw_data, delta_logit_delta, logit_delta, np_logit_delta):
    #             output_dataset.append({ "delta_abs": delta_logit_delta_.item(), "logit_delta" :logit_delta_.item(), "np_logit_delta":np_logit_delta_.item(), "raw_data": raw_data})

    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # with jsonlines.open(args.output_path + f"{args.dataset.split('/')[-2]}_{args.dataset.split('/')[-1].split('.')[0]}{args.pretrain.split('/')[-1]}.json" + str(strategy.get_rank()), mode="w") as writer:
    #     writer.write_all(output_dataset)
    # # wait unitl all processes generate done
    # dist.barrier()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="score", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    parser.add_argument("--prompt_key", type=str, default='prompt')
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    if args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()
