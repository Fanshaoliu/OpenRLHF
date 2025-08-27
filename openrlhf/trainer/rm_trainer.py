from contextlib import redirect_stdout
import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Optional
import json, os
from datetime import datetime

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
        mask_cpu = mask.detach().to("cpu").bool()
    except:
        ids_cpu  = [ids]
        mask_cpu = [mask]

    texts = []
    for seq, m in zip(ids_cpu, mask_cpu):
        valid = seq[m]             # 只保留有效 token
        if valid.numel() == 0:
            texts.append("")       # 空序列兜底
            continue
        text = tokenizer.decode(
            valid.tolist(),
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

class LayerwiseTokenGradLogger:
    def __init__(
        self,
        tokenizer,
        out_jsonl,
        log_every=10,
        only_rank0=True,
        only_chosen=True,
        store_vectors=False,   # 为 True 时也写出每个 token 的梯度向量，体积很大，默认 False
        decode_tokens=False,   # 为 True 时额外写出 tokens 的可读文本
        norm_type="l2",        # "l2" | "l1" | "max"
    ):
        self.tokenizer = tokenizer
        self.out_jsonl = out_jsonl
        self.log_every = log_every
        self.only_rank0 = only_rank0
        self.only_chosen = only_chosen
        self.store_vectors = store_vectors
        self.decode_tokens = decode_tokens
        self.norm_type = norm_type

        self.hooks = []
        self.num_layers = 0

        # 运行态
        self.active = False
        self._step = None
        self._chosen_bs = None
        self._ids_cpu = None
        self._mask_cpu = None
        self._collected = None  # {layer_idx: {"norms": [Bc,S], "vectors": [Bc,S,D]?}}

    def _rank_is(self):
        return int(dist.get_rank())

    def _compute_norm(self, g):
        # g: [B,S,D] float
        if self.norm_type == "l1":
            return g.abs().sum(dim=-1)
        elif self.norm_type == "max":
            return g.abs().amax(dim=-1)
        else:
            return torch.linalg.vector_norm(g, dim=-1)

    def _make_bwd_hook(self, layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output 是 tuple，取第一个是 hidden_states 的梯度: [B,S,D]
            if not self.active:
                return

            g = grad_output[0]
            if g is None:
                return
            # 只保留 chosen 部分（前 Bc 条），避免和 reject 混在一起
            if self.only_chosen and self._chosen_bs is not None:
                g = g[: self._chosen_bs]
            g = g.float()  # 混合精度下先转 float 以稳定范数计算
            norms = self._compute_norm(g).detach().cpu()  # [Bc,S]
            payload = {"norms": norms}
            if self.store_vectors:
                payload["vectors"] = g.detach().cpu()  # [Bc,S,D] 非常大，谨慎开启
            self._collected[layer_idx] = payload
        return hook

    def register_on_model(self, model):
        # 找到 transformer 层列表（兼容常见结构）
        m = model.module if hasattr(model, "module") else model

        layers = None
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            print('Found "model.layers" in the model structure.')
            layers = m.model.layers
        elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            print('Found "transformer.h" in the model structure.')
            layers = m.transformer.h
        elif hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            print('Found "gpt_neox.layers" in the model structure.')
            layers = m.gpt_neox.layers
        elif hasattr(m, "backbone") and hasattr(m.backbone, "layers"):
            print('Found "backbone.layers" in the model structure.')
            layers = m.backbone.layers

        if layers is None:
            raise RuntimeError("Cannot locate transformer blocks; please adapt `register_on_model` to your model.")

        self.num_layers = len(layers)
        for li, layer in enumerate(layers):
            h = layer.register_full_backward_hook(self._make_bwd_hook(li))
            self.hooks.append(h)

    def begin(self, chosen_ids, reject_ids, c_mask, r_mask, step, chosen_len, reject_len):
        # 只有 rank0 才写盘（避免多进程并发写）
        if self.only_rank0:
            if self._rank_is() != 0:
                self.active = False
                return

        self.active = (step % self.log_every == 0)
        if not self.active:
            return

        self._step = int(step)
        self._chosen_bs = chosen_ids.size(0)
        self.chosen_ids_cpu = chosen_ids.detach().cpu()
        self.reject_ids_cpu = reject_ids.detach().cpu()
        self.chosen_mask_cpu = c_mask.detach().cpu()
        self.reject_mask_cpu = r_mask.detach().cpu()
        self._collected = {}
        self.chosen_len = chosen_len
        self.reject_len = reject_len

    def _align_list_leftpad(self, seq_list, target_len, pad_value=0):
        # 将 1D 列表左侧补齐到 target_len；若过长则截取右端
        L = len(seq_list)
        if L == target_len:
            return seq_list
        if L < target_len:
            return [pad_value] * (target_len - L) + seq_list
        else:
            return seq_list[-target_len:]

    def end(self):
        if not self.active:
            return

        os.makedirs(os.path.dirname(self.out_jsonl), exist_ok=True)
        Bc = self.chosen_ids_cpu.shape[0]

        # 可选：tokens 文本
        tokens_text = None
        if self.decode_tokens:
            try:
                tokens_text = [self.tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in self.chosen_ids_cpu]
            except Exception:
                tokens_text = [
                    [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in seq.tolist()]
                    for seq in self.chosen_ids_cpu
                ]

        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            for b in range(Bc):
                rec = {
                    "step": self._step,
                    "seq_idx": b,
                    "token_ids": self.chosen_ids_cpu[b].tolist(),  # 若你也想与梯度长度对齐，可在此对齐（见下方注释）
                }
                if tokens_text is not None:
                    rec["tokens"] = tokens_text[b]

                per_layer = []
                for li in range(self.num_layers):
                    item = self._collected.get(li)
                    if item is None:
                        per_layer.append(None)
                        continue
                    # 该层该样本的梯度范数（长度为 S_grad）
                    norms_t = item["norms"][b]          # tensor [S_grad]
                    norms = norms_t.tolist()[:self.chosen_len]
                    S_grad = len(norms)
                    
                    # 对齐 mask 到 S_grad（左填充或截断右端），再应用 mask
                    if self.chosen_mask_cpu is not None:
                        mask_b = self.chosen_mask_cpu[b].tolist()        # 可能长度为 S_mask != S_grad
                        mask_b = self._align_list_leftpad(mask_b, S_grad, pad_value=0)
                        # 应用 mask：pad 位置置 0
                        norms = [float(v) if mask_b[i] == 1 else 0.0 for i, v in enumerate(norms)]

                    if self.store_vectors:
                        vec = item["vectors"][b]  # tensor [S_grad, D] (cpu)
                        if self.chosen_mask_cpu is not None:
                            # 对齐后的 mask 转为 [S_grad,1] 以广播到向量
                            mask_aligned = torch.tensor(mask_b, dtype=vec.dtype).view(-1, 1)
                            vec = (vec * mask_aligned)
                        per_layer.append({"norm": norms, "vec": vec.tolist()})
                    else:
                        per_layer.append(norms)

                rec["layers"] = per_layer

                # 如需让 token_ids 也与梯度长度一致，可使用与 mask 相同的左填充/截断逻辑：
                # 如果想对齐，请取消下两行注释：
                # S_ref = next((len(x) for x in per_layer if isinstance(x, list)), len(rec["token_ids"]))
                # rec["token_ids"] = self._align_list_leftpad(rec["token_ids"], S_ref, pad_value=self.tokenizer.pad_token_id or 0)

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 清理
        self._collected = None
        self.chosen_ids_cpu = None
        self.reject_ids_cpu = None
        self.chosen_mask_cpu = None
        self.reject_mask_cpu = None
        self.active = False

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

import csv
import numpy as np
import os, json
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class RewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        lambda_cmi: float = 0.0,
        clap_cap: float = 1.0,
        loss="sigmoid",
        disable_ds_ckpt=False,
        save_hf_ckpt=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.disable_ds_ckpt = disable_ds_ckpt
        self.save_hf_ckpt = save_hf_ckpt
        self.lambda_cmi = lambda_cmi
        self.clap_cap = clap_cap

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
                settings=wandb.Settings(init_timeout=300)  # 增加超时设置
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        # # 在 __init__ 的最后添加（路径请改成你自己的）：
        # self.layer_grad_logger = LayerwiseTokenGradLogger(
        #     tokenizer=self.tokenizer,
        #     out_jsonl="/mnt/data/user/liu_shaofan/OpenRLHF/output/layer_chosen_hidden_grads.jsonl",
        #     log_every=10,          # 每 100 step 记录一次
        #     only_rank0=True,        # 只在 rank0 写文件
        #     only_chosen=True,       # 只记录 chosen 半部分，避免和 reject 混在一起
        #     store_vectors=False,    # 如需写出整条向量可改 True（非常大）
        #     decode_tokens=True,    # 如要按特殊 token 分段再聚合，可改 True
        #     norm_type="l2",
        # )
        # raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # self.layer_grad_logger.register_on_model(raw_model)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        acc_sum = 0
        loss_sum = 0
            
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # [DEBUG] 打印数据
                debug_preview_batch(self.tokenizer, chosen_ids, c_mask, reject_ids, r_mask,
                                    n_preview=2, show_special=True)

                chosen_reward, reject_reward, aux_loss, hidden_states = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                # self.layer_grad_logger.begin(chosen_ids, reject_ids, c_mask, r_mask, step, chosen_ids.shape[1], reject_ids.shape[1])

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                
                
                # --- 关键：自动推断 prompt_len，并做 R-only 二次前向得到 a(R) ---
                prompt_len = self.compute_prompt_len_by_pair(chosen_ids, c_mask, reject_ids, r_mask)  # [B]
                a_logit = self.forward_R_only(
                    self.model,
                    chosen_ids, c_mask, reject_ids, r_mask,
                    prompt_len=prompt_len
                )

                # 条件互信息近似项：CE(y, σ(Δ)) - CE(y, σ(a))
                if margin is not None:
                    ce_a = -F.logsigmoid(a_logit - margin).mean()
                else:
                    a_logit_claped = torch.clamp(a_logit, -self.clap_cap, self.clap_cap)
                    ce_a = -F.logsigmoid(a_logit_claped).mean()

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                # loss = preference_loss + aux_loss * self.args.aux_loss_coef + self.lambda_cmi * loss_cmi
                loss = (1+self.lambda_cmi)*preference_loss - self.lambda_cmi * ce_a + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # # 结束并写出这一 step 的层/Token 梯度（如果该步处于采样步）
                # self.layer_grad_logger.end()

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # optional rm info
                logs_dict = {
                    "bt_loss": preference_loss.item(),
                    # "ce_delta_loss": ce_delta.item(),
                    "ce_a": ce_a.item(),
                    # "loss_cmi": loss_cmi.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0, model_name='', data_name='', eval_result_path=''):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, _, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                loss = self.loss_fn(chosen_reward, reject_reward, margin)

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            if eval_result_path!="":
                saved_bar_dict = {
                    "model_name": model_name,
                    "data_name": data_name,
                    "eval_loss": loss_mean,
                    "acc_mean": acc_mean,
                    "reward_mean": reward_mean.item(),
                    "reward_std": reward_std.item(),
                    # 可选加个时间戳，方便追踪
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }

                append_jsonl(f"./output/eval_log/{eval_result_path}.jsonl", saved_bar_dict)

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        # ============================ 修改点 ============================
        # 增加 output_hidden_states=True 来获取所有层的输出
        # `return_output=True` 似乎是您自定义的参数，这里我们确保 transformers 的原生参数被正确传递
        # 假设您的模型内部最终会调用一个Hugging Face的预训练模型
        rewards, output = model(
            input_ids, 
            attention_mask=att_masks, 
            output_hidden_states=True,  # <--- 核心修改
            return_output=True          # 保持您原有的逻辑
        )
        
        # 解包输出
        # Hugging Face模型的标准输出是一个包含多个元素的元组或特定对象
        # all_values 是您的RM的最终打分
        # hidden_states 将是一个元组，包含了embedding层和之后每一层Transformer的输出
        all_values = rewards  # 假设您的模型输出对象有一个 .rewards 属性
        hidden_states = output.hidden_states # 假设输出对象有 .hidden_states 属性
        # ==============================================================

        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        
        # 将 hidden_states 也一并返回，以便在训练循环中使用
        return chosen_rewards, rejected_rewards, aux_loss, hidden_states

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        # if dist.get_rank() == 0:
        #     print("chosen_ids shape:", chosen_ids.shape)
        #     print("reject_ids shape:", reject_ids.shape)
        #     print("c_mask shape:", c_mask.shape)
        #     print("r_mask shape:", r_mask.shape)
        #     print("Concatenated inputs shape:", inputs_ids.shape, att_masks.shape)
        # hosen_ids shape: torch.Size([1, 96]) 
        # reject_ids shape: torch.Size([1, 78])
        # c_mask shape: torch.Size([1, 96])
        # r_mask shape: torch.Size([1, 78])
        # Concatenated inputs shape: torch.Size([2, 96]) torch.Size([2, 96])

        return inputs_ids, att_masks

    def forward_R_only(
        self,
        model,
        chosen_ids, c_mask, reject_ids, r_mask,
        prompt_len: torch.Tensor
    ):
        """
        返回：a_logit = s_c^{R-only} - s_r^{R-only}  （不看 prompt 的 logit 差）
        注意：这里不需要 hidden_states，减少开销。
        """

        B = chosen_ids.shape[0]

        chosen_ids_C, c_mask_C = self.extract_bos_plus_last_assistant(chosen_ids, c_mask)
        reject_ids_R, r_mask_R = self.extract_bos_plus_last_assistant(reject_ids, r_mask)

        # 先拿到拼接后的 input 与原始 mask
        input_ids, att_masks = self.concatenated_inputs(chosen_ids_C, c_mask_C, reject_ids_R, r_mask_R)  # [2B, L]
        
        # 为拼接后的前 B（chosen）与后 B（reject）分别屏蔽各自的 prompt
        att_masks_ro = att_masks.clone()
        # att_masks_ro[:B] = self.zero_first_k_valid(att_masks_ro[:B], prompt_len)   # chosen 半批
        # att_masks_ro[B:] = self.zero_first_k_valid(att_masks_ro[B:], prompt_len)   # reject 半批

        # 二次前向（R-only）：禁用 hidden_states 以节省显存
        rewards_ro, out_ro = model(
            input_ids,
            attention_mask=att_masks_ro,
            output_hidden_states=False,
            return_output=True
        )
        s_c_ro = out_ro['log_y_r_values_reward'][:B]
        s_r_ro = out_ro['log_y_r_values_reward'][B:]
        a_logit = s_c_ro - s_r_ro
        return a_logit
        
    def compute_prompt_len_by_pair(self, chosen_ids, c_mask, reject_ids, r_mask):
        """
        chosen_ids, reject_ids: [B, L]  (Long)
        c_mask, r_mask:        [B, L]  (0/1)
        return: prompt_len     [B]     (Long)
        """
        B, L = chosen_ids.shape
        prompt_len = torch.zeros(B, dtype=torch.long, device=chosen_ids.device)

        for b in range(B):

            c_seq = chosen_ids[b][c_mask[b].bool()]  # 有效 token 序列
            r_seq = reject_ids[b][r_mask[b].bool()]
            K = min(c_seq.numel(), r_seq.numel())
            if K == 0:
                prompt_len[b] = 0
                continue
            # 逐位比较前缀，直到首次不相等
            eq = (c_seq[:K] == r_seq[:K])
            # 找到第一个 False 的下标；若不存在，说明完整前缀相同
            mismatch = (~eq).nonzero(as_tuple=False)
            pl = int(mismatch[0]) if mismatch.numel() > 0 else K
            prompt_len[b] = pl - 4 # 减去\n, 以及<|start_header_id|>assistant<|end_header_id|>一共四个 token
            pl-=4
            debug_preview_batch(self.tokenizer, c_seq[:pl], [True for _ in c_seq[:pl]], r_seq[:pl], [True for _ in r_seq[:pl]], file='out2.txt')
            debug_preview_batch(self.tokenizer, c_seq[pl:], [True for _ in c_seq[pl:]], r_seq[pl:], [True for _ in r_seq[pl:]], file='out3.txt')
            debug_preview_batch(self.tokenizer, c_seq, [True for _ in c_seq], r_seq, [True for _ in r_seq], file='out4.txt')
        return prompt_len

    def _tok_id(self, tok_str: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(tok_str)
        if tid is None or tid == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer has no id for token: {tok_str!r}")
        return tid

    def _find_last_assistant_content_span(self, valid_ids: list[int], tokenizer) -> tuple[int, int]:
        """
        单次扫描：返回最后一条 assistant 消息的内容区间 [content_start, content_end)（不含 EOT）。
        若不存在，返回 (len(valid_ids), len(valid_ids))。
        """
        SID = self._tok_id(self.tokenizer, "<|start_header_id|>")
        EID = self._tok_id(self.tokenizer, "<|end_header_id|>")
        EOT = self._tok_id(self.tokenizer, "<|eot_id|>")

        n = len(valid_ids)
        # 维护解析 header 的状态
        in_header = False
        header_start = -1

        # 当前候选（最近一次解析到的 assistant 头）
        have_active = False           # 是否存在“活跃”的 assistant 段（已看到头，尚未遇到其 EOT）
        waiting_ws = False            # 头之后是否还在跳过空白
        current_content_start = None  # 该候选的内容起点

        # 最终答案（最后一个“完成的”或到序列末尾的 assistant 段）
        last_start = n
        last_end = n

        for i, t in enumerate(valid_ids):
            if t == SID:
                # 新 header 开始：取消尚未完成的候选，转而解析新的 header
                in_header = True
                header_start = i
                have_active = False
                waiting_ws = False
                current_content_start = None
                continue

            if t == EID and in_header:
                # 结束一个 header，判断角色
                role_text = tokenizer.decode(valid_ids[header_start + 1:i]).strip().lower()
                in_header = False
                if role_text == "assistant":
                    # 激活一个候选段，随后跳空白寻找内容起点
                    have_active = True
                    waiting_ws = True
                    current_content_start = None
                else:
                    have_active = False
                    waiting_ws = False
                    current_content_start = None
                continue

            # 非 SID/EID：如果有活跃候选，推进其内容区间
            if have_active:
                if waiting_ws:
                    # 空白直接跳过；若 EOT 在空白期即到来，视为零长度内容（start==end==i）
                    if t == EOT:
                        current_content_start = i  # 与“跳空白后的索引”一致，形成空区间
                        last_start = current_content_start
                        last_end = i
                        have_active = False
                        waiting_ws = False
                        current_content_start = None
                        continue
                    ch = tokenizer.decode([t])
                    if not ch.isspace():
                        current_content_start = i
                        waiting_ws = False

                # 已确定起点后，遇到第一个 EOT 即完成一个段
                if not waiting_ws and current_content_start is not None and t == EOT:
                    last_start = current_content_start
                    last_end = i
                    have_active = False
                    current_content_start = None
                    # 不清 waiting_ws（已不使用）

        # 扫描结束：若还留有活跃候选但未遇到 EOT，则收尾到序列末尾
        if have_active:
            # 若一直在等空白且没遇到任何非空白/EOT，content_start 可能仍为 None：
            # 将其视为“空内容在末尾”
            if current_content_start is None:
                current_content_start = n
            last_start = current_content_start
            last_end = n

        # 若从未找到任何候选，返回空
        return (last_start, last_end)

    def extract_assistant_only_batch(self, ids: torch.Tensor, mask: torch.Tensor):
        """
        输入：
        ids:  [B, L]  左填充的token id
        mask: [B, L]  0/1，有效位置为1（左填充）
        输出：
        new_ids:  [B, L_max'] 左填充的新序列（只包含BOS + 最后assistant内容 + EOT + EOS）
        new_mask: [B, L_max'] 对应的mask（有效位置为1）
        说明：
        - 依赖 self.tokenizer，且其词表需包含 <|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>, <|end_of_text|>
        - 若某条样本内找不到 assistant 头，则内容为空，只输出 BOS + EOT + EOS
        """
        assert ids.dim() == 2 and mask.shape == ids.shape
        device = ids.device

        BOS = _tok_id(self.tokenizer, "<|begin_of_text|>")
        EOT = _tok_id(self.tokenizer, "<|eot_id|>")
        EOS = _tok_id(self.tokenizer, "<|end_of_text|>")

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            # 常见做法：没有pad就用eos充当pad
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            # 兜底：用 end_of_text 充当 pad
            pad_id = EOS

        B, L = ids.shape
        out_sequences = []

        for b in range(B):
            # 去掉左pad：保留mask==1的有效token
            valid_ids = ids[b][mask[b].bool()].tolist()

            # 找最后一条assistant内容区间
            s, e = self._find_last_assistant_content_span(valid_ids, self.tokenizer)
            assistant_content = valid_ids[s:e]  # 可能为空

            # 目标序列：BOS + content + EOT + EOS
            new_seq = [BOS] + assistant_content + [EOT, EOS]
            out_sequences.append(torch.tensor(new_seq, dtype=torch.long, device=device))

        # 左填充到同一长度
        max_len = max(seq.numel() for seq in out_sequences)
        new_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        new_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)

        for b, seq in enumerate(out_sequences):
            n = seq.numel()
            new_ids[b, max_len - n:] = seq
            new_mask[b, max_len - n:] = 1

        debug_preview_batch(new_ids, new_mask,file='out5.txt')

        return new_ids, new_mask