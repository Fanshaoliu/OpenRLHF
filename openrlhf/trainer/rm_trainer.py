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
        
        grad_save_path = '/mnt/data/user/liu_shaofan/OpenRLHF/output/output.jsonl'
        csv_path = '/mnt/data/user/liu_shaofan/OpenRLHF/output/output.csv'

        def make_grad_hook(input_ids):
            def hook_fn(grad_weight):
                # grad_weight: (vocab_size, hidden_dim)
                # --- 替换用的代码段（把原来的 batch_grads 写 jsonl 的部分替换成下面） ---
                # 如果 csv 不存在则写 header
                write_header = not os.path.exists(csv_path)

                num_segments = 10  # prompt/response 各分 10 段

                def segment_average(values, n_segments):
                    L = len(values)
                    if L == 0:
                        return [0.0] * n_segments
                    seg_avgs = []
                    for i in range(n_segments):
                        start = int(np.floor(i * L / n_segments))
                        end   = int(np.floor((i + 1) * L / n_segments))
                        if start >= L or end <= start:
                            seg_avgs.append(0.0)
                        else:
                            seg_avgs.append(float(np.mean(values[start:end])))
                    return seg_avgs

                # 准备 labels（CSV 的列头，顺序固定）
                labels = []
                labels.append("<|im_start|>1")
                labels.append("i_s1")
                labels.append("<|im_end|>1")
                labels.append("<|im_start|>2")
                for i in range(num_segments):
                    labels.append(f"p_s{i+1}")
                labels.append("<|im_end|>2")
                labels.append("<|im_start|>3")
                for i in range(num_segments):
                    labels.append(f"r_s{i+1}")
                labels.append("<|im_end|>3")
                labels.append("<|endoftext|>")

                # 为 batch 中每个序列构造一行（长度 28）
                rows_to_write = []
                for seq in input_ids:
                    # 收集 token 文本与对应的 grad norm
                    token_texts = []
                    token_grads = []
                    for token_id in seq:
                        # token_id 可能是 tensor，也可能是 python int
                        if hasattr(token_id, "item"):
                            tid = int(token_id.item())
                        else:
                            tid = int(token_id)
                        # grad_weight[tid] -> 梯度向量 (tensor)
                        grad_vec = grad_weight[tid]
                        # 有时 grad_vec 在 GPU 上，直接 norm().item() 通常可用；若报错可改为 grad_vec.cpu().norm().item()
                        grad_norm = float(grad_vec.norm().item())
                        token_texts.append(self.tokenizer.decode([tid], skip_special_tokens=False))
                        token_grads.append(grad_norm)

                    # 找出所有分隔符出现的位置（按出现顺序）
                    im_start_idxs = [i for i, t in enumerate(token_texts) if t == "<|im_start|>"]
                    im_end_idxs   = [i for i, t in enumerate(token_texts) if t == "<|im_end|>"]
                    try:
                        eot_idx = token_texts.index("<|endoftext|>")
                    except ValueError:
                        eot_idx = len(token_texts) - 1

                    # 根据出现次数取前三个（若不足则用空段 / 0 填充）
                    if len(im_start_idxs) >= 3 and len(im_end_idxs) >= 3:
                        s1, s2, s3 = im_start_idxs[0], im_start_idxs[1], im_start_idxs[2]
                        e1, e2, e3 = im_end_idxs[0],   im_end_idxs[1],   im_end_idxs[2]
                    else:
                        # 若不满足三对分隔符，尽量容错：把 values 填 0（保证每条输出都是 28 列）
                        # 你可以改这里的容错策略为跳过该样本
                        rows_to_write.append([0.0] * 28)
                        continue

                    # 提取阶段 token 的梯度（不包含分隔符本身）
                    instruct_tokens = token_grads[s1+1 : e1] if e1 > s1+1 else []
                    prompt_tokens   = token_grads[s2+1 : e2] if e2 > s2+1 else []
                    response_tokens = token_grads[s3+1 : e3] if e3 > s3+1 else []

                    # 阶段整体平均
                    instruct_avg = float(np.mean(instruct_tokens)) if len(instruct_tokens) > 0 else 0.0
                    prompt_avg   = float(np.mean(prompt_tokens))   if len(prompt_tokens)   > 0 else 0.0
                    response_avg = float(np.mean(response_tokens)) if len(response_tokens) > 0 else 0.0

                    # prompt / response 各自分段 10 段平均
                    prompt_seg_avg   = segment_average(prompt_tokens, num_segments)
                    response_seg_avg = segment_average(response_tokens, num_segments)

                    # 安全获取特殊标记位置的梯度（若索引越界返回 0）
                    def safe_get(idx):
                        if 0 <= idx < len(token_grads):
                            return token_grads[idx]
                        return 0.0

                    grad_s1 = safe_get(s1); grad_e1 = safe_get(e1)
                    grad_s2 = safe_get(s2); grad_e2 = safe_get(e2)
                    grad_s3 = safe_get(s3); grad_e3 = safe_get(e3)
                    grad_eot = safe_get(eot_idx)

                    # 组装 28 个值（顺序与 labels 一致）
                    vals = []
                    vals.append(grad_s1)            # <|im_start|>1
                    vals.append(instruct_avg)       # i_s1
                    vals.append(grad_e1)            # <|im_end|>1

                    vals.append(grad_s2)            # <|im_start|>2
                    vals.extend(prompt_seg_avg)     # p_s1..p_s10
                    vals.append(grad_e2)            # <|im_end|>2

                    vals.append(grad_s3)            # <|im_start|>3
                    vals.extend(response_seg_avg)   # r_s1..r_s10
                    vals.append(grad_e3)            # <|im_end|>3

                    vals.append(grad_eot)           # <|endoftext|>

                    # 最后长度保证
                    if len(vals) != 28:
                        # 容错：若长度不对，补 0 到 28
                        if len(vals) < 28:
                            vals.extend([0.0] * (28 - len(vals)))
                        else:
                            vals = vals[:28]

                    rows_to_write.append(vals)

                # 写入 CSV（只在 rank 0 写，避免多进程冲突）
                # if int(os.environ.get("RANK", 0)) == 0 and rows_to_write:
                if rows_to_write:
                    # rows_to_write_group_mean = np.mean(rows_to_write_group, axis=1).tolist()
                    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf)
                        if write_header:
                            writer.writerow(labels)
                        for row in rows_to_write:
                            writer.writerow([f"{float(v):.8g}" for v in row])
                # --- end replacement ---

            return hook_fn
            
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

                if step%100==0:
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    embedding_layer = raw_model.model.embed_tokens
                    embedding_layer.weight.register_hook(make_grad_hook(chosen_ids))

                chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )


                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
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

    def evaluate(self, eval_dataloader, steps=0):
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

                chosen_reward, reject_reward, _ = self.concatenated_forward(
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
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

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
        return inputs_ids, att_masks
