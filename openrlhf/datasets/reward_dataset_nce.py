from typing import Callable, List, Tuple, Any, Dict
from torch.utils.data import Dataset
from openrlhf.datasets.utils import exist_and_not_none
from openrlhf.utils.utils import zero_pad_sequences

# 如果真的要用缓存，可再启用
# from datasets import load_from_disk


def preprocess_data(
    data: Dict[str, Any],
    input_template: str | None = None,
    prompt_key: str | None = None,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    apply_chat_template: Callable | None = None,
    is_dpo: bool = False,         # 顺序：注意 is_dpo 在前
    n_fake_prompt: int = 3,       #       n_fake_prompt 在后（你原来传反了）
    nce_hard_level: str | None = None,
    nce_grad_type: str | None = None
) -> Tuple[str, str, str, float, List[str]]:
    """
    返回:
      prompt            : 已经加上 generation prompt 的 "用户侧前缀"（字符串）
      chosen / rejected : 仅包含 assistant 侧的内容（字符串），可与 prompt 拼接成完整样本
      margin            : float/int
      fake_prompt_list  : 长度 <= n_fake_prompt 的字符串列表，每个元素是 "fake prompt" 的前缀（已含 generation prompt）
    """
    if apply_chat_template is None:
        raise ValueError("apply_chat_template is None")
    if not prompt_key:
        raise ValueError("prompt_key is None")

    # 1) 真 prompt 前缀（含 generation prompt）
    prompt = apply_chat_template(
        data[prompt_key],
        tokenize=False,
        add_generation_prompt=True
    )

    # 2) 真 response（仅 assistant 段），通过裁掉前缀得到
    full_chosen   = apply_chat_template(data[prompt_key] + data[chosen_key],   tokenize=False)
    full_rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)
    chosen   = full_chosen[len(prompt):]
    rejected = full_rejected[len(prompt):]

        # for nce_hard_level in easy middle hard ; do
        #     for nce_grad_type in rm_grad rm_wo_grad two_head_grad ; do

    # 3) 假 prompt 列表（如果缺少某些 key，就跳过）
    fake_prompt_list: List[str] = []
    if nce_hard_level == 'easy' :
        for i in range(1, n_fake_prompt + 1):
            fk = f"{prompt_key}_{i}"
            if fk in data and exist_and_not_none(data, fk):
                fp = apply_chat_template(
                    data[fk],
                    tokenize=False,
                    add_generation_prompt=True
                )
                fake_prompt_list.append(fp)
    elif nce_hard_level == 'middle':
        assert n_fake_prompt==3, "n_fake_prompt must be 3 for nce_hard_level middle"
        for i in range(1, n_fake_prompt + 1):
            fk = f"{prompt_key}_{i}"
            # i=1,2的样本使用随机prompt
            if (fk in data) and (exist_and_not_none(data, fk)) and (i!=3):
                fp = apply_chat_template(
                    data[fk],
                    tokenize=False,
                    add_generation_prompt=True
                )
                fake_prompt_list.append(fp)
            # i=3的样本使用原prompt，供后续-1/5操作
            elif i==3:
                fake_prompt_list.append(prompt)
    elif nce_hard_level == 'hard':
        assert n_fake_prompt==3, "n_fake_prompt must be 3 for nce_hard_level hard"
        # 所有hard样本使用原prompt，供后续-1/5操作
        for i in range(1, n_fake_prompt + 1):
            fk = f"{prompt_key}_{i}"
            fake_prompt_list.append(prompt)

    # 4) margin（若无则给 0）
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin, fake_prompt_list


class RewardDatasetNCE(Dataset):
    """
    Dataset for reward model (NCE/CMI variant)

    Returns from __getitem__:
      (
        chosen_ids, chosen_mask,
        reject_ids, rejects_mask,
        extra,   # margin
        fake_prompt_response_input_list  # List of tuples length n_fake_prompt:
                                         #   [
                                         #     (fake_chosen_ids, fake_chosen_mask,
                                         #      fake_reject_ids, fake_rejects_mask),
                                         #     ...
                                         #   ]
      )
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template: str | None = None,
        is_dpo: bool = False,
        num_processors: int = 8,
        n_fake_prompt: int = 3,
        nce_hard_level=None,
        nce_grad_type=None
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat_template / keys
        self.input_template = input_template
        self.prompt_key   = getattr(self.strategy.args, "prompt_key",   None) or "prompt"
        self.chosen_key   = getattr(self.strategy.args, "chosen_key",   None) or "chosen"
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None) or "rejected"
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        self.n_fake_prompt = n_fake_prompt
        self.nce_hard_level = nce_hard_level
        self.nce_grad_type = nce_grad_type

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        else:
            # 这里强制要求 chat_template，因为你后续逻辑依赖 "prompt + chosen"
            raise ValueError("apply_chat_template must be True for this dataset.")

        # 预处理
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Materialize
        self.prompts           = processed_dataset["prompt"]
        self.chosens           = processed_dataset["chosen"]
        self.rejects           = processed_dataset["reject"]
        self.extras            = processed_dataset["extra"]              # margin
        self.fake_prompt_lists = processed_dataset["fake_prompt_list"]   # List[List[str]]

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt, chosen, reject, margin, fake_prompt_list = preprocess_data(
            data=data,
            input_template=self.input_template,
            prompt_key=self.prompt_key,
            chosen_key=self.chosen_key,
            rejected_key=self.rejected_key,
            apply_chat_template=self.apply_chat_template,
            is_dpo=self.is_dpo,                    # ✅ 参数顺序已修正
            n_fake_prompt=self.n_fake_prompt,      # ✅
            nce_hard_level=self.nce_hard_level,
            nce_grad_type=self.nce_grad_type
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra":  margin,                       # 统一用 margin；以后要 DPO 再改
            "fake_prompt_list": fake_prompt_list,
        }

    def __len__(self) -> int:
        return len(self.chosens)

    def _ensure_eos(self, text: str) -> str:
        text = text.rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        return text

    def __getitem__(self, idx: int):
        prompt  = self.prompts[idx]
        chosen  = self.chosens[idx]
        reject  = self.rejects[idx]
        extra   = self.extras[idx]
        fake_prompt_list = list(self.fake_prompt_lists[idx])

        # 保证 fake_prompt_list 长度 == n_fake_prompt（不足回填为真 prompt）
        if len(fake_prompt_list) < self.n_fake_prompt:
            fake_prompt_list += [prompt] * (self.n_fake_prompt - len(fake_prompt_list))
        else:
            fake_prompt_list = fake_prompt_list[: self.n_fake_prompt]

        # 真样本：prompt + chosen / reject
        chosen_text = self._ensure_eos(prompt + chosen)
        reject_text = self._ensure_eos(prompt + reject)

        chosen_response_text = self._ensure_eos(chosen)
        reject_response_text = self._ensure_eos(reject)
        chosen_response_token = self.tokenizer(
            chosen_response_text, max_length=self.max_length, padding=False,
            truncation=True, return_tensors="pt", add_special_tokens=False
        )
        reject_response_token = self.tokenizer(
            reject_response_text, max_length=self.max_length, padding=False,
            truncation=True, return_tensors="pt", add_special_tokens=False
        )
        chosen_response_length = len(chosen_response_token)
        reject_response_length = len(reject_response_token)

        chosen_token = self.tokenizer(
            chosen_text, max_length=self.max_length, padding=False,
            truncation=True, return_tensors="pt", add_special_tokens=False
        )
        reject_token = self.tokenizer(
            reject_text, max_length=self.max_length, padding=False,
            truncation=True, return_tensors="pt", add_special_tokens=False
        )

        # 避免 EOS 被截断: 把最后一个 token 强制成 EOS，并确保其 mask=1
        chosen_token["input_ids"][0][-1]        = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1]        = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1]   = True
        reject_token["attention_mask"][0][-1]   = True

        # 假样本（每个 fake_prompt 与真 response 组合）
        fake_prompt_response_input_list: List[Tuple[Any, Any, Any, Any]] = []
        for fp in fake_prompt_list:
            fake_chosen_text = self._ensure_eos(fp + chosen)
            fake_reject_text = self._ensure_eos(fp + reject)

            fake_chosen_token = self.tokenizer(
                fake_chosen_text, max_length=self.max_length, padding=False,
                truncation=True, return_tensors="pt", add_special_tokens=False
            )
            fake_reject_token = self.tokenizer(
                fake_reject_text, max_length=self.max_length, padding=False,
                truncation=True, return_tensors="pt", add_special_tokens=False
            )

            fake_chosen_token["input_ids"][0][-1]      = self.tokenizer.eos_token_id
            fake_reject_token["input_ids"][0][-1]      = self.tokenizer.eos_token_id
            fake_chosen_token["attention_mask"][0][-1] = True
            fake_reject_token["attention_mask"][0][-1] = True

            fake_prompt_response_input_list.append((
                fake_chosen_token["input_ids"],
                fake_chosen_token["attention_mask"],
                fake_reject_token["input_ids"],
                fake_reject_token["attention_mask"],
            ))

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
            fake_prompt_response_input_list,   # << 供 collate_fn 聚合
            chosen_response_length,
            reject_response_length
        )

    def collate_fn(self, item_list: List[Tuple]):
        """
        输出：
          chosen_ids, chosen_masks, reject_ids, rejects_masks, extras,
          fake_prompt_response_input_list_batched
            = [
                (fake_chosen_ids, fake_chosen_masks, fake_reject_ids, fake_rejects_masks),
                ...
              ]  # 长度 = n_fake_prompt
        """
        # 真样本容器
        chosen_ids, chosen_masks = [], []
        reject_ids, rejects_masks = [], []
        chosen_response_lengths = []
        reject_response_lengths = []
        extras = []

        # 假样本容器（动态长度）
        #   每个条目都是 (list_ids, list_masks, list_ids, list_masks)
        fake_buckets = [ ([], [], [], []) for _ in range(self.n_fake_prompt) ]

        # 拆包
        for (c_id, c_ms, r_id, r_ms, extra, fake_list, chosen_response_length, reject_response_length) in item_list:
            chosen_ids.append(c_id)
            chosen_masks.append(c_ms)
            reject_ids.append(r_id)
            rejects_masks.append(r_ms)
            extras.append(extra)
            chosen_response_lengths.append(chosen_response_length)
            reject_response_lengths.append(reject_response_length)

            # fake_list: List[ (ids, mask, ids, mask) ]，长度==n_fake_prompt（前面已保证）
            assert len(fake_list) == self.n_fake_prompt, \
                f"fake_list len {len(fake_list)} != n_fake_prompt {self.n_fake_prompt}"
            for k in range(self.n_fake_prompt):
                f_c_id, f_c_ms, f_r_id, f_r_ms = fake_list[k]
                fake_buckets[k][0].append(f_c_id)
                fake_buckets[k][1].append(f_c_ms)
                fake_buckets[k][2].append(f_r_id)
                fake_buckets[k][3].append(f_r_ms)

        # padding 方向
        padding_side = "right" if self.is_dpo else "left"

        # 真样本 padding
        chosen_ids      = zero_pad_sequences(chosen_ids,      side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks    = zero_pad_sequences(chosen_masks,    side=padding_side)
        reject_ids      = zero_pad_sequences(reject_ids,      side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks   = zero_pad_sequences(rejects_masks,   side=padding_side)

        # 假样本逐桶 padding 后组装
        fake_prompt_response_input_list_batched = []
        for k in range(self.n_fake_prompt):
            (f_c_ids, f_c_ms, f_r_ids, f_r_ms) = fake_buckets[k]
            f_c_ids = zero_pad_sequences(f_c_ids, side=padding_side, value=self.tokenizer.pad_token_id)
            f_c_ms  = zero_pad_sequences(f_c_ms,  side=padding_side)
            f_r_ids = zero_pad_sequences(f_r_ids, side=padding_side, value=self.tokenizer.pad_token_id)
            f_r_ms  = zero_pad_sequences(f_r_ms,  side=padding_side)
            fake_prompt_response_input_list_batched.append(
                (f_c_ids, f_c_ms, f_r_ids, f_r_ms)
            )

        return (
            chosen_ids,
            chosen_masks,
            reject_ids,
            rejects_masks,
            extras,   # margin list
            fake_prompt_response_input_list_batched,  # ✅ 符合你的训练循环期望
            chosen_response_lengths,
            reject_response_lengths,
        )
