from typing import Optional

import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

        # set_z3_leaf_modules is required for MoE models
        for m in model.modules():
            # https://github.com/microsoft/DeepSpeed/pull/4966
            if "SparseMoeBlock" in m.__class__.__name__:
                deepspeed.utils.set_z3_leaf_modules(model, [m.__class__])
                print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                break

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))
            
            self.log_y_r_head_prefix = "y_r_score"
            setattr(self, self.log_y_r_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.log_b_score_head_prefix = "b_score"
            setattr(self, self.log_b_score_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def get_prompt_mask(self, attention_mask, c_mask_R_only, r_mask_R_only):
            # ---- 基于 attention_mask 与 {c,r}_mask_R_only 推断 prompt 位置，得到 P_mask ----
            assert c_mask_R_only is not None and r_mask_R_only is not None, "需要提供 c_mask_R_only 与 r_mask_R_only"

            B, S = attention_mask.shape
            half = B // 2
            device = attention_mask.device
            am_dtype = attention_mask.dtype

            # response_mask: 与 attention_mask 同形状，response 位置为 1，其余为 0
            response_mask = torch.zeros_like(attention_mask, dtype=am_dtype)

            # 将每条样本的 R-only 掩码按“从后向前对齐”的方式贴到 response_mask 上
            for i in range(B):
                if i < half:
                    r_only = c_mask_R_only[i]
                else:
                    r_only = r_mask_R_only[i - half]

                # 确保是一维张量 + 正确设备/类型
                if not isinstance(r_only, torch.Tensor):
                    r_only = torch.tensor(r_only, device=device, dtype=am_dtype)
                else:
                    r_only = r_only.to(device=device, dtype=am_dtype)
                r_only = r_only.view(-1)

                L = int(r_only.numel())
                if L == 0:
                    continue

                # 右对齐写入到最后 L 个位置（等价于“从后向前对比/对齐”）
                response_mask[i, S - L:S] = r_only

            # 只保留真实 token 处（去掉 padding 的影响）
            response_mask = (response_mask & attention_mask).to(am_dtype)

            # prompt 区域 = 有 token 且不在 response 的位置
            P_mask = (attention_mask.to(torch.int8) & (~response_mask.to(torch.bool))).to(am_dtype)

            # # 需要的话，存到 outputs 里供后续采样或可视化使用
            # outputs["P_mask"] = P_mask
            # outputs["response_mask"] = response_mask
            return P_mask

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
            **kwargs
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids, **kwargs
            )
            last_hidden_states = outputs["last_hidden_state"]
            # print(last_hidden_states.shape)  # shape (4, 283, 2048)

            # # --- 推断 prompt位置，采样计算Var(B) ---
            # estimation_var_sample_num = kwargs.pop("estimation_var_sample_num", 0)
            # estimation_var_sample_rate = kwargs.pop("estimation_var_sample_rate", 0.0)
            # if estimation_var_sample_num>0 and estimation_var_sample_rate!=0.0:
            #     outputs['b_score_reward'] = []
            #     c_mask_R_only = kwargs.pop("c_mask_R_only", None)
            #     r_mask_R_only = kwargs.pop("r_mask_R_only", None)

            #     P_mask = self.get_prompt_mask(attention_mask, c_mask_R_only, r_mask_R_only)
            #     P_mask_tensor = P_mask

            #     B2, N2, H = last_hidden_states.shape
            #     assert P_mask_tensor.shape[0] == B2 and P_mask_tensor.shape[1] == N2, \
            #         f"P_mask 形状 {tuple(P_mask_tensor.shape)} 与 last_hidden_states {tuple(last_hidden_states.shape)} 不一致"

            #     device = last_hidden_states.device
            #     P_mask_bool = P_mask_tensor.to(torch.bool).to(device)

            #     with torch.no_grad():
            #         # # 基础 mask（含原生 padding），每次采样在此基础上置 0
            #         # base_am = attention_mask.to(device)  # [B,N]
            #         for _ in range(estimation_var_sample_num):

            #             randu = torch.rand(B2, N2, device=device)
            #             sample_mask = P_mask_bool & (randu < estimation_var_sample_rate)

            #             # 生成本次采样的 attention mask：把被抽中的 prompt 位置置 0（作为 key/value 屏蔽）
            #             mc_attention_mask = attention_mask.clone()
            #             mc_attention_mask = mc_attention_mask.masked_fill(sample_mask, 0)

            #             # 再前向一次（不传 input_embeds，保持与常规路径一致）
            #             mc_outputs = getattr(self, self.base_model_prefix)(
            #                 input_ids=input_ids,
            #                 attention_mask=mc_attention_mask,
            #                 position_ids=position_ids,
            #                 **kwargs
            #             )
            #             mc_last_hidden = mc_outputs["last_hidden_state"]  # [B,N,H]

            #             # 方案1：用本次的有效 token（mc_attention_mask==1）做 masked-mean 池化
            #             # 容易输出层被hack
            #             # valid = mc_attention_mask.to(mc_last_hidden.dtype).unsqueeze(-1)  # [B,N,1]
            #             # summed = (mc_last_hidden * valid).sum(dim=1)                      # [B,H]
            #             # denom = valid.sum(dim=1).clamp_min(1e-6)                          # [B,1]
            #             # pooled = summed / denom                                           # [B,H]

            #             # 方案2：直接复用输出的线性层，但是不传梯度
            #             last_layer_weight = getattr(self, self.value_head_prefix).weight.detach()
            #             b_score = nn.functional.linear(mc_last_hidden, last_layer_weight, None).squeeze(-1)                    # [B,N]

            #             if self.packing_samples:
            #                 b_score = gather_and_pad_tensor(b_score, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            #             b_score_reward = b_score.gather(dim=1, index=eos_indices).squeeze(1)

            #             outputs['b_score_reward'].append(b_score_reward)

            last_layer_weight = getattr(self, self.value_head_prefix).weight.detach()
            nce_logits_wo_grad_raw = nn.functional.linear(last_hidden_states, last_layer_weight, None).squeeze(-1)                    # [B,N]

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # shape (4, 283)
            log_y_r_values = getattr(self, self.log_y_r_head_prefix)(last_hidden_states).squeeze(-1)  # shape (4, 283)
            
            # last_layer_weight = getattr(self, self.value_head_prefix).weight.detach()
            # log_y_r_values = nn.functional.linear(last_hidden_states, last_layer_weight, None).squeeze(-1)
            
            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
                log_y_r_values = gather_and_pad_tensor(log_y_r_values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
                nce_logits_wo_grad_raw = gather_and_pad_tensor(nce_logits_wo_grad_raw, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
                

            reward = values.gather(dim=1, index=eos_indices).squeeze(1)
            log_y_r_values_reward = log_y_r_values.gather(dim=1, index=eos_indices).squeeze(1)
            nce_logits_wo_grad = nce_logits_wo_grad_raw.gather(dim=1, index=eos_indices).squeeze(1)
            

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std
                log_y_r_values_reward = (log_y_r_values_reward - self.mean) / self.std
                # 此处b_score的正则应该是让\Delta的均值为0
            outputs['reward'] = reward
            outputs['log_y_r_values_reward'] = log_y_r_values_reward
            outputs['nce_logits_wo_grad'] = nce_logits_wo_grad
            
            # hidden_states = outputs['hidden_states'] if 'hidden_states' in outputs else None

            return (reward, outputs) if return_output else reward

        def _tok_id(self, tokenizer=None, tok_str="") -> int:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is None or tid == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer has no id for token: {tok_str!r}")
            return tid


        def _find_last_assistant_content_span(self, valid_ids: list[int], tokenizer) -> tuple[int, int]:
            '''
            针对的是llama 3的chat template, 找到最后一条assistant的内容区间
            '''
            SID = self._tok_id(tokenizer, "<|start_header_id|>")
            EID = self._tok_id(tokenizer, "<|end_header_id|>")
            EOT = self._tok_id(tokenizer, "<|eot_id|>")

            last_header_end = -1
            cur_header_start = -1
            length = len(valid_ids)
            # 扫 header
            for i, t in enumerate(valid_ids[::-1]):
                if t == EID and tokenizer.decode(valid_ids[::-1][i+1]).strip().lower() == 'assistant':
                    rev_header_start = i
                    content_start = len(valid_ids) - 3 - rev_header_start
                    return (content_start, length)
            return (length, length)

        def _find_last_assistant_content_span_wo_chat_template(self, valid_ids: list[int], tokenizer) -> tuple[int, int]:
            '''
            针对的是llama 3的chat template, 找到最后一条assistant的内容区间
            '''
            mark_token = self._tok_id(":")

            last_header_end = -1
            cur_header_start = -1
            length = len(valid_ids)
            # 扫 header
            for i, t in enumerate(valid_ids[::-1]):
                if t == mark_token and tokenizer.decode(valid_ids[::-1][i+1]).strip().lower() == 'Assistant':
                    rev_header_start = i
                    content_start = len(valid_ids) - 2 - rev_header_start
                    return (content_start, length)
            return (length, length)

        def extract_assistant_only_batch(self, ids: torch.Tensor, mask: torch.Tensor, tokenizer=None, apply_chat_template=True):
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
            self.apply_chat_template = apply_chat_template
            assert ids.dim() == 2 and mask.shape == ids.shape
            device = ids.device

            BOS = self._tok_id(tokenizer, "<|begin_of_text|>")
            EOT = self._tok_id(tokenizer, "<|eot_id|>")
            EOS = self._tok_id(tokenizer, "<|end_of_text|>")

            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                # 常见做法：没有pad就用eos充当pad
                pad_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None:
                # 兜底：用 end_of_text 充当 pad
                pad_id = EOS

            B, L = ids.shape
            out_sequences = []

            for b in range(B):
                # 去掉左pad：保留mask==1的有效token
                valid_ids = ids[b][mask[b].bool()].tolist()

                if self.apply_chat_template:
                    # 找最后一条assistant内容区间
                    s, e = self._find_last_assistant_content_span(valid_ids, tokenizer)
                    assistant_content = valid_ids[s:e]  # 可能为空

                    # 目标序列：BOS + content + EOT + EOS
                    new_seq = [BOS] + assistant_content
                else:
                    # 找最后一条assistant内容区间
                    s, e = self._find_last_assistant_content_span_wo_chat_template(valid_ids, tokenizer)
                    assistant_content = valid_ids[s:e]  # 可能为空

                    # 目标序列：BOS + content + EOT + EOS
                    new_seq = assistant_content
                out_sequences.append(torch.tensor(new_seq, dtype=torch.long, device=device))

            # 左填充到同一长度
            max_len = max(seq.numel() for seq in out_sequences)
            new_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            new_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)

            for b, seq in enumerate(out_sequences):
                n = seq.numel()
                new_ids[b, max_len - n:] = seq
                new_mask[b, max_len - n:] = 1

            return new_ids, new_mask


        def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, tokenizer):
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
                    pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
                    pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
                ),
                dim=0,
            )
            max_length = max(c_mask.shape[1], r_mask.shape[1])
            att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)

            return inputs_ids, att_masks

        def forward_R_only(
            self,
            tokenizer,
            chosen_ids, 
            c_mask, 
            reject_ids, 
            r_mask
        ):
            B = chosen_ids.shape[0]
            chosen_ids_R_only, c_mask_R_only = self.extract_assistant_only_batch(chosen_ids, c_mask, tokenizer)
            reject_ids_R_only, r_mask_R_only = self.extract_assistant_only_batch(reject_ids, r_mask, tokenizer)
            # debug_preview_batch(self.tokenizer, chosen_ids_R_only, c_mask_R_only, reject_ids_R_only, r_mask_R_only, file='output/data_log/cliped_response.txt', show_special=True)

            # 先拿到拼接后的 input 与原始 mask
            input_ids, att_masks = self.concatenated_inputs(chosen_ids_R_only, c_mask_R_only, reject_ids_R_only, r_mask_R_only, tokenizer)  # [2B, L]
            
            # 为拼接后的前 B（chosen）与后 B（reject）分别屏蔽各自的 prompt
            att_masks_ro = att_masks.clone()

            # 二次前向（R-only）：禁用 hidden_states 以节省显存
            rewards_ro, out_ro = self.forward(
                input_ids,
                attention_mask=att_masks_ro,
                output_hidden_states=False,
                return_output=True
            )
            s_c_ro_a = out_ro['log_y_r_values_reward'][:B]
            s_r_ro_a = out_ro['log_y_r_values_reward'][B:]
            a_logit = s_c_ro_a - s_r_ro_a

            r_logit = rewards_ro[:B] - rewards_ro[B:]

            output = {
                'a_logit': a_logit,
                'r_logit': r_logit,
                'c_mask_R_only': c_mask_R_only,
                'r_mask_R_only': r_mask_R_only
            }
            return output

        def concatenated_forward(self, tokenizer, chosen_ids, c_mask, reject_ids, r_mask, **kwargs):
            """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

            We do this to avoid doing two forward passes, because it's faster for FSDP.
            """
            input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, tokenizer)
            # ============================ 修改点 ============================
            # 增加 output_hidden_states=True 来获取所有层的输出
            # `return_output=True` 似乎是您自定义的参数，这里我们确保 transformers 的原生参数被正确传递
            # 假设您的模型内部最终会调用一个Hugging Face的预训练模型
            rewards, output = self.forward(
                input_ids, 
                attention_mask=att_masks, 
                output_hidden_states=False,  # <--- 核心修改
                return_output=True,          # 保持您原有的逻辑
                **kwargs
            )

            chosen_rewards = rewards[: chosen_ids.shape[0]]
            rejected_rewards = rewards[chosen_ids.shape[0] :]
            aux_loss = output.aux_loss if "aux_loss" in output else []

            # 将 hidden_states 也一并返回，以便在训练循环中使用
            return chosen_rewards, rejected_rewards, aux_loss


    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # (1, total_seqs)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
