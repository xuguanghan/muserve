"""Qwen3.5-397B 完整模型 forward（60 层）。"""

import os
import torch
import torch_musa

from muserve.config import (
    NUM_LAYERS, HIDDEN_SIZE, VOCAB_SIZE, TP_SIZE,
    GDN_NUM_V_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM, GDN_CONV_KERNEL,
)
from muserve.distributed import all_gather_into_tensor, get_tp_rank
from muserve.model.prefix_cache import PrefixCache

_USE_V2 = os.environ.get("MUSERVE_LAYER_V2", "1") == "1"

if _USE_V2:
    from muserve.model.qwen35_layer_v2 import (
        rms_norm_gemma as rms_norm,
        layer_forward_decode,
        layer_forward_prefill,
        _QKV_DIM_LOCAL,
    )
else:
    from muserve.model.qwen35_layer import rms_norm, layer_forward_decode, layer_forward_prefill
    _QKV_DIM_LOCAL = None


def _init_gdn_states(
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """初始化所有层的 GDN state，全零。"""
    return [
        torch.zeros(
            batch_size, GDN_NUM_V_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM,
            device=device, dtype=torch.float32,
        )
        for _ in range(NUM_LAYERS)
    ]


def _init_conv_states(device: torch.device) -> list[torch.Tensor | None]:
    """初始化所有层的 conv state (v2 only)。GDN 层有 conv state，Attention 层为 None。"""
    if not _USE_V2 or _QKV_DIM_LOCAL is None:
        return [None] * NUM_LAYERS
    return [
        torch.zeros(_QKV_DIM_LOCAL, GDN_CONV_KERNEL - 1, device=device, dtype=torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]


class Qwen35Model:
    """Qwen3.5-397B-A17B-FP8 完整模型。

    持有所有层的权重引用，不做额外封装。
    """

    def __init__(
        self,
        embed_weights: dict,
        layer_weights: list[dict],
    ):
        self.embed_weights = embed_weights
        self.layer_weights = layer_weights
        self.rank = get_tp_rank()
        self.prefix_cache = PrefixCache(max_entries=64, storage="cpu")

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding。input_ids: [B, T] → [B, T, HIDDEN]。"""
        w = self.embed_weights["embed_tokens.weight"]  # [VOCAB/TP, HIDDEN]
        # 每卡只有 vocab 的 1/TP，需要 gather
        vocab_per_rank = VOCAB_SIZE // TP_SIZE
        local_ids = input_ids - self.rank * vocab_per_rank
        mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
        local_emb = torch.zeros(
            *input_ids.shape, HIDDEN_SIZE,
            device=input_ids.device, dtype=w.dtype,
        )
        valid = local_ids[mask]
        if valid.numel() > 0:
            local_emb[mask] = w[valid]
        # AllReduce sum（每卡只有自己负责的 token 有值）
        from muserve.distributed import all_reduce
        all_reduce(local_emb)
        return local_emb

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """LM head：[..., HIDDEN] → [..., VOCAB/TP]（TP 切分后的 logits）。"""
        w = self.embed_weights["lm_head.weight"]  # [VOCAB/TP, HIDDEN]
        norm_w = self.embed_weights["norm.weight"]
        x = rms_norm(hidden, norm_w)
        return torch.nn.functional.linear(x.bfloat16(), w.bfloat16())

    def forward_decode(
        self,
        input_ids: torch.Tensor,        # [B, 1]
        gdn_states: list[torch.Tensor], # NUM_LAYERS × [B, V_heads, V_dim, K_dim]
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        conv_states: list[torch.Tensor | None] | None = None,
        positions: torch.Tensor | None = None,  # [B] int64, 当前 decode step 的绝对位置
    ) -> tuple:
        """Decode step forward。返回 (logits, new_gdn_states, new_kv_caches, new_conv_states)。"""
        hidden = self.embed(input_ids)  # [B, 1, HIDDEN]

        new_states = []
        new_kv_caches = []
        new_conv_states = []
        for i in range(len(self.layer_weights)):
            past_kv = kv_caches[i] if kv_caches is not None else None
            conv_st = conv_states[i] if conv_states is not None else None
            if _USE_V2:
                hidden, new_state, new_kv, new_conv = layer_forward_decode(
                    hidden, gdn_states[i], self.layer_weights[i],
                    kv_cache=past_kv, conv_state=conv_st, positions=positions,
                )
            else:
                hidden, new_state, new_kv = layer_forward_decode(
                    hidden, gdn_states[i], self.layer_weights[i], kv_cache=past_kv
                )
                new_conv = conv_st
            new_states.append(new_state)
            new_kv_caches.append(new_kv)
            new_conv_states.append(new_conv)

        logits = self.lm_head(hidden[:, -1, :])  # [B, VOCAB/TP]
        return logits, new_states, new_kv_caches, new_conv_states

    def forward_prefill(
        self,
        input_ids: torch.Tensor,        # [total_tokens]
        cu_seqlens: torch.Tensor,       # [num_seqs+1] int64
        cache_prefix_len: int = 0,      # 缓存前缀长度（0=不缓存）
    ) -> tuple:
        """Prefill forward. 返回 (logits, gdn_states, kv_caches, conv_states)。"""
        device = input_ids.device
        total_tokens = len(input_ids)

        # Build positions tensor: per-sequence position indices
        positions = self._build_positions(cu_seqlens, device)

        # Check prefix cache
        cache_hit = self.prefix_cache.lookup(input_ids, device)
        if cache_hit is not None:
            prefix_len, cached_hidden, cached_gdn_states = cache_hit
            suffix_ids = input_ids[prefix_len:]
            if len(suffix_ids) == 0:
                hidden = cached_hidden
                gdn_states = cached_gdn_states
                kv_caches = [None] * len(self.layer_weights)
                conv_states = _init_conv_states(device)
            else:
                suffix_hidden = self.embed(suffix_ids.unsqueeze(0)).squeeze(0)
                hidden = suffix_hidden
                suffix_cu = torch.tensor([0, len(suffix_ids)], device=device, dtype=torch.int64)
                suffix_positions = torch.arange(prefix_len, prefix_len + len(suffix_ids), device=device)
                gdn_states = []
                kv_caches = []
                conv_states = []
                for i in range(len(self.layer_weights)):
                    initial_state = cached_gdn_states[i] if cached_gdn_states[i] is not None else None
                    if _USE_V2:
                        hidden, final_state, new_kv, new_conv = layer_forward_prefill(
                            hidden, suffix_cu, self.layer_weights[i],
                            initial_gdn_state=initial_state,
                            positions=suffix_positions,
                        )
                    else:
                        hidden, final_state, new_kv = layer_forward_prefill(
                            hidden, suffix_cu, self.layer_weights[i],
                            initial_gdn_state=initial_state,
                        )
                        new_conv = None
                    gdn_states.append(final_state)
                    kv_caches.append(new_kv)
                    conv_states.append(new_conv)
        else:
            hidden = self.embed(input_ids.unsqueeze(0)).squeeze(0)
            gdn_states = []
            kv_caches = []
            conv_states = []
            for i in range(len(self.layer_weights)):
                if _USE_V2:
                    hidden, final_state, new_kv, new_conv = layer_forward_prefill(
                        hidden, cu_seqlens, self.layer_weights[i],
                        positions=positions,
                    )
                else:
                    hidden, final_state, new_kv = layer_forward_prefill(
                        hidden, cu_seqlens, self.layer_weights[i],
                        layer_idx=i,
                    )
                    new_conv = None
                gdn_states.append(final_state)
                kv_caches.append(new_kv)
                conv_states.append(new_conv)

            # Store to cache if requested
            if cache_prefix_len > 0 and cache_prefix_len <= total_tokens:
                self.prefix_cache.store(
                    input_ids, cache_prefix_len, hidden, gdn_states,
                )

        # 取每个序列最后一个 token 的 hidden 做 lm_head
        if cache_hit is not None and len(suffix_ids) > 0:
            last_hidden = hidden[-1:].unsqueeze(0).squeeze(0)
        else:
            last_positions = cu_seqlens[1:] - 1
            last_hidden = hidden[last_positions]
        logits = self.lm_head(last_hidden)

        # Reshape attention KV cache from flat [total, H, D] to per-batch [B, S, H, D].
        kv_caches = self._reshape_kv_caches_for_decode(kv_caches, cu_seqlens)

        return logits, gdn_states, kv_caches, conv_states

    @staticmethod
    def _build_positions(cu_seqlens: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Build per-token position indices from cu_seqlens."""
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        positions = torch.cat([torch.arange(s, device=device) for s in seqlens.tolist()])
        return positions

    @staticmethod
    def _reshape_kv_caches_for_decode(kv_caches, cu_seqlens):
        """Convert flat-prefill KV → batched-decode KV [B, S, H, D]."""
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        B = seqlens.shape[0]
        if B == 0:
            return kv_caches
        S = int(seqlens[0].item())
        if not torch.all(seqlens == S).item():
            raise NotImplementedError(
                "KV cache decode only supports equal-length prefill sequences"
            )
        reshaped = []
        for kv in kv_caches:
            if kv is None:
                reshaped.append(None)
                continue
            k, v = kv
            # Handle both [total, H, D] and [total, 1, H, D] from _attn_forward
            if k.dim() == 4 and k.shape[1] == 1:
                k = k.squeeze(1)
                v = v.squeeze(1)
            H, D = k.shape[-2], k.shape[-1]
            k_b = k.reshape(B, S, H, D).contiguous()
            v_b = v.reshape(B, S, H, D).contiguous()
            reshaped.append((k_b, v_b))
        return reshaped

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy sampling from TP-sharded logits。返回 token ids [B]。"""
        local_max_val, local_max_idx = logits.max(dim=-1)  # [B]
        vocab_per_rank = VOCAB_SIZE // TP_SIZE
        global_idx = local_max_idx + self.rank * vocab_per_rank

        # 跨卡找全局最大值
        all_vals = torch.zeros(
            logits.shape[0], TP_SIZE,
            device=logits.device, dtype=logits.dtype,
        )
        all_idxs = torch.zeros(
            logits.shape[0], TP_SIZE,
            device=logits.device, dtype=torch.float32,
        )
        all_vals[:, self.rank] = local_max_val
        all_idxs[:, self.rank] = global_idx.float()
        from muserve.distributed import all_reduce
        all_reduce(all_vals)
        all_reduce(all_idxs)

        # 选全局最大值对应的 token
        best_rank = all_vals.argmax(dim=-1)  # [B]
        next_tokens = all_idxs[
            torch.arange(logits.shape[0], device=logits.device), best_rank
        ]
        return next_tokens.long()
