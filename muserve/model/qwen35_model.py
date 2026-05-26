"""Qwen3.5-397B 完整模型 forward（60 层）。"""

import torch
import torch_musa

from muserve.config import (
    NUM_LAYERS, HIDDEN_SIZE, VOCAB_SIZE, TP_SIZE,
    GDN_NUM_V_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM,
)
from muserve.distributed import all_gather_into_tensor, get_tp_rank
from muserve.model.qwen35_layer import rms_norm, layer_forward_decode, layer_forward_prefill


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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode step forward。返回 (logits [B, VOCAB/TP], new_gdn_states)。"""
        hidden = self.embed(input_ids)  # [B, 1, HIDDEN]

        new_states = []
        for i in range(len(self.layer_weights)):
            hidden, new_state = layer_forward_decode(
                hidden, gdn_states[i], self.layer_weights[i]
            )
            new_states.append(new_state)

        logits = self.lm_head(hidden[:, -1, :])  # [B, VOCAB/TP]
        return logits, new_states

    def forward_prefill(
        self,
        input_ids: torch.Tensor,        # [total_tokens]
        cu_seqlens: torch.Tensor,       # [num_seqs+1] int64
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Prefill forward。返回 (logits [num_seqs, VOCAB/TP], final_gdn_states)。"""
        hidden = self.embed(input_ids.unsqueeze(0)).squeeze(0)  # [total, HIDDEN]

        gdn_states = []
        for i in range(len(self.layer_weights)):
            hidden, final_state = layer_forward_prefill(
                hidden, cu_seqlens, self.layer_weights[i],
            )
            gdn_states.append(final_state)

        # 取每个序列最后一个 token 的 hidden 做 lm_head
        last_positions = cu_seqlens[1:] - 1  # [num_seqs]
        last_hidden = hidden[last_positions]  # [num_seqs, HIDDEN]
        logits = self.lm_head(last_hidden)    # [num_seqs, VOCAB/TP]
        return logits, gdn_states

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy sampling from TP-sharded logits。返回 token ids [B]。"""
        # 每卡有 VOCAB/TP 的 logits，找本地 argmax 再 AllReduce
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
            device=logits.device, dtype=torch.int64,
        )
        all_vals[:, self.rank] = local_max_val
        all_idxs[:, self.rank] = global_idx
        from muserve.distributed import all_reduce
        all_reduce(all_vals)
        all_reduce(all_idxs.float())

        # 选全局最大值对应的 token
        best_rank = all_vals.argmax(dim=-1)  # [B]
        next_tokens = all_idxs[
            torch.arange(logits.shape[0], device=logits.device), best_rank
        ]
        return next_tokens.long()
