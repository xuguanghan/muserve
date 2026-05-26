"""Phase 3: KV Cache for Attention Layers.

Qwen3.5 有 15 个标准 attention 层（每 4 层一个），需要 KV cache。
GDN 层使用 recurrent state（已在 Phase 1 实现），不需要 KV cache。

KV cache 参数：
  - num_kv_heads = 2
  - head_dim = 256
  - 每层 KV cache: 2 heads × 256 dim × 2 (K+V) = 1024 元素/token
  - 15 层总计: 15360 元素/token × 2 bytes (bf16) = 30 KB/token
  - 32K context: 30 KB × 32768 = ~960 MB/sequence
"""

import torch
from muserve.config import TP_SIZE

_ATTN_NUM_KV_HEADS = 2
_ATTN_HEAD_DIM = 256
_NUM_ATTN_LAYERS = 15
_ATTN_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59]


class KVCache:
    """Per-sequence KV cache for attention layers."""

    def __init__(
        self,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.current_len = 0

        # [num_attn_layers, 2(K+V), max_seq_len, num_kv_heads, head_dim]
        self.cache = torch.zeros(
            _NUM_ATTN_LAYERS, 2, max_seq_len,
            _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM,
            device=device, dtype=dtype,
        )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """追加新的 K/V 到 cache。
        k, v: [num_new_tokens, num_kv_heads, head_dim]
        """
        attn_idx = _ATTN_LAYER_INDICES.index(layer_idx)
        num_new = k.shape[0]
        start = self.current_len
        end = start + num_new
        assert end <= self.max_seq_len, f"KV cache overflow: {end} > {self.max_seq_len}"
        self.cache[attn_idx, 0, start:end] = k
        self.cache[attn_idx, 1, start:end] = v

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取当前 cache 中的 K/V。
        返回: (k, v) 各 [current_len, num_kv_heads, head_dim]
        """
        attn_idx = _ATTN_LAYER_INDICES.index(layer_idx)
        k = self.cache[attn_idx, 0, :self.current_len]
        v = self.cache[attn_idx, 1, :self.current_len]
        return k, v

    def advance(self, num_tokens: int):
        """推进 cache 位置（在所有层都 append 完后调用）。"""
        self.current_len += num_tokens

    def reset(self):
        """重置 cache。"""
        self.current_len = 0

    @property
    def memory_bytes(self) -> int:
        return self.cache.numel() * self.cache.element_size()
