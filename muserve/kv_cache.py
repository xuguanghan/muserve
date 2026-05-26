"""Task 0.3：Paged KV Cache 分配器。

page_size=16 固定，不做抽象。
KV cache 布局：[num_layers, 2, num_pages, page_size, num_kv_heads, head_dim]
  - dim 0: layer index
  - dim 1: 0=key, 1=value
  - dim 2: page index (物理页)
  - dim 3: token slot within page
  - dim 4: kv head
  - dim 5: head dim
"""

import torch
import torch_musa

from muserve.config import (
    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, KV_PAGE_SIZE, KV_DTYPE, TP_SIZE,
)
from muserve.distributed import get_tp_rank


_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class PagedKVCache:
    """GPU paged KV cache，支持动态 page 分配/释放。"""

    def __init__(self, num_gpu_pages: int | None = None):
        rank = get_tp_rank()
        device = torch.device(f"musa:{rank}")
        dtype = _DTYPE_MAP[KV_DTYPE]

        # KV heads 按 TP 切分
        kv_heads_per_rank = max(1, NUM_KV_HEADS // TP_SIZE)

        if num_gpu_pages is None:
            num_gpu_pages = self._estimate_pages(device, kv_heads_per_rank, dtype)

        self.num_pages = num_gpu_pages
        self.page_size = KV_PAGE_SIZE
        self.device = device

        # 物理 KV cache 存储
        # [num_layers, 2, num_pages, page_size, kv_heads_per_rank, head_dim]
        self.cache = torch.zeros(
            NUM_LAYERS, 2, num_gpu_pages, KV_PAGE_SIZE,
            kv_heads_per_rank, HEAD_DIM,
            dtype=dtype, device=device,
        )

        # 空闲页列表（栈结构，O(1) 分配/释放）
        self._free_pages = list(range(num_gpu_pages))
        self._num_free = num_gpu_pages

        if rank == 0:
            cache_gb = self.cache.numel() * self.cache.element_size() / 1e9
            print(f"[kv_cache] Allocated {num_gpu_pages} pages "
                  f"({cache_gb:.1f} GB, {kv_heads_per_rank} kv_heads/rank)")

    def _estimate_pages(self, device, kv_heads_per_rank: int,
                        dtype: torch.dtype) -> int:
        """根据可用 HBM 估算最大 page 数，保留 10% 给激活值。"""
        free_bytes = torch.musa.mem_get_info(device)[0]
        reserved = int(free_bytes * 0.10)
        available = free_bytes - reserved

        # 单页大小：num_layers × 2 × page_size × kv_heads × head_dim × bytes
        elem_size = 2 if dtype == torch.float16 else 2  # fp16/bf16 = 2 bytes
        page_bytes = (NUM_LAYERS * 2 * KV_PAGE_SIZE *
                      kv_heads_per_rank * HEAD_DIM * elem_size)

        num_pages = max(1024, available // page_bytes)
        return int(num_pages)

    # ── 分配 / 释放 ────────────────────────────────────────────────────────────

    def alloc_pages(self, n: int) -> list[int]:
        """分配 n 个物理页，返回 page_idx 列表。OOM 时抛 RuntimeError。"""
        if n > self._num_free:
            raise RuntimeError(
                f"KV cache OOM: requested {n} pages, only {self._num_free} free"
            )
        pages = self._free_pages[-n:]
        del self._free_pages[-n:]
        self._num_free -= n
        return pages

    def free_pages(self, page_ids: list[int]) -> None:
        """释放 page_ids 中的物理页。"""
        self._free_pages.extend(page_ids)
        self._num_free += len(page_ids)

    @property
    def num_free_pages(self) -> int:
        return self._num_free

    def num_free_tokens(self) -> int:
        return self._num_free * self.page_size

    # ── KV 读写 ────────────────────────────────────────────────────────────────

    def write_kv(
        self,
        layer_idx: int,
        page_idx: int,
        slot: int,
        k: torch.Tensor,   # [num_tokens, kv_heads_per_rank, head_dim]
        v: torch.Tensor,
    ) -> None:
        """写入 KV 到指定 page 的 slot 位置。"""
        self.cache[layer_idx, 0, page_idx, slot] = k
        self.cache[layer_idx, 1, page_idx, slot] = v

    def get_kv_by_layer(self, layer_idx: int):
        """返回 (k_cache, v_cache) 用于 flash_attn_with_kvcache。
        shape: [num_pages, page_size, kv_heads_per_rank, head_dim]
        """
        return self.cache[layer_idx, 0], self.cache[layer_idx, 1]


class SequenceKVState:
    """单个序列的 KV cache 状态：page_table + 当前长度。"""

    def __init__(self, max_pages: int = 4096):
        self.page_table: list[int] = []   # 物理页 idx 列表
        self.seq_len: int = 0             # 已填充的 token 数
        self.max_pages = max_pages

    @property
    def num_pages_used(self) -> int:
        return len(self.page_table)

    @property
    def last_page_slot(self) -> int:
        """当前最后一个 token 在其 page 内的 slot 位置。"""
        if self.seq_len == 0:
            return 0
        return (self.seq_len - 1) % KV_PAGE_SIZE

    def needs_new_page(self) -> bool:
        """下一个 token 是否需要分配新页。"""
        if not self.page_table:
            return True
        return self.seq_len % KV_PAGE_SIZE == 0

    def current_page(self) -> int:
        return self.page_table[-1]

    def to_page_table_tensor(
        self, max_pages: int, device: torch.device
    ) -> torch.Tensor:
        """转换为 flash_attn_with_kvcache 需要的 page_table tensor。
        shape: [1, max_pages] int32，未使用的位置填 0。
        """
        t = torch.zeros(1, max_pages, dtype=torch.int32, device=device)
        for i, p in enumerate(self.page_table):
            t[0, i] = p
        return t
