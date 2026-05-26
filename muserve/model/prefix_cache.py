"""Prefix Cache: 缓存已处理前缀的 hidden state 和 GDN states。

支持 CPU 内存缓存（offload），大幅提升可缓存前缀数量和并发能力。
缓存命中时 CPU→GPU 传输仅需 ~25ms，vs 重新计算 7.48s。
"""
import hashlib
from collections import OrderedDict
from typing import Optional

import torch


class PrefixCache:
    """LRU prefix cache with CPU memory offload.

    Stores (hidden_state, gdn_states) for processed prefixes in CPU memory.
    On cache hit, transfers back to GPU (~25ms for 32K prefix).

    With 2TB system RAM (8×S5000 server), can cache ~3000 prefixes of 32K tokens.

    Args:
        max_entries: Maximum number of cached prefixes.
        storage: 'cpu' for CPU RAM (high capacity), 'gpu' for GPU HBM (low latency).
    """

    def __init__(self, max_entries: int = 1024, storage: str = "cpu"):
        self.max_entries = max_entries
        self.storage = storage
        self._cache: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _hash_tokens(token_ids: torch.Tensor) -> str:
        data = token_ids.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _to_storage(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.storage == "cpu":
            return tensor.cpu()
        return tensor.clone()

    def _to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.device != device:
            return tensor.to(device, non_blocking=True)
        return tensor

    def lookup(
        self, input_ids: torch.Tensor, device: torch.device
    ) -> Optional[tuple[int, torch.Tensor, list]]:
        """Find longest cached prefix match.

        Returns (prefix_len, hidden_on_gpu, gdn_states_on_gpu) or None.
        Transfer from CPU→GPU is non_blocking (~25ms for 32K prefix).
        """
        best_key = None
        best_len = 0

        for key, entry in self._cache.items():
            prefix_len = entry["prefix_len"]
            if prefix_len > len(input_ids):
                continue
            if prefix_len <= best_len:
                continue
            candidate_hash = self._hash_tokens(input_ids[:prefix_len])
            if candidate_hash == key:
                best_key = key
                best_len = prefix_len

        if best_key is None:
            return None

        self._cache.move_to_end(best_key)
        entry = self._cache[best_key]

        hidden = self._to_device(entry["hidden"], device)
        gdn_states = [
            self._to_device(s, device) if s is not None else None
            for s in entry["gdn_states"]
        ]
        return best_len, hidden, gdn_states

    def store(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        hidden: torch.Tensor,
        gdn_states: list,
    ):
        """Cache state after processing prefix_len tokens (offload to CPU)."""
        key = self._hash_tokens(input_ids[:prefix_len])

        if key in self._cache:
            self._cache.move_to_end(key)
            return

        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "prefix_len": prefix_len,
            "hidden": self._to_storage(hidden),
            "gdn_states": [
                self._to_storage(s) if s is not None else None
                for s in gdn_states
            ],
        }

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def memory_mb(self) -> float:
        total = 0
        for entry in self._cache.values():
            total += entry["hidden"].nelement() * entry["hidden"].element_size()
            for s in entry["gdn_states"]:
                if s is not None:
                    total += s.nelement() * s.element_size()
        return total / 1024 / 1024
