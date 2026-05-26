"""Phase 4: Production-ready inference engine.

集成所有组件：
  - Model forward (Phase 1)
  - Batched MoE (Phase 2)
  - KV Cache + Continuous Batching (Phase 3)
  - Monitoring + Health checks (Phase 4)
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class InferenceMetrics:
    """推理性能指标收集器。"""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_prefill_tokens: int = 0
    total_prefill_time_ms: float = 0.0
    total_decode_time_ms: float = 0.0
    active_requests: int = 0
    peak_batch_size: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_prefill(self, num_tokens: int, time_ms: float):
        with self._lock:
            self.total_prefill_tokens += num_tokens
            self.total_prefill_time_ms += time_ms

    def record_decode(self, num_tokens: int, time_ms: float):
        with self._lock:
            self.total_tokens_generated += num_tokens
            self.total_decode_time_ms += time_ms

    def record_request_start(self):
        with self._lock:
            self.total_requests += 1
            self.active_requests += 1
            self.peak_batch_size = max(self.peak_batch_size, self.active_requests)

    def record_request_end(self):
        with self._lock:
            self.active_requests -= 1

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_prefill_ms_per_token(self) -> float:
        if self.total_prefill_tokens == 0:
            return 0.0
        return self.total_prefill_time_ms / self.total_prefill_tokens

    @property
    def avg_decode_ms_per_token(self) -> float:
        if self.total_tokens_generated == 0:
            return 0.0
        return self.total_decode_time_ms / self.total_tokens_generated

    @property
    def decode_throughput(self) -> float:
        if self.total_decode_time_ms == 0:
            return 0.0
        return self.total_tokens_generated / (self.total_decode_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "peak_batch_size": self.peak_batch_size,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prefill_tokens": self.total_prefill_tokens,
            "avg_prefill_ms_per_token": round(self.avg_prefill_ms_per_token, 1),
            "avg_decode_ms_per_token": round(self.avg_decode_ms_per_token, 1),
            "decode_throughput_tok_s": round(self.decode_throughput, 2),
        }


# Global metrics instance
metrics = InferenceMetrics()
