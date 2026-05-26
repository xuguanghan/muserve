"""Phase 3: Continuous Batching Scheduler.

支持动态 batch 调度：
  - 新请求随时加入 running batch
  - 完成的请求随时移出
  - Prefill 和 Decode 分离调度
  - GDN state 和 KV cache 按 request 管理
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque

import torch

from muserve.memory.kv_cache import KVCache


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.0
    state: RequestState = RequestState.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    gdn_states: Optional[list] = None
    kv_cache: Optional[KVCache] = None
    created_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def total_len(self) -> int:
        return len(self.input_ids) + len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.state == RequestState.FINISHED


class ContinuousBatchScheduler:
    """Continuous batching scheduler for Qwen3.5."""

    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 32768,
        device: torch.device = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.waiting_queue: deque[InferenceRequest] = deque()
        self.running_batch: list[InferenceRequest] = []
        self.lock = threading.Lock()

    def add_request(self, request: InferenceRequest):
        with self.lock:
            self.waiting_queue.append(request)

    def schedule_step(self) -> tuple[list[InferenceRequest], list[InferenceRequest]]:
        """调度一步：返回 (prefill_requests, decode_requests)。

        策略：
          - 优先处理 decode（已在 running batch 中的请求）
          - 如果 batch 未满，从 waiting queue 取新请求做 prefill
          - Prefill 和 Decode 不混合（简化实现）
        """
        with self.lock:
            # 移除已完成的请求
            self.running_batch = [r for r in self.running_batch if not r.is_done]

            # 如果有 running requests，优先 decode
            decode_requests = list(self.running_batch)

            # 如果 batch 未满，取新请求做 prefill
            prefill_requests = []
            while (
                self.waiting_queue
                and len(self.running_batch) + len(prefill_requests) < self.max_batch_size
            ):
                req = self.waiting_queue.popleft()
                req.state = RequestState.PREFILLING
                prefill_requests.append(req)

            return prefill_requests, decode_requests

    def finish_prefill(self, requests: list[InferenceRequest]):
        """Prefill 完成后，将请求加入 running batch。"""
        with self.lock:
            for req in requests:
                req.state = RequestState.DECODING
                req.first_token_at = time.time()
                self.running_batch.append(req)

    def finish_request(self, request: InferenceRequest):
        """标记请求完成。"""
        request.state = RequestState.FINISHED
        request.finished_at = time.time()

    @property
    def num_waiting(self) -> int:
        return len(self.waiting_queue)

    @property
    def num_running(self) -> int:
        return len(self.running_batch)

    @property
    def is_idle(self) -> bool:
        return self.num_waiting == 0 and self.num_running == 0
