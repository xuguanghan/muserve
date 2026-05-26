"""Multi-rank inference loop: all ranks run the same event loop.

Rank 0 receives requests from Flask (via queue), broadcasts to all ranks,
all ranks execute forward pass together, rank 0 returns results.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch_musa

from muserve.distributed import get_tp_rank, broadcast_pyobj, barrier


@dataclass
class InferenceRequest:
    """A single inference request to be processed by all ranks."""
    request_id: str
    input_ids: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.0
    cache_prefix_len: int = 0
    # Result queue for streaming tokens back to Flask thread
    result_queue: queue.Queue = field(default_factory=queue.Queue)


@dataclass
class BroadcastPayload:
    """Serializable payload broadcast from rank 0 to all ranks."""
    action: str  # "prefill", "decode", "shutdown"
    request_id: str = ""
    input_ids: Optional[list[int]] = None
    max_new_tokens: int = 0
    temperature: float = 0.0
    cache_prefix_len: int = 0


class InferenceLoop:
    """Inference event loop running on all TP ranks.

    Rank 0 additionally runs Flask in a separate thread and feeds
    requests into this loop via a thread-safe queue.
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = get_tp_rank()
        self.device = torch.device(f"musa:{self.rank}")
        self.request_queue: queue.Queue = queue.Queue()
        self._running = True
        self._active_requests: dict[str, InferenceRequest] = {}

    def submit_request(self, req: InferenceRequest):
        """Submit a request from Flask thread (rank 0 only)."""
        self._active_requests[req.request_id] = req
        self.request_queue.put(req)

    def run(self):
        """Main inference loop. All ranks must call this simultaneously."""
        barrier()
        while self._running:
            # Rank 0: check for new request from Flask thread
            if self.rank == 0:
                try:
                    req = self.request_queue.get(timeout=0.01)
                    payload = BroadcastPayload(
                        action="prefill",
                        request_id=req.request_id,
                        input_ids=req.input_ids,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        cache_prefix_len=req.cache_prefix_len,
                    )
                except queue.Empty:
                    payload = None
            else:
                payload = None

            # Broadcast: all ranks synchronize here
            payload = broadcast_pyobj(payload, src=0)

            if payload is None:
                continue

            if payload.action == "shutdown":
                self._running = False
                break

            if payload.action == "prefill":
                self._handle_generation(payload)

    def _handle_generation(self, payload: BroadcastPayload):
        """Execute prefill + decode loop. All ranks participate."""
        input_ids = payload.input_ids
        ids_tensor = torch.tensor(input_ids, device=self.device, dtype=torch.long)
        cu_seqlens = torch.tensor([0, len(input_ids)], device=self.device, dtype=torch.int64)

        # Prefill
        logits, gdn_states = self.model.forward_prefill(
            ids_tensor, cu_seqlens, cache_prefix_len=payload.cache_prefix_len,
        )

        # Sample first token
        next_token = self.model.greedy_sample(logits)
        token_id = next_token.item()

        if self.rank == 0:
            self._send_token(payload.request_id, token_id)

        # Decode loop
        for step in range(payload.max_new_tokens - 1):
            if self.tokenizer and token_id == self.tokenizer.eos_token_id:
                break

            decode_ids = next_token.unsqueeze(0).unsqueeze(0)
            logits, gdn_states = self.model.forward_decode(decode_ids, gdn_states)
            next_token = self.model.greedy_sample(logits)
            token_id = next_token.item()

            if self.rank == 0:
                self._send_token(payload.request_id, token_id)

        if self.rank == 0:
            self._send_done(payload.request_id)

    def _send_token(self, request_id: str, token_id: int):
        """Send a generated token back to the Flask thread via result queue."""
        req = self._active_requests.get(request_id)
        if req:
            req.result_queue.put(("token", token_id))

    def _send_done(self, request_id: str):
        """Signal generation complete."""
        req = self._active_requests.get(request_id)
        if req:
            req.result_queue.put(("done", None))
            del self._active_requests[request_id]

    def shutdown(self):
        """Signal the loop to stop."""
        self._running = False
        if self.rank == 0:
            self.request_queue.put(None)
