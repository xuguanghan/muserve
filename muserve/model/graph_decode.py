"""MUSA Graph captured decode step — 消除 CPU dispatch overhead。

用法：
    graphed = GraphedDecodeStep(model, batch_size=8, device=device)
    graphed.capture(sample_ids, gdn_states)  # 首次 capture
    logits = graphed.run(input_ids, gdn_states)  # 后续 replay（返回 logits）
    # 或用 step() 实现全自动 decode（含 sampling + input 更新）
    graphed.step()
"""

import torch
import torch_musa

from muserve.model.qwen35_layer import layer_forward_decode
from muserve.distributed import all_reduce


class GraphedDecodeStep:
    """将完整 decode step（60 层 + sampling）capture 到单个 MUSA Graph。"""

    def __init__(self, model, batch_size: int, device: torch.device):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.graph = None
        self._input_ids_buf = None
        self._logits_buf = None
        self._next_tokens_buf = None
        self._captured = False

    def capture(self, sample_ids: torch.Tensor, gdn_states: list[torch.Tensor]):
        """Capture full decode step (layers + sampling) into MUSA Graph.

        Args:
            sample_ids: [B, 1] sample input for capture
            gdn_states: list of GDN state tensors (updated in-place on replay)
        """
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        B = self.batch_size
        assert sample_ids.shape == (B, 1)

        self._input_ids_buf = sample_ids.clone()

        # Warmup on default stream (MCCL requires same stream across ranks)
        for i in range(3):
            if rank == 0:
                print(f"[graph_capture] warmup {i+1}/3...", flush=True)
            self._run_full_step(self._input_ids_buf, gdn_states)
        torch.musa.synchronize()
        if rank == 0:
            print(f"[graph_capture] warmup done, starting capture...", flush=True)

        # Capture
        pool = torch.musa.graph_pool_handle()
        self.graph = torch.musa.MUSAGraph()
        with torch.musa.graph(self.graph, pool=pool,
                              capture_error_mode="relaxed"):
            self._run_full_step(self._input_ids_buf, gdn_states)
        torch.musa.synchronize()
        self._captured = True
        if rank == 0:
            print(f"[graph_capture] capture complete.", flush=True)

    def step(self):
        """Replay one decode step. input_ids_buf is auto-updated with next tokens."""
        assert self._captured, "call capture() first"
        self.graph.replay()

    def get_next_tokens(self) -> torch.Tensor:
        """Get the last generated token IDs after step()."""
        return self._next_tokens_buf

    def run(self, input_ids: torch.Tensor,
            gdn_states: list[torch.Tensor]) -> torch.Tensor:
        """Replay with explicit input_ids (for compatibility). Returns logits."""
        assert self._captured, "call capture() first"
        self._input_ids_buf.copy_(input_ids)
        self.graph.replay()
        return self._logits_buf

    def _run_full_step(self, input_ids: torch.Tensor,
                       gdn_states: list[torch.Tensor]):
        """Execute decode + sampling + input update (for warmup and capture)."""
        hidden = self.model.embed(input_ids)  # [B, 1, HIDDEN]

        for i in range(len(self.model.layer_weights)):
            hidden, _ = layer_forward_decode(
                hidden, gdn_states[i], self.model.layer_weights[i]
            )

        logits = self.model.lm_head(hidden[:, -1, :])  # [B, VOCAB/TP]
        self._logits_buf = logits

        # Greedy sample (included in graph to avoid MCCL sequence issues)
        next_tokens = self.model.greedy_sample(logits)
        self._next_tokens_buf = next_tokens

        # Update input_ids_buf for next step (self-feeding loop)
        input_ids[:, 0] = next_tokens
