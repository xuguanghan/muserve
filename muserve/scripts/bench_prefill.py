"""Benchmark prefill (TTFT) performance."""
import torch
import torch_musa
import time
import os
import sys

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

NUM_LAYERS = 10

weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
model = Qwen35Model(embed_w, layer_ws)
barrier()

if rank == 0:
    print(f"[prefill_bench] Model loaded: {NUM_LAYERS} layers, TP=8")
    print(f"[prefill_bench] Measuring prefill TTFT...")

for seq_len in [128, 512, 1024, 4096, 8192]:
    input_ids = torch.randint(0, 10000, (seq_len,), device=device)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int64)

    # Warmup
    model.forward_prefill(input_ids, cu_seqlens)
    torch.musa.synchronize()
    barrier()

    # Measure (3 runs, take median)
    times = []
    for _ in range(3):
        torch.musa.synchronize()
        t0 = time.time()
        model.forward_prefill(input_ids, cu_seqlens)
        torch.musa.synchronize()
        times.append(time.time() - t0)

    t_med = sorted(times)[1]
    t60 = t_med * 60 / NUM_LAYERS

    if rank == 0:
        print(f"  seq={seq_len:>5}: {NUM_LAYERS}L={t_med*1000:.0f}ms, "
              f"est_60L={t60:.2f}s, "
              f"tok/s={seq_len/t60:.0f}")
