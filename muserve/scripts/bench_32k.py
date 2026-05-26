"""Benchmark prefill for large sequences (up to 32K)."""
import torch
import torch_musa
import time
import sys

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.qwen35_layer import layer_forward_prefill, rms_norm, moe_forward
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

NUM_LAYERS = 5
weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
model = Qwen35Model(embed_w, layer_ws)
barrier()

if rank == 0:
    print(f"[32K_bench] {NUM_LAYERS} layers, TP=8")
    print(f"[32K_bench] Testing large sequence prefill...")

for seq_len in [4096, 8192, 16384, 32768]:
    try:
        input_ids = torch.randint(0, 10000, (seq_len,), device=device)
        cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int64)

        model.forward_prefill(input_ids, cu_seqlens)
        torch.musa.synchronize()
        barrier()

        times = []
        for _ in range(3):
            torch.musa.synchronize()
            t0 = time.time()
            model.forward_prefill(input_ids, cu_seqlens)
            torch.musa.synchronize()
            times.append(time.time() - t0)

        t = sorted(times)[1]
        t60 = t * 60 / NUM_LAYERS
        per_layer = t / NUM_LAYERS * 1000
        if rank == 0:
            print(f"  seq={seq_len:>5}: {NUM_LAYERS}L={t*1000:.0f}ms, "
                  f"per_layer={per_layer:.1f}ms, "
                  f"est_60L={t60:.2f}s, "
                  f"tok/s={seq_len/t60:.0f}")
    except torch.musa.OutOfMemoryError:
        if rank == 0:
            print(f"  seq={seq_len:>5}: OOM")
        torch.musa.empty_cache()
        barrier()
    except Exception as e:
        if rank == 0:
            print(f"  seq={seq_len:>5}: ERROR - {str(e)[:60]}")
        barrier()

# Profile 32K per-component if it fits in memory
if rank == 0:
    print(f"\n[32K_bench] Per-component profiling (seq=32768)...")

try:
    seq_len = 32768
    input_ids = torch.randint(0, 10000, (seq_len,), device=device)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int64)
    hidden = model.embed(input_ids.unsqueeze(0)).squeeze(0)
    torch.musa.synchronize()

    weights = layer_ws[0]
    norm_w = weights["input_layernorm.weight"]
    norm_w2 = weights["post_attention_layernorm.weight"]

    # Full layer
    torch.musa.synchronize()
    t0 = time.time()
    layer_forward_prefill(hidden, cu_seqlens, weights)
    torch.musa.synchronize()
    t_layer = (time.time() - t0) * 1000

    # MoE only
    x2 = rms_norm(hidden, norm_w2)
    torch.musa.synchronize()
    t0 = time.time()
    moe_forward(x2, weights)
    torch.musa.synchronize()
    t_moe = (time.time() - t0) * 1000

    t_gdn = t_layer - t_moe
    if rank == 0:
        print(f"  Full layer:  {t_layer:.0f}ms")
        print(f"  MoE:         {t_moe:.0f}ms ({t_moe/t_layer*100:.0f}%)")
        print(f"  GDN+norms:   {t_gdn:.0f}ms ({t_gdn/t_layer*100:.0f}%)")
except torch.musa.OutOfMemoryError:
    if rank == 0:
        print(f"  32K profiling: OOM")
    torch.musa.empty_cache()
except Exception as e:
    if rank == 0:
        print(f"  32K profiling: ERROR - {str(e)[:60]}")
