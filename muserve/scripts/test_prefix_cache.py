"""Test Prefix Cache: verify correctness and measure speedup."""
import torch
import torch_musa
import time
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
    print(f"[prefix_cache_test] {NUM_LAYERS} layers, TP=8")

# Warmup (trigger JIT compilation)
warmup_ids = torch.randint(0, 10000, (512,), device=device)
warmup_cu = torch.tensor([0, 512], device=device, dtype=torch.int64)
model.forward_prefill(warmup_ids, warmup_cu)
torch.musa.synchronize()
barrier()
model.prefix_cache.clear()

# Test scenario: prefix=3072, suffix=1024 (total=4096)
PREFIX_LEN = 3072
SUFFIX_LEN = 1024
TOTAL = PREFIX_LEN + SUFFIX_LEN

input_ids = torch.randint(0, 10000, (TOTAL,), device=device)
cu_seqlens = torch.tensor([0, TOTAL], device=device, dtype=torch.int64)

# 1. Full prefill baseline (no cache)
torch.musa.synchronize()
t0 = time.time()
logits_full, _ = model.forward_prefill(input_ids, cu_seqlens)
torch.musa.synchronize()
t_full = (time.time() - t0) * 1000

if rank == 0:
    print(f"\n  Full prefill ({TOTAL} tokens): {t_full:.0f}ms")

# 2. Cache the prefix
model.prefix_cache.clear()
prefix_ids = input_ids[:PREFIX_LEN]
prefix_cu = torch.tensor([0, PREFIX_LEN], device=device, dtype=torch.int64)
model.forward_prefill(prefix_ids, prefix_cu, cache_prefix_len=PREFIX_LEN)
torch.musa.synchronize()

if rank == 0:
    print(f"  Cache stored: {len(model.prefix_cache)} entries, "
          f"{model.prefix_cache.memory_mb():.0f} MB")

# 3. Cache hit: same prefix + new suffix
new_suffix = torch.randint(0, 10000, (SUFFIX_LEN,), device=device)
new_input = torch.cat([prefix_ids, new_suffix])
new_cu = torch.tensor([0, TOTAL], device=device, dtype=torch.int64)

# Warmup cached path
model.forward_prefill(new_input, new_cu)
torch.musa.synchronize()

# Measure
torch.musa.synchronize()
t0 = time.time()
logits_cached, _ = model.forward_prefill(new_input, new_cu)
torch.musa.synchronize()
t_cached = (time.time() - t0) * 1000

if rank == 0:
    print(f"  Cached prefill (prefix={PREFIX_LEN} hit, suffix={SUFFIX_LEN}): {t_cached:.0f}ms")
    print(f"  Speedup: {t_full/t_cached:.1f}x")
    print(f"\n  === 60-layer estimates ===")
    print(f"  Full 4K:    {t_full*60/NUM_LAYERS/1000:.2f}s")
    print(f"  Cached 1K:  {t_cached*60/NUM_LAYERS/1000:.2f}s")
