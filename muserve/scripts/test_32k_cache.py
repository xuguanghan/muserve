"""32K Prefix Cache full scenario verification.

Scenario: 28K prefix (document) + 4K suffix (question) = 32K total.
Tests: first request (cache miss) → store → second request (cache hit).
"""
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

NUM_LAYERS = 5
weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
model = Qwen35Model(embed_w, layer_ws)
barrier()

if rank == 0:
    print(f"[32K_cache] {NUM_LAYERS} layers, TP=8")
    print(f"[32K_cache] Scenario: 28K prefix + 4K suffix = 32K total")

# Warmup
warmup = torch.randint(0, 10000, (1024,), device=device)
model.forward_prefill(warmup, torch.tensor([0, 1024], device=device, dtype=torch.int64))
torch.musa.synchronize()
barrier()
model.prefix_cache.clear()

PREFIX_LEN = 28672  # 28K
SUFFIX_LEN = 4096   # 4K
TOTAL = PREFIX_LEN + SUFFIX_LEN  # 32K

prefix_ids = torch.randint(0, 10000, (PREFIX_LEN,), device=device)

# === Step 1: First request - full 32K prefill (cache miss) ===
if rank == 0:
    print(f"\n--- Step 1: First request (full {TOTAL} tokens, cache miss) ---")

suffix_1 = torch.randint(0, 10000, (SUFFIX_LEN,), device=device)
input_1 = torch.cat([prefix_ids, suffix_1])
cu_1 = torch.tensor([0, TOTAL], device=device, dtype=torch.int64)

torch.musa.synchronize()
t0 = time.time()
logits_1, states_1 = model.forward_prefill(input_1, cu_1, cache_prefix_len=PREFIX_LEN)
torch.musa.synchronize()
t_first = (time.time() - t0) * 1000

if rank == 0:
    print(f"  Time: {t_first:.0f}ms ({NUM_LAYERS}L)")
    print(f"  Est 60L: {t_first*60/NUM_LAYERS/1000:.2f}s")
    print(f"  Cache: {len(model.prefix_cache)} entries, {model.prefix_cache.memory_mb():.0f} MB")

# === Step 2: Second request - same prefix, new suffix (cache hit) ===
if rank == 0:
    print(f"\n--- Step 2: Second request (prefix={PREFIX_LEN} cached, suffix={SUFFIX_LEN} new) ---")

suffix_2 = torch.randint(0, 10000, (SUFFIX_LEN,), device=device)
input_2 = torch.cat([prefix_ids, suffix_2])
cu_2 = torch.tensor([0, TOTAL], device=device, dtype=torch.int64)

# Warmup cached path
model.forward_prefill(input_2, cu_2)
torch.musa.synchronize()

# Measure
torch.musa.synchronize()
t0 = time.time()
logits_2, states_2 = model.forward_prefill(input_2, cu_2)
torch.musa.synchronize()
t_cached = (time.time() - t0) * 1000

if rank == 0:
    print(f"  Time: {t_cached:.0f}ms ({NUM_LAYERS}L)")
    print(f"  Est 60L: {t_cached*60/NUM_LAYERS/1000:.2f}s")
    print(f"  Speedup vs full: {t_first/t_cached:.1f}x")

# === Step 3: Third request - different prefix (cache miss) ===
if rank == 0:
    print(f"\n--- Step 3: Different prefix (cache miss) ---")

new_prefix = torch.randint(0, 10000, (PREFIX_LEN,), device=device)
input_3 = torch.cat([new_prefix, suffix_2])
cu_3 = torch.tensor([0, TOTAL], device=device, dtype=torch.int64)

torch.musa.synchronize()
t0 = time.time()
logits_3, _ = model.forward_prefill(input_3, cu_3)
torch.musa.synchronize()
t_miss = (time.time() - t0) * 1000

if rank == 0:
    print(f"  Time: {t_miss:.0f}ms ({NUM_LAYERS}L)")
    print(f"  Est 60L: {t_miss*60/NUM_LAYERS/1000:.2f}s (no cache benefit)")

# === Summary ===
if rank == 0:
    print(f"\n=== 32K Prefix Cache Summary ===")
    print(f"  First request (32K, cache miss):   est_60L = {t_first*60/NUM_LAYERS/1000:.2f}s")
    print(f"  Cached request (4K suffix only):   est_60L = {t_cached*60/NUM_LAYERS/1000:.2f}s")
    print(f"  Target: < 1.5s")
    cached_60 = t_cached * 60 / NUM_LAYERS / 1000
    if cached_60 < 1.5:
        print(f"  RESULT: PASSED ({cached_60:.2f}s < 1.5s)")
    else:
        print(f"  RESULT: FAILED ({cached_60:.2f}s > 1.5s)")
