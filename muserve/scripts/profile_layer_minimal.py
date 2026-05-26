"""Minimal test: measure layer decode time with/without GDN kernel."""
import time, os, sys
import torch
import torch_musa

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29566")

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_embedding_weights, load_layer_weights
from muserve.model.qwen35_layer import (
    layer_forward_decode, gdn_prefill_forward, rms_norm,
    gdn_decode_forward, moe_forward,
)
from muserve.config import DEFAULT_MODEL_PATH, HIDDEN_SIZE

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")

idx = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, idx)
w = load_layer_weights(DEFAULT_MODEL_PATH, 0, idx)
barrier()

B = 8
# Get real GDN state via prefill
prefill_ids = torch.randint(1, 1000, (B * 4,), device=device)
cu = torch.arange(0, (B + 1) * 4, 4, device=device, dtype=torch.int64)
embed = embed_w["embed_tokens.weight"][prefill_ids]
norm_w = w["input_layernorm.weight"]
x = rms_norm(embed, norm_w)
_, gdn_state = gdn_prefill_forward(x, cu, w)
torch.musa.synchronize()

hidden = torch.randn(B, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

# Warmup full layer
for _ in range(3):
    layer_forward_decode(hidden, gdn_state, w)
torch.musa.synchronize()

# Measure full layer (with GDN kernel)
N = 10
torch.musa.synchronize()
t0 = time.time()
for _ in range(N):
    layer_forward_decode(hidden, gdn_state, w)
torch.musa.synchronize()
t_full = (time.time() - t0) / N * 1000

# Measure MoE only
x_moe = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
for _ in range(3):
    moe_forward(x_moe, w)
torch.musa.synchronize()
torch.musa.synchronize()
t0 = time.time()
for _ in range(N):
    moe_forward(x_moe, w)
torch.musa.synchronize()
t_moe = (time.time() - t0) / N * 1000

# Measure GDN decode only
for _ in range(3):
    gdn_decode_forward(hidden, gdn_state, w)
torch.musa.synchronize()
torch.musa.synchronize()
t0 = time.time()
for _ in range(N):
    gdn_decode_forward(hidden, gdn_state, w)
torch.musa.synchronize()
t_gdn = (time.time() - t0) / N * 1000

if rank == 0:
    print(f"\n{'='*50}")
    print(f"LAYER DECODE TIMING (B={B}, repeat={N})")
    print(f"{'='*50}")
    print(f"Full layer_forward_decode: {t_full:.2f} ms")
    print(f"MoE only:                  {t_moe:.2f} ms")
    print(f"GDN decode only:           {t_gdn:.2f} ms")
    print(f"Sum (MoE+GDN):             {t_moe+t_gdn:.2f} ms")
    print(f"Unaccounted:               {t_full-t_moe-t_gdn:.2f} ms")
    print(f"{'='*50}")
