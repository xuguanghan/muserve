"""Profile prefill path: measure time breakdown per component."""
import torch
import torch_musa
import time
import sys

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.qwen35_layer import (
    layer_forward_prefill, rms_norm, moe_forward,
    gdn_prefill_forward, fp8_linear, bf16_linear,
)
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

NUM_LAYERS = 10
SEQ_LEN = 4096

weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
model = Qwen35Model(embed_w, layer_ws)
barrier()

if rank == 0:
    print(f"[profile] {NUM_LAYERS} layers, TP=8, seq_len={SEQ_LEN}")

input_ids = torch.randint(0, 10000, (SEQ_LEN,), device=device)
cu_seqlens = torch.tensor([0, SEQ_LEN], device=device, dtype=torch.int64)

# Warmup
model.forward_prefill(input_ids, cu_seqlens)
torch.musa.synchronize()
barrier()

# Profile individual components within a single layer
if rank == 0:
    print(f"\n=== Per-component profiling (seq={SEQ_LEN}) ===")

hidden = model.embed(input_ids.unsqueeze(0)).squeeze(0)  # [total, HIDDEN]
torch.musa.synchronize()

# Measure per-layer time by layer type
for layer_idx in [0, 1, 4]:
    weights = layer_ws[layer_idx]
    is_attn = "attn.q_proj.weight" in weights
    ltype = "attention" if is_attn else "GDN"

    # Full layer
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(3):
        layer_forward_prefill(hidden, cu_seqlens, weights)
    torch.musa.synchronize()
    t_layer = (time.time() - t0) / 3 * 1000

    # MoE only
    norm_w2 = weights["post_attention_layernorm.weight"]
    x2 = rms_norm(hidden, norm_w2)
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(3):
        moe_forward(x2, weights)
    torch.musa.synchronize()
    t_moe = (time.time() - t0) / 3 * 1000

    # RMSNorm only
    norm_w = weights["input_layernorm.weight"]
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(10):
        rms_norm(hidden, norm_w)
    torch.musa.synchronize()
    t_norm = (time.time() - t0) / 10 * 1000

    t_attn_gdn = t_layer - t_moe - 2 * t_norm

    if rank == 0:
        print(f"\n  Layer {layer_idx} ({ltype}): total={t_layer:.1f}ms")
        print(f"    {ltype}:    {t_attn_gdn:.1f}ms ({t_attn_gdn/t_layer*100:.0f}%)")
        print(f"    MoE:       {t_moe:.1f}ms ({t_moe/t_layer*100:.0f}%)")
        print(f"    2x norm:   {2*t_norm:.1f}ms ({2*t_norm/t_layer*100:.0f}%)")

# Embed overhead
torch.musa.synchronize()
t0 = time.time()
for _ in range(10):
    model.embed(input_ids.unsqueeze(0)).squeeze(0)
torch.musa.synchronize()
t_embed = (time.time() - t0) / 10 * 1000

if rank == 0:
    print(f"\n  embed: {t_embed:.1f}ms")
    print(f"\n=== 60-layer TTFT estimate (seq={SEQ_LEN}) ===")
    # ~56 GDN layers + 4 attention layers
    est = 56 * t_layer + 4 * t_layer + t_embed
    print(f"  {est:.0f}ms = {est/1000:.2f}s")

