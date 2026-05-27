"""Debug: print exact shapes/strides of tensors passed to GDN decode kernel."""
import torch
import torch_musa
import sys

sys.path.insert(0, "/workspace")
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import load_layer_weights, _load_index
from muserve.config import DEFAULT_MODEL_PATH, GDN_KEY_DIM, GDN_VALUE_DIM
from muserve.model.qwen35_layer import bf16_linear, fp8_linear, _split_qkv

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)
barrier()

weight_index = _load_index(DEFAULT_MODEL_PATH)
weights = load_layer_weights(DEFAULT_MODEL_PATH, 0, weight_index)

B, T = 1, 1
hidden = torch.randn(B, T, 4096, device=device, dtype=torch.bfloat16)

# Reproduce gdn_decode_forward computation step by step
w_a = weights["gdn.in_proj_a.weight"]
w_b = weights["gdn.in_proj_b.weight"]
w_ab = torch.cat([w_a, w_b], dim=0)

ab = bf16_linear(hidden, w_ab)
a_raw = ab[..., :ab.shape[-1]//2]
a = a_raw.clone()
a_sq = a.squeeze(1)
a_contig = a_sq.contiguous()

if rank == 0:
    print(f"w_a: {w_a.shape}")
    print(f"w_b: {w_b.shape}")
    print(f"w_ab: {w_ab.shape}")
    print(f"ab: {ab.shape}, stride={ab.stride()}")
    print(f"a_raw (slice): {a_raw.shape}, stride={a_raw.stride()}")
    print(f"a (clone): {a.shape}, stride={a.stride()}, contig={a.is_contiguous()}")
    print(f"a.squeeze(1): {a_sq.shape}, stride={a_sq.stride()}, contig={a_sq.is_contiguous()}")
    print(f"a.squeeze(1).contiguous(): {a_contig.shape}, stride={a_contig.stride()}")
    print(f"\nExpected kernel stride[0] = v_heads_local = {w_a.shape[0]}")
    print(f"Actual stride[0] = {a_sq.stride()[0]}")
    if a_sq.stride()[0] == w_a.shape[0]:
        print("MATCH - stride is correct, issue is elsewhere")
    else:
        print(f"MISMATCH - kernel expects {w_a.shape[0]}, got {a_sq.stride()[0]}")
