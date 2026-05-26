"""Profile full decode step breakdown: GDN vs Attention vs MoE."""
import argparse
import time
import torch
import torch_musa

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_embedding_weights, load_layer_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.qwen35_layer import (
    rms_norm, moe_forward, gdn_decode_forward, _attn_forward,
)
from muserve.config import DEFAULT_MODEL_PATH, HIDDEN_SIZE, TP_SIZE


def time_sync(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.musa.synchronize()
    return (time.time() - t0) / repeat * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    idx = _load_index(DEFAULT_MODEL_PATH)
    embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, idx)
    layer_ws = [load_layer_weights(DEFAULT_MODEL_PATH, i, idx) for i in range(args.layers)]
    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    B = args.batch
    seq_len = 4
    all_ids = torch.randint(1, 1000, (B * seq_len,), device=device)
    cu = torch.arange(0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int64)

    # Prefill to get gdn_states
    logits, gdn_states = model.forward_prefill(all_ids, cu)
    torch.musa.synchronize()

    # 取第 0 层的 weights 和 gdn_state 做分解测试
    w = layer_ws[0]
    hidden = torch.randn(B, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    gdn_st = gdn_states[0] if gdn_states else None

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"LAYER BREAKDOWN: batch={B}, layer 0")
        print(f"{'='*60}")

    # 1. RMSNorm
    norm_w = w["input_layernorm.weight"]
    t_norm = time_sync(lambda: rms_norm(hidden.squeeze(1), norm_w))

    # 2. GDN decode (layer 0 is GDN layer)
    x = rms_norm(hidden.squeeze(1), norm_w)
    is_attn = "attn.q_proj.weight" in w
    if not is_attn and gdn_st is not None:
        t_gdn = time_sync(lambda: gdn_decode_forward(x.unsqueeze(1), gdn_st, w))
    else:
        t_gdn = -1

    # 3. Attention (use an attn layer if available)
    attn_layer_idx = -1
    for i in range(args.layers):
        if "attn.q_proj.weight" in layer_ws[i]:
            attn_layer_idx = i
            break
    if attn_layer_idx >= 0:
        aw = layer_ws[attn_layer_idx]
        norm_aw = aw["input_layernorm.weight"]
        ax = rms_norm(hidden.squeeze(1), norm_aw)
        t_attn = time_sync(lambda: _attn_forward(ax, aw))
    else:
        t_attn = -1

    # 4. MoE
    norm_w2 = w["post_attention_layernorm.weight"]
    x_moe = rms_norm(hidden.squeeze(1), norm_w2)
    t_moe = time_sync(lambda: moe_forward(x_moe, w))

    # 5. Full layer forward decode
    from muserve.model.qwen35_layer import layer_forward_decode
    t_full = time_sync(lambda: layer_forward_decode(hidden, gdn_st, w))

    if rank == 0:
        print(f"{'Component':<35} {'Time (ms)':>10}")
        print("-" * 47)
        print(f"{'Full layer_forward_decode':<35} {t_full:>10.2f}")
        print(f"{'  RMSNorm':<35} {t_norm:>10.2f}")
        if t_gdn >= 0:
            print(f"{'  GDN decode':<35} {t_gdn:>10.2f}")
        if t_attn >= 0:
            print(f"{'  Attention (sdpa)':<35} {t_attn:>10.2f}")
        print(f"{'  MoE forward':<35} {t_moe:>10.2f}")
        accounted = t_norm * 2 + t_moe + (t_gdn if t_gdn >= 0 else t_attn if t_attn >= 0 else 0)
        print(f"{'  Sum of parts':<35} {accounted:>10.2f}")
        print(f"{'  Unaccounted (Python overhead)':<35} {t_full - accounted:>10.2f}")
        print()
        print(f"60-layer estimated: {t_full*60:.0f} ms")
        print(f"Throughput (B={B}): {B/(t_full*60/1000):.2f} tok/s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

