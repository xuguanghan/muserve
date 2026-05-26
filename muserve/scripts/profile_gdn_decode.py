"""Profile GDN decode breakdown: 定位 123ms 中哪个子操作是瓶颈。"""
import argparse
import time
import torch
import torch_musa

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_layer_weights
from muserve.model.qwen35_layer import (
    fp8_linear, bf16_linear, rms_norm, _split_qkv, _gdn_allgather_out_proj,
    gdn_decode_forward,
)
from muserve.config import (
    DEFAULT_MODEL_PATH, HIDDEN_SIZE, GDN_KEY_DIM, GDN_VALUE_DIM,
)
import mate.gdn_decode as gdn_dec


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
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    idx = _load_index(DEFAULT_MODEL_PATH)
    w = load_layer_weights(DEFAULT_MODEL_PATH, 0, idx)
    barrier()

    B = args.batch
    hidden = torch.randn(B, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # 通过 prefill 获取真实 GDN state
    from muserve.model.qwen35_layer import gdn_prefill_forward
    from muserve.loader import load_embedding_weights
    embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, idx)

    prefill_ids = torch.randint(1, 1000, (B * 4,), device=device)
    cu = torch.arange(0, (B + 1) * 4, 4, device=device, dtype=torch.int64)

    # 做一次 prefill 获取 state
    embed = embed_w["embed_tokens.weight"][prefill_ids]
    from muserve.model.qwen35_layer import rms_norm
    norm_w = w["input_layernorm.weight"]
    x = rms_norm(embed, norm_w)
    _, state = gdn_prefill_forward(x, cu, w)
    torch.musa.synchronize()

    # 从 state 推导维度
    v_heads_local = state.shape[1]
    _V_DIM_LOCAL = v_heads_local * GDN_VALUE_DIM

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"GDN DECODE BREAKDOWN: batch={B}")
        print(f"State shape: {state.shape}, v_heads_local={v_heads_local}")
        print(f"{'='*60}")

    # 1. 完整 gdn_decode_forward
    t_full = time_sync(lambda: gdn_decode_forward(hidden, state, w))

    # 2. QKV projection
    w_qkv = w["gdn.in_proj_qkv.weight"]
    s_qkv = w.get("gdn.in_proj_qkv.weight_scale_inv")
    has_scale = s_qkv is not None
    t_qkv = time_sync(lambda: fp8_linear(hidden.view(B, HIDDEN_SIZE), w_qkv, s_qkv) if has_scale else bf16_linear(hidden.view(B, HIDDEN_SIZE), w_qkv))

    # 3. A, B projections
    t_ab = time_sync(lambda: (
        bf16_linear(hidden.view(B, HIDDEN_SIZE), w["gdn.in_proj_a.weight"]),
        bf16_linear(hidden.view(B, HIDDEN_SIZE), w["gdn.in_proj_b.weight"]),
    ))

    # 4. GDN decode kernel itself
    qkv = fp8_linear(hidden.view(B, HIDDEN_SIZE), w_qkv, s_qkv) if has_scale else bf16_linear(hidden.view(B, HIDDEN_SIZE), w_qkv)
    q, k, v = _split_qkv(qkv)
    q = q.reshape(B, 1, -1, GDN_KEY_DIM).contiguous()
    k = k.reshape(B, 1, -1, GDN_KEY_DIM).contiguous()
    v = v.reshape(B, 1, -1, GDN_VALUE_DIM).contiguous()
    a = bf16_linear(hidden.view(B, HIDDEN_SIZE), w["gdn.in_proj_a.weight"])
    b = bf16_linear(hidden.view(B, HIDDEN_SIZE), w["gdn.in_proj_b.weight"])
    A_log = w["gdn.A_log"][rank * v_heads_local:(rank + 1) * v_heads_local]
    dt_bias = w["gdn.dt_bias"][rank * v_heads_local:(rank + 1) * v_heads_local]

    t_kernel = time_sync(lambda: gdn_dec.gated_delta_rule_decode(
        q, k, v, state.clone(), A_log, a, dt_bias, b
    ))

    # 5. AllGather + out_proj
    out, _ = gdn_dec.gated_delta_rule_decode(q, k, v, state.clone(), A_log, a, dt_bias, b)
    out_local = out.reshape(B, 1, _V_DIM_LOCAL)
    t_outproj = time_sync(lambda: _gdn_allgather_out_proj(out_local, w))

    if rank == 0:
        print(f"{'Sub-operation':<40} {'Time (ms)':>10}")
        print("-" * 52)
        print(f"{'Full gdn_decode_forward':<40} {t_full:>10.2f}")
        print(f"{'  QKV projection (fp8_linear)':<40} {t_qkv:>10.2f}")
        print(f"{'  A + B projections (bf16_linear x2)':<40} {t_ab:>10.2f}")
        print(f"{'  gated_delta_rule_decode kernel':<40} {t_kernel:>10.2f}")
        print(f"{'  AllGather + out_proj':<40} {t_outproj:>10.2f}")
        accounted = t_qkv + t_ab + t_kernel + t_outproj
        print(f"{'  Sum of parts':<40} {accounted:>10.2f}")
        print(f"{'  Unaccounted':<40} {t_full - accounted:>10.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
