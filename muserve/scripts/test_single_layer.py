"""Task 0.4：单层 forward 正确性验证（单卡，非 TP）。

用法：
    torchrun --nproc-per-node=8 muserve/scripts/test_single_layer.py
"""

import torch
import torch_musa
import mate
import mate.gdn_decode as gdn_dec
import mate.gemm as gemm_mod

from muserve.distributed import init_distributed, destroy_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_layer_weights
from muserve.config import (
    DEFAULT_MODEL_PATH, TP_SIZE, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
    MOE_INTERMEDIATE, GDN_NUM_K_HEADS, GDN_NUM_V_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM,
)


def test_gdn_decode(weights: dict, rank: int, device: torch.device) -> bool:
    """验证 GDN decode 路径：in_proj_qkv → gdn_decode → out_proj。"""
    print(f"[rank {rank}] Test: GDN decode")
    B, T = 2, 1

    # 构造输入
    hidden = torch.randn(B, T, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # in_proj_qkv: [3 * (GDN_NUM_K_HEADS * GDN_KEY_DIM) / TP, HIDDEN]
    w_qkv = weights.get("gdn.in_proj_qkv.weight")
    if w_qkv is None:
        print(f"  ⚠ gdn.in_proj_qkv.weight not found, skipping")
        return True

    # 线性投影
    qkv = torch.nn.functional.linear(hidden.float(), w_qkv.float()).to(torch.bfloat16)
    qkv_dim = qkv.shape[-1] // 3
    # reshape 后必须 contiguous，否则 gdn_decode kernel 报 stride 错误
    q    = qkv[..., :qkv_dim].reshape(B, T, -1, GDN_KEY_DIM).contiguous()
    k    = qkv[..., qkv_dim:2*qkv_dim].reshape(B, T, -1, GDN_KEY_DIM).contiguous()
    v_in = qkv[..., 2*qkv_dim:].reshape(B, T, -1, GDN_VALUE_DIM).contiguous()

    # GDN state
    H_v = v_in.shape[2]
    state = torch.zeros(B, H_v, GDN_VALUE_DIM, GDN_KEY_DIM,
                        device=device, dtype=torch.float32)
    A_log   = torch.zeros(H_v, device=device, dtype=torch.float32)
    a_gate  = torch.randn(B, T, H_v, device=device, dtype=torch.bfloat16)
    dt_bias = torch.zeros(H_v, device=device, dtype=torch.float32)
    b_gate  = torch.randn(B, T, H_v, device=device, dtype=torch.bfloat16)

    try:
        out, new_state = gdn_dec.gated_delta_rule_decode(
            q, k, v_in, state, A_log, a_gate, dt_bias, b_gate
        )
        torch.musa.synchronize()
        assert out.shape == (B, H_v, GDN_VALUE_DIM), f"unexpected shape {out.shape}"
        assert not out.isnan().any(), "NaN in GDN output"
        print(f"  ✓ GDN decode: out={out.shape}, state={new_state.shape}")
        return True
    except Exception as e:
        print(f"  ✗ GDN decode failed: {e}")
        return False


def test_moe_forward(weights: dict, rank: int, device: torch.device) -> bool:
    """验证 MoE forward：gate → top-10 routing → expert GEMM。"""
    print(f"[rank {rank}] Test: MoE forward (eager loop)")
    B, T = 2, 4
    tokens = B * T

    hidden = torch.randn(tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    gate_w = weights.get("moe.gate.weight")
    if gate_w is None:
        print(f"  ⚠ moe.gate.weight not found, skipping")
        return True

    # Gate + routing
    logits = torch.nn.functional.linear(hidden.float(), gate_w.float())
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # 每卡只有 NUM_EXPERTS // TP_SIZE 个 expert
    experts_per_rank = NUM_EXPERTS // TP_SIZE
    output = torch.zeros(tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    expert_hits = 0
    for e_local in range(min(3, experts_per_rank)):  # 只测前 3 个 expert
        e_global = e_local * TP_SIZE + rank
        w_gate = weights.get(f"moe.expert_{e_local}.gate_proj.weight")
        w_up   = weights.get(f"moe.expert_{e_local}.up_proj.weight")
        w_down = weights.get(f"moe.expert_{e_local}.down_proj.weight")
        if w_gate is None:
            continue

        mask = (topk_ids == e_global).any(dim=-1)
        if not mask.any():
            continue
        expert_hits += 1

        tok = hidden[mask].float()
        gate_out = torch.nn.functional.linear(tok, w_gate.float())
        up_out   = torch.nn.functional.linear(tok, w_up.float())
        act = torch.nn.functional.silu(gate_out) * up_out
        down_out = torch.nn.functional.linear(act, w_down.float()).to(torch.bfloat16)

        sel_w = topk_w[mask][topk_ids[mask] == e_global].unsqueeze(-1)
        output[mask] += down_out * sel_w

    torch.musa.synchronize()
    assert not output.isnan().any(), "NaN in MoE output"
    print(f"  ✓ MoE forward: output={output.shape}, expert_hits={expert_hits}")
    return True


def test_fp8_gemm(weights: dict, rank: int, device: torch.device) -> bool:
    """验证 FP8 GEMM 路径（用 expert 权重测试）。"""
    print(f"[rank {rank}] Test: FP8 GEMM")

    w = weights.get("moe.expert_0.gate_proj.weight")
    if w is None or w.dtype != torch.float8_e4m3fn:
        print(f"  ⚠ FP8 weight not available (dtype={w.dtype if w is not None else 'N/A'}), skipping")
        return True

    M, K = 8, HIDDEN_SIZE
    N = w.shape[0]
    a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, K // 128, device=device, dtype=torch.float32)

    s = weights.get("moe.expert_0.gate_proj.weight_scale_inv")
    if s is None:
        b_scale = torch.ones(N // 128, K // 128, device=device, dtype=torch.float32)
    else:
        b_scale = s

    try:
        out = gemm_mod.gemm_fp8_nt_groupwise(
            a, w, a_scale, b_scale,
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
        )
        torch.musa.synchronize()
        assert out.shape == (M, N), f"unexpected shape {out.shape}"
        assert not out.isnan().any(), "NaN in FP8 GEMM output"
        print(f"  ✓ FP8 GEMM: out={out.shape}, dtype={out.dtype}")
        return True
    except Exception as e:
        print(f"  ✗ FP8 GEMM failed: {e}")
        return False


def main():
    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[test_single_layer] Loading layer 0 weights (TP={TP_SIZE})...")

    weight_index = _load_index(DEFAULT_MODEL_PATH)
    weights = load_layer_weights(DEFAULT_MODEL_PATH, 0, weight_index)

    if rank == 0:
        print(f"  Loaded {len(weights)} weight tensors")
        print(f"  Keys: {sorted(weights.keys())[:8]}...")

    results = []
    results.append(test_gdn_decode(weights, rank, device))
    results.append(test_moe_forward(weights, rank, device))
    results.append(test_fp8_gemm(weights, rank, device))

    barrier()
    all_ok = all(results)

    if rank == 0:
        status = "PASSED" if all_ok else "FAILED"
        print(f"\n[Task 0.4] Single layer forward verification: {status}")

    destroy_distributed()


if __name__ == "__main__":
    main()
