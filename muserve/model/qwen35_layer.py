"""Qwen3.5-397B 单层 forward。

权重结构（已验证）：
  in_proj_qkv:  [12288, 4096] fp8  → q[2048]+k[2048]+v[8192]，col shard → [1536/rank, 4096]
                per rank: q_dim=256, k_dim=256, v_dim=1024
  in_proj_a:    [64, 4096]    bf16 → [GDN_NUM_V_HEADS, HIDDEN]，col shard → [8/rank, 4096]
  in_proj_b:    [64, 4096]    bf16 → 同 in_proj_a
  in_proj_z:    [8192, 4096]  fp8  → col shard → [1024/rank, 4096]
  out_proj:     [4096, 8192]  fp8  → 不切分（每卡持有完整权重）
                GDN 输出 AllGather 到 [*, 8192] 再做 out_proj
  A_log:        [64]          fp32
  dt_bias:      [64]          fp32
  norm.weight:  [128]         fp32
  conv1d.weight:[4, 64, 1]    bf16
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_musa
import mate.gdn_decode as gdn_dec
import mate.gdn_prefill as gdn_pre
import mate.gemm as gemm_mod

from muserve.config import (
    HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, MOE_INTERMEDIATE,
    GDN_NUM_K_HEADS, GDN_NUM_V_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM, TP_SIZE,
)
from muserve.distributed import all_reduce

# Phase 2: 使用 batched MoE 替代 naive 循环
_USE_BATCHED_MOE = True

# q/k/v 维度（全局，未切分）
_Q_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM   # 16 * 128 = 2048
_K_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM   # 16 * 128 = 2048
_V_DIM = GDN_NUM_V_HEADS * GDN_VALUE_DIM # 64 * 128 = 8192
# per rank（col shard ÷ TP_SIZE）
_Q_DIM_LOCAL = _Q_DIM // TP_SIZE   # 256
_K_DIM_LOCAL = _K_DIM // TP_SIZE   # 256
_V_DIM_LOCAL = _V_DIM // TP_SIZE   # 1024


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)


def fp8_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 linear via gemm_fp8_nt_groupwise。x: [..., K], w: [N, K]"""
    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = w.shape[0]
    x_bf16 = x.reshape(M, K).bfloat16()
    # 动态 per-block scaling：避免激活溢出 fp8e4m3 范围（±448）
    # x_scale shape: [M, K//128]，每个 128-element block 用同一个 scale
    x_blocks = x_bf16.reshape(M, K // 128, 128)
    x_scale = x_blocks.abs().amax(dim=-1).float().clamp(min=1e-12) / 448.0  # [M, K//128] fp32
    x_scaled = (x_bf16 / x_scale.repeat_interleave(128, dim=-1)).to(torch.float8_e4m3fn)
    out = gemm_mod.gemm_fp8_nt_groupwise(
        x_scaled, w, x_scale, w_scale,
        scale_granularity_mnk=(1, 128, 128),
        out_dtype=out_dtype,
    )
    return out.reshape(orig_shape[:-1] + (N,))


def bf16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x.bfloat16(), w.bfloat16())


# ── GDN ──────────────────────────────────────────────────────────────────────

def _split_qkv(qkv: torch.Tensor) -> tuple:
    """从 col-sharded in_proj_qkv 输出中切出 q/k/v（维度不等）。"""
    q = qkv[..., :_Q_DIM_LOCAL]
    k = qkv[..., _Q_DIM_LOCAL:_Q_DIM_LOCAL + _K_DIM_LOCAL]
    v = qkv[..., _Q_DIM_LOCAL + _K_DIM_LOCAL:]
    return q, k, v


def _gdn_allgather_out_proj(out_local: torch.Tensor, weights: dict) -> torch.Tensor:
    """AllGather GDN 局部输出 → 完整 v_dim，再做 out_proj。"""
    # out_local: [..., V_DIM_LOCAL=1024]
    orig_shape = out_local.shape
    flat = out_local.reshape(-1, _V_DIM_LOCAL)          # [M, 1024]
    parts = [torch.zeros_like(flat) for _ in range(TP_SIZE)]
    dist.all_gather(parts, flat)
    gathered = torch.cat(parts, dim=-1)                  # [M, 8192]
    gathered = gathered.reshape(orig_shape[:-1] + (_V_DIM,))

    w_out = weights["gdn.out_proj.weight"]
    s_out = weights.get("gdn.out_proj.weight_scale_inv")
    if s_out is not None:
        return fp8_linear(gathered, w_out, s_out)
    return bf16_linear(gathered, w_out)


_GDN_DECODE_KERNEL_FN = None  # 缓存编译好的 tilelang kernel，避免每次 121ms 的 cache lookup


def _init_gdn_decode_kernel(B, Hq, HV, K, V):
    """首次调用：编译 kernel 并缓存 JITKernel 对象。"""
    global _GDN_DECODE_KERNEL_FN
    import mate.gdn_kernels.tilelang.gdn_decode as _tl_mod
    scale = float(K ** -0.5)
    _GDN_DECODE_KERNEL_FN = _tl_mod._get_decode_fp32_vk_kernel(
        batch=B, qk_head=Hq, head=HV, dim_k=K, dim_v=V,
        input_dtype="bfloat16", gate_batch_dtype="bfloat16",
        scale=scale, use_qk_l2norm=True,
        num_stages=3, threads=128, v_tile=8,
    )


def gdn_decode_forward(
    hidden: torch.Tensor,   # [B, 1, HIDDEN]
    state: torch.Tensor,    # [B, V_heads_local, V_dim, K_dim] fp32
    weights: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _GDN_DECODE_KERNEL_FN
    if hidden.dim() != 3:
        hidden = hidden.reshape(-1, 1, hidden.shape[-1])
    B, T, _ = hidden.shape

    w_qkv = weights["gdn.in_proj_qkv.weight"]
    s_qkv = weights.get("gdn.in_proj_qkv.weight_scale_inv")
    qkv = fp8_linear(hidden, w_qkv, s_qkv) if s_qkv is not None else bf16_linear(hidden, w_qkv)

    q, k, v = _split_qkv(qkv)
    q = q.reshape(B, T, -1, GDN_KEY_DIM).contiguous()
    k = k.reshape(B, T, -1, GDN_KEY_DIM).contiguous()
    v = v.reshape(B, T, -1, GDN_VALUE_DIM).contiguous()

    w_ab = torch.cat([weights["gdn.in_proj_a.weight"], weights["gdn.in_proj_b.weight"]], dim=0)
    ab = bf16_linear(hidden, w_ab)
    a = ab[..., :ab.shape[-1]//2].clone()
    b = ab[..., ab.shape[-1]//2:].clone()
    v_heads_local = v.shape[2]
    rank = weights["gdn.A_log"].device.index or 0
    A_log   = weights["gdn.A_log"][rank * v_heads_local:(rank + 1) * v_heads_local]
    dt_bias = weights["gdn.dt_bias"][rank * v_heads_local:(rank + 1) * v_heads_local]

    if _GDN_DECODE_KERNEL_FN is None:
        Hq = q.shape[2]
        _init_gdn_decode_kernel(B, Hq, v_heads_local, GDN_KEY_DIM, GDN_VALUE_DIM)

    # 直接调用缓存的 JITKernel（0.06ms），跳过 mate API（121ms）
    output = torch.empty(B, v_heads_local, GDN_VALUE_DIM,
                         device=hidden.device, dtype=hidden.dtype)
    _GDN_DECODE_KERNEL_FN(
        q.squeeze(1), k.squeeze(1), v.squeeze(1),
        A_log, a.squeeze(1), dt_bias, b.squeeze(1),
        state, output,
    )

    out_local = output.reshape(B, 1, _V_DIM_LOCAL)
    out_proj = _gdn_allgather_out_proj(out_local, weights)
    return out_proj, state


def gdn_prefill_forward(
    hidden: torch.Tensor,       # [total, HIDDEN]
    cu_seqlens: torch.Tensor,
    weights: dict,
    initial_state=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    total, _ = hidden.shape
    w_qkv = weights["gdn.in_proj_qkv.weight"]
    s_qkv = weights.get("gdn.in_proj_qkv.weight_scale_inv")
    qkv = fp8_linear(hidden, w_qkv, s_qkv) if s_qkv is not None else bf16_linear(hidden, w_qkv)

    q, k, v = _split_qkv(qkv)
    q = q.reshape(total, -1, GDN_KEY_DIM).contiguous()
    k = k.reshape(total, -1, GDN_KEY_DIM).contiguous()
    v = v.reshape(total, -1, GDN_VALUE_DIM).contiguous()

    out, final_state = gdn_pre.chunk_gated_delta_rule(
        q, k, v, cu_seqlens=cu_seqlens,
        initial_state=initial_state, output_final_state=True,
    )
    # out: [total, local_V_heads, V_dim] → [total, V_DIM_LOCAL]
    out_local = out.reshape(total, _V_DIM_LOCAL)
    out_proj = _gdn_allgather_out_proj(out_local, weights)  # [total, HIDDEN]
    return out_proj, final_state


# ── Standard Attention（GQA + output gate，每 4 层一个）───────────────────────
# config: num_q_heads=32(+32 gate), num_kv_heads=2, head_dim=256
# q_proj: [16384, 4096] → col shard → [2048, 4096] (4 Q heads + 4 gate heads per rank)
# k_proj, v_proj: [512, 4096] → 不切分
# o_proj: [4096, 8192] → 不切分

_ATTN_NUM_Q_HEADS = 32
_ATTN_NUM_KV_HEADS = 2
_ATTN_HEAD_DIM = 256
_ATTN_Q_DIM = _ATTN_NUM_Q_HEADS * _ATTN_HEAD_DIM  # 8192
_ATTN_Q_HEADS_LOCAL = _ATTN_NUM_Q_HEADS // TP_SIZE  # 4
_ATTN_GQA_RATIO = _ATTN_NUM_Q_HEADS // _ATTN_NUM_KV_HEADS  # 16


def _attn_forward(
    hidden: torch.Tensor,   # [..., HIDDEN]
    weights: dict,
) -> torch.Tensor:          # [..., HIDDEN]
    """Standard GQA attention with output gate. Works for both prefill and decode."""
    orig_shape = hidden.shape
    if hidden.dim() == 3:
        B, T, H = hidden.shape
        x = hidden.reshape(B * T, H)
    else:
        x = hidden
        B, T = x.shape[0], 1

    total = x.shape[0]

    # Q projection (col-sharded, includes gate): [total, 2048] = 4 Q heads + 4 gate heads
    w_q = weights["attn.q_proj.weight"]
    s_q = weights.get("attn.q_proj.weight_scale_inv")
    qg = fp8_linear(x, w_q, s_q) if s_q is not None else bf16_linear(x, w_q)

    # Split Q and gate (each has local_heads × head_dim per rank)
    local_qg_dim = qg.shape[-1]
    local_q_dim = local_qg_dim // 2
    q_local = qg[..., :local_q_dim]           # [total, 4*256=1024]
    gate_local = qg[..., local_q_dim:]        # [total, 1024]

    # K, V projections (not sharded): [total, 512] = 2 heads × 256
    w_k = weights["attn.k_proj.weight"]
    s_k = weights.get("attn.k_proj.weight_scale_inv")
    k = fp8_linear(x, w_k, s_k) if s_k is not None else bf16_linear(x, w_k)

    w_v = weights["attn.v_proj.weight"]
    s_v = weights.get("attn.v_proj.weight_scale_inv")
    v = fp8_linear(x, w_v, s_v) if s_v is not None else bf16_linear(x, w_v)

    # Reshape into heads
    q_heads = q_local.reshape(total, _ATTN_Q_HEADS_LOCAL, _ATTN_HEAD_DIM)  # [T, 4, 256]
    k_heads = k.reshape(total, _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM)         # [T, 2, 256]
    v_heads = v.reshape(total, _ATTN_NUM_KV_HEADS, _ATTN_HEAD_DIM)         # [T, 2, 256]

    # RMS norm on Q and K (per-head)
    if "attn.q_norm.weight" in weights:
        q_heads = rms_norm(q_heads, weights["attn.q_norm.weight"])
    if "attn.k_norm.weight" in weights:
        k_heads = rms_norm(k_heads, weights["attn.k_norm.weight"])

    # GQA: expand K/V to match local Q heads
    # local_q_heads=4, kv_heads=2, local_gqa_ratio=2
    local_gqa_ratio = _ATTN_Q_HEADS_LOCAL // _ATTN_NUM_KV_HEADS
    k_expanded = k_heads.repeat_interleave(local_gqa_ratio, dim=1)  # [T, 4, 256]
    v_expanded = v_heads.repeat_interleave(local_gqa_ratio, dim=1)  # [T, 4, 256]

    # Scaled dot-product attention (causal, simple implementation)
    # For decode (T=1): no causal mask needed
    # For prefill: need causal mask, but using simple implementation for now
    scale = _ATTN_HEAD_DIM ** -0.5
    # [total, heads, dim] → [total, heads, 1] @ [total, heads, dim]^T won't work for multi-token
    # Use einsum for single-step or simple matmul
    # For eager baseline, use PyTorch's scaled_dot_product_attention
    q_t = q_heads.transpose(0, 1).unsqueeze(0)  # [1, heads, total, dim]
    k_t = k_expanded.transpose(0, 1).unsqueeze(0)
    v_t = v_expanded.transpose(0, 1).unsqueeze(0)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=(T > 1)
    )  # [1, heads, total, dim]
    attn_out = attn_out.squeeze(0).transpose(0, 1)  # [total, heads, dim]
    attn_flat = attn_out.reshape(total, -1)  # [total, local_q_dim=1024]

    # Apply output gate (sigmoid)
    gate = torch.sigmoid(gate_local)
    gated = (attn_flat * gate).contiguous()

    # AllGather: 汇聚所有卡的 attention 输出 → [total, 8192]
    parts = [torch.zeros_like(gated) for _ in range(TP_SIZE)]
    dist.all_gather(parts, gated)
    gathered = torch.cat(parts, dim=-1)  # [total, 8192]

    # O projection (not sharded): [total, 4096]
    w_o = weights["attn.o_proj.weight"]
    s_o = weights.get("attn.o_proj.weight_scale_inv")
    output = fp8_linear(gathered, w_o, s_o) if s_o is not None else bf16_linear(gathered, w_o)

    return output.reshape(orig_shape)


_FLAT_TOKEN_IDX_CACHE = {}


def _get_flat_token_idx(total: int, topk: int, device: torch.device) -> torch.Tensor:
    """缓存 arange.repeat_interleave，避免 Graph 内重复计算。"""
    key = (total, topk, device)
    if key not in _FLAT_TOKEN_IDX_CACHE:
        _FLAT_TOKEN_IDX_CACHE[key] = torch.arange(
            total, device=device).repeat_interleave(topk)
    return _FLAT_TOKEN_IDX_CACHE[key]


_FUSED_GATE_UP_CACHE = {}


def _get_fused_gate_up_weights(weights: dict):
    """拼接 gate_proj + up_proj 权重，存回 weights dict 并释放原始权重。"""
    if "moe.experts._gate_up_proj.weight" in weights:
        return weights["moe.experts._gate_up_proj.weight"], weights["moe.experts._gate_up_proj.weight_scale_inv"]
    w_gate = weights["moe.experts.gate_proj.weight"]
    w_up = weights["moe.experts.up_proj.weight"]
    s_gate = weights["moe.experts.gate_proj.weight_scale_inv"]
    s_up = weights["moe.experts.up_proj.weight_scale_inv"]
    fused_w = torch.cat([w_gate, w_up], dim=1)
    fused_s = torch.cat([s_gate, s_up], dim=1)
    weights["moe.experts._gate_up_proj.weight"] = fused_w
    weights["moe.experts._gate_up_proj.weight_scale_inv"] = fused_s
    del weights["moe.experts.gate_proj.weight"]
    del weights["moe.experts.gate_proj.weight_scale_inv"]
    del weights["moe.experts.up_proj.weight"]
    del weights["moe.experts.up_proj.weight_scale_inv"]
    return fused_w, fused_s


_FUSED_AB_CACHE = {}


def _get_fused_ab_weight(weights: dict) -> torch.Tensor:
    """拼接 in_proj_a + in_proj_b 权重（bf16，极小：16×4096=128KB）。"""
    w_a = weights["gdn.in_proj_a.weight"]
    key = id(w_a)
    if key not in _FUSED_AB_CACHE:
        w_b = weights["gdn.in_proj_b.weight"]
        _FUSED_AB_CACHE[key] = torch.cat([w_a, w_b], dim=0)
    return _FUSED_AB_CACHE[key]


def moe_forward(
    hidden: torch.Tensor,   # [total_tokens, HIDDEN]
    weights: dict,
) -> torch.Tensor:          # [total_tokens, HIDDEN]
    """MoE forward：无隐式 GPU→CPU 同步版本。
    所有 tensor 大小在 Python 层面已知（total × NUM_EXPERTS_PER_TOK），
    不使用 nonzero/any 等数据依赖操作。
    """
    total = hidden.shape[0]
    device = hidden.device
    experts_per_rank = NUM_EXPERTS // TP_SIZE
    rank = _get_rank(device)

    # ── 1. Gate routing（无同步）──
    gate_w = weights["moe.gate.weight"]
    logits = bf16_linear(hidden, gate_w).float()
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    # ── 2. 展开为固定大小 [total*topk, ...] ──
    # 每个 token 有 topk 个 slot，全部展开（不筛选本卡 expert）
    topk = NUM_EXPERTS_PER_TOK
    flat_ids = topk_ids.view(-1)          # [total*topk] 全局 expert id
    flat_weights = topk_w.view(-1)        # [total*topk]
    flat_token_idx = _get_flat_token_idx(total, topk, device)

    # 本卡负责的 expert mask（不触发同步，保持 GPU tensor）
    local_mask = (flat_ids % TP_SIZE == rank)  # [total*topk] bool
    local_expert_ids = flat_ids // TP_SIZE     # [total*topk] local expert id

    # 将非本卡的 expert 标记为 -1（ragged_gemm 会忽略 -1）
    local_expert_ids = torch.where(local_mask, local_expert_ids, torch.full_like(local_expert_ids, -1))
    flat_weights = torch.where(local_mask, flat_weights, torch.zeros_like(flat_weights))

    # 按 expert 排序（固定大小，无同步）
    sort_order = local_expert_ids.argsort(stable=True)
    sorted_expert_ids = local_expert_ids[sort_order]
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Gather hidden states（固定大小 [total*topk, HIDDEN]）
    expanded_hidden = hidden[sorted_token_idx]
    m_indices = sorted_expert_ids.to(torch.int32)

    # ── 3. Expert GEMM ──
    num_expanded = total * topk  # 固定大小，Python 已知

    use_batched = (
        "moe.experts.gate_proj.weight" in weights
        and weights["moe.experts.gate_proj.weight"].dtype == torch.float8_e4m3fn
    )

    if use_batched:
        from mate.deep_gemm import ragged_m_moe_gemm_8bit

        w_gate = weights["moe.experts.gate_proj.weight"]
        s_gate = weights["moe.experts.gate_proj.weight_scale_inv"]
        w_up = weights["moe.experts.up_proj.weight"]
        s_up = weights["moe.experts.up_proj.weight_scale_inv"]
        w_down = weights["moe.experts.down_proj.weight"]
        s_down = weights["moe.experts.down_proj.weight_scale_inv"]

        a_fp8, a_scale = _fast_fp8_quantize(expanded_hidden)

        gate_out = torch.empty(num_expanded, MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
        ragged_m_moe_gemm_8bit(
            (a_fp8, a_scale), (w_gate, s_gate), m_indices, gate_out,
        )

        up_out = torch.empty(num_expanded, MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
        ragged_m_moe_gemm_8bit(
            (a_fp8, a_scale), (w_up, s_up), m_indices, up_out,
        )

        act = F.silu(gate_out) * up_out

        act_fp8, act_scale = _fast_fp8_quantize(act)
        down_out = torch.empty(num_expanded, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        ragged_m_moe_gemm_8bit(
            (act_fp8, act_scale), (w_down, s_down), m_indices, down_out,
        )
    else:
        down_out = torch.zeros(num_expanded, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        for e_local in range(experts_per_rank):
            emask = (m_indices == e_local)
            if not emask.any():
                continue
            tok = expanded_hidden[emask]
            w_g = weights[f"moe.expert_{e_local}.gate_proj.weight"]
            w_u = weights[f"moe.expert_{e_local}.up_proj.weight"]
            w_d = weights[f"moe.expert_{e_local}.down_proj.weight"]
            s_g = weights.get(f"moe.expert_{e_local}.gate_proj.weight_scale_inv")
            s_u = weights.get(f"moe.expert_{e_local}.up_proj.weight_scale_inv")
            s_d = weights.get(f"moe.expert_{e_local}.down_proj.weight_scale_inv")
            g = fp8_linear(tok, w_g, s_g) if s_g is not None else bf16_linear(tok, w_g)
            u = fp8_linear(tok, w_u, s_u) if s_u is not None else bf16_linear(tok, w_u)
            a = F.silu(g) * u
            d = fp8_linear(a, w_d, s_d) if s_d is not None else bf16_linear(a, w_d)
            down_out[emask] = d

    # ── 4. 加权 scatter 回原位（无同步）──
    # m_indices == -1 的行 down_out 可能有垃圾值，用 sorted_weights 置零
    weighted = down_out * sorted_weights.unsqueeze(-1)
    output = torch.zeros(total, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    output.scatter_add_(0, sorted_token_idx.unsqueeze(-1).expand_as(weighted), weighted)

    # AllReduce
    all_reduce(output)

    # ── 5. Shared expert ──
    if "moe.shared_expert.gate_proj.weight" in weights:
        w_g = weights["moe.shared_expert.gate_proj.weight"]
        w_u = weights["moe.shared_expert.up_proj.weight"]
        w_d = weights["moe.shared_expert.down_proj.weight"]
        s_g = weights.get("moe.shared_expert.gate_proj.weight_scale_inv")
        s_u = weights.get("moe.shared_expert.up_proj.weight_scale_inv")
        s_d = weights.get("moe.shared_expert.down_proj.weight_scale_inv")

        gate_out = fp8_linear(hidden, w_g, s_g) if s_g is not None else bf16_linear(hidden, w_g)
        up_out = fp8_linear(hidden, w_u, s_u) if s_u is not None else bf16_linear(hidden, w_u)
        act = F.silu(gate_out) * up_out
        shared_out = fp8_linear(act, w_d, s_d) if s_d is not None else bf16_linear(act, w_d)

        if "moe.shared_expert_gate.weight" in weights:
            gate_val = torch.sigmoid(
                bf16_linear(hidden, weights["moe.shared_expert_gate.weight"])
            )
            shared_out = shared_out * gate_val

        output = output + shared_out

    return output


def _fast_fp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 量化：per-block scale（block=128），满足 ragged_m_moe_gemm_8bit 要求。
    返回 (fp8_data [M, K], scale [M, K//128])。
    """
    BLOCK = 128
    M, K = x.shape
    K_blocks = (K + BLOCK - 1) // BLOCK
    # Pad K if needed
    if K % BLOCK != 0:
        x = F.pad(x, (0, BLOCK * K_blocks - K))
    x_blocks = x.reshape(M, K_blocks, BLOCK)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [M, K_blocks]
    scale = (amax / 448.0).to(torch.float32)  # [M, K_blocks]
    x_scaled = x_blocks / amax.unsqueeze(-1) * 448.0
    x_fp8 = x_scaled.reshape(M, K_blocks * BLOCK).to(torch.float8_e4m3fn)
    if K % BLOCK != 0:
        x_fp8 = x_fp8[:, :K].contiguous()
    return x_fp8, scale


def _quantize_activation_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 bf16 activation 量化为 FP8 e4m3fn + per-block scale。
    block_size = 128（与 mate.deep_gemm 的 scale_granularity 对齐）。
    """
    BLOCK = 128
    M, K = x.shape
    K_blocks = (K + BLOCK - 1) // BLOCK

    # Pad K to multiple of BLOCK
    if K % BLOCK != 0:
        x = F.pad(x, (0, BLOCK - K % BLOCK))

    x_blocks = x.reshape(M, K_blocks, BLOCK)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 448.0).to(torch.float32).squeeze(-1)  # [M, K_blocks]
    x_scaled = x_blocks / amax * 448.0
    x_fp8 = x_scaled.to(torch.float8_e4m3fn).reshape(M, K_blocks * BLOCK)

    # 截断回原始 K（如果有 padding）
    if K % BLOCK != 0:
        x_fp8 = x_fp8[:, :K].contiguous()

    return x_fp8, scale


def _get_rank(device: torch.device) -> int:
    """从 device index 获取 TP rank。"""
    return device.index if device.index is not None else 0


# ── 完整单层 forward ───────────────────────────────────────────────────────────

def layer_forward_decode(
    hidden: torch.Tensor,       # [B, 1, HIDDEN]
    gdn_state: torch.Tensor | None,
    weights: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """单层 decode forward。自动检测 GDN 或 attention 层。"""
    is_attn_layer = "attn.q_proj.weight" in weights

    # 1. Pre-norm + attention/GDN
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm(hidden, norm_w)

    if is_attn_layer:
        attn_out = _attn_forward(x, weights)
        hidden = hidden + attn_out
        new_state = gdn_state  # attention 层不更新 GDN state
    else:
        gdn_out, new_state = gdn_decode_forward(x, gdn_state, weights)
        hidden = hidden + gdn_out

    # 2. Post-norm + MoE
    norm_w2 = weights["post_attention_layernorm.weight"]
    x = rms_norm(hidden, norm_w2)
    if x.dim() == 3:
        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)
        moe_out = moe_forward(x_flat, weights)
        hidden = hidden + moe_out.reshape(B, T, H)
    else:
        moe_out = moe_forward(x, weights)
        hidden = hidden + moe_out

    return hidden, new_state


def layer_forward_prefill(
    hidden: torch.Tensor,       # [total_tokens, HIDDEN]
    cu_seqlens: torch.Tensor,   # [num_seqs+1] int64
    weights: dict,
    initial_gdn_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """单层 prefill forward。自动检测 GDN 或 attention 层。"""
    is_attn_layer = "attn.q_proj.weight" in weights

    # 1. Pre-norm + attention/GDN
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm(hidden, norm_w)

    if is_attn_layer:
        attn_out = _attn_forward(x, weights)
        hidden = hidden + attn_out
        final_state = initial_gdn_state  # attention 层不更新 GDN state
    else:
        gdn_out, final_state = gdn_prefill_forward(x, cu_seqlens, weights, initial_gdn_state)
        hidden = hidden + gdn_out

    # 2. Post-norm + MoE
    norm_w2 = weights["post_attention_layernorm.weight"]
    x = rms_norm(hidden, norm_w2)
    moe_out = moe_forward(x, weights)
    hidden = hidden + moe_out

    return hidden, final_state
