"""Qwen3.5-397B 单层 forward — V2 (rebase from sglang qwen3_5.py).

基于 sglang/python/sglang/srt/models/qwen3_5.py 的数学逻辑重写，
使用 mate kernel 作为底层计算。去除所有诊断代码。

数据流严格对齐 sglang:
  GDN layer:  input_layernorm → qkvz_proj → conv1d+silu → GDN_kernel → RMSNormGated(out,z) → out_proj → residual
  Attn layer: input_layernorm → qkv_proj(含gate) → QK_norm → RoPE → Attention → gate*output → o_proj → residual
  MoE:        post_attention_layernorm → gate → topk → batched_expert_gemm → shared_expert → residual

mate kernel API:
  decode: gated_delta_rule_decode(q[B,1,H,K], k[B,1,H,K], v[B,1,HV,V], state[B,HV,V,K],
          A_log[HV], a[B,1,HV], dt_bias[HV], b[B,1,HV]) — kernel 内部算 g/beta
  prefill: chunk_gated_delta_rule(q[S,H,D], k[S,H,D], v[S,HV,D], g[S,HV](float32 alpha),
           beta[S,HV](float32), cu_seqlens, use_qk_l2norm_in_kernel=True)
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
    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, GDN_CONV_KERNEL,
)
from muserve.distributed import all_reduce

_USE_BATCHED_MOE = True

# GDN dimensions (global)
_Q_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM    # 2048
_K_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM    # 2048
_V_DIM = GDN_NUM_V_HEADS * GDN_VALUE_DIM  # 8192
# GDN per rank (col shard / TP_SIZE)
_Q_DIM_LOCAL = _Q_DIM // TP_SIZE    # 256
_K_DIM_LOCAL = _K_DIM // TP_SIZE    # 256
_V_DIM_LOCAL = _V_DIM // TP_SIZE    # 1024
_QKV_DIM_LOCAL = _Q_DIM_LOCAL + _K_DIM_LOCAL + _V_DIM_LOCAL  # 1536

# GDN head dims per rank
_GDN_K_HEADS_LOCAL = GDN_NUM_K_HEADS // TP_SIZE   # 2
_GDN_V_HEADS_LOCAL = GDN_NUM_V_HEADS // TP_SIZE   # 8

# Attention dimensions
_ATTN_Q_HEADS_LOCAL = NUM_Q_HEADS // TP_SIZE      # 4
_ATTN_KV_HEADS = NUM_KV_HEADS                      # 2 (not sharded, too few)
_ATTN_HEAD_DIM = HEAD_DIM                          # 256


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def rms_norm_gemma(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMS norm: weight 存储相对于 1.0 的偏移。"""
    w = (1.0 + weight.float()).to(x.dtype)
    return F.rms_norm(x, (x.shape[-1],), w, eps).to(x.dtype)


def rms_norm_plain(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain RMS norm: weight 直接乘."""
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    return F.rms_norm(x, (x.shape[-1],), w, eps).to(x.dtype)


def _fast_fp8_quantize(x: torch.Tensor, block_size: int = 128) -> tuple:
    """FP8 activation 量化 (per-block amax)。返回 (fp8_tensor, x_scale)。

    x_scale = amax/448 (per-block, [M, K//128]) — 同 v1 约定，直接传 gemm_fp8_nt_groupwise。
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

    x_blocks = x_2d.view(M, K // block_size, block_size).float()
    x_scale = x_blocks.abs().amax(dim=-1).clamp(min=1e-12) / 448.0  # [M, K//128]

    x_scaled = (x_blocks / x_scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn).view(M, K).contiguous()
    return x_fp8, x_scale.contiguous()


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """FP8 GEMM: x [..., K] @ w [N, K] → [..., N].

    w_fp8:  [N, K] fp8_e4m3
    w_scale: [N//128, K//128] float32 (block-wise, amax/448 convention)
    Mirrors v1 fp8_linear: gemm_fp8_nt_groupwise(x_fp8, w, x_scale, w_scale, ...)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N = w_fp8.shape[0]

    x_fp8, x_scale = _fast_fp8_quantize(x_2d)

    out = gemm_mod.gemm_fp8_nt_groupwise(
        x_fp8, w_fp8, x_scale, w_scale,
        scale_granularity_mnk=(1, 128, 128),
        out_dtype=torch.bfloat16,
    )
    return out.reshape(*orig_shape[:-1], N)


def bf16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 GEMM: x [..., K] @ w.T [N, K] → [..., N]."""
    return F.linear(x, w)


def linear_maybe_fp8(x: torch.Tensor, weights: dict, prefix: str) -> torch.Tensor:
    """根据是否有 weight_scale_inv 自动 dispatch fp8 / bf16 linear."""
    w = weights[f"{prefix}.weight"]
    s = weights.get(f"{prefix}.weight_scale_inv")
    return fp8_linear(x, w, s) if s is not None else bf16_linear(x, w)


# ═══════════════════════════════════════════════════════════════════════════════
# GDN (GatedDeltaNet) Forward
# ═══════════════════════════════════════════════════════════════════════════════
# 数据流（对齐 sglang Qwen3_5GatedDeltaNet.forward）：
#   1. in_proj_qkv(hidden) → mixed_qkv [S, q_local+k_local+v_local]
#      in_proj_z(hidden)   → z [S, v_local]，reshape 为 [S, v_heads_local, head_v_dim]
#   2. in_proj_b(hidden)   → b [S, v_heads_local]
#      in_proj_a(hidden)   → a [S, v_heads_local]
#   3. conv1d(mixed_qkv) (causal, kernel=4, depthwise) + silu
#   4. split mixed_qkv 为 (q, k, v)，reshape 为 [S, heads, head_dim]
#   5. GDN kernel:
#      - decode: gated_delta_rule_decode(q[B,1,H,K], k, v, state, A_log, a, dt_bias, b)
#                kernel 内部计算 g/beta
#      - prefill: chunk_gated_delta_rule(q[S,H,D], k, v, g[S,HV] linear-space alpha,
#                 beta[S,HV], cu_seqlens, use_qk_l2norm_in_kernel=True)
#   6. RMSNormGated(out, z): out = rms_norm(out) * silu(z)（norm_before_gate=True）
#   7. AllGather + out_proj (out_proj 不切分 dim=K，每卡都有完整 weight)


def _gdn_in_proj(hidden: torch.Tensor, weights: dict) -> tuple:
    """GDN 入口投影：返回 (mixed_qkv, z, b, a)。

    mixed_qkv: [S, q_local + k_local + v_local]
    z:         [S, v_heads_local, head_v_dim]
    b, a:      [S, v_heads_local]
    """
    # qkv 三段投影（HF checkpoint 中存为合并的 in_proj_qkv）
    mixed_qkv = linear_maybe_fp8(hidden, weights, "gdn.in_proj_qkv")  # [S, _QKV_DIM_LOCAL]

    # z 投影（HF checkpoint 单独存）
    z = linear_maybe_fp8(hidden, weights, "gdn.in_proj_z")  # [S, _V_DIM_LOCAL]
    S = z.shape[0]
    z = z.reshape(S, _GDN_V_HEADS_LOCAL, GDN_VALUE_DIM)

    # b, a 投影（bf16，存为 in_proj_b / in_proj_a，每个 head 一个标量）
    b = bf16_linear(hidden, weights["gdn.in_proj_b.weight"]).contiguous()  # [S, _GDN_V_HEADS_LOCAL]
    a = bf16_linear(hidden, weights["gdn.in_proj_a.weight"]).contiguous()

    return mixed_qkv, z, b, a


def _gdn_conv1d_apply(qkv: torch.Tensor, conv_weight: torch.Tensor,
                      conv_state: torch.Tensor | None = None) -> tuple:
    """Causal depthwise conv1d (kernel=4) + SiLU.

    Args:
      qkv:          [S, C] 当前 token 的 mixed_qkv
      conv_weight:  [C, 1, K=4]
      conv_state:   [C, K-1=3] 之前 3 个 token 的 qkv（decode 路径），或 None（prefill）

    Returns:
      (out [S, C], new_conv_state [C, 3] or None)
    """
    S, C = qkv.shape
    K = conv_weight.shape[-1]
    orig_dtype = qkv.dtype

    # F.conv1d on MUSA does not support bf16 — upcast to float32 for the op,
    # then cast back to the original dtype afterwards.
    qkv_f = qkv.float()
    w_f = conv_weight.float()

    # [1, C, S]
    qkv_t = qkv_f.transpose(0, 1).unsqueeze(0).contiguous()

    if conv_state is not None:
        # decode 路径：prepend conv_state（最近 K-1 个 token）
        state_t = conv_state.float().unsqueeze(0)
        padded = torch.cat([state_t, qkv_t], dim=-1)  # [1, C, K-1+S]
        out = F.conv1d(padded, w_f, groups=C)          # [1, C, S]
        new_conv_state = padded[0, :, -(K - 1):].to(orig_dtype).contiguous()
    else:
        # prefill 路径：causal left-pad K-1 个 0
        padded = F.pad(qkv_t, (K - 1, 0))             # [1, C, S+K-1]
        out = F.conv1d(padded, w_f, groups=C)          # [1, C, S]
        new_conv_state = padded[0, :, -(K - 1):].to(orig_dtype).contiguous() if S >= K - 1 else None

    out = F.silu(out)
    out = out.squeeze(0).transpose(0, 1).to(orig_dtype).contiguous()  # [S, C], back to bf16
    return out, new_conv_state


def _gdn_split_qkv(mixed_qkv: torch.Tensor) -> tuple:
    """Split mixed_qkv [S, q_local+k_local+v_local] → (q, k, v).

    sglang fix_query_key_value_ordering 把 qkvz 按 [k_tp, k_tp, v_tp, v_tp] split,
    然后 q/k 保持 flat (S, k_dim_local), v reshape 成 (S, num_v_heads_local, head_v_dim).
    muserve 这里 qkv 是 [q_local|k_local|v_local]，需要 reshape 成 head 维度。
    """
    q = mixed_qkv[:, :_Q_DIM_LOCAL].contiguous()                              # [S, q_local]
    k = mixed_qkv[:, _Q_DIM_LOCAL:_Q_DIM_LOCAL + _K_DIM_LOCAL].contiguous()   # [S, k_local]
    v = mixed_qkv[:, _Q_DIM_LOCAL + _K_DIM_LOCAL:].contiguous()               # [S, v_local]
    S = mixed_qkv.shape[0]
    q = q.reshape(S, _GDN_K_HEADS_LOCAL, GDN_KEY_DIM)
    k = k.reshape(S, _GDN_K_HEADS_LOCAL, GDN_KEY_DIM)
    v = v.reshape(S, _GDN_V_HEADS_LOCAL, GDN_VALUE_DIM)
    return q, k, v


def _gdn_output_norm_gate(out: torch.Tensor, z: torch.Tensor, norm_weight: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """sglang RMSNormGated (norm_before_gate=True): y = rms_norm(out, norm.weight) * silu(z).

    out: [S, v_heads_local, head_v_dim]
    z:   [S, v_heads_local, head_v_dim]
    norm_weight: [head_v_dim]  per-head RMS norm weight
    """
    # rms_norm 在最后一维（head_v_dim）
    out_normed = rms_norm_plain(out, norm_weight, eps)  # weight 直接用，不是 Gemma-style
    return out_normed * F.silu(z)


def _gdn_allgather_and_out_proj(out_local: torch.Tensor, weights: dict) -> torch.Tensor:
    """AllGather GDN 局部输出（v_local）→ 完整 v_dim，再 out_proj。

    out_local: [..., v_local=1024]，out_proj.weight: [hidden=4096, v_dim=8192] (不切分)
    """
    orig_shape = out_local.shape
    flat = out_local.reshape(-1, _V_DIM_LOCAL)  # [M, 1024]
    parts = [torch.zeros_like(flat) for _ in range(TP_SIZE)]
    dist.all_gather(parts, flat)
    gathered = torch.cat(parts, dim=-1)  # [M, 8192]
    gathered = gathered.reshape(orig_shape[:-1] + (_V_DIM,))
    return linear_maybe_fp8(gathered, weights, "gdn.out_proj")


def gdn_decode_forward(
    hidden: torch.Tensor,  # [B, 1, HIDDEN] or [B, HIDDEN]
    state: torch.Tensor,   # [B, HV_local, V, K] float32
    weights: dict,
    conv_state: torch.Tensor | None = None,  # [C_local, K-1=3] 之前的 qkv
) -> tuple:
    """GDN decode forward. Returns (output [B, 1, HIDDEN], new_state, new_conv_state)."""
    if hidden.dim() == 3:
        B, T, _ = hidden.shape
        assert T == 1, f"decode expects T=1, got T={T}"
        h2d = hidden.reshape(B, HIDDEN_SIZE)
    else:
        B = hidden.shape[0]
        h2d = hidden

    # 1. Input projections
    mixed_qkv, z, b, a = _gdn_in_proj(h2d, weights)  # [B, qkv_local], [B, V_HEADS_LOCAL, head_v_dim], ...

    # 2. Conv1d + silu (with conv_state for decode)
    conv_w = weights["gdn.conv1d.weight"]  # [C_local, 1, K=4]
    mixed_qkv, new_conv_state = _gdn_conv1d_apply(mixed_qkv, conv_w, conv_state)

    # 3. Split q/k/v and reshape into heads
    q, k, v = _gdn_split_qkv(mixed_qkv)  # [B, k_heads, head_k_dim], ..., [B, v_heads, head_v_dim]

    # 4. mate decode kernel: [B, T=1, H, D] shape, kernel 内部算 g/beta
    q4 = q.unsqueeze(1)  # [B, 1, k_heads, head_k_dim]
    k4 = k.unsqueeze(1)
    v4 = v.unsqueeze(1)
    a4 = a.unsqueeze(1).to(q.dtype).contiguous()  # [B, 1, V_HEADS_LOCAL]
    b4 = b.unsqueeze(1).to(q.dtype).contiguous()

    # A_log, dt_bias: loader 存全量 [GDN_NUM_V_HEADS=64]，运行时按 rank 切片
    rank = dist.get_rank() if dist.is_initialized() else 0
    A_log = weights["gdn.A_log"][rank * _GDN_V_HEADS_LOCAL:(rank + 1) * _GDN_V_HEADS_LOCAL]
    dt_bias = weights["gdn.dt_bias"][rank * _GDN_V_HEADS_LOCAL:(rank + 1) * _GDN_V_HEADS_LOCAL]

    # mate decode kernel — supports GQA: HV % H == 0 ✓ (V_HEADS_LOCAL=8, K_HEADS_LOCAL=2, 4x ratio)
    out, new_state = gdn_dec.gated_delta_rule_decode(
        q=q4, k=k4, v=v4,
        state=state,
        A_log=A_log,
        a=a4,
        dt_bias=dt_bias,
        b=b4,
        state_layout="VK",
        use_qk_l2norm=True,
    )
    # out: [B, V_HEADS_LOCAL, head_v_dim] (decode 路径，T=1 被压掉)
    # 但有些代码返回 [B, T=1, V_HEADS_LOCAL, head_v_dim]，做兼容
    if out.dim() == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    # out: [B, V_HEADS_LOCAL, head_v_dim]

    # 5. RMSNormGated: rms_norm(out, norm.weight) * silu(z)
    out_gated = _gdn_output_norm_gate(out, z, weights["gdn.norm.weight"])
    # [B, V_HEADS_LOCAL, head_v_dim] → [B, v_local]
    out_local = out_gated.reshape(B, _V_DIM_LOCAL)

    # 6. AllGather + out_proj
    out_proj = _gdn_allgather_and_out_proj(out_local, weights)  # [B, HIDDEN]
    out_proj = out_proj.reshape(B, 1, HIDDEN_SIZE) if hidden.dim() == 3 else out_proj

    return out_proj, new_state, new_conv_state


def gdn_prefill_forward(
    hidden: torch.Tensor,        # [total, HIDDEN] (varlen flat)
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int32
    weights: dict,
    initial_state: torch.Tensor | None = None,
) -> tuple:
    """GDN prefill forward. Returns (output [total, HIDDEN], final_state, conv_state)."""
    total = hidden.shape[0]

    # 1. Input projections
    mixed_qkv, z, b_raw, a_raw = _gdn_in_proj(hidden, weights)

    # 2. Conv1d + silu (prefill, 无 prior conv_state)
    conv_w = weights["gdn.conv1d.weight"]
    mixed_qkv, new_conv_state = _gdn_conv1d_apply(mixed_qkv, conv_w, conv_state=None)

    # 3. Split q/k/v
    q, k, v = _gdn_split_qkv(mixed_qkv)

    # 4. 计算 g/beta（sglang fused_gdn_gating 公式）：
    #    g_log = -exp(A_log) * softplus(a + dt_bias)  [log-space, ≤ 0]
    #    g = exp(g_log)                                [linear-space alpha (0, 1]]
    #    beta = sigmoid(b)
    # mate chunk_gated_delta_rule 要求 g 是 linear-space alpha (kernel 内部转 log-space)。
    # loader 存全量 [GDN_NUM_V_HEADS=64]，运行时按 rank 切片
    rank = dist.get_rank() if dist.is_initialized() else 0
    A_log = weights["gdn.A_log"][rank * _GDN_V_HEADS_LOCAL:(rank + 1) * _GDN_V_HEADS_LOCAL].float()
    dt_bias = weights["gdn.dt_bias"][rank * _GDN_V_HEADS_LOCAL:(rank + 1) * _GDN_V_HEADS_LOCAL].float()

    a_f = a_raw.float()
    b_f = b_raw.float()
    softplus_a = F.softplus(a_f + dt_bias)
    g_log = -torch.exp(A_log) * softplus_a  # [total, V_HEADS_LOCAL]
    g = torch.exp(g_log).clamp(min=1.1754943508222875e-38).contiguous()
    beta = torch.sigmoid(b_f).contiguous()

    # 5. mate prefill kernel (varlen)
    out, final_state = gdn_pre.chunk_gated_delta_rule(
        q=q, k=k, v=v,
        g=g, beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out: [total, V_HEADS_LOCAL, head_v_dim]

    # 6. RMSNormGated
    out_gated = _gdn_output_norm_gate(out, z, weights["gdn.norm.weight"])
    out_local = out_gated.reshape(total, _V_DIM_LOCAL)

    # 7. AllGather + out_proj
    out_proj = _gdn_allgather_and_out_proj(out_local, weights)  # [total, HIDDEN]
    return out_proj, final_state, new_conv_state


# ═══════════════════════════════════════════════════════════════════════════════
# Attention (GQA + output gate + RoPE) Forward
# ═══════════════════════════════════════════════════════════════════════════════
# sglang Qwen3_5AttentionDecoderLayer 数据流:
#   1. qkv_proj(hidden) → split [q_gate, k, v]
#      q_gate: [S, num_heads*2*head_dim] → reshape [S, num_heads, 2*head_dim] → chunk → (q, gate)
#   2. Q/K per-head GemmaRMSNorm
#   3. RoPE(positions, q, k)
#   4. Attention (RadixAttention / manual GQA)
#   5. output * sigmoid(gate)
#   6. o_proj (RowParallelLinear, reduce_results=False → AllGather 后做)

# RoPE 参数 (Qwen3.5-397B config):
#   rope_theta = 1000000.0
#   max_position_embeddings = 262144
#   head_dim = 256
#   partial_rotary_factor = 1.0 (全部 head_dim 都做 RoPE)
_ROPE_THETA = 1000000.0


def _build_rope_cache(max_seq_len: int, head_dim: int, theta: float = _ROPE_THETA,
                      device: torch.device = None) -> tuple:
    """预计算 RoPE cos/sin cache。返回 (cos_cache, sin_cache) 各 [max_seq_len, head_dim//2]。"""
    dim_half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, device=device).float() / dim_half))
    t = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # [max_seq_len, dim_half]
    return torch.cos(angles), torch.sin(angles)


_ROPE_COS_CACHE = None
_ROPE_SIN_CACHE = None


def _get_rope_cache(seq_len: int, device: torch.device) -> tuple:
    """获取或扩展 RoPE cache。"""
    global _ROPE_COS_CACHE, _ROPE_SIN_CACHE
    if _ROPE_COS_CACHE is None or _ROPE_COS_CACHE.shape[0] < seq_len or _ROPE_COS_CACHE.device != device:
        max_len = max(seq_len, 8192)
        _ROPE_COS_CACHE, _ROPE_SIN_CACHE = _build_rope_cache(max_len, _ATTN_HEAD_DIM, device=device)
    return _ROPE_COS_CACHE[:seq_len], _ROPE_SIN_CACHE[:seq_len]


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x [..., head_dim]. cos/sin: [seq_len, head_dim//2]."""
    dim_half = x.shape[-1] // 2
    x1 = x[..., :dim_half]
    x2 = x[..., dim_half:]
    # Cast cos/sin to x.dtype to avoid float32 promotion on MUSA
    cos = cos.to(x.dtype).unsqueeze(-2)  # [total, 1, dim_half]
    sin = sin.to(x.dtype).unsqueeze(-2)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attn_forward(
    hidden: torch.Tensor,   # [B*T, HIDDEN] or [B, T, HIDDEN]
    weights: dict,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    positions: torch.Tensor | None = None,  # [total] int64, token 在序列中的绝对位置
) -> tuple:
    """Standard GQA attention with output gate + RoPE. Returns (output, updated_kv)."""
    orig_shape = hidden.shape
    if hidden.dim() == 3:
        B, T, H = hidden.shape
        x = hidden.reshape(B * T, H)
    else:
        x = hidden
        B, T = x.shape[0], 1

    total = x.shape[0]

    # 1. Q projection (col-sharded, includes gate)
    # sglang QKVParallelLinear 布局: q_gate 在每个 head 内交错 [head0_q|head0_gate|head1_q|...]
    qg = linear_maybe_fp8(x, weights, "attn.q_proj")  # [total, 2048]
    qg_per_head = qg.reshape(total, _ATTN_Q_HEADS_LOCAL, 2 * _ATTN_HEAD_DIM)
    q_per_head, gate_per_head = torch.chunk(qg_per_head, 2, dim=-1)
    # q_per_head: [total, 4, 256], gate_per_head: [total, 4, 256]

    # K, V projections (not sharded)
    k = linear_maybe_fp8(x, weights, "attn.k_proj")  # [total, 512]
    v = linear_maybe_fp8(x, weights, "attn.v_proj")  # [total, 512]

    # Reshape into heads
    q_heads = q_per_head  # [total, 4, 256]
    k_heads = k.reshape(total, _ATTN_KV_HEADS, _ATTN_HEAD_DIM)  # [total, 2, 256]
    v_heads = v.reshape(total, _ATTN_KV_HEADS, _ATTN_HEAD_DIM)  # [total, 2, 256]

    # 2. Per-head GemmaRMSNorm on Q and K
    q_heads = rms_norm_gemma(q_heads, weights["attn.q_norm.weight"])
    k_heads = rms_norm_gemma(k_heads, weights["attn.k_norm.weight"])

    # 3. RoPE
    if positions is not None:
        max_pos = int(positions.max().item()) + 1
        cos, sin = _get_rope_cache(max_pos, x.device)
        # 按 positions 索引 cos/sin
        cos_pos = cos[positions]  # [total, dim_half]
        sin_pos = sin[positions]
        q_heads = _apply_rotary_emb(q_heads, cos_pos, sin_pos)
        k_heads = _apply_rotary_emb(k_heads, cos_pos, sin_pos)

    scale = _ATTN_HEAD_DIM ** -0.5

    # 4. Attention — 用 past_kv 是否为 None 区分 prefill/decode，
    #    不用 T>1，避免单 token prefill（T=1）走错 decode 路径。
    is_prefill = (past_kv is None)
    if is_prefill:
        # Prefill: causal FMHA，支持 GQA（q:4 heads, k/v:2 heads）
        from mate.jit.attention.fmha import _fmha_fwd
        q_4d = q_heads.unsqueeze(0)   # [1, total, 4, 256]
        k_4d = k_heads.unsqueeze(0)   # [1, total, 2, 256]
        v_4d = v_heads.unsqueeze(0)   # [1, total, 2, 256]
        attn_out, _ = _fmha_fwd(q_4d, k_4d, v_4d, softmax_scale=scale, is_causal=True)
        attn_out = attn_out.squeeze(0)  # [total, 4, 256]
        new_kv = (k_heads, v_heads)
    else:
        # Decode: GQA with KV cache
        k_cur = k_heads.unsqueeze(1)  # [B, 1, 2, 256]
        v_cur = v_heads.unsqueeze(1)
        past_k, past_v = past_kv
        k_full = torch.cat([past_k, k_cur], dim=1)
        v_full = torch.cat([past_v, v_cur], dim=1)
        new_kv = (k_full, v_full)
        S = k_full.shape[1]
        local_gqa_ratio = _ATTN_Q_HEADS_LOCAL // _ATTN_KV_HEADS
        k_exp = k_full.repeat_interleave(local_gqa_ratio, dim=2)  # [B, S, 4, 256]
        v_exp = v_full.repeat_interleave(local_gqa_ratio, dim=2)
        q_4d = q_heads.unsqueeze(2)           # [B, 4, 1, 256]
        k_4d = k_exp.permute(0, 2, 1, 3)     # [B, 4, S, 256]
        scores = torch.matmul(q_4d, k_4d.transpose(-1, -2)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        v_4d = v_exp.permute(0, 2, 1, 3)
        attn_out = torch.matmul(attn_weights, v_4d).squeeze(2)  # [B, 4, 256]

    # 5. Output gate: sigmoid(gate) * attn_output
    gate = torch.sigmoid(gate_per_head)  # [total, 4, 256]
    gated = (attn_out * gate).reshape(total, -1).contiguous()  # [total, 1024]

    # 6. AllGather + o_proj
    parts = [torch.zeros_like(gated) for _ in range(TP_SIZE)]
    dist.all_gather(parts, gated)
    gathered = torch.cat(parts, dim=-1)  # [total, 8192]
    output = linear_maybe_fp8(gathered, weights, "attn.o_proj")  # [total, 4096]

    return output.reshape(orig_shape), new_kv


# ═══════════════════════════════════════════════════════════════════════════════
# MoE (Mixture of Experts) Forward
# ═══════════════════════════════════════════════════════════════════════════════
# sglang Qwen2MoeSparseMoeBlock 数据流:
#   1. gate(hidden) → logits [total, num_experts]
#   2. softmax → topk → renormalize weights
#   3. dispatch tokens to experts (batched GEMM)
#   4. weighted sum of expert outputs
#   5. shared expert (if any) + AllReduce
#
# muserve 优化:
#   - ragged_m_moe_gemm_8bit (mate) 替代 Python for 循环
#   - unsort+reshape+sum 替代 scatter_add_ (2.6x faster)
#   - 每卡只处理 expert_id % TP_SIZE == rank 的 experts

_FLAT_TOKEN_IDX_CACHE = {}
_FUSED_GATE_UP_CACHE = {}


def _get_flat_token_idx(total: int, topk: int, device: torch.device) -> torch.Tensor:
    key = (total, topk, device)
    if key not in _FLAT_TOKEN_IDX_CACHE:
        _FLAT_TOKEN_IDX_CACHE[key] = torch.arange(total, device=device).repeat_interleave(topk)
    return _FLAT_TOKEN_IDX_CACHE[key]


def _get_fused_gate_up(weights: dict) -> tuple:
    """延迟融合 gate_proj + up_proj → gate_up_proj，结果缓存在 weights dict 中。"""
    if "moe.experts.gate_up_proj.weight" in weights:
        return weights["moe.experts.gate_up_proj.weight"], weights["moe.experts.gate_up_proj.weight_scale_inv"]
    w_gate = weights["moe.experts.gate_proj.weight"]
    w_up   = weights["moe.experts.up_proj.weight"]
    s_gate = weights["moe.experts.gate_proj.weight_scale_inv"]
    s_up   = weights["moe.experts.up_proj.weight_scale_inv"]
    # [E, N, K] cat on dim=1 → [E, 2N, K]
    fused_w = torch.cat([w_gate, w_up], dim=1)
    fused_s = torch.cat([s_gate, s_up], dim=1)
    weights["moe.experts.gate_up_proj.weight"] = fused_w
    weights["moe.experts.gate_up_proj.weight_scale_inv"] = fused_s
    del weights["moe.experts.gate_proj.weight"], weights["moe.experts.gate_proj.weight_scale_inv"]
    del weights["moe.experts.up_proj.weight"],   weights["moe.experts.up_proj.weight_scale_inv"]
    return fused_w, fused_s


def moe_forward(hidden: torch.Tensor, weights: dict) -> torch.Tensor:
    """MoE forward. hidden: [total, HIDDEN] → output: [total, HIDDEN]."""
    total = hidden.shape[0]
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 1. Gate: bf16 linear → softmax → topk → renormalize
    gate_w = weights["moe.gate.weight"]  # [NUM_EXPERTS, HIDDEN] bf16
    logits = bf16_linear(hidden, gate_w).float()  # [total, 512]
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_w = topk_w.to(torch.bfloat16)

    # 2. Expand tokens: [total*topk, HIDDEN]
    topk = NUM_EXPERTS_PER_TOK
    flat_ids = topk_ids.reshape(-1)  # [total*topk]
    token_idx = _get_flat_token_idx(total, topk, hidden.device)
    expanded = hidden[token_idx]  # [total*topk, HIDDEN]

    # 3. Filter to local experts (expert_id % TP_SIZE == rank)
    local_mask = (flat_ids % TP_SIZE) == rank
    local_expert_ids = flat_ids[local_mask] // TP_SIZE  # local expert index
    local_tokens = expanded[local_mask]  # [num_local, HIDDEN]
    num_local = local_tokens.shape[0]

    # 4. Sort by expert ID for batched GEMM
    sort_order = local_expert_ids.argsort(stable=True)
    sorted_tokens = local_tokens[sort_order]
    sorted_expert_ids = local_expert_ids[sort_order].to(torch.int32)  # ragged_m_moe_gemm_8bit requires int32
    inverse_order = sort_order.argsort()

    # 5. Batched FP8 GEMM (gate_up fused)
    num_experts_local = NUM_EXPERTS // TP_SIZE  # 64
    alignment = 128
    padded_m = ((num_local + alignment - 1) // alignment) * alignment
    if padded_m > num_local:
        pad_tokens = torch.zeros(padded_m - num_local, HIDDEN_SIZE,
                                 dtype=sorted_tokens.dtype, device=sorted_tokens.device)
        sorted_tokens_padded = torch.cat([sorted_tokens, pad_tokens], dim=0)
        pad_ids = torch.full((padded_m - num_local,), -1,
                             dtype=sorted_expert_ids.dtype, device=sorted_expert_ids.device)
        sorted_expert_ids_padded = torch.cat([sorted_expert_ids, pad_ids], dim=0)
    else:
        sorted_tokens_padded = sorted_tokens
        sorted_expert_ids_padded = sorted_expert_ids

    if _USE_BATCHED_MOE:
        # gate_up fused: [num_experts_local, 2*MOE_INTERMEDIATE, HIDDEN]
        w_gate_up, s_gate_up = _get_fused_gate_up(weights)
        gate_up_out = torch.empty(padded_m, 2 * MOE_INTERMEDIATE,
                                  dtype=torch.bfloat16, device=hidden.device)
        x_fp8, x_scale = _fast_fp8_quantize(sorted_tokens_padded)
        gemm_mod.ragged_m_moe_gemm_8bit(
            (x_fp8, x_scale),
            (w_gate_up, s_gate_up),
            sorted_expert_ids_padded,
            gate_up_out,
        )
        gate_up_out = gate_up_out[:num_local]

        # SiLU activation: silu(gate) * up  (SwiGLU)
        gate_out = gate_up_out[:, :MOE_INTERMEDIATE]
        up_out   = gate_up_out[:, MOE_INTERMEDIATE:]
        activated = F.silu(gate_out) * up_out

        # Down projection
        w_down = weights["moe.experts.down_proj.weight"]
        s_down = weights["moe.experts.down_proj.weight_scale_inv"]
        down_out = torch.empty(padded_m, HIDDEN_SIZE,
                               dtype=torch.bfloat16, device=hidden.device)
        if padded_m > num_local:
            activated_padded = torch.zeros(padded_m, MOE_INTERMEDIATE,
                                           dtype=activated.dtype, device=activated.device)
            activated_padded[:num_local] = activated
        else:
            activated_padded = activated
        act_fp8, act_scale = _fast_fp8_quantize(activated_padded)
        gemm_mod.ragged_m_moe_gemm_8bit(
            (act_fp8, act_scale),
            (w_down, s_down),
            sorted_expert_ids_padded,
            down_out,
        )
        down_out = down_out[:num_local]
    else:
        raise NotImplementedError("Non-batched MoE not implemented in v2")

    # 6. Unsort + weighted sum (replaces scatter_add_, 2.6x faster)
    unsorted = down_out[inverse_order]  # [num_local, HIDDEN]

    # Reconstruct full output: scatter local results back to [total*topk, HIDDEN]
    full_out = torch.zeros(total * topk, HIDDEN_SIZE, dtype=hidden.dtype, device=hidden.device)
    local_indices = torch.where(local_mask)[0]
    full_out[local_indices] = unsorted

    # AllReduce across TP ranks (each rank computed different experts)
    all_reduce(full_out)

    # Weighted sum: reshape [total, topk, HIDDEN] → sum with topk_w
    full_out = full_out.reshape(total, topk, HIDDEN_SIZE)
    topk_w_3d = topk_w.unsqueeze(-1)  # [total, topk, 1]
    output = (full_out * topk_w_3d).sum(dim=1)  # [total, HIDDEN]

    # 7. Shared expert (if configured)
    if "moe.shared_expert.gate_proj.weight" in weights:
        shared_gate = linear_maybe_fp8(hidden, weights, "moe.shared_expert.gate_proj")
        shared_up = linear_maybe_fp8(hidden, weights, "moe.shared_expert.up_proj")
        shared_out = F.silu(shared_gate) * shared_up
        shared_down = linear_maybe_fp8(shared_out, weights, "moe.shared_expert.down_proj")
        if "moe.shared_expert_gate.weight" in weights:
            gate_val = torch.sigmoid(bf16_linear(hidden, weights["moe.shared_expert_gate.weight"]))
            shared_down = shared_down * gate_val
        output = output + shared_down

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# Layer Forward Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def layer_forward_decode(
    hidden: torch.Tensor,       # [B, 1, HIDDEN] or [B, HIDDEN]
    gdn_state: torch.Tensor | None,   # [B, HV_local, V, K] float32
    weights: dict,
    kv_cache: tuple | None = None,
    conv_state: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> tuple:
    """单层 decode forward. Returns (hidden, new_gdn_state, new_kv_cache, new_conv_state)."""
    is_attn_layer = "attn.q_proj.weight" in weights
    squeeze_3d = hidden.dim() == 3

    # 1. Pre-norm + attention/GDN
    h_flat = hidden.reshape(-1, HIDDEN_SIZE) if squeeze_3d else hidden
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm_gemma(h_flat, norm_w)

    new_kv = kv_cache
    new_gdn_state = gdn_state
    new_conv_state = conv_state

    if is_attn_layer:
        attn_out, new_kv = attn_forward(x, weights, past_kv=kv_cache, positions=positions)
        h_flat = h_flat + attn_out.reshape(-1, HIDDEN_SIZE)
    else:
        gdn_out, new_gdn_state, new_conv_state = gdn_decode_forward(
            x, gdn_state, weights, conv_state=conv_state)
        gdn_out_flat = gdn_out.reshape(-1, HIDDEN_SIZE)
        h_flat = h_flat + gdn_out_flat

    # 2. Post-norm + MoE
    norm_w2 = weights["post_attention_layernorm.weight"]
    x2 = rms_norm_gemma(h_flat, norm_w2)
    moe_out = moe_forward(x2, weights)
    h_flat = h_flat + moe_out

    if squeeze_3d:
        h_flat = h_flat.reshape(hidden.shape)
    return h_flat, new_gdn_state, new_kv, new_conv_state


def layer_forward_prefill(
    hidden: torch.Tensor,       # [total, HIDDEN] (varlen flat)
    cu_seqlens: torch.Tensor,   # [num_seqs+1] int32
    weights: dict,
    initial_gdn_state: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> tuple:
    """单层 prefill forward. Returns (hidden, gdn_state, kv_cache, conv_state)."""
    is_attn_layer = "attn.q_proj.weight" in weights

    # 1. Pre-norm + attention/GDN
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm_gemma(hidden, norm_w)

    new_kv = None
    final_state = initial_gdn_state
    new_conv_state = None

    if is_attn_layer:
        x_3d = x.unsqueeze(0)
        attn_out, new_kv = attn_forward(x_3d, weights, positions=positions)
        hidden = hidden + attn_out.squeeze(0)
    else:
        gdn_out, final_state, new_conv_state = gdn_prefill_forward(
            x, cu_seqlens, weights, initial_gdn_state)
        hidden = hidden + gdn_out

    # 2. Post-norm + MoE
    norm_w2 = weights["post_attention_layernorm.weight"]
    x2 = rms_norm_gemma(hidden, norm_w2)
    moe_out = moe_forward(x2, weights)
    hidden = hidden + moe_out

    return hidden, final_state, new_kv, new_conv_state
