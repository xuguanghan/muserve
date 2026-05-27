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
    # Qwen3.5 uses Gemma-style RMS norm: weight stores offset from 1.0
    w = (1.0 + weight.float()).to(x.dtype)
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)


def rms_norm_direct(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Standard RMS norm using weight directly (for RMSNormGated in GDN)
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


def _gdn_allgather_out_proj(out_local: torch.Tensor, weights: dict, _diag=None) -> torch.Tensor:
    """AllGather GDN 局部输出 → 完整 v_dim，再做 out_proj。"""
    # out_local: [..., V_DIM_LOCAL=1024]
    orig_shape = out_local.shape
    flat = out_local.reshape(-1, _V_DIM_LOCAL)          # [M, 1024]
    if _diag: _diag("og.flat", flat)
    parts = [torch.zeros_like(flat) for _ in range(TP_SIZE)]
    dist.all_gather(parts, flat)
    gathered = torch.cat(parts, dim=-1)                  # [M, 8192]
    if _diag: _diag("og.gathered", gathered)
    gathered = gathered.reshape(orig_shape[:-1] + (_V_DIM,))

    w_out = weights["gdn.out_proj.weight"]
    s_out = weights.get("gdn.out_proj.weight_scale_inv")
    if _diag:
        _diag("og.w_out", w_out.float())
        if s_out is not None: _diag("og.s_out", s_out.float())
    if s_out is not None:
        return fp8_linear(gathered, w_out, s_out)
    return bf16_linear(gathered, w_out)


_GDN_DECODE_KERNEL_FN = None  # 缓存编译好的 tilelang kernel，避免每次 cache lookup
_GDN_DECODE_SCALE = None


def _init_gdn_decode_kernel(B, Hq, HV, K, V):
    """首次调用：编译 kernel 并缓存 JITKernel 对象（mate 0.2.1 API）。"""
    global _GDN_DECODE_KERNEL_FN, _GDN_DECODE_SCALE
    import mate.gdn_kernels.tilelang.gdn_decode as _tl_mod
    _GDN_DECODE_SCALE = float(K ** -0.5)
    config = _tl_mod._resolve_autotuned_kernel_config(B)
    _GDN_DECODE_KERNEL_FN = _tl_mod._get_decode_fp32_vk_kernel(
        qk_head=Hq, head=HV, dim_k=K, dim_v=V,
        input_dtype="bfloat16", gate_batch_dtype="bfloat16",
        dt_bias_dtype="bfloat16", output_dtype="bfloat16",
        use_qk_l2norm=True,
        v_tile=config["v_tile"],
        num_blocks_per_state=config["num_blocks_per_state"],
        stage=config["stage"],
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
    # reshape+contiguous 强制 stride 规范化
    # MUSA 上 ab[..., :half] 是 [B,1,8] stride (16,16,1)，reshape 到 [B,8] 只是 view
    # 得到 stride (16,1)，kernel 要求 stride[0]=8，必须 contiguous() 强制重新分配
    half = ab.shape[-1] // 2
    a = ab[..., :half].reshape(B, half).contiguous()
    b = ab[..., half:].reshape(B, half).contiguous()
    v_heads_local = v.shape[2]
    rank = weights["gdn.A_log"].device.index or 0
    A_log   = weights["gdn.A_log"][rank * v_heads_local:(rank + 1) * v_heads_local]
    dt_bias = weights["gdn.dt_bias"][rank * v_heads_local:(rank + 1) * v_heads_local].to(torch.bfloat16)

    if _GDN_DECODE_KERNEL_FN is None:
        Hq = q.shape[2]
        _init_gdn_decode_kernel(B, Hq, v_heads_local, GDN_KEY_DIM, GDN_VALUE_DIM)

    output = torch.empty(B, v_heads_local, GDN_VALUE_DIM,
                         device=hidden.device, dtype=hidden.dtype)
    _GDN_DECODE_KERNEL_FN(
        q.squeeze(1), k.squeeze(1), v.squeeze(1),
        A_log, a, dt_bias, b,
        _GDN_DECODE_SCALE,
        state, output,
    )

    # Apply output norm (per-head RMS norm on V_dim) — Gemma-style (1+w)
    norm_w = weights["gdn.norm.weight"]
    output = rms_norm(output, norm_w)

    # Z-gate: silu(z) * normed_output
    w_z = weights["gdn.in_proj_z.weight"]
    s_z = weights.get("gdn.in_proj_z.weight_scale_inv")
    z = fp8_linear(hidden, w_z, s_z) if s_z is not None else bf16_linear(hidden, w_z)
    z = F.silu(z)  # [B, 1, V_DIM_LOCAL]

    out_local = output.reshape(B, 1, _V_DIM_LOCAL)
    out_local = z * out_local

    out_proj = _gdn_allgather_out_proj(out_local, weights)
    return out_proj, state


def gdn_prefill_forward(
    hidden: torch.Tensor,       # [total, HIDDEN]
    cu_seqlens: torch.Tensor,
    weights: dict,
    initial_state=None,
    layer_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _gd(tag, t):
        if layer_idx is None:
            return
        ft = t.float()
        nan_n = torch.isnan(ft).sum().item()
        inf_n = torch.isinf(ft).sum().item()
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        # 在 L01 强制所有 rank 打印（定位 mate kernel 哪个 rank 输入异常）
        force_all = (layer_idx == 1)
        if (nan_n or inf_n) or force_all:
            r = dist.get_rank() if dist.is_initialized() else 0
            flag = f" nan={nan_n} inf={inf_n}" if (nan_n or inf_n) else ""
            print(f"  [GDN L{layer_idx:02d} r{r} {tag}] norm={ft.norm().item():.3e} max={ft.abs().max().item():.3e}{flag} shape={tuple(t.shape)}", flush=True)
        elif is_rank0:
            print(f"  [GDN L{layer_idx:02d} {tag}] norm={ft.norm().item():.3e} max={ft.abs().max().item():.3e} shape={tuple(t.shape)}", flush=True)

    total, _ = hidden.shape
    _gd("hidden_in", hidden)
    w_qkv = weights["gdn.in_proj_qkv.weight"]
    s_qkv = weights.get("gdn.in_proj_qkv.weight_scale_inv")
    qkv = fp8_linear(hidden, w_qkv, s_qkv) if s_qkv is not None else bf16_linear(hidden, w_qkv)
    _gd("qkv_post_fp8", qkv)

    # Causal conv1d + SiLU (depthwise, kernel_size=4, groups=C)
    conv_w = weights["gdn.conv1d.weight"]  # [C, 1, 4]
    C = qkv.shape[-1]
    qkv_conv = qkv.unsqueeze(0).transpose(1, 2)  # [1, C, total]
    qkv_conv = F.pad(qkv_conv, (3, 0))  # causal left-pad
    qkv_conv = F.conv1d(qkv_conv, conv_w, groups=C)  # [1, C, total]
    qkv_conv = F.silu(qkv_conv)
    qkv = qkv_conv.transpose(1, 2).squeeze(0)  # [total, C]
    _gd("qkv_post_conv", qkv)

    q, k, v = _split_qkv(qkv)
    q = q.reshape(total, -1, GDN_KEY_DIM).contiguous()
    k = k.reshape(total, -1, GDN_KEY_DIM).contiguous()
    v = v.reshape(total, -1, GDN_VALUE_DIM).contiguous()
    _gd("q", q); _gd("k", k); _gd("v", v)

    # Compute alpha (g) and beta from hidden states (same as decode kernel)
    # decode: x = a + dt_bias; softplus_x = softplus(x); g = -exp(A_log) * softplus_x
    #         alpha = exp(g); beta = sigmoid(b_raw)
    w_a = weights["gdn.in_proj_a.weight"]
    w_b = weights["gdn.in_proj_b.weight"]
    a_raw = bf16_linear(hidden, w_a).float()  # [total, num_v_heads_local]
    b_raw = bf16_linear(hidden, w_b).float()
    _gd("a_raw", a_raw); _gd("b_raw", b_raw)
    num_v_heads_local = v.shape[1]
    rank = w_a.device.index or 0
    A_log = weights["gdn.A_log"][rank * num_v_heads_local:(rank + 1) * num_v_heads_local].float()
    dt_bias = weights["gdn.dt_bias"][rank * num_v_heads_local:(rank + 1) * num_v_heads_local].float()
    x = a_raw + dt_bias  # [total, H]
    softplus_x = F.softplus(x)
    g_log = -torch.exp(A_log) * softplus_x  # [total, H]
    # Clamp to FLT_MIN to avoid underflow → 0 → log(0)=-inf in mate kernel
    # chunk_local_cumsum 仅处理 (0, FLT_MIN) 的 subnormal，alpha==0 会产生 -inf 传染 NaN
    g = torch.exp(g_log).clamp(min=1.1754943508222875e-38).contiguous()  # alpha in (0,1]
    beta = torch.sigmoid(b_raw).contiguous()
    _gd("g", g); _gd("beta", beta)

    out, final_state = gdn_pre.chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens,
        initial_state=initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out: [total, local_V_heads, V_dim]
    _gd("chunk_out", out); _gd("final_state", final_state)

    # Apply output norm (per-head RMS norm on V_dim) — uses weight directly (not Gemma-style)
    norm_w = weights["gdn.norm.weight"]
    out = rms_norm_direct(out, norm_w)
    _gd("post_norm", out)

    # Z-gate: silu(z) * normed_output
    w_z = weights["gdn.in_proj_z.weight"]
    s_z = weights.get("gdn.in_proj_z.weight_scale_inv")
    z = fp8_linear(hidden, w_z, s_z) if s_z is not None else bf16_linear(hidden, w_z)
    z = F.silu(z)  # [total, V_DIM_LOCAL]
    _gd("z_silu", z)

    out_local = out.reshape(total, _V_DIM_LOCAL)
    out_local = z * out_local
    _gd("z_gated", out_local)

    out_proj = _gdn_allgather_out_proj(out_local, weights, _diag=_gd)  # [total, HIDDEN]
    _gd("out_proj", out_proj)
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
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Standard GQA attention with output gate. Returns (output, updated_kv)."""
    orig_shape = hidden.shape
    if hidden.dim() == 3:
        B, T, H = hidden.shape
        x = hidden.reshape(B * T, H)
    else:
        x = hidden
        B, T = x.shape[0], 1

    total = x.shape[0]

    # Q projection (col-sharded, includes gate): [total, 2048] = 4 heads × 2 (q+gate) × 256
    # checkpoint 中 q+gate 在每个 head 内交错存放（sglang QKVParallelLinear 的布局，
    # 已通过 diag_weight_layout.py 验证：q 子块 RMS ≈ 0.0001, gate 子块 RMS ≈ 0.0002，
    # 在每个 head 内部交错出现）：
    #   [head0_q | head0_gate | head1_q | head1_gate | ...]
    # 必须先 reshape 成 [..., local_heads, 2*head_dim] 再 chunk(2, dim=-1) 出 q/gate。
    w_q = weights["attn.q_proj.weight"]
    s_q = weights.get("attn.q_proj.weight_scale_inv")
    qg = fp8_linear(x, w_q, s_q) if s_q is not None else bf16_linear(x, w_q)

    qg_per_head = qg.reshape(total, _ATTN_Q_HEADS_LOCAL, 2 * _ATTN_HEAD_DIM)
    q_per_head, gate_per_head = torch.chunk(qg_per_head, 2, dim=-1)
    q_local = q_per_head.reshape(total, _ATTN_Q_HEADS_LOCAL * _ATTN_HEAD_DIM)
    gate_local = gate_per_head.reshape(total, _ATTN_Q_HEADS_LOCAL * _ATTN_HEAD_DIM)

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

    scale = _ATTN_HEAD_DIM ** -0.5
    if T > 1:
        # Prefill: use mate FMHA (supports head_dim=256, handles GQA)
        from mate.jit.attention.fmha import _fmha_fwd
        q_4d = q_heads.unsqueeze(0)   # [1, total, 4, 256]
        k_4d = k_heads.unsqueeze(0)   # [1, total, 2, 256]
        v_4d = v_heads.unsqueeze(0)   # [1, total, 2, 256]
        attn_out, _ = _fmha_fwd(q_4d, k_4d, v_4d, softmax_scale=scale, is_causal=True)
        attn_out = attn_out.squeeze(0)  # [total, 4, 256]
        new_kv = (k_heads, v_heads)
    else:
        # Decode (T=1): GQA attention with KV cache
        # past_kv shape: [B, past_len, num_kv_heads, head_dim] or None
        # k_heads, v_heads: [B, num_kv_heads, head_dim]
        k_cur = k_heads.unsqueeze(1)  # [B, 1, 2, 256]
        v_cur = v_heads.unsqueeze(1)  # [B, 1, 2, 256]
        if past_kv is not None:
            past_k, past_v = past_kv
            k_full = torch.cat([past_k, k_cur], dim=1)  # [B, past+1, 2, 256]
            v_full = torch.cat([past_v, v_cur], dim=1)  # [B, past+1, 2, 256]
        else:
            k_full = k_cur
            v_full = v_cur
        new_kv = (k_full, v_full)
        S = k_full.shape[1]
        local_gqa_ratio = _ATTN_Q_HEADS_LOCAL // _ATTN_NUM_KV_HEADS
        k_exp = k_full.repeat_interleave(local_gqa_ratio, dim=2)  # [B, S, 4, 256]
        v_exp = v_full.repeat_interleave(local_gqa_ratio, dim=2)  # [B, S, 4, 256]
        # q_heads: [B, 4, 256] → [B, 4, 1, 256]
        q_4d = q_heads.unsqueeze(2)
        # k_exp: [B, S, 4, 256] → [B, 4, S, 256]
        k_4d = k_exp.permute(0, 2, 1, 3)
        scores = torch.matmul(q_4d, k_4d.transpose(-1, -2)) * scale  # [B, 4, 1, S]
        attn_weights = F.softmax(scores, dim=-1)
        v_4d = v_exp.permute(0, 2, 1, 3)  # [B, 4, S, 256]
        attn_out = torch.matmul(attn_weights, v_4d).squeeze(2)  # [B, 4, 256]
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

    return output.reshape(orig_shape), new_kv


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
    # HF Qwen3_5MoeTopKRouter: top-k 概率重归一化为和=1（HF modeling_qwen3_5_moe.py:778）
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_w = topk_w.to(torch.bfloat16)

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

    # 预计算 inverse_order 用于后续 unsort（替代 scatter_add）
    inverse_order = torch.empty_like(sort_order)
    inverse_order[sort_order] = torch.arange(total * topk, device=device)

    # Gather hidden states（固定大小 [total*topk, HIDDEN]）
    expanded_hidden = hidden[sorted_token_idx]
    m_indices = sorted_expert_ids.to(torch.int32)

    # ── 3. Expert GEMM ──
    num_expanded = total * topk  # 固定大小，Python 已知

    use_batched = (
        "moe.experts.gate_proj.weight" in weights
        or "moe.experts._gate_up_proj.weight" in weights
    ) and (
        weights.get("moe.experts.gate_proj.weight", weights.get("moe.experts._gate_up_proj.weight")).dtype == torch.float8_e4m3fn
    )

    if use_batched:
        from mate.deep_gemm import ragged_m_moe_gemm_8bit

        w_gate_up, s_gate_up = _get_fused_gate_up_weights(weights)
        w_down = weights["moe.experts.down_proj.weight"]
        s_down = weights["moe.experts.down_proj.weight_scale_inv"]

        a_fp8, a_scale = _fast_fp8_quantize(expanded_hidden)

        gate_up_out = torch.empty(num_expanded, MOE_INTERMEDIATE * 2, device=device, dtype=torch.bfloat16)
        ragged_m_moe_gemm_8bit(
            (a_fp8, a_scale), (w_gate_up, s_gate_up), m_indices, gate_up_out,
        )
        gate_out, up_out = gate_up_out.split(MOE_INTERMEDIATE, dim=-1)

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

    # ── 4. Unsort + reshape + weighted sum（替代 scatter_add，快 2.6x）──
    unsorted = down_out[inverse_order]  # 恢复到 token 顺序 [total*topk, HIDDEN]
    reshaped = unsorted.reshape(total, topk, HIDDEN_SIZE)
    output = (reshaped * flat_weights.reshape(total, topk, 1)).sum(dim=1)

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
    if K % BLOCK != 0:
        x = F.pad(x, (0, BLOCK * K_blocks - K))
    x_blocks = x.reshape(M, K_blocks, BLOCK)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [M, K_blocks]
    scale = (amax / 448.0).to(torch.float32)  # [M, K_blocks]
    x_fp8 = (x_blocks * (448.0 / amax).unsqueeze(-1)).reshape(M, K_blocks * BLOCK).to(torch.float8_e4m3fn)
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
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
    """单层 decode forward。返回 (hidden, gdn_state, kv_cache)。"""
    is_attn_layer = "attn.q_proj.weight" in weights

    # 1. Pre-norm + attention/GDN
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm(hidden, norm_w)

    new_kv = None
    if is_attn_layer:
        attn_out, new_kv = _attn_forward(x, weights, past_kv=kv_cache)
        hidden = hidden + attn_out
        new_state = gdn_state
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

    return hidden, new_state, new_kv


def layer_forward_prefill(
    hidden: torch.Tensor,       # [total_tokens, HIDDEN]
    cu_seqlens: torch.Tensor,   # [num_seqs+1] int64
    weights: dict,
    initial_gdn_state: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
    """单层 prefill forward。返回 (hidden, gdn_state, kv_cache)。"""
    is_attn_layer = "attn.q_proj.weight" in weights

    def _diag(tag: str, t: torch.Tensor):
        if layer_idx is None:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        ft = t.float()
        nan_n = torch.isnan(ft).sum().item()
        inf_n = torch.isinf(ft).sum().item()
        if nan_n or inf_n:
            print(f"[DIAG L{layer_idx:02d} {tag}] norm={ft.norm().item():.3e} nan={nan_n} inf={inf_n} shape={tuple(t.shape)}", flush=True)
        else:
            print(f"[DIAG L{layer_idx:02d} {tag}] norm={ft.norm().item():.3e} max={ft.abs().max().item():.3e} shape={tuple(t.shape)}", flush=True)

    _diag(f"in   ({'ATTN' if is_attn_layer else 'GDN '})", hidden)

    # 1. Pre-norm + attention/GDN
    norm_w = weights["input_layernorm.weight"]
    x = rms_norm(hidden, norm_w)
    _diag("post-norm1   ", x)

    new_kv = None
    if is_attn_layer:
        # Pass as 3D [1, total, HIDDEN] so _attn_forward uses FMHA (T > 1)
        x_3d = x.unsqueeze(0)
        attn_out, new_kv = _attn_forward(x_3d, weights)
        _diag("attn_out     ", attn_out)
        hidden = hidden + attn_out.squeeze(0)
        final_state = initial_gdn_state
    else:
        gdn_out, final_state = gdn_prefill_forward(x, cu_seqlens, weights, initial_gdn_state, layer_idx=layer_idx)
        _diag("gdn_out      ", gdn_out)
        hidden = hidden + gdn_out
    _diag("post-residual1", hidden)

    # 2. Post-norm + MoE
    norm_w2 = weights["post_attention_layernorm.weight"]
    x = rms_norm(hidden, norm_w2)
    _diag("post-norm2   ", x)
    moe_out = moe_forward(x, weights)
    _diag("moe_out      ", moe_out)
    hidden = hidden + moe_out
    _diag("out          ", hidden)

    return hidden, final_state, new_kv
