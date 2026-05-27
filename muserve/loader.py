"""Task 0.2：FP8 safetensors 权重加载 + TP sharding。

只支持 Qwen3.5-397B-A17B-FP8，TP=8 固定。
"""

import os
import json
from pathlib import Path
from typing import Dict

import torch
import torch_musa
from safetensors import safe_open

from muserve.config import (
    TP_SIZE, HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
    NUM_EXPERTS, MOE_INTERMEDIATE, GDN_NUM_K_HEADS, GDN_NUM_V_HEADS,
    GDN_KEY_DIM, GDN_VALUE_DIM, DEFAULT_MODEL_PATH,
)
from muserve.distributed import get_tp_rank


def _device() -> torch.device:
    return torch.device(f"musa:{get_tp_rank()}")


def _load_index(model_path: str) -> Dict[str, str]:
    """返回 {weight_name: shard_filename} 映射。"""
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        return json.load(f)["weight_map"]


# 实际 checkpoint 结构（已验证）：
# 主层前缀：  model.language_model.layers.{i}.
# 主层组件：  linear_attn.*（GDN），mlp.*（MoE），input_layernorm，post_attention_layernorm
# 注意：主层无 self_attn，标准 attention 只在 mtp.layers 里
# Embedding：  model.language_model.embed_tokens.weight
# GDN 投影：  in_proj_qkv（Q+K+V 合并），in_proj_a，in_proj_b，in_proj_z，out_proj


def _open_shard(model_path: str, filename: str):
    path = str(Path(model_path) / filename)
    return safe_open(path, framework="pt", device="cpu")


# ── TP sharding 规则 ──────────────────────────────────────────────────────────
# 列切分（Column Parallel）：沿 dim=0 切，每卡取 [N//TP, K]
# 行切分（Row Parallel）：沿 dim=0 切（权重已转置），每卡取 [N, K//TP]
# Expert 分配：expert_idx % TP_SIZE == rank 的 expert 归该卡

def _col_shard(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """列切分：dim=0，取第 rank 段。"""
    n = tensor.shape[0] // TP_SIZE
    return tensor[rank * n:(rank + 1) * n].contiguous()


def _row_shard(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """行切分：dim=-1，取第 rank 段（适用于 [out, in] 格式）。"""
    k = tensor.shape[-1] // TP_SIZE
    return tensor[..., rank * k:(rank + 1) * k].contiguous()


def _merged_col_shard(tensor: torch.Tensor, rank: int, segments: list[int]) -> torch.Tensor:
    """Merged column 切分：dim=0 上的多段（如 [q|k|v]）各自独立 TP 切再拼接。

    对应 sglang MergedColumnParallelLinear 的语义：output_sizes 中的每段独立切。
    简单 _col_shard 会把 dim=0 当作单一连续段，导致 q/k/v 错位。
    """
    assert sum(segments) == tensor.shape[0], (
        f"segments sum {sum(segments)} != tensor.shape[0] {tensor.shape[0]}"
    )
    shards = []
    offset = 0
    for seg in segments:
        sub = tensor[offset:offset + seg]
        assert seg % TP_SIZE == 0, f"segment {seg} not divisible by TP_SIZE {TP_SIZE}"
        per_rank = seg // TP_SIZE
        shards.append(sub[rank * per_rank:(rank + 1) * per_rank])
        offset += seg
    return torch.cat(shards, dim=0).contiguous()


def load_layer_weights(
    model_path: str,
    layer_idx: int,
    weight_index: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """加载单层权重，按 TP rank 切分后放到 MUSA 设备。

    实际 checkpoint 结构（已验证）：
      前缀：model.language_model.layers.{i}.
      主层无 self_attn，只有 linear_attn（GDN）+ mlp（MoE）
      GDN 投影：in_proj_qkv（Q+K+V 合并），in_proj_a，in_proj_b，in_proj_z，out_proj
    """
    rank = get_tp_rank()
    device = _device()
    prefix = f"model.language_model.layers.{layer_idx}."

    # 收集本层所有 key
    layer_keys = {k: v for k, v in weight_index.items() if k.startswith(prefix)}

    # 按 shard 文件分组，减少重复打开
    shard_to_keys: Dict[str, list] = {}
    for key, shard_file in layer_keys.items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    raw: Dict[str, torch.Tensor] = {}
    for shard_file, keys in shard_to_keys.items():
        f = _open_shard(model_path, shard_file)
        for key in keys:
            raw[key] = f.get_tensor(key)

    weights: Dict[str, torch.Tensor] = {}
    p = prefix

    def get(name: str) -> torch.Tensor:
        return raw[p + name]

    def maybe(name: str) -> torch.Tensor | None:
        t = raw.get(p + name)
        # gemm_fp8_nt_groupwise 要求 scale 必须是 float32
        if t is not None and "scale_inv" in name:
            t = t.float()
        return t

    # ── GDN（GatedDeltaNet 线性注意力）────────────────────────────────────────
    # in_proj_qkv: [12288, 4096] fp8  → 分段列切分 → [1536, 4096]
    #   checkpoint 布局: [q_full(2048) | k_full(2048) | v_full(8192)] dim=0
    #   每段独立按 TP 切再拼接：rank r 拿到 [q(256) | k(256) | v(1024)]
    # in_proj_a:   [64, 4096]    bf16 → 列切分 → [8, 4096]（gate a，本卡 8 heads）
    # in_proj_b:   [64, 4096]    bf16 → 列切分 → [8, 4096]（gate b）
    # in_proj_z:   [8192, 4096]  fp8  → 列切分 → [1024, 4096]
    # out_proj:    [4096, 8192]  fp8  → 不切分，每卡持有完整权重
    #   GDN 输出先 AllGather 到 [B, 8192]，再做完整 out_proj → [B, 4096]
    #   不需要 AllReduce（out_proj 不切分）
    _Q_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM    # 2048
    _K_DIM = GDN_NUM_K_HEADS * GDN_KEY_DIM    # 2048
    _V_DIM = GDN_NUM_V_HEADS * GDN_VALUE_DIM  # 8192
    _QKV_SEGMENTS = [_Q_DIM, _K_DIM, _V_DIM]
    t = maybe("linear_attn.in_proj_qkv.weight")
    if t is not None:
        weights["gdn.in_proj_qkv.weight"] = _merged_col_shard(t, rank, _QKV_SEGMENTS).to(device)
    s = maybe("linear_attn.in_proj_qkv.weight_scale_inv")
    if s is not None:
        # fp8 weight_scale_inv 是 [N//128, K//128] 的 block-wise scale
        # 每段在 dim=0 上的块数 = seg // 128
        scale_segments = [seg // 128 for seg in _QKV_SEGMENTS]
        weights["gdn.in_proj_qkv.weight_scale_inv"] = _merged_col_shard(s, rank, scale_segments).to(device)

    for proj, shard_fn in [
        ("in_proj_a",   _col_shard),
        ("in_proj_b",   _col_shard),
        ("in_proj_z",   _col_shard),
    ]:
        t = maybe(f"linear_attn.{proj}.weight")
        if t is not None:
            weights[f"gdn.{proj}.weight"] = shard_fn(t, rank).to(device)
        s = maybe(f"linear_attn.{proj}.weight_scale_inv")
        if s is not None:
            weights[f"gdn.{proj}.weight_scale_inv"] = shard_fn(s, rank).to(device)

    # out_proj 不切分
    t = maybe("linear_attn.out_proj.weight")
    if t is not None:
        weights["gdn.out_proj.weight"] = t.to(device)
    s = maybe("linear_attn.out_proj.weight_scale_inv")
    if s is not None:
        weights["gdn.out_proj.weight_scale_inv"] = s.to(device)

    # GDN 标量参数（A_log/dt_bias/norm.weight 与 qkv 通道无关，保留全量）
    # A_log, dt_bias 必须是 float32（gated_delta_rule_decode 要求）
    for param in ("A_log", "dt_bias", "norm.weight"):
        t = maybe(f"linear_attn.{param}")
        if t is not None:
            if param in ("A_log", "dt_bias"):
                t = t.float()
            weights[f"gdn.{param}"] = t.to(device)

    # conv1d.weight: depthwise conv，通道数 = qkv 总维度，必须跟 in_proj_qkv 同样分段切
    # 全量 [12288, 1, 4] → 每卡 [1536, 1, 4]（q256 + k256 + v1024 通道顺序）
    t = maybe("linear_attn.conv1d.weight")
    if t is not None:
        weights["gdn.conv1d.weight"] = _merged_col_shard(t, rank, _QKV_SEGMENTS).to(device)

    # ── Self-Attention（标准 GQA，每 4 层一个）────────────────────────────────────
    # q_proj: [16384, 4096] fp8 → col shard → [2048, 4096]（含 gate）
    # k_proj: [512, 4096] fp8   → 不切分（只有 2 个 KV heads，太小）
    # v_proj: [512, 4096] fp8   → 不切分
    # o_proj: [4096, 8192] fp8  → 不切分（AllGather 后用完整权重）
    # q_norm, k_norm: [256] bf16 → 不切分
    for proj in ("q_proj",):
        t = maybe(f"self_attn.{proj}.weight")
        if t is not None:
            weights[f"attn.{proj}.weight"] = _col_shard(t, rank).to(device)
        s = maybe(f"self_attn.{proj}.weight_scale_inv")
        if s is not None:
            weights[f"attn.{proj}.weight_scale_inv"] = _col_shard(s, rank).to(device)

    for proj in ("k_proj", "v_proj", "o_proj"):
        t = maybe(f"self_attn.{proj}.weight")
        if t is not None:
            weights[f"attn.{proj}.weight"] = t.to(device)
        s = maybe(f"self_attn.{proj}.weight_scale_inv")
        if s is not None:
            weights[f"attn.{proj}.weight_scale_inv"] = s.to(device)

    for norm in ("q_norm", "k_norm"):
        t = maybe(f"self_attn.{norm}.weight")
        if t is not None:
            weights[f"attn.{norm}.weight"] = t.to(device)
    # gate（router）：[num_experts, hidden]  → 不切分，每卡都有完整 gate
    t = maybe("mlp.gate.weight")
    if t is not None:
        weights["moe.gate.weight"] = t.to(device)

    # shared_expert（不切分，每卡都有）
    for proj in ("gate_proj", "up_proj", "down_proj"):
        t = maybe(f"mlp.shared_expert.{proj}.weight")
        if t is not None:
            weights[f"moe.shared_expert.{proj}.weight"] = t.to(device)
        s = maybe(f"mlp.shared_expert.{proj}.weight_scale_inv")
        if s is not None:
            weights[f"moe.shared_expert.{proj}.weight_scale_inv"] = s.to(device)

    t = maybe("mlp.shared_expert_gate.weight")
    if t is not None:
        weights["moe.shared_expert_gate.weight"] = t.to(device)

    # routed experts：每卡负责 expert_idx % TP_SIZE == rank 的 expert
    # 堆叠为 [num_experts_local, N, K] 供 ragged_m_moe_gemm_8bit 使用
    my_experts = [e for e in range(NUM_EXPERTS) if e % TP_SIZE == rank]
    experts_per_rank = len(my_experts)

    # 先按 expert 逐个加载
    for e_global in my_experts:
        e_local = e_global // TP_SIZE
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t = maybe(f"mlp.experts.{e_global}.{proj}.weight")
            if t is not None:
                weights[f"moe.expert_{e_local}.{proj}.weight"] = t.to(device)
            s = maybe(f"mlp.experts.{e_global}.{proj}.weight_scale_inv")
            if s is not None:
                weights[f"moe.expert_{e_local}.{proj}.weight_scale_inv"] = s.to(device)

    # 堆叠 expert 权重为 [num_experts_local, N, K] 供 batched GEMM 使用
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w_list = []
        s_list = []
        for e_local in range(experts_per_rank):
            wk = f"moe.expert_{e_local}.{proj}.weight"
            sk = f"moe.expert_{e_local}.{proj}.weight_scale_inv"
            if wk in weights:
                w_list.append(weights[wk])
            if sk in weights:
                s_list.append(weights[sk])
        if w_list:
            weights[f"moe.experts.{proj}.weight"] = torch.stack(w_list, dim=0)
        if s_list:
            weights[f"moe.experts.{proj}.weight_scale_inv"] = torch.stack(s_list, dim=0)
        # 删除单独的 expert 权重释放显存
        for e_local in range(experts_per_rank):
            weights.pop(f"moe.expert_{e_local}.{proj}.weight", None)
            weights.pop(f"moe.expert_{e_local}.{proj}.weight_scale_inv", None)

    # ── LayerNorm ──────────────────────────────────────────────────────────────
    for norm in ("input_layernorm", "post_attention_layernorm"):
        t = maybe(f"{norm}.weight")
        if t is not None:
            weights[f"{norm}.weight"] = t.to(device)

    return weights


def load_embedding_weights(
    model_path: str,
    weight_index: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """加载 embedding 和 lm_head 权重。"""
    rank = get_tp_rank()
    device = _device()
    weights = {}

    # 实际 key 名（已验证）
    name_map = {
        "model.language_model.embed_tokens.weight": "embed_tokens.weight",
        "lm_head.weight":                           "lm_head.weight",
        "model.language_model.norm.weight":         "norm.weight",
    }
    for src_name, dst_name in name_map.items():
        if src_name not in weight_index:
            continue
        f = _open_shard(model_path, weight_index[src_name])
        t = f.get_tensor(src_name)
        if dst_name in ("embed_tokens.weight", "lm_head.weight"):
            weights[dst_name] = _col_shard(t, rank).to(device)
        else:
            weights[dst_name] = t.to(device)

    return weights


def load_all_weights(
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[Dict[str, torch.Tensor], list[Dict[str, torch.Tensor]]]:
    """加载完整模型权重。返回 (embedding_weights, [layer_0_weights, ...])。"""
    rank = get_tp_rank()
    if rank == 0:
        print(f"[loader] Loading weights from {model_path} (rank {rank}/{TP_SIZE})")

    weight_index = _load_index(model_path)

    embed_weights = load_embedding_weights(model_path, weight_index)

    from muserve.config import NUM_LAYERS
    layer_weights = []
    for i in range(NUM_LAYERS):
        if rank == 0 and i % 10 == 0:
            print(f"[loader] Loading layer {i}/{NUM_LAYERS}...")
        layer_weights.append(load_layer_weights(model_path, i, weight_index))

    if rank == 0:
        print("[loader] Done.")

    return embed_weights, layer_weights
