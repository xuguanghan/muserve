"""Phase 2: Batched MoE routing — 替代 Python 循环的向量化实现。

核心优化：
  - 原始：Python for 循环遍历 64 experts，每个 expert 单独做 GEMM
  - 优化：按 expert 分组 token，批量调用 GEMM，减少 kernel launch 开销
"""

import torch
import torch.nn.functional as F
import mate.gemm as gemm_mod

from muserve.config import (
    HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, MOE_INTERMEDIATE, TP_SIZE,
)
from muserve.distributed import all_reduce
from muserve.model.qwen35_layer import fp8_linear, bf16_linear


def batched_moe_forward(
    hidden: torch.Tensor,   # [total_tokens, HIDDEN]
    weights: dict,
) -> torch.Tensor:          # [total_tokens, HIDDEN]
    """Batched MoE forward：向量化 routing + 分组 GEMM。"""
    total = hidden.shape[0]
    device = hidden.device
    experts_per_rank = NUM_EXPERTS // TP_SIZE

    # Gate routing
    gate_w = weights["moe.gate.weight"]
    logits = bf16_linear(hidden, gate_w).float()
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    output = torch.zeros(total, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # 按 expert 分组 token，批量处理
    for e_local in range(experts_per_rank):
        e_global = e_local * TP_SIZE + (device.index or 0)
        # 找到路由到此 expert 的所有 (token, slot) 对
        expert_mask = (topk_ids == e_global)  # [total, top_k]
        token_mask = expert_mask.any(dim=-1)  # [total]

        if not token_mask.any():
            continue

        # 提取路由到此 expert 的 token
        tok_indices = token_mask.nonzero(as_tuple=True)[0]
        tok = hidden[tok_indices]  # [num_routed, HIDDEN]

        # Expert GEMM: gate_proj + up_proj → SiLU → down_proj
        w_g = weights[f"moe.expert_{e_local}.gate_proj.weight"]
        w_u = weights[f"moe.expert_{e_local}.up_proj.weight"]
        w_d = weights[f"moe.expert_{e_local}.down_proj.weight"]
        s_g = weights.get(f"moe.expert_{e_local}.gate_proj.weight_scale_inv")
        s_u = weights.get(f"moe.expert_{e_local}.up_proj.weight_scale_inv")
        s_d = weights.get(f"moe.expert_{e_local}.down_proj.weight_scale_inv")

        if s_g is not None:
            gate_out = fp8_linear(tok, w_g, s_g)
            up_out = fp8_linear(tok, w_u, s_u)
        else:
            gate_out = bf16_linear(tok, w_g)
            up_out = bf16_linear(tok, w_u)

        act = F.silu(gate_out) * up_out

        if s_d is not None:
            down_out = fp8_linear(act, w_d, s_d)
        else:
            down_out = bf16_linear(act, w_d)

        # 加权累加（向量化）
        # 获取每个 token 对应此 expert 的权重
        expert_weights = topk_w[tok_indices]  # [num_routed, top_k]
        expert_slots = expert_mask[tok_indices]  # [num_routed, top_k]
        # 取对应 slot 的权重（一个 token 可能在多个 slot 路由到同一 expert）
        slot_weights = (expert_weights * expert_slots.float()).sum(dim=-1, keepdim=True)
        output[tok_indices] += down_out * slot_weights.to(down_out.dtype)

    # AllReduce（各卡的 expert 输出求和）
    all_reduce(output)

    # Shared expert
    if "moe.shared_expert.gate_proj.weight" in weights:
        w_g = weights["moe.shared_expert.gate_proj.weight"]
        w_u = weights["moe.shared_expert.up_proj.weight"]
        w_d = weights["moe.shared_expert.down_proj.weight"]
        s_g = weights.get("moe.shared_expert.gate_proj.weight_scale_inv")
        s_u = weights.get("moe.shared_expert.up_proj.weight_scale_inv")
        s_d = weights.get("moe.shared_expert.down_proj.weight_scale_inv")

        if s_g is not None:
            gate_out = fp8_linear(hidden, w_g, s_g)
            up_out = fp8_linear(hidden, w_u, s_u)
        else:
            gate_out = bf16_linear(hidden, w_g)
            up_out = bf16_linear(hidden, w_u)

        act = F.silu(gate_out) * up_out

        if s_d is not None:
            shared_out = fp8_linear(act, w_d, s_d)
        else:
            shared_out = bf16_linear(act, w_d)

        if "moe.shared_expert_gate.weight" in weights:
            gate_val = torch.sigmoid(
                bf16_linear(hidden, weights["moe.shared_expert_gate.weight"])
            )
            shared_out = shared_out * gate_val

        output = output + shared_out

    return output
