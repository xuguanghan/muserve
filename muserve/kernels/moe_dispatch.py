"""MoE routing + expert GEMM — mate fp8 gemm."""
import torch
import mate
import mate.gemm as _gemm
from muserve.config import NUM_EXPERTS, NUM_EXPERTS_PER_TOK, MOE_INTERMEDIATE, HIDDEN_SIZE


def moe_gate(
    hidden: torch.Tensor,   # [total_tokens, HIDDEN_SIZE]
    gate_weight: torch.Tensor,  # [NUM_EXPERTS, HIDDEN_SIZE]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute routing scores and select top-k experts.

    Returns:
        topk_ids:    [total_tokens, NUM_EXPERTS_PER_TOK] int32
        topk_weights:[total_tokens, NUM_EXPERTS_PER_TOK] float32 (softmax-normalized)
    """
    # [total_tokens, NUM_EXPERTS]
    logits = torch.nn.functional.linear(hidden.float(), gate_weight.float())
    scores = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    # renormalize
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights


def moe_experts_fp8(
    hidden: torch.Tensor,           # [total_tokens, HIDDEN_SIZE] bf16
    topk_ids: torch.Tensor,         # [total_tokens, NUM_EXPERTS_PER_TOK] int32
    topk_weights: torch.Tensor,     # [total_tokens, NUM_EXPERTS_PER_TOK] fp32
    gate_up_weight: torch.Tensor,   # [NUM_EXPERTS, 2*MOE_INTERMEDIATE, HIDDEN_SIZE] fp8
    gate_up_scale: torch.Tensor,    # scale for gate_up_weight
    down_weight: torch.Tensor,      # [NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE] fp8
    down_scale: torch.Tensor,       # scale for down_weight
) -> torch.Tensor:                  # [total_tokens, HIDDEN_SIZE] bf16
    """Expert computation with FP8 GEMM. Fallback: loop over active experts."""
    total_tokens = hidden.shape[0]
    output = torch.zeros_like(hidden)

    # Group tokens by expert
    for expert_idx in range(NUM_EXPERTS):
        mask = (topk_ids == expert_idx).any(dim=-1)  # [total_tokens]
        if not mask.any():
            continue

        tokens = hidden[mask]  # [n_tokens, HIDDEN_SIZE]
        # gate_up proj: [n_tokens, 2*MOE_INTERMEDIATE]
        w_gu = gate_up_weight[expert_idx]   # [2*MOE_INTERMEDIATE, HIDDEN_SIZE] fp8
        s_gu = gate_up_scale[expert_idx]
        gate_up = _gemm.gemm_fp8_nt_groupwise(
            tokens.to(torch.float8_e4m3fn),
            w_gu,
            torch.ones(tokens.shape[0], HIDDEN_SIZE // 128,
                       device=hidden.device, dtype=torch.float32),
            s_gu,
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up  # [n_tokens, MOE_INTERMEDIATE]

        # down proj: [n_tokens, HIDDEN_SIZE]
        w_d = down_weight[expert_idx]   # [HIDDEN_SIZE, MOE_INTERMEDIATE] fp8
        s_d = down_scale[expert_idx]
        expert_out = _gemm.gemm_fp8_nt_groupwise(
            act.to(torch.float8_e4m3fn),
            w_d,
            torch.ones(act.shape[0], MOE_INTERMEDIATE // 128,
                       device=hidden.device, dtype=torch.float32),
            s_d,
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
        )

        # weighted accumulate
        weights = topk_weights[mask][topk_ids[mask] == expert_idx].unsqueeze(-1)
        output[mask] += expert_out * weights

    return output
