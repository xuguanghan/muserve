"""GatedDeltaNet kernel wrappers — mate.gdn_decode / mate.gdn_prefill."""
import torch
import mate.gdn_decode as _dec
import mate.gdn_prefill as _pre
from muserve.config import (
    GDN_NUM_K_HEADS, GDN_NUM_V_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM,
)


def gdn_decode(
    q: torch.Tensor,        # [B, 1, GDN_NUM_K_HEADS, GDN_KEY_DIM]
    k: torch.Tensor,        # [B, 1, GDN_NUM_K_HEADS, GDN_KEY_DIM]
    v: torch.Tensor,        # [B, 1, GDN_NUM_V_HEADS, GDN_VALUE_DIM]
    state: torch.Tensor,    # [B, GDN_NUM_V_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM] fp32
    A_log: torch.Tensor,    # [GDN_NUM_V_HEADS] fp32
    a: torch.Tensor,        # [B, 1, GDN_NUM_V_HEADS]
    dt_bias: torch.Tensor,  # [GDN_NUM_V_HEADS] fp32
    b: torch.Tensor,        # [B, 1, GDN_NUM_V_HEADS]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step GDN decode. Returns (output, new_state)."""
    return _dec.gated_delta_rule_decode(q, k, v, state, A_log, a, dt_bias, b)


def gdn_prefill(
    q: torch.Tensor,            # [total_tokens, GDN_NUM_K_HEADS, GDN_KEY_DIM]
    k: torch.Tensor,            # [total_tokens, GDN_NUM_K_HEADS, GDN_KEY_DIM]
    v: torch.Tensor,            # [total_tokens, GDN_NUM_V_HEADS, GDN_VALUE_DIM]
    cu_seqlens: torch.Tensor,   # [num_seqs + 1] int64
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Chunked GDN prefill (varlen). Returns output or (output, final_state)."""
    return _pre.chunk_gated_delta_rule(
        q, k, v,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
