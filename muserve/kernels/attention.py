"""Flash attention wrappers — mate flash_attn."""
import torch
import mate


def attn_prefill(
    q: torch.Tensor,            # [total_tokens, NUM_Q_HEADS, HEAD_DIM]
    k: torch.Tensor,            # [total_tokens, NUM_KV_HEADS, HEAD_DIM]
    v: torch.Tensor,            # [total_tokens, NUM_KV_HEADS, HEAD_DIM]
    cu_seqlens_q: torch.Tensor, # [num_seqs + 1] int32
    cu_seqlens_k: torch.Tensor, # [num_seqs + 1] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:              # [total_tokens, NUM_Q_HEADS, HEAD_DIM]
    return mate.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def attn_decode(
    q: torch.Tensor,                    # [B, 1, NUM_Q_HEADS, HEAD_DIM]
    k_cache: torch.Tensor,              # [num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM]
    v_cache: torch.Tensor,              # [num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM]
    cache_seqlens: torch.Tensor,        # [B] int32 — current kv length per seq
    page_table: torch.Tensor | None,    # [B, max_pages] int32 — paged kv index
    softmax_scale: float | None = None,
) -> torch.Tensor:                      # [B, 1, NUM_Q_HEADS, HEAD_DIM]
    return mate.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
    )
