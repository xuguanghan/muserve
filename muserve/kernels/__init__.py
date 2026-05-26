"""muserve kernel layer.

Kernel selection strategy:
  - Triton:   standalone ops where S5000 hardware primitives (SQMMA) matter
              fp8_matmul, sparse_attn
  - tilelang: fused ops and persistent kernels where cross-op fusion matters
              fused_decode (RMSNorm+Linear), fused_moe_gate, persistent_layer
  - mate:     pre-built kernels already tuned for S5000
              gdn (GatedDeltaNet), attention (flash_attn)
"""

from muserve.kernels.fp8_matmul import w8a8_block_fp8_matmul
from muserve.kernels.sparse_attn import sparse_attn
from muserve.kernels.gdn import gdn_decode, gdn_prefill
from muserve.kernels.attention import attn_prefill, attn_decode

__all__ = [
    # Triton — ported from FlagGems _mthreads backend
    "w8a8_block_fp8_matmul",   # W8A8 block FP8 GEMM, SQMMA path for S5000
    "sparse_attn",             # top-k sparse attention with sink, SQMMA

    # mate wrappers
    "gdn_decode",              # GatedDeltaNet single-step decode
    "gdn_prefill",             # GatedDeltaNet chunked prefill
    "attn_prefill",            # flash attention varlen (prefill)
    "attn_decode",             # flash attention with KV cache (decode)
]
