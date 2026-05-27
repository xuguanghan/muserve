"""Fused RMSNorm + Linear kernel (tilelang).

Uses GEMV-style per-thread accumulation (proven pattern from tilelang examples).
Each thread computes one output element, accumulates full dot product over K.
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def fused_rmsnorm_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_N: int = 128,
    block_K: int = 128,
    eps: float = 1e-6,
):
    dtype = T.bfloat16
    acc_dtype = T.float

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W_linear: T.Tensor((N, K), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), M, threads=block_N) as (bn, bm):
            tn = T.get_thread_binding(0)

            # Each thread computes Out[bm, bn*block_N + tn]
            X_shared = T.alloc_shared((block_K,), dtype)
            W_local = T.alloc_local((1,), dtype)
            acc_val = T.alloc_local((1,), acc_dtype)
            sq_sum = T.alloc_local((1,), acc_dtype)

            T.clear(acc_val)
            T.clear(sq_sum)

            n_idx = bn * block_N + tn

            for bk in T.serial(T.ceildiv(K, block_K)):
                # Load X row tile into shared memory
                for tk in T.serial(block_K):
                    X_shared[tk] = X[bm, bk * block_K + tk]
                # Accumulate dot product and x^2
                for tk in T.serial(block_K):
                    x_val = X_shared[tk].astype(acc_dtype)
                    sq_sum[0] += x_val * x_val
                    W_local[0] = W_linear[n_idx, bk * block_K + tk]
                    acc_val[0] += x_val * W_local[0].astype(acc_dtype)

            # Apply RMS normalization
            rms_inv = T.rsqrt(sq_sum[0] / K + eps)
            Out[bm, n_idx] = (acc_val[0] * rms_inv).astype(dtype)

    return main


def fused_rmsnorm_linear(x, norm_weight, w_linear, eps=1e-6):
    """Fused RMSNorm + Linear.

    Args:
        x: [M, K] bf16
        norm_weight: [K] f32
        w_linear: [N, K] bf16
    Returns:
        [M, N] bf16 = RMSNorm(x, norm_weight) @ w_linear^T
    """
    M, K = x.shape
    N = w_linear.shape[0]
    w_scaled = w_linear * norm_weight.unsqueeze(0).to(w_linear.dtype)
    kernel = fused_rmsnorm_linear_kernel(M, K, N, block_N=128, block_K=128)
    return kernel(x, w_scaled)
