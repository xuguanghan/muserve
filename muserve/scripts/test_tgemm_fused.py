"""Test T.gemm based fused RMSNorm+Linear with block_M>=16."""
import torch
import torch_musa
import time
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def rmsnorm_gemm_kernel(
    M: int, K: int, N: int,
    block_M: int = 32, block_N: int = 64, block_K: int = 128,
    eps: float = 1e-6,
):
    """Fused RMSNorm + GEMM using T.gemm (requires block_M >= 16)."""
    dtype = T.bfloat16
    acc_dtype = T.float

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),  # transposed weight (K, N)
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), acc_dtype)

            # RMS computation buffers
            x_sq_acc = T.alloc_fragment((block_M, block_K), acc_dtype)
            x_sq_sum = T.alloc_fragment((block_M,), acc_dtype)

            num_k = T.ceildiv(K, block_K)

            # Pass 1: compute RMS scale
            T.clear(x_sq_acc)
            for k in range(num_k):
                T.copy(X[by * block_M, k * block_K], X_shared)
                for i, j in T.Parallel(block_M, block_K):
                    x_sq_acc[i, j] += (
                        X_shared[i, j].astype(acc_dtype)
                        * X_shared[i, j].astype(acc_dtype)
                    )
            T.reduce_sum(x_sq_acc, x_sq_sum, dim=1)
            for i in T.Parallel(block_M):
                x_sq_sum[i] = T.rsqrt(x_sq_sum[i] / K + eps)

            # Pass 2: GEMM with T.gemm
            T.clear(acc)
            for k in T.Pipelined(num_k, num_stages=2):
                T.copy(X[by * block_M, k * block_K], X_shared)
                T.copy(W[k * block_K, bx * block_N], W_shared)
                T.gemm(X_shared, W_shared, acc)

            # Apply RMS scale
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] *= x_sq_sum[i]

            T.copy(acc, Out[by * block_M, bx * block_N])

    return main


def test_and_bench(M, K, N):
    device = torch.device("musa:0")
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    w_t = w.T.contiguous()  # (K, N) for T.gemm

    # Compile kernel
    block_M = min(M, 32)
    try:
        kernel = rmsnorm_gemm_kernel(M, K, N, block_M=block_M, block_N=64, block_K=128)
    except Exception as e:
        print(f"  M={M}: compile FAILED: {e}")
        return

    out = kernel(x, w_t)

    # Reference
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (x_f * rms).bfloat16() @ w.T
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    status = "PASS" if rel < 0.02 else "FAIL"
    print(f"  M={M}: correctness rel={rel:.6f} {status}")

    if status == "FAIL":
        return

    # Benchmark fused
    for _ in range(10):
        kernel(x, w_t)
    torch.musa.synchronize()
    R = 500
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(R):
        kernel(x, w_t)
    torch.musa.synchronize()
    t_fused = (time.time() - t0) / R * 1000

    # Benchmark separate
    w_norm_bf16 = torch.ones(K, device=device, dtype=torch.bfloat16)
    for _ in range(10):
        xn = torch.nn.functional.rms_norm(x, (K,), w_norm_bf16, 1e-6)
        torch.nn.functional.linear(xn, w)
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(R):
        xn = torch.nn.functional.rms_norm(x, (K,), w_norm_bf16, 1e-6)
        torch.nn.functional.linear(xn, w)
    torch.musa.synchronize()
    t_sep = (time.time() - t0) / R * 1000

    speedup = t_sep / t_fused
    print(f"  M={M}: fused={t_fused:.4f}ms, separate={t_sep:.4f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=== T.gemm fused RMSNorm+Linear benchmark ===")
    for M in [16, 32, 64, 128]:
        test_and_bench(M, 4096, 512)
