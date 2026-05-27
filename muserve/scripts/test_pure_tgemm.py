"""Test pure T.gemm on S5000 (no RMS, isolate T.gemm compatibility)."""
import torch
import torch_musa
import time
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def pure_gemm(M, N, K, block_M=32, block_N=64, block_K=128):
    dtype = T.bfloat16
    acc_dtype = T.float

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_fragment((block_M, block_N), acc_dtype)

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)

            T.copy(acc, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    device = torch.device("musa:0")

    for M in [32, 64, 128]:
        K, N = 4096, 512
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        w_t = w.T.contiguous()

        try:
            kernel = pure_gemm(M, N, K, block_M=min(M, 32), block_N=64, block_K=128)
            out = kernel(a, w_t)
            ref = a @ w_t
            rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
            correct = "PASS" if rel < 0.02 else "FAIL"

            # Benchmark
            for _ in range(10):
                kernel(a, w_t)
            torch.musa.synchronize()
            R = 500
            torch.musa.synchronize()
            t0 = time.time()
            for _ in range(R):
                kernel(a, w_t)
            torch.musa.synchronize()
            t_tl = (time.time() - t0) / R * 1000

            # Compare with F.linear
            for _ in range(10):
                torch.nn.functional.linear(a, w)
            torch.musa.synchronize()
            t0 = time.time()
            for _ in range(R):
                torch.nn.functional.linear(a, w)
            torch.musa.synchronize()
            t_torch = (time.time() - t0) / R * 1000

            print(f"M={M}: {correct} (rel={rel:.5f}), tilelang={t_tl:.4f}ms, torch={t_torch:.4f}ms, ratio={t_tl/t_torch:.2f}x")
        except Exception as e:
            err = str(e).split('\n')[-1][:100]
            print(f"M={M}: COMPILE FAILED: {err}")
