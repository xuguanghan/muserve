"""Test fused RMSNorm + Linear kernel correctness."""
import torch
import torch_musa
import sys
sys.path.insert(0, "/workspace")
from muserve.kernels.fused_rmsnorm_linear import fused_rmsnorm_linear

M, K, N = 8, 4096, 512
print(f"Testing fused_rmsnorm_linear M={M}, K={K}, N={N}")

device = torch.device("musa:0")
x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
w_norm = torch.ones(K, device=device, dtype=torch.float32)
w_linear = torch.randn(N, K, device=device, dtype=torch.bfloat16)

out = fused_rmsnorm_linear(x, w_norm, w_linear)
print(f"Output shape: {out.shape}")

# Reference: RMSNorm(x) @ W^T (norm_weight=1)
x_f = x.float()
rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)
x_norm = (x_f * rms).bfloat16()
ref = x_norm @ w_linear.T

diff = (out.float() - ref.float()).abs().max().item()
ref_max = ref.float().abs().max().item()
rel = diff / ref_max if ref_max > 0 else diff
print(f"Max abs diff: {diff:.4f}, ref_max: {ref_max:.2f}, relative: {rel:.6f}")

if rel < 0.02:
    print("PASSED")
else:
    print(f"FAILED (relative diff {rel:.4f} > 0.02)")
    raw_ref = x.bfloat16() @ w_linear.T
    if raw_ref[0, 0] != 0:
        ratio = out.float()[0, 0].item() / raw_ref.float()[0, 0].item()
        print(f"  out[0,0]={out[0,0].item():.4f}")
        print(f"  raw_gemm[0,0]={raw_ref[0,0].item():.4f}")
        print(f"  ratio={ratio:.4f}")
        print(f"  expected rms_inv[0]={torch.rsqrt(x.float().pow(2).mean(-1)+1e-6)[0].item():.4f}")
