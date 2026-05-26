"""Profile Scatter + AllReduce sub-components."""
import torch
import torch_musa
import time
import sys

sys.path.insert(0, "/workspace")
from muserve.distributed import init_distributed, get_tp_rank, barrier

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)
barrier()

total, topk, HIDDEN = 4096, 10, 4096
num_expanded = total * topk  # 40960

down_out = torch.randn(num_expanded, HIDDEN, device=device, dtype=torch.bfloat16)
sorted_weights = torch.randn(num_expanded, device=device, dtype=torch.bfloat16)
sorted_token_idx = torch.randint(0, total, (num_expanded,), device=device)
idx_expanded = sorted_token_idx.unsqueeze(-1).expand(num_expanded, HIDDEN)

R = 20

# Warmup
for _ in range(3):
    w = down_out * sorted_weights.unsqueeze(-1)
    o = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    o.scatter_add_(0, idx_expanded, w)
    torch.distributed.all_reduce(o)
torch.musa.synchronize()
barrier()

# 1. Weighted multiply
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    weighted = down_out * sorted_weights.unsqueeze(-1)
torch.musa.synchronize()
t_mul = (time.time() - t0) / R * 1000

# 2. scatter_add
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    output = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    output.scatter_add_(0, idx_expanded, weighted)
torch.musa.synchronize()
t_scatter = (time.time() - t0) / R * 1000

# 3. AllReduce
output2 = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
barrier()
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    torch.distributed.all_reduce(output2)
torch.musa.synchronize()
t_ar = (time.time() - t0) / R * 1000

if rank == 0:
    tt = t_mul + t_scatter + t_ar
    print(f"=== Scatter+AllReduce breakdown ===")
    print(f"  weighted mul:  {t_mul:.2f} ms ({t_mul/tt*100:.0f}%)")
    print(f"  scatter_add:   {t_scatter:.2f} ms ({t_scatter/tt*100:.0f}%)")
    print(f"  AllReduce:     {t_ar:.2f} ms ({t_ar/tt*100:.0f}%)")
    print(f"  TOTAL:         {tt:.2f} ms")
    print(f"  Data size:     {total*HIDDEN*2/1024/1024:.1f} MB")
