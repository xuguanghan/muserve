"""Test scatter_add alternatives with realistic MoE data flow."""
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
num_expanded = total * topk
experts_per_rank = 64
TP_SIZE = 8
NUM_EXPERTS = 512

# Simulate realistic MoE routing
logits = torch.randn(total, NUM_EXPERTS, device=device)
scores = torch.softmax(logits, dim=-1)
topk_w, topk_ids = torch.topk(scores, topk, dim=-1)
topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

flat_ids = topk_ids.view(-1)
flat_weights = topk_w.view(-1)
flat_token_idx = torch.arange(total, device=device).unsqueeze(1).expand(total, topk).reshape(-1)

local_mask = (flat_ids % TP_SIZE == rank)
local_expert_ids = flat_ids // TP_SIZE
local_expert_ids = torch.where(local_mask, local_expert_ids, torch.full_like(local_expert_ids, -1))
flat_weights = torch.where(local_mask, flat_weights, torch.zeros_like(flat_weights))

sort_order = local_expert_ids.argsort(stable=True)
sorted_expert_ids = local_expert_ids[sort_order]
sorted_token_idx = flat_token_idx[sort_order]
sorted_weights = flat_weights[sort_order]

# Simulate GEMM output
down_out = torch.randn(num_expanded, HIDDEN, device=device, dtype=torch.bfloat16)

# Precompute
idx_expanded = sorted_token_idx.unsqueeze(-1).expand(num_expanded, HIDDEN)
inverse_order = torch.empty_like(sort_order)
inverse_order[sort_order] = torch.arange(num_expanded, device=device)

R = 30

# Warmup
for _ in range(5):
    w = down_out * sorted_weights.unsqueeze(-1)
    o = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    o.scatter_add_(0, idx_expanded, w)
torch.musa.synchronize()

# Method 1: Current (weighted * scatter_add)
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    weighted = down_out * sorted_weights.unsqueeze(-1)
    output1 = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    output1.scatter_add_(0, idx_expanded, weighted)
torch.musa.synchronize()
t1 = (time.time() - t0) / R * 1000

# Method 2: unsort + reshape + weighted sum (bmm style)
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    unsorted = down_out[inverse_order]
    reshaped = unsorted.reshape(total, topk, HIDDEN)
    w_reshaped = flat_weights.reshape(total, topk, 1)  # original order weights
    output2 = (reshaped * w_reshaped).sum(dim=1)
torch.musa.synchronize()
t2 = (time.time() - t0) / R * 1000

# Method 3: index_put with accumulate (alternative to scatter_add)
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    weighted = down_out * sorted_weights.unsqueeze(-1)
    output3 = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    output3.index_add_(0, sorted_token_idx, weighted)
torch.musa.synchronize()
t3 = (time.time() - t0) / R * 1000

# Method 4: segment_reduce if available
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    # Use scatter with reduce='sum' (might be same as scatter_add)
    weighted = down_out * sorted_weights.unsqueeze(-1)
    output4 = torch.zeros(total, HIDDEN, device=device, dtype=torch.bfloat16)
    output4.scatter_(0, idx_expanded, weighted, reduce='add')
torch.musa.synchronize()
t4 = (time.time() - t0) / R * 1000

if rank == 0:
    print(f"=== scatter_add alternatives (realistic MoE, {total} tokens) ===")
    print(f"  M1 scatter_add:          {t1:.2f} ms")
    print(f"  M2 unsort+reshape+sum:   {t2:.2f} ms  ({t1/t2:.2f}x)")
    print(f"  M3 index_add:            {t3:.2f} ms  ({t1/t3:.2f}x)")
    print(f"  M4 scatter reduce=add:   {t4:.2f} ms  ({t1/t4:.2f}x)")
