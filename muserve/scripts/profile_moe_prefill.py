"""Fine-grained MoE prefill profiling: routing vs GEMM breakdown."""
import torch
import torch_musa
import time
import sys

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.qwen35_layer import (
    moe_forward, rms_norm, bf16_linear, _fast_fp8_quantize,
    _get_flat_token_idx, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
    MOE_INTERMEDIATE, HIDDEN_SIZE, TP_SIZE,
)
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = [load_layer_weights(DEFAULT_MODEL_PATH, 0, weight_index)]
barrier()

weights = layer_ws[0]
SEQ_LEN = 4096
hidden = torch.randn(SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

# Warmup
norm_w2 = weights["post_attention_layernorm.weight"]
x = rms_norm(hidden, norm_w2)
moe_forward(x, weights)
torch.musa.synchronize()
barrier()

if rank == 0:
    print(f"[moe_profile] seq={SEQ_LEN}, experts={NUM_EXPERTS}, topk={NUM_EXPERTS_PER_TOK}, TP={TP_SIZE}")
    print(f"[moe_profile] expanded tokens: {SEQ_LEN * NUM_EXPERTS_PER_TOK}")

# Now profile each component separately
from mate.deep_gemm import ragged_m_moe_gemm_8bit

total = SEQ_LEN
topk = NUM_EXPERTS_PER_TOK
experts_per_rank = NUM_EXPERTS // TP_SIZE
R = 10

def _get_rank(dev):
    return dev.index or 0

# --- 1. Gate routing ---
gate_w = weights["moe.gate.weight"]
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    logits = bf16_linear(x, gate_w).float()
    scores = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
torch.musa.synchronize()
t_routing = (time.time() - t0) / R * 1000

# --- 2. Token dispatch (expand + sort + gather) ---
flat_ids = topk_ids.view(-1)
flat_weights = topk_w.view(-1)
flat_token_idx = _get_flat_token_idx(total, topk, device)
local_mask = (flat_ids % TP_SIZE == rank)
local_expert_ids = flat_ids // TP_SIZE
local_expert_ids = torch.where(local_mask, local_expert_ids, torch.full_like(local_expert_ids, -1))
flat_weights_masked = torch.where(local_mask, flat_weights, torch.zeros_like(flat_weights))

torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    sort_order = local_expert_ids.argsort(stable=True)
    sorted_expert_ids = local_expert_ids[sort_order]
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_weights = flat_weights_masked[sort_order]
    expanded_hidden = x[sorted_token_idx]
torch.musa.synchronize()
t_dispatch = (time.time() - t0) / R * 1000

# --- 3. FP8 quantize ---
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    a_fp8, a_scale = _fast_fp8_quantize(expanded_hidden)
torch.musa.synchronize()
t_quant1 = (time.time() - t0) / R * 1000

# --- 4. Expert GEMMs (gate + up + down) ---
m_indices = sorted_expert_ids.to(torch.int32)
num_expanded = total * topk
w_gate = weights["moe.experts.gate_proj.weight"]
s_gate = weights["moe.experts.gate_proj.weight_scale_inv"]
w_up = weights["moe.experts.up_proj.weight"]
s_up = weights["moe.experts.up_proj.weight_scale_inv"]
w_down = weights["moe.experts.down_proj.weight"]
s_down = weights["moe.experts.down_proj.weight_scale_inv"]

gate_out = torch.empty(num_expanded, MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
up_out = torch.empty(num_expanded, MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
down_out = torch.empty(num_expanded, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    ragged_m_moe_gemm_8bit((a_fp8, a_scale), (w_gate, s_gate), m_indices, gate_out)
    ragged_m_moe_gemm_8bit((a_fp8, a_scale), (w_up, s_up), m_indices, up_out)
torch.musa.synchronize()
t_gemm_gate_up = (time.time() - t0) / R * 1000

# --- 5. SiLU + mul + quant ---
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    act = torch.nn.functional.silu(gate_out) * up_out
    act_fp8, act_scale = _fast_fp8_quantize(act)
torch.musa.synchronize()
t_act = (time.time() - t0) / R * 1000

# --- 6. Down GEMM ---
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    ragged_m_moe_gemm_8bit((act_fp8, act_scale), (w_down, s_down), m_indices, down_out)
torch.musa.synchronize()
t_gemm_down = (time.time() - t0) / R * 1000

# --- 7. Scatter + AllReduce ---
torch.musa.synchronize()
t0 = time.time()
for _ in range(R):
    weighted = down_out * sorted_weights.unsqueeze(-1)
    output = torch.zeros(total, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    output.scatter_add_(0, sorted_token_idx.unsqueeze(-1).expand_as(weighted), weighted)
    torch.distributed.all_reduce(output)
torch.musa.synchronize()
t_scatter = (time.time() - t0) / R * 1000

total_time = t_routing + t_dispatch + t_quant1 + t_gemm_gate_up + t_act + t_gemm_down + t_scatter

if rank == 0:
    print(f"\n=== MoE Prefill Breakdown (seq={SEQ_LEN}) ===")
    print(f"  1. Gate routing (linear+softmax+topk): {t_routing:.2f} ms ({t_routing/total_time*100:.0f}%)")
    print(f"  2. Token dispatch (sort+gather):       {t_dispatch:.2f} ms ({t_dispatch/total_time*100:.0f}%)")
    print(f"  3. FP8 quantize input:                 {t_quant1:.2f} ms ({t_quant1/total_time*100:.0f}%)")
    print(f"  4. Gate+Up GEMM (2x ragged):           {t_gemm_gate_up:.2f} ms ({t_gemm_gate_up/total_time*100:.0f}%)")
    print(f"  5. SiLU*mul + FP8 quant:               {t_act:.2f} ms ({t_act/total_time*100:.0f}%)")
    print(f"  6. Down GEMM (1x ragged):              {t_gemm_down:.2f} ms ({t_gemm_down/total_time*100:.0f}%)")
    print(f"  7. Scatter + AllReduce:                {t_scatter:.2f} ms ({t_scatter/total_time*100:.0f}%)")
    print(f"  TOTAL:                                 {total_time:.2f} ms")
    print(f"\n  GEMM total (4+6): {t_gemm_gate_up + t_gemm_down:.2f} ms ({(t_gemm_gate_up+t_gemm_down)/total_time*100:.0f}%)")
    print(f"  Non-GEMM total:   {total_time - t_gemm_gate_up - t_gemm_down:.2f} ms ({(total_time-t_gemm_gate_up-t_gemm_down)/total_time*100:.0f}%)")
