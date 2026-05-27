"""Correctness test: short prompt with attention KV cache (eager, no graph).

Verifies the fix for the attention-layer-decode-without-KV-cache bug.
Before fix: all 20 decode tokens collapse to token 220 (space).
After fix: model should produce sensible output containing "4" or coherent text.
"""
import sys; sys.path.insert(0, "/workspace")
import torch, torch_musa
from transformers import AutoTokenizer
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS, VOCAB_SIZE, TP_SIZE
from muserve.model.qwen35_model import Qwen35Model

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)
wi = _load_index(DEFAULT_MODEL_PATH)
model = Qwen35Model(
    load_embedding_weights(DEFAULT_MODEL_PATH, wi),
    [load_layer_weights(DEFAULT_MODEL_PATH, i, wi) for i in range(NUM_LAYERS)],
)
barrier()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
prompt = "2+2="
input_ids = tokenizer.encode(prompt)
seq_len = len(input_ids)

if rank == 0:
    print(f"Prompt: '{prompt}' ({seq_len} tokens)", flush=True)

# Prefill: B=8, short sequences
B = 8
ids_flat = torch.tensor(input_ids * B, device=device)
cu = torch.arange(0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int64)
logits, gdn, kv_caches = model.forward_prefill(ids_flat, cu)
torch.musa.synchronize(); barrier()

if rank == 0:
    print(
        f"Prefill logits: min={logits.min():.4f} max={logits.max():.4f} "
        f"nan={logits.isnan().any().item()}",
        flush=True,
    )
    # Report which layers have KV cache
    attn_layer_idx = [i for i, kv in enumerate(kv_caches) if kv is not None]
    print(f"Attention layers with KV cache: {attn_layer_idx}", flush=True)
    if attn_layer_idx:
        k0, v0 = kv_caches[attn_layer_idx[0]]
        print(f"  layer {attn_layer_idx[0]} K shape={tuple(k0.shape)} V shape={tuple(v0.shape)}", flush=True)

# Decode 20 tokens (eager, no graph)
next_tokens = model.greedy_sample(logits)
generated = [next_tokens[0].item()]
decode_ids = next_tokens.view(B, 1)
for step in range(20):
    logits, gdn, kv_caches = model.forward_decode(decode_ids, gdn, kv_caches=kv_caches)
    next_tokens = model.greedy_sample(logits)
    tok_id = next_tokens[0].item()
    generated.append(tok_id)
    decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()
    if rank == 0:
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
        print(f"  step {step+1}: token={tok_id} -> '{tok_text}'", flush=True)

output = tokenizer.decode(generated, skip_special_tokens=True)
if rank == 0:
    print(f"\nGenerated: '{output}'", flush=True)
    unique_tokens = len(set(generated))
    print(f"Unique tokens: {unique_tokens}/21", flush=True)
    if "4" in output or "four" in output.lower():
        print("PASSED: output contains '4'", flush=True)
    elif unique_tokens <= 2:
        print(f"FAILED: degenerate output (only {unique_tokens} unique tokens)", flush=True)
    else:
        print(f"CHECK: output has {unique_tokens} unique tokens but no '4'", flush=True)
