"""Debug GDN state shape/stride mismatch between prefill and decode."""
import torch
import torch_musa
import sys

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.qwen35_layer import gdn_decode_forward
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

NUM_LAYERS = 5
weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
model = Qwen35Model(embed_w, layer_ws)
barrier()

# Prefill with B=1
input_ids = torch.randint(0, 10000, (64,), device=device)
cu_seqlens = torch.tensor([0, 64], device=device, dtype=torch.int64)
logits, gdn_states = model.forward_prefill(input_ids, cu_seqlens)

if rank == 0:
    print(f"[debug] Prefill done. logits: {logits.shape}")
    for i, s in enumerate(gdn_states):
        if s is not None:
            print(f"  state[{i}]: shape={s.shape}, stride={s.stride()}, dtype={s.dtype}, contig={s.is_contiguous()}")
        else:
            print(f"  state[{i}]: None")

# Try decode with B=1
next_token = model.greedy_sample(logits)
decode_ids = next_token.view(1, 1)

if rank == 0:
    print(f"\n[debug] decode_ids: {decode_ids.shape}")
    print(f"[debug] Attempting decode...")

try:
    logits2, gdn_states2 = model.forward_decode(decode_ids, gdn_states)
    if rank == 0:
        print(f"[debug] Decode SUCCESS! logits2: {logits2.shape}")
except Exception as e:
    if rank == 0:
        print(f"[debug] Decode FAILED: {e}")

    # Try with contiguous states
    gdn_states_c = [s.contiguous() if s is not None else None for s in gdn_states]
    if rank == 0:
        for i, s in enumerate(gdn_states_c):
            if s is not None:
                print(f"  state_c[{i}]: shape={s.shape}, stride={s.stride()}, contig={s.is_contiguous()}")

    try:
        logits3, _ = model.forward_decode(decode_ids, gdn_states_c)
        if rank == 0:
            print(f"[debug] Decode with contiguous: SUCCESS!")
    except Exception as e2:
        if rank == 0:
            print(f"[debug] Decode with contiguous: FAILED: {e2}")
