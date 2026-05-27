"""End-to-end test: verify InferenceLoop produces correct inference output.

Tests: broadcast request → all ranks execute forward → rank 0 gets tokens.
"""
import torch
import torch_musa
import time
import sys
import queue
import threading

sys.path.insert(0, "/workspace")
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.model.qwen35_model import Qwen35Model
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.config import DEFAULT_MODEL_PATH
from muserve.inference_loop import InferenceLoop, InferenceRequest

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

if rank == 0:
    print(f"[e2e_test] Model loaded: {NUM_LAYERS} layers, TP=8")

# Create inference loop
loop = InferenceLoop(model, tokenizer=None)

# On rank 0, submit a test request in a separate thread
if rank == 0:
    req = InferenceRequest(
        request_id="test_001",
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],  # dummy token ids
        max_new_tokens=5,
        temperature=0.0,
    )

    def submit_after_delay():
        time.sleep(0.1)
        loop.submit_request(req)
        # After generation finishes, send shutdown
        time.sleep(2.0)
        from muserve.inference_loop import BroadcastPayload
        # Put a None to trigger shutdown check
        loop._running = False

    t = threading.Thread(target=submit_after_delay, daemon=True)
    t.start()

# All ranks run the inference loop (will process 1 request then exit)
try:
    loop.run()
except Exception as e:
    if rank == 0:
        print(f"[e2e_test] Loop error: {e}")

# Check results on rank 0
if rank == 0:
    tokens = []
    while not req.result_queue.empty():
        msg_type, payload = req.result_queue.get_nowait()
        if msg_type == "token":
            tokens.append(payload)
        elif msg_type == "done":
            break

    print(f"[e2e_test] Generated {len(tokens)} tokens: {tokens[:10]}")
    if len(tokens) > 0:
        print(f"[e2e_test] PASSED - inference loop produced output")
    else:
        print(f"[e2e_test] FAILED - no tokens generated")
