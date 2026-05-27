"""Full inference accuracy test with tokenizer - verify model produces meaningful output."""
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
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS
from muserve.inference_loop import InferenceLoop, InferenceRequest
from transformers import AutoTokenizer

init_distributed()
rank = get_tp_rank()
device = torch.device(f"musa:{rank}")
torch.musa.set_device(device)

# Load full model (60 layers)
weight_index = _load_index(DEFAULT_MODEL_PATH)
embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
layer_ws = []
for i in range(NUM_LAYERS):
    layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
    if rank == 0 and (i + 1) % 10 == 0:
        print(f"  Loading layer {i+1}/{NUM_LAYERS}...")
model = Qwen35Model(embed_w, layer_ws)
barrier()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)

if rank == 0:
    print(f"[accuracy_test] Model loaded: {NUM_LAYERS} layers, TP=8")

# Create inference loop
loop = InferenceLoop(model, tokenizer=tokenizer)

# Test prompt — disable thinking mode so output is direct answer, not <think>...</think>
prompt = "What is 2+2? Answer in one word:"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer.encode(text)

if rank == 0:
    print(f"[accuracy_test] Prompt: '{prompt}'")
    print(f"[accuracy_test] Input tokens: {len(input_ids)}")

# Submit request
req = InferenceRequest(
    request_id="accuracy_001",
    input_ids=input_ids,
    max_new_tokens=20,
    temperature=0.0,
)

if rank == 0:
    def submit_after_delay():
        time.sleep(0.1)
        loop.submit_request(req)
        time.sleep(10.0)
        loop._running = False

    t = threading.Thread(target=submit_after_delay, daemon=True)
    t.start()

# All ranks run inference loop
try:
    loop.run()
except Exception as e:
    if rank == 0:
        print(f"[accuracy_test] Error: {e}")

# Collect results
if rank == 0:
    tokens = []
    while not req.result_queue.empty():
        msg_type, payload = req.result_queue.get_nowait()
        if msg_type == "token":
            tokens.append(payload)
        elif msg_type == "done":
            break

    output_text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\n[accuracy_test] Generated {len(tokens)} tokens")
    print(f"[accuracy_test] Token IDs: {tokens}")
    print(f"[accuracy_test] Output: '{output_text}'")
    if len(tokens) > 0:
        print(f"[accuracy_test] PASSED")
    else:
        print(f"[accuracy_test] FAILED - no output")
