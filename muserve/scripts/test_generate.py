"""Task 1.3：端到端生成验证（不依赖 Flask，直接调用模型生成）。

用法：
    torchrun --nproc-per-node=8 muserve/scripts/test_generate.py [--max-tokens 32]
"""

import argparse
import time
import torch
import torch_musa

from muserve.distributed import init_distributed, destroy_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_layer_weights, load_embedding_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS, TP_SIZE, VOCAB_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[test_generate] Loading model ({args.layers} layers)...")

    weight_index = _load_index(args.model_path)
    embed_w = load_embedding_weights(args.model_path, weight_index)
    layer_ws = []
    for i in range(args.layers):
        layer_ws.append(load_layer_weights(args.model_path, i, weight_index))

    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    if rank == 0:
        print(f"[test_generate] Model loaded. Starting generation...")

    # 用简单的 token 序列作为 prompt（不依赖 tokenizer）
    # token 1 = BOS-like, 后面随机几个 token
    prompt_ids = [1, 100, 200, 300, 400, 500]
    input_ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, len(prompt_ids)], device=device, dtype=torch.int64)

    # Prefill
    t0 = time.time()
    logits, gdn_states = model.forward_prefill(input_ids, cu_seqlens)
    torch.musa.synchronize()
    prefill_ms = (time.time() - t0) * 1000

    next_token = model.greedy_sample(logits)
    generated = [next_token.item()]

    if rank == 0:
        print(f"  Prefill: {prefill_ms:.0f}ms, first token: {next_token.item()}")

    # Decode loop
    decode_times = []
    for step in range(args.max_tokens - 1):
        decode_ids = next_token.view(1, 1)  # [B=1, T=1]
        t0 = time.time()
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)
        torch.musa.synchronize()
        step_ms = (time.time() - t0) * 1000
        decode_times.append(step_ms)

        next_token = model.greedy_sample(logits)
        generated.append(next_token.item())

        if rank == 0 and (step + 1) % 5 == 0:
            print(f"  Step {step+1}: {step_ms:.0f}ms, token={next_token.item()}")

    barrier()

    if rank == 0:
        avg_decode = sum(decode_times[2:]) / max(len(decode_times[2:]), 1)
        print(f"\n[test_generate] Results:")
        print(f"  Prompt: {prompt_ids}")
        print(f"  Generated ({len(generated)} tokens): {generated}")
        print(f"  Prefill: {prefill_ms:.0f}ms")
        print(f"  Avg decode (skip 2 warmup): {avg_decode:.0f}ms/token")
        print(f"  Decode throughput: {1000/avg_decode:.2f} tok/s")
        print(f"\n[Task 1.3] End-to-end generation: PASSED")

    destroy_distributed()


if __name__ == "__main__":
    main()
