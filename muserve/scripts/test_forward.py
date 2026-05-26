"""Task 1.1 验证脚本：完整 60 层 forward（TP=8）。

用法：
    torchrun --nproc-per-node=8 muserve/scripts/test_forward.py [--seqlen N] [--batch N]
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
    parser.add_argument("--seqlen", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--layers", type=int, default=NUM_LAYERS,
                        help="Number of layers to test (default: all 60)")
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[test_forward] batch={args.batch}, seqlen={args.seqlen}, "
              f"layers={args.layers}, TP={TP_SIZE}")
        print(f"[test_forward] Loading weights...")

    t0 = time.time()
    weight_index = _load_index(args.model_path)
    embed_w = load_embedding_weights(args.model_path, weight_index)
    layer_ws = []
    for i in range(args.layers):
        layer_ws.append(load_layer_weights(args.model_path, i, weight_index))
        if rank == 0 and (i + 1) % 10 == 0:
            used = torch.musa.memory_allocated(device) / 1e9
            print(f"  layer {i+1}/{args.layers}, GPU mem: {used:.1f} GB")

    if rank == 0:
        print(f"[test_forward] Weights loaded in {time.time()-t0:.1f}s")

    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    # ── Prefill test ──────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n[test_forward] Prefill: batch={args.batch}, seqlen={args.seqlen}")

    B, T = args.batch, args.seqlen
    input_ids = torch.randint(0, VOCAB_SIZE // TP_SIZE, (B * T,), device=device)
    cu_seqlens = torch.tensor(
        [i * T for i in range(B + 1)], dtype=torch.int64, device=device
    )

    t0 = time.time()
    logits, gdn_states = model.forward_prefill(input_ids, cu_seqlens)
    torch.musa.synchronize()
    prefill_ms = (time.time() - t0) * 1000

    assert logits.shape == (B, VOCAB_SIZE // TP_SIZE), \
        f"logits shape mismatch: {logits.shape}"
    assert not logits.isnan().any(), "NaN in prefill logits"
    assert len(gdn_states) == args.layers

    if rank == 0:
        print(f"  ✓ logits={logits.shape}, no NaN")
        print(f"  ✓ prefill time: {prefill_ms:.1f} ms ({T} tokens × {B} seqs)")
        ttft_s = prefill_ms / 1000
        print(f"  TTFT estimate: {ttft_s:.2f}s (extrapolated for 32K: "
              f"{ttft_s * 32768 / T:.1f}s)")

    # ── Decode test ───────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n[test_forward] Decode: batch={args.batch}, 10 steps")

    decode_ids = torch.randint(0, VOCAB_SIZE // TP_SIZE, (B, 1), device=device)
    current_states = gdn_states

    step_times = []
    for step in range(10):
        t0 = time.time()
        logits, current_states = model.forward_decode(decode_ids, current_states)
        torch.musa.synchronize()
        step_ms = (time.time() - t0) * 1000
        step_times.append(step_ms)

        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.unsqueeze(1)

        assert not logits.isnan().any(), f"NaN in decode logits at step {step}"

    if rank == 0:
        avg_ms = sum(step_times[2:]) / len(step_times[2:])  # skip warmup
        tok_per_s = 1000 / avg_ms * B
        print(f"  ✓ 10 decode steps, no NaN")
        print(f"  avg step time (skip 2 warmup): {avg_ms:.1f} ms")
        print(f"  decode throughput (batch={B}): {tok_per_s:.1f} tok/s")
        print(f"  single-stream (batch=1): {1000/avg_ms:.1f} tok/s")

    barrier()

    if rank == 0:
        print(f"\n[Task 1.1] Full {args.layers}-layer forward: PASSED")

    destroy_distributed()


if __name__ == "__main__":
    main()
