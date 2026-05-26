"""Batch decode benchmark: 验证 batch=8 时 ragged_m_moe_gemm_8bit 的加速效果。

不需要完整 continuous batching，直接模拟 batch=8 的 decode step。
"""
import argparse
import time
import torch
import torch_musa
import torch.distributed as dist

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_embedding_weights, load_layer_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.config import DEFAULT_MODEL_PATH, VOCAB_SIZE, TP_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layers", type=int, default=60)
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[bench_batch_decode] batch={args.batch}, steps={args.steps}, layers={args.layers}")
        print(f"[bench_batch_decode] Loading model...")

    weight_index = _load_index(args.model_path if hasattr(args, 'model_path') else DEFAULT_MODEL_PATH)
    model_path = DEFAULT_MODEL_PATH
    embed_w = load_embedding_weights(model_path, weight_index)
    layer_ws = []
    for i in range(args.layers):
        layer_ws.append(load_layer_weights(model_path, i, weight_index))

    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    if rank == 0:
        print(f"[bench_batch_decode] Model loaded. Running prefill...")

    B = args.batch
    seq_len = 4

    # Prefill: B 个短序列
    all_ids = torch.randint(1, VOCAB_SIZE // TP_SIZE, (B * seq_len,), device=device)
    cu_seqlens = torch.arange(0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int64)

    logits, gdn_states = model.forward_prefill(all_ids, cu_seqlens)
    torch.musa.synchronize()

    next_tokens = model.greedy_sample(logits)  # [B]

    if rank == 0:
        print(f"[bench_batch_decode] Prefill done. Starting batch decode...")

    # Warmup decode
    decode_ids = next_tokens.view(B, 1)
    for _ in range(2):
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)
        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()

    # Timed decode
    t0 = time.time()
    for step in range(args.steps):
        decode_ids = next_tokens.view(B, 1)
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)
        next_tokens = model.greedy_sample(logits)
    torch.musa.synchronize()
    elapsed = time.time() - t0

    total_tokens = args.steps * B
    tok_per_sec = total_tokens / elapsed
    ms_per_step = elapsed / args.steps * 1000

    if rank == 0:
        print(f"[bench_batch_decode] Results:")
        print(f"  Batch size: {B}")
        print(f"  Steps: {args.steps}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Throughput: {tok_per_sec:.2f} tok/s")
        print(f"  Latency: {ms_per_step:.0f} ms/step")
        if tok_per_sec >= 10:
            print(f"[Task A.3] Batch decode benchmark: PASSED (>= 10 tok/s)")
        else:
            print(f"[Task A.3] Batch decode benchmark: BELOW TARGET (< 10 tok/s)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
