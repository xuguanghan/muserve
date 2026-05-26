"""Benchmark: MUSA Graph captured decode vs eager decode."""
import argparse, time, os, sys
import torch
import torch_musa

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_embedding_weights, load_layer_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.graph_decode import GraphedDecodeStep
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
        print(f"[graph_bench] batch={args.batch}, steps={args.steps}, layers={args.layers}")
        print(f"[graph_bench] Loading model...")

    weight_index = _load_index(DEFAULT_MODEL_PATH)
    embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
    layer_ws = []
    for i in range(args.layers):
        layer_ws.append(load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index))
    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    B = args.batch
    seq_len = 4

    # Prefill
    if rank == 0:
        print(f"[graph_bench] Running prefill...")
    all_ids = torch.randint(1, VOCAB_SIZE // TP_SIZE, (B * seq_len,), device=device)
    cu_seqlens = torch.arange(0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int64)
    logits, gdn_states = model.forward_prefill(all_ids, cu_seqlens)
    next_tokens = model.greedy_sample(logits)
    torch.musa.synchronize()

    # Eager warmup (2 steps)
    if rank == 0:
        print(f"[graph_bench] Eager warmup...")
    decode_ids = next_tokens.view(B, 1)
    for _ in range(2):
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)
        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()

    # Capture graph
    if rank == 0:
        print(f"[graph_bench] Capturing MUSA Graph...")
    graphed = GraphedDecodeStep(model, batch_size=B, device=device)
    graphed.capture(decode_ids, gdn_states)
    torch.musa.synchronize()
    if rank == 0:
        print(f"[graph_bench] Graph captured. Running benchmark...")

    # Graph warmup
    for _ in range(2):
        graphed.step()
    torch.musa.synchronize()

    # Timed graph decode (zero Python dispatch per step)
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        graphed.step()
    torch.musa.synchronize()
    t_graph = time.time() - t0

    graph_toks = args.steps * B / t_graph
    graph_ms = t_graph / args.steps * 1000

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"MUSA Graph Decode (B={B}, layers={args.layers})")
        print(f"{'='*50}")
        print(f"  Graph: {graph_toks:.2f} tok/s, {graph_ms:.1f} ms/step")
        print(f"{'='*50}")
        if graph_toks >= 100:
            print(f"  PASSED (>= 100 tok/s)")
        else:
            print(f"  Target: 100+ tok/s")


if __name__ == "__main__":
    main()
