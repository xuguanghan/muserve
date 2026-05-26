"""Measure greedy_sample overhead outside MUSA Graph."""
import time, os, sys
import torch
import torch_musa

from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_embedding_weights, load_layer_weights
from muserve.model.qwen35_model import Qwen35Model
from muserve.model.graph_decode import GraphedDecodeStep
from muserve.config import DEFAULT_MODEL_PATH, VOCAB_SIZE, TP_SIZE


def main():
    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    weight_index = _load_index(DEFAULT_MODEL_PATH)
    embed_w = load_embedding_weights(DEFAULT_MODEL_PATH, weight_index)
    layer_ws = [load_layer_weights(DEFAULT_MODEL_PATH, i, weight_index) for i in range(10)]
    model = Qwen35Model(embed_w, layer_ws)
    barrier()

    B = 8
    all_ids = torch.randint(1, VOCAB_SIZE // TP_SIZE, (B * 4,), device=device)
    cu = torch.arange(0, (B + 1) * 4, 4, device=device, dtype=torch.int64)
    logits, gdn_states = model.forward_prefill(all_ids, cu)
    next_tokens = model.greedy_sample(logits)
    decode_ids = next_tokens.view(B, 1)

    for _ in range(2):
        logits, gdn_states = model.forward_decode(decode_ids, gdn_states)
        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()

    graphed = GraphedDecodeStep(model, batch_size=B, device=device)
    graphed.capture(decode_ids, gdn_states)
    torch.musa.synchronize()

    # Measure graph replay only
    N = 100
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(N):
        logits = graphed.run(decode_ids, gdn_states)
    torch.musa.synchronize()
    t_replay = (time.time() - t0) / N * 1000

    # Measure graph + greedy_sample
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(N):
        logits = graphed.run(decode_ids, gdn_states)
        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()
    t_full = (time.time() - t0) / N * 1000

    # Measure greedy_sample alone
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(N):
        next_tokens = model.greedy_sample(logits)
        decode_ids = next_tokens.view(B, 1)
    torch.musa.synchronize()
    t_sample = (time.time() - t0) / N * 1000

    if rank == 0:
        print(f"Graph replay only:  {t_replay:.2f} ms")
        print(f"Graph + sample:     {t_full:.2f} ms")
        print(f"Sample alone:       {t_sample:.2f} ms")
        print(f"Sample overhead:    {t_full - t_replay:.2f} ms")
        print(f"Replay tok/s:       {B / t_replay * 1000:.1f}")
        print(f"Full tok/s:         {B / t_full * 1000:.1f}")


if __name__ == "__main__":
    main()
