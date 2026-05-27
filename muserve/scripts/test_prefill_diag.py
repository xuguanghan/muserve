"""Diagnose prefill logits: print top-10 tokens from each rank."""
import sys, time; sys.path.insert(0, "/workspace")
import torch, torch_musa
import torch.distributed as dist
from transformers import AutoTokenizer
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS, VOCAB_SIZE, TP_SIZE
from muserve.model.qwen35_model import Qwen35Model

def main():
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
    vocab_per_rank = VOCAB_SIZE // TP_SIZE

    for prompt in ["2+2=", "The capital of France is"]:
        input_ids = tokenizer.encode(prompt)
        # B=1 single sequence to avoid cross-seq attention issues
        ids_flat = torch.tensor(input_ids, device=device)
        cu = torch.tensor([0, len(input_ids)], device=device, dtype=torch.int64)

        logits, _, _ = model.forward_prefill(ids_flat, cu)
        torch.musa.synchronize(); barrier()

        # Each rank: get top-5 local logits and their global token ids
        local_vals, local_idxs = logits[0].topk(5)
        global_idxs = local_idxs + rank * vocab_per_rank

        # Print from each rank
        for r in range(TP_SIZE):
            barrier()
            if rank == r:
                tokens_info = []
                for j in range(5):
                    tid = global_idxs[j].item()
                    val = local_vals[j].item()
                    text = tokenizer.decode([tid])
                    tokens_info.append(f"{tid}({text!r})={val:.3f}")
                print(f"  rank{r} top5: {', '.join(tokens_info)}", flush=True)
            barrier()

        # Check expected tokens (only rank 0 since expected tokens are in rank 0 shard)
        if rank == 0:
            print(f"\n=== Prompt: {prompt!r} ===", flush=True)
            print(f"Logits shape={logits.shape} range=[{logits.min():.3f}, {logits.max():.3f}]", flush=True)
            expected = {"2+2=": [19, 220, 18, 20], "The capital of France is": [11751, 220, 8923, 279]}
            for tid in expected.get(prompt, []):
                if tid < vocab_per_rank:
                    val = logits[0, tid].item()
                    text = tokenizer.decode([tid])
                    print(f"  token {tid} ({text!r}): logit={val:.4f}", flush=True)

    barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
