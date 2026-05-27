"""Layer-by-layer prefill diagnosis: run N layers and check logits quality."""
import sys; sys.path.insert(0, "/workspace")
import torch, torch_musa
import torch.distributed as dist
from transformers import AutoTokenizer
from muserve.distributed import init_distributed, get_tp_rank, barrier
from muserve.loader import load_embedding_weights, load_layer_weights, _load_index
from muserve.config import DEFAULT_MODEL_PATH, NUM_LAYERS, VOCAB_SIZE, TP_SIZE
from muserve.model.qwen35_layer import rms_norm, layer_forward_prefill
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

    prompt = "2+2="
    input_ids = tokenizer.encode(prompt)
    ids_flat = torch.tensor(input_ids, device=device)
    cu = torch.tensor([0, len(input_ids)], device=device, dtype=torch.int64)

    if rank == 0:
        print(f"Prompt: {prompt!r} ids={input_ids}", flush=True)
        print(f"Target: token 19 ('4'), vocab_per_rank={vocab_per_rank}", flush=True)
        print(f"NUM_LAYERS={NUM_LAYERS}", flush=True)

    # Run prefill layer by layer, check logits at checkpoints
    hidden = model.embed(ids_flat.unsqueeze(0)).squeeze(0)  # [total, HIDDEN]
    barrier()

    if rank == 0:
        print(f"\nAfter embed: hidden shape={hidden.shape} "
              f"norm={hidden.norm():.2f} min={hidden.min():.4f} max={hidden.max():.4f}", flush=True)

    checkpoints = list(range(1, 61))  # every layer
    for i in range(NUM_LAYERS):
        hidden, _, _ = layer_forward_prefill(
            hidden, cu, model.layer_weights[i],
        )

        if (i + 1) in checkpoints:
            torch.musa.synchronize(); barrier()
            if rank == 0:
                h_norm = hidden.norm().item()
                h_max = hidden.abs().max().item()
                is_attn = "attn.q_proj.weight" in model.layer_weights[i]
                ltype = "ATTN" if is_attn else "GDN "
                print(
                    f"Layer {i+1:2d} [{ltype}]: h_norm={h_norm:.2f} h_max={h_max:.3f}",
                    flush=True,
                )

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
