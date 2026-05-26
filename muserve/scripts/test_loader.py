"""Task 0.2 验证脚本：FP8 权重加载 + TP sharding 验证。

用法：
    torchrun --nproc-per-node=8 muserve/scripts/test_loader.py [--layers N]
"""

import argparse
import torch
import torch_musa
from muserve.distributed import init_distributed, destroy_distributed, get_tp_rank, barrier
from muserve.loader import _load_index, load_layer_weights, load_embedding_weights
from muserve.config import (
    DEFAULT_MODEL_PATH, TP_SIZE, HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
    HEAD_DIM, NUM_EXPERTS, MOE_INTERMEDIATE,
)


def check_shape(name, tensor, expected_shape, rank):
    actual = tuple(tensor.shape)
    ok = actual == tuple(expected_shape)
    if not ok:
        print(f"[rank {rank}] SHAPE MISMATCH {name}: got {actual}, expected {expected_shape}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of layers to load and verify")
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    if rank == 0:
        print(f"[test_loader] model={args.model_path}, layers={args.layers}, TP={TP_SIZE}")

    weight_index = _load_index(args.model_path)
    all_ok = True

    # ── Embedding ──────────────────────────────────────────────────────────────
    if rank == 0:
        print("[test_loader] Loading embedding weights...")
    embed_w = load_embedding_weights(args.model_path, weight_index)

    from muserve.config import VOCAB_SIZE
    vocab_per_rank = VOCAB_SIZE // TP_SIZE
    for name, expected in [
        ("model.embed_tokens.weight", (vocab_per_rank, HIDDEN_SIZE)),
        ("lm_head.weight",            (vocab_per_rank, HIDDEN_SIZE)),
        ("model.norm.weight",         (HIDDEN_SIZE,)),
    ]:
        if name in embed_w:
            ok = check_shape(name, embed_w[name], expected, rank)
            all_ok = all_ok and ok
            if rank == 0 and ok:
                print(f"  ✓ {name}: {tuple(embed_w[name].shape)}")

    # ── Layers ─────────────────────────────────────────────────────────────────
    q_per_rank  = (NUM_Q_HEADS  * HEAD_DIM) // TP_SIZE
    kv_per_rank = (NUM_KV_HEADS * HEAD_DIM) // TP_SIZE
    experts_per_rank = NUM_EXPERTS // TP_SIZE

    for layer_idx in range(args.layers):
        if rank == 0:
            print(f"[test_loader] Loading layer {layer_idx}...")
        w = load_layer_weights(args.model_path, layer_idx, weight_index)

        checks = [
            # Attention projections
            ("attn.q_proj.weight", (q_per_rank,  HIDDEN_SIZE)),
            ("attn.k_proj.weight", (kv_per_rank, HIDDEN_SIZE)),
            ("attn.v_proj.weight", (kv_per_rank, HIDDEN_SIZE)),
            ("attn.o_proj.weight", (HIDDEN_SIZE, q_per_rank)),
            # MoE gate (full, not sharded)
            ("moe.gate.weight",    (NUM_EXPERTS, HIDDEN_SIZE)),
            # Expert 0 on this rank
            ("moe.expert_0.gate_proj.weight", (MOE_INTERMEDIATE, HIDDEN_SIZE)),
            ("moe.expert_0.up_proj.weight",   (MOE_INTERMEDIATE, HIDDEN_SIZE)),
            ("moe.expert_0.down_proj.weight", (HIDDEN_SIZE, MOE_INTERMEDIATE)),
        ]

        for name, expected in checks:
            if name not in w:
                if rank == 0:
                    print(f"  ⚠ {name}: NOT FOUND (may be optional)")
                continue
            ok = check_shape(name, w[name], expected, rank)
            all_ok = all_ok and ok
            if rank == 0 and ok:
                print(f"  ✓ layer {layer_idx} {name}: {tuple(w[name].shape)}")

        # 验证 expert 数量
        my_expert_count = sum(1 for k in w if k.startswith("moe.expert_") and k.endswith(".gate_proj.weight"))
        if my_expert_count != experts_per_rank:
            if rank == 0:
                print(f"  ✗ layer {layer_idx}: expected {experts_per_rank} experts/rank, got {my_expert_count}")
            all_ok = False
        elif rank == 0:
            print(f"  ✓ layer {layer_idx} experts/rank: {my_expert_count}")

        # 内存使用
        if rank == 0:
            used_gb = torch.musa.memory_allocated(device) / 1e9
            print(f"  GPU memory after layer {layer_idx}: {used_gb:.2f} GB")

    barrier()

    if rank == 0:
        status = "PASSED" if all_ok else "FAILED"
        print(f"\n[Task 0.2] Weight loading verification: {status}")

    destroy_distributed()


if __name__ == "__main__":
    main()
