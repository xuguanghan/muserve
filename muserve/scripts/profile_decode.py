"""Profile decode step: 精确测量 MoE 各操作耗时，确认瓶颈。"""
import argparse
import time
import torch
import torch_musa

from muserve.distributed import init_distributed, get_tp_rank, barrier, all_reduce
from muserve.loader import _load_index, load_layer_weights
from muserve.model.qwen35_layer import (
    rms_norm, moe_forward, fp8_linear, _fast_fp8_quantize,
)
from muserve.config import (
    DEFAULT_MODEL_PATH, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
    TP_SIZE, MOE_INTERMEDIATE,
)


def time_op(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.musa.synchronize()
    return (time.time() - t0) / repeat * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    init_distributed()
    rank = get_tp_rank()
    device = torch.device(f"musa:{rank}")

    idx = _load_index(DEFAULT_MODEL_PATH)
    w = load_layer_weights(DEFAULT_MODEL_PATH, 0, idx)
    barrier()

    B = args.batch
    hidden = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[PROFILE] Batch={B}, single layer MoE breakdown")
        print(f"{'='*60}\n")

    # 1. 完整 MoE forward
    t_moe = time_op(lambda: moe_forward(hidden, w))

    # 2. Gate (softmax + topk)
    gate_w = w["moe.gate.weight"]
    def gate_op():
        logits = torch.mm(hidden.float(), gate_w.float().t())
        scores = torch.softmax(logits, dim=-1)
        return torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)
    t_gate = time_op(gate_op)

    # 3. FP8 量化
    t_quant = time_op(lambda: _fast_fp8_quantize(hidden))

    # 4. 单次 fp8_linear (1 token × 1 expert)
    tok1 = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    w_e = w.get("moe.experts.gate_proj.weight")
    s_e = w.get("moe.experts.gate_proj.weight_scale_inv")
    if w_e is not None:
        t_1gemm = time_op(lambda: fp8_linear(tok1, w_e[0], s_e[0]))
    else:
        t_1gemm = -1

    # 5. 单次 fp8_linear (B tokens × 1 expert)
    if w_e is not None:
        t_bgemm = time_op(lambda: fp8_linear(hidden, w_e[0], s_e[0]))
    else:
        t_bgemm = -1

    # 6. ragged_m_moe_gemm_8bit (all experts, 1 call)
    from mate.deep_gemm import ragged_m_moe_gemm_8bit
    # 模拟：B tokens 各去 1 个 expert
    m_indices = torch.arange(B, device=device, dtype=torch.int32) % (NUM_EXPERTS // TP_SIZE)
    a_fp8, a_scale = _fast_fp8_quantize(hidden)
    out = torch.empty(B, MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
    def ragged_op():
        ragged_m_moe_gemm_8bit(
            (a_fp8, a_scale), (w_e, s_e), m_indices, out,
        )
    t_ragged = time_op(ragged_op)

    # 7. AllReduce
    dummy = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    t_ar = time_op(lambda: all_reduce(dummy.clone()))

    # 8. scatter_add
    indices = torch.randint(0, B, (B,), device=device).unsqueeze(-1).expand(B, HIDDEN_SIZE)
    src = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    output = torch.zeros(B, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    t_scatter = time_op(lambda: output.zero_().scatter_add_(0, indices, src))

    # 9. argsort
    ids = torch.randint(0, 64, (B * 10,), device=device, dtype=torch.int32)
    t_sort = time_op(lambda: ids.argsort(stable=True))

    if rank == 0:
        print(f"{'Operation':<45} {'Time (ms)':>10}")
        print("-" * 57)
        print(f"{'MoE forward (complete)':<45} {t_moe:>10.2f}")
        print(f"{'  Gate (mm + softmax + topk)':<45} {t_gate:>10.2f}")
        print(f"{'  FP8 quantize [B, 4096]':<45} {t_quant:>10.2f}")
        print(f"{'  fp8_linear [1, 4096]x[4096, 1024]':<45} {t_1gemm:>10.2f}")
        print(f"{'  fp8_linear [B, 4096]x[4096, 1024]':<45} {t_bgemm:>10.2f}")
        print(f"{'  ragged_m_moe_gemm_8bit [B experts]':<45} {t_ragged:>10.2f}")
        print(f"{'  AllReduce [B, 4096] bf16':<45} {t_ar:>10.2f}")
        print(f"{'  scatter_add [B, 4096]':<45} {t_scatter:>10.2f}")
        print(f"{'  argsort [B*10]':<45} {t_sort:>10.2f}")
        print()
        print(f"[PROFILE] 60 layers MoE estimated: {t_moe*60:.0f} ms")
        print(f"[PROFILE] Throughput (B={B}): {B/(t_moe*60/1000):.2f} tok/s")
        if t_1gemm > 0:
            n_active = min(B * NUM_EXPERTS_PER_TOK * (NUM_EXPERTS//TP_SIZE) // NUM_EXPERTS, NUM_EXPERTS//TP_SIZE)
            print(f"[PROFILE] Active experts/layer: ~{n_active}")
            print(f"[PROFILE] If loop {n_active}×3 fp8_linear: {t_1gemm*n_active*3:.0f} ms")
            print(f"[PROFILE] If 3× ragged_gemm: {t_ragged*3:.0f} ms")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
