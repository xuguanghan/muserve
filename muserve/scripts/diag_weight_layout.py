"""诊断 checkpoint 中权重的真实布局，确认 4 个候选根因。

每个根因对应一个独立的检查函数，输出可读结论。运行：
    docker exec sglang-musa5-dsv4 python3 /workspace/muserve/scripts/diag_weight_layout.py

不需要 torchrun，单进程即可。只读 checkpoint 元数据 + 部分权重。
"""
import sys
sys.path.insert(0, "/workspace")

import torch
from safetensors import safe_open
from pathlib import Path
import json

from muserve.config import (
    DEFAULT_MODEL_PATH, TP_SIZE,
    GDN_NUM_K_HEADS, GDN_NUM_V_HEADS, GDN_KEY_DIM, GDN_VALUE_DIM,
)

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 256


def _load_tensor(model_path: str, key: str) -> torch.Tensor:
    """直接从 safetensors 加载单个 tensor（不上 GPU）。"""
    idx_path = Path(model_path) / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    shard_file = idx[key]
    with safe_open(Path(model_path) / shard_file, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _segment_stats(t: torch.Tensor, segments: list[tuple[str, int]], dim: int = 0) -> list[dict]:
    """对 tensor 的 dim 维度按 segments 切分，统计每段 RMS/max/min/std。"""
    results = []
    offset = 0
    t_float = t.float()
    for name, size in segments:
        sub = t_float.narrow(dim, offset, size)
        results.append({
            "name": name,
            "range": (offset, offset + size),
            "rms": sub.pow(2).mean().sqrt().item(),
            "max": sub.abs().max().item(),
            "mean": sub.mean().item(),
            "std": sub.std().item(),
        })
        offset += size
    return results


def _fmt_stats(stats: list[dict], indent: str = "  ") -> str:
    lines = []
    for s in stats:
        lines.append(
            f"{indent}{s['name']:>10s} [{s['range'][0]:>5d}:{s['range'][1]:<5d}] "
            f"rms={s['rms']:.4f} max={s['max']:.4f} std={s['std']:.4f}"
        )
    return "\n".join(lines)


def diagnose_in_proj_qkv(model_path: str, layer_idx: int = 0):
    """根因 1: in_proj_qkv 真实布局是 [q|k|v] 拼接 还是 head-interleaved？

    Qwen3.5-397B GDN: num_k_heads=16, num_v_heads=64, head_dim=128
      hypothesis A (fused [q|k|v]):  [q_full(2048) | k_full(2048) | v_full(8192)]
      hypothesis B (head-interleaved): 每 head 占 (k_dim + v_dim) 段

    判定：fp8 weight 没法直接看，但 weight_scale_inv 是 [N//128, K//128] fp32，
          每段在 dim=0 上的 RMS/max 特征足以区分。
    """
    print("=" * 70)
    print("[根因 1] GDN in_proj_qkv 真实布局")
    print("=" * 70)

    key = f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight_scale_inv"
    try:
        s = _load_tensor(model_path, key)
    except KeyError:
        print(f"  ERROR: key not found: {key}")
        return
    print(f"  weight_scale_inv shape: {tuple(s.shape)} dtype={s.dtype}")
    # 期望 shape: [12288//128=96, 4096//128=32]
    assert s.shape[0] == 96, f"expected 96 row-blocks, got {s.shape[0]}"

    # 假设 A: [q(2048/128=16), k(16), v(8192/128=64)]
    seg_A = [("q_full", 16), ("k_full", 16), ("v_full", 64)]
    stats_A = _segment_stats(s, seg_A, dim=0)
    print("\n  假设 A: [q_full(2048) | k_full(2048) | v_full(8192)] 拼接")
    print(_fmt_stats(stats_A))

    # 假设 B: head-interleaved，每 v_head 占 (head_dim*2 for k+v?) ——
    # 但 K_heads=16, V_heads=64，无法简单 head-interleave 成统一 head 块。
    # 实际可能是 K_heads-grouped: 每组里 [q(k_heads_per_group) | k(k_heads_per_group) | v(v_heads_per_group)]
    # gqa_ratio = 64/16 = 4, 每组 1 q_head + 1 k_head + 4 v_heads = 6 * head_dim = 768 → 768/128=6 blocks
    # 16 组 × 768 = 12288 ✓
    # blocks_per_group = 6 (1 q + 1 k + 4 v) 每个 128 行
    seg_B = []
    for g in range(16):
        seg_B.append((f"g{g}_q", 1))
        seg_B.append((f"g{g}_k", 1))
        seg_B.append((f"g{g}_v", 4))
    stats_B = _segment_stats(s, seg_B, dim=0)
    print("\n  假设 B: K-head 分组交错 [q1|k1|v4] × 16 (GQA-style 16 groups)")
    # 只打印前 3 组对比
    print(_fmt_stats(stats_B[:18]))
    # 计算每种假设的 within-segment / cross-segment variance ratio
    def cross_var(stats):
        rmss = torch.tensor([s["rms"] for s in stats])
        return rmss.std().item() / (rmss.mean().item() + 1e-9)
    print(f"\n  假设 A cross-segment rms-CV = {cross_var(stats_A):.4f} (3 segments)")
    print(f"  假设 B cross-segment rms-CV = {cross_var(stats_B):.4f} (96 segments)")

    # 真正的判断：在 fused 假设 A 下，q_full 内的 16 个 head 块应该 RMS 相近；
    # k_full 同理；v_full 同理。如果 q_full 内部 16 块差异大，说明不是 fused。
    print("\n  ➜ 判定：")
    if stats_A[0]["rms"] > stats_A[1]["rms"] * 1.3 or stats_A[0]["rms"] < stats_A[1]["rms"] * 0.7 \
       or stats_A[2]["rms"] / stats_A[0]["rms"] > 1.3 or stats_A[2]["rms"] / stats_A[0]["rms"] < 0.7:
        print("    q/k/v 段 RMS 差异显著 → 强烈支持假设 A（fused [q|k|v]）")
    else:
        print("    q/k/v 段 RMS 相近，需要看每组内分布判断")

    # 看 q_full 内部 16 个 head 块的 RMS 变异
    q_blocks = s[:16].float()
    q_block_rms = [q_blocks[i].pow(2).mean().sqrt().item() for i in range(16)]
    print(f"  q_full 16 个 head 块 RMS: min={min(q_block_rms):.4f} max={max(q_block_rms):.4f} "
          f"std={torch.tensor(q_block_rms).std().item():.4f}")
    print(f"  v_full 64 个 head 块 RMS sample: " + ", ".join(f"{s[16+i].float().pow(2).mean().sqrt().item():.4f}" for i in [0, 16, 32, 48, 63]))


def diagnose_q_proj(model_path: str, layer_idx: int):
    """根因 2: self_attn.q_proj 真实布局是 [q_full|gate_full] 还是 head-interleaved？

    Qwen3.5 主层标准 attention: num_q_heads=32, head_dim=256, q_proj=[16384, 4096]
      hypothesis A (fused [q|gate]): [q_full(8192) | gate_full(8192)]
      hypothesis B (head-interleaved): [h0_q(256)|h0_gate(256)|h1_q|h1_gate|...]
    """
    print("\n" + "=" * 70)
    print(f"[根因 2] Layer {layer_idx} self_attn.q_proj 真实布局")
    print("=" * 70)

    # 标准 attention 只在某些层（每 4 层一个）。先尝试常见的 attention 层
    key_w = f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"
    key_s = f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight_scale_inv"
    try:
        s = _load_tensor(model_path, key_s)
    except KeyError:
        print(f"  Layer {layer_idx} 没有 self_attn.q_proj（可能不是 attention 层）")
        return False
    print(f"  weight_scale_inv shape: {tuple(s.shape)} dtype={s.dtype}")
    # 期望 [16384//128=128, 4096//128=32]
    assert s.shape[0] == 128, f"expected 128 row-blocks, got {s.shape[0]}"

    # 假设 A: [q_full(8192) | gate_full(8192)] → 各 64 block
    seg_A = [("q_full", 64), ("gate_full", 64)]
    stats_A = _segment_stats(s, seg_A, dim=0)
    print("\n  假设 A: [q_full(8192) | gate_full(8192)] 拼接")
    print(_fmt_stats(stats_A))

    # 假设 B: 32 heads × 2 (q,gate) × head_dim(256) = 32 × 4 = 128 blocks
    # 每个 head 占 4 blocks (2 q + 2 gate per head, 因为 head_dim=256 = 2*128)
    seg_B = []
    for h in range(32):
        seg_B.append((f"h{h}_q", 2))
        seg_B.append((f"h{h}_gate", 2))
    stats_B = _segment_stats(s, seg_B, dim=0)
    print("\n  假设 B: head-interleaved [h_q|h_gate] × 32 (sglang QKVParallelLinear)")
    print(_fmt_stats(stats_B[:8]))  # 前 2 个 head

    # 关键判定：q_full 内的 64 块（覆盖 32 heads × 2 = 64 sub-blocks，head_dim=256=2*128）
    # 如果是假设 A：q_full 64 块都是 q 权重，应该 RMS 相近
    # 如果是假设 B：实际 q_full 段会是 [h0_q(2)|h0_gate(2)|h1_q(2)|h1_gate(2)|...|h15_gate(2)]
    #              即 q 和 gate 交错出现，奇偶 head 子块 RMS 会有差异

    # 取 q_full 段（前 64 行块），按每 2 个一组（每个 head 的 q 子块），看偶数组(h_q) vs 奇数组(h_gate)
    q_seg = s[:64].float()  # [64, 32]
    # 假设 B: 每 4 block = 1 head, 前 2 是 q，后 2 是 gate
    q_subblocks_in_qseg = []  # 假设B下，q_seg 内的 q 子块
    gate_subblocks_in_qseg = []  # 假设B下，q_seg 内的 gate 子块
    for i in range(0, 64, 4):
        q_subblocks_in_qseg.append(q_seg[i:i+2].pow(2).mean().sqrt().item())
        gate_subblocks_in_qseg.append(q_seg[i+2:i+4].pow(2).mean().sqrt().item())
    q_rms_assumed_B_q = sum(q_subblocks_in_qseg) / len(q_subblocks_in_qseg)
    q_rms_assumed_B_gate = sum(gate_subblocks_in_qseg) / len(gate_subblocks_in_qseg)
    diff_B = abs(q_rms_assumed_B_q - q_rms_assumed_B_gate) / (q_rms_assumed_B_q + 1e-9)
    print(f"\n  假设 B 验证：q_full 段内按 [q,q,gate,gate] 拆分后")
    print(f"    'q 子块' avg RMS = {q_rms_assumed_B_q:.4f}")
    print(f"    'gate 子块' avg RMS = {q_rms_assumed_B_gate:.4f}")
    print(f"    相对差异 = {diff_B*100:.2f}%")
    print(f"  如果差异 > 30% → 强烈支持假设 B (head-interleaved)")
    print(f"  如果差异 < 10% → 强烈支持假设 A (fused)，因为 q_full 内全是 q")

    # 同样检查段间差异
    rms_qfull = stats_A[0]["rms"]
    rms_gfull = stats_A[1]["rms"]
    diff_A = abs(rms_qfull - rms_gfull) / (rms_qfull + 1e-9)
    print(f"\n  假设 A 验证：q_full vs gate_full RMS 相对差异 = {diff_A*100:.2f}%")
    print(f"  如果差异 > 30% → 强烈支持假设 A（q 和 gate 整体属性不同）")
    return True


def diagnose_conv1d(model_path: str, layer_idx: int = 0):
    """根因 3: conv1d.weight 是否与 in_proj_qkv 同布局？"""
    print("\n" + "=" * 70)
    print(f"[根因 3] Layer {layer_idx} GDN conv1d.weight 布局")
    print("=" * 70)
    key = f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight"
    try:
        w = _load_tensor(model_path, key)
    except KeyError:
        print(f"  ERROR: not found: {key}")
        return
    print(f"  conv1d.weight shape: {tuple(w.shape)} dtype={w.dtype}")
    # 期望 [12288, 1, 4]
    seg = [("q_full", 2048), ("k_full", 2048), ("v_full", 8192)]
    stats = _segment_stats(w, seg, dim=0)
    print("  按 [q|k|v] 分段统计：")
    print(_fmt_stats(stats))

    # 关键判定：q_full 内 16 heads × 128 channels 的 RMS 相近性
    q_part = w[:2048].float()
    k_part = w[2048:4096].float()
    v_part = w[4096:].float()
    # 每 128 行（1 个 head）一组
    q_head_rms = [q_part[i*128:(i+1)*128].pow(2).mean().sqrt().item() for i in range(16)]
    v_head_rms = [v_part[i*128:(i+1)*128].pow(2).mean().sqrt().item() for i in range(64)]
    print(f"\n  q_full 16 个 head 块 RMS: min={min(q_head_rms):.4f} max={max(q_head_rms):.4f}")
    print(f"  v_full 64 个 head 块 RMS: min={min(v_head_rms):.4f} max={max(v_head_rms):.4f}")
    print(f"\n  ➜ q/k/v 三段 RMS 差异 (q={stats[0]['rms']:.4f}, k={stats[1]['rms']:.4f}, v={stats[2]['rms']:.4f})")
    if abs(stats[0]['rms'] - stats[2]['rms']) / (stats[0]['rms'] + 1e-9) > 0.2:
        print("    q/k/v 段 RMS 差异显著 → conv1d 布局与 in_proj_qkv 同布局 [q|k|v]")
    else:
        print("    q/k/v 段 RMS 相近 → 无法直接判定")


def diagnose_moe_jump(model_path: str, layer_idx: int = 22):
    """根因 4: L22→L23 hidden norm 跳变是否源自 MoE 路径异常？

    检查 L22 的 MoE 关键权重统计：gate weight 范围、experts weight 范围。
    """
    print("\n" + "=" * 70)
    print(f"[根因 4] Layer {layer_idx} MoE 权重统计（hidden norm 跳变嫌疑）")
    print("=" * 70)

    # MoE gate (router)
    gate_key = f"model.language_model.layers.{layer_idx}.mlp.gate.weight"
    try:
        g = _load_tensor(model_path, gate_key).float()
        print(f"  mlp.gate.weight: shape={tuple(g.shape)} rms={g.pow(2).mean().sqrt().item():.6f} "
              f"max={g.abs().max().item():.4f} std={g.std().item():.4f}")
    except KeyError:
        print(f"  gate key not found: {gate_key}")

    # 几个 expert 的 down_proj weight_scale_inv（看哪个 expert 异常）
    for expert_id in [0, 1, 10, 50, 100, 200]:
        key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight_scale_inv"
        try:
            s = _load_tensor(model_path, key).float()
            print(f"  expert{expert_id:>3d}.down_proj.scale_inv: shape={tuple(s.shape)} "
                  f"rms={s.pow(2).mean().sqrt().item():.4f} max={s.abs().max().item():.4f}")
        except KeyError:
            if expert_id == 0:
                print(f"  expert weight key pattern not matching, skip")
                break

    # 对比 L21 / L22 / L23 的 input_layernorm + post_attention_layernorm
    print("\n  各层 layernorm.weight 统计对比：")
    for lid in [layer_idx - 1, layer_idx, layer_idx + 1, layer_idx + 2]:
        for norm in ("input_layernorm", "post_attention_layernorm"):
            k = f"model.language_model.layers.{lid}.{norm}.weight"
            try:
                w = _load_tensor(model_path, k).float()
                print(f"    L{lid:>2d}.{norm}: rms={w.pow(2).mean().sqrt().item():.4f} "
                      f"max={w.abs().max().item():.4f} mean={w.mean().item():+.4f}")
            except KeyError:
                pass

    # 比较各层 MoE gate 权重 norm（看是否有某层异常大/小）
    print("\n  L20-L28 mlp.gate.weight RMS 对比（查找异常）：")
    for lid in range(20, 29):
        k = f"model.language_model.layers.{lid}.mlp.gate.weight"
        try:
            w = _load_tensor(model_path, k).float()
            print(f"    L{lid:>2d}.gate: rms={w.pow(2).mean().sqrt().item():.6f} "
                  f"max={w.abs().max().item():.4f}")
        except KeyError:
            pass


def main():
    model_path = DEFAULT_MODEL_PATH
    print(f"Model path: {model_path}")
    print(f"TP_SIZE={TP_SIZE}, GDN: K_heads={GDN_NUM_K_HEADS}, V_heads={GDN_NUM_V_HEADS}, "
          f"K_dim={GDN_KEY_DIM}, V_dim={GDN_VALUE_DIM}")
    print(f"Attention: Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}\n")

    # 根因 1: GDN in_proj_qkv 布局
    diagnose_in_proj_qkv(model_path, layer_idx=0)

    # 根因 2: q_proj 布局（attention 层只在某些 layer 出现，多试几层）
    for lid in [3, 7, 11, 15]:
        if diagnose_q_proj(model_path, lid):
            break

    # 根因 3: conv1d 布局
    diagnose_conv1d(model_path, layer_idx=0)

    # 根因 4: MoE 跳变
    diagnose_moe_jump(model_path, layer_idx=22)


if __name__ == "__main__":
    main()
