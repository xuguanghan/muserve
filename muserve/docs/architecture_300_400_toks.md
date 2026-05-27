# 实现 300~420 tok/s 单流的架构设计

**适用场景**：Qwen3.5-397B-A17B-FP8，8×S5000（80GB HBM/卡），MUSA SDK 5.1.0
**硬件上限**：~420 tok/s（由 S5000 HBM 带宽决定，17B active params × 1 byte / 7.2 TB/s）
**当前进展**：166.23 tok/s（MUSA Graph capture，mate 0.2.1，2026-05-27）

---

## 一、瓶颈演进

### 已解决：Host Dispatch Overhead（通过 MUSA Graph）

传统推理框架每个算子独立 launch，~1900 次/step × ~100μs = ~190ms overhead。

**MUSA Graph 实测效果**（SDK 5.1.0，`pool + capture_error_mode="relaxed"`）：
```
Eager:  234ms/step, 34 tok/s
Graph:  59.7ms/step, 134 tok/s (3.9x speedup)
```

Graph 将 ~1900 次独立 launch 合并为 1 次 replay，消除了所有 CPU dispatch overhead。

### 当前瓶颈：HBM 带宽利用率

Graph 消除 dispatch overhead 后，59.7ms/step 几乎全是 GPU 计算 + HBM 读写：
- 理论最小值（纯权重加载）：17B × 1 byte / 7.2 TB/s = **2.4ms**
- 实际：**59.7ms** = 2.4ms(权重) + ~57ms(activation HBM 读写)
- 差距原因：每层 ~10 次 HBM 读写中间结果（RMSNorm→HBM→Linear→HBM→GDN→HBM→...）

**结论：要到 300-420 tok/s，必须通过算子融合减少 activation 的 HBM 读写次数。**

---

## 二、两层架构设计（修订版）

### 第一层：MUSA Graph（✓ 已实现，消除 Launch Overhead）

**核心思想**：用 `torch.musa.graph(pool=pool, capture_error_mode="relaxed")` 将整个 decode step capture 到单个 Graph，replay 时零 CPU dispatch。

**实测效果**：
```
Eager (1900 launches):  234ms/step → 34 tok/s
Graph (1 replay):       59.7ms/step → 134 tok/s
Speedup: 3.9x
```

**关键配置**：
```python
pool = torch.musa.graph_pool_handle()
g = torch.musa.MUSAGraph()
with torch.musa.graph(g, pool=pool, capture_error_mode="relaxed"):
    logits = full_decode_step(input_ids, gdn_states)  # 60 层全部 capture
g.replay()  # 零 CPU dispatch
```

**约束**：
- 固定 batch size（Graph capture 时确定）
- 动态 routing（topk、scatter_add_）通过 `pool + relaxed` 模式支持
- MCCL collective（AllReduce、AllGather）需在 default stream 上 capture

> **原方案对比**：原文档设计了 Persistent Kernel（tilelang `T.sync_grid()`）来消除 launch overhead。
> MUSA Graph 以更低的工程成本（~100 行 Python vs 3-4 周 tilelang 开发）达到了相同效果。
> Persistent Kernel 仍有价值（可进一步减少 kernel 间的 L2 cache miss），但优先级降低。

---

### 第二层：算子融合（减少 HBM 读写）— 下一步

**核心思想**：相邻算子的中间结果不写回 HBM，直接在 register/shared memory 传递。
这是从 134 tok/s 到 300-420 tok/s 的关键。

tilelang_musa 已有完整示例：
- `tilelang_musa/examples/gemm/example_gemm_persistent.py`
- `tilelang_musa/examples/deepseek_mla/example_mla_decode_persistent.py`

#### 关键机制：`T.sync_grid()`

`T.sync_grid()` 是 persistent kernel 的核心原语，允许所有 SM 在 kernel 内部同步，然后进行第二阶段计算（如 split-K 的 partial sum reduction）。这在传统 kernel 里需要两次独立 launch 才能实现。

#### 对 Qwen3.5-397B 的应用

```
目标：把 60 层 × ~32 launches/层 = 1900 次 → 1 次（整个 decode step）

单层 persistent kernel 结构：
  Phase 1（所有 SM 并行）：
    - RMSNorm + QKV_proj + GDN_decode（fused）
    - MoE_gate + top10_select（fused）
    - Expert_GEMM × 10（batched，wave 调度）
    - Attention（split-K persistent）
  T.sync_grid()
  Phase 2（归约）：
    - split-K attention reduction
    - AllReduce（TP=8 通信）
    - residual add
```

---

### 第二层：算子融合（减少 Global Memory 读写）

**核心思想**：相邻算子的中间结果不写回 global memory，直接在 register/shared memory 传递。

#### 内存层次与延迟

| 存储层次 | 带宽 | 延迟 |
|----------|------|------|
| Register | ~100 TB/s | 1 cycle |
| Shared Memory | ~20 TB/s | ~20 cycles |
| L2 Cache | ~7 TB/s | ~200 cycles |
| HBM（global memory） | ~900 GB/s | ~500 cycles |

每次写回 global memory 再读回，带宽利用率损失 100x。

#### 融合目标（按收益排序）

**融合 1：RMSNorm + QKV_proj**
```
未融合：RMSNorm → [写 HBM, 4096×bf16] → QKV_proj → [写 HBM]
融合后：RMSNorm → [register] → QKV_proj → [写 HBM]
节省：1 次 HBM 读写（4096 × batch × 2 bytes）
```

**融合 2：MoE gate + top-10 routing**
```
未融合：gate_linear → [写 HBM, 512×fp32] → softmax → [写 HBM] → topk → [写 HBM]
融合后：gate_linear → softmax → topk → [写 HBM, 10×int32]
节省：2 次 HBM 读写，512 expert 的 softmax 在 register 内完成
```

**融合 3：Expert GEMM batching**
```
未融合：10 次独立 gemm_fp8_nt_groupwise（10 次 launch，10 次权重读取）
融合后：1 次 batched GEMM（wave 调度，权重读取可 pipeline）
节省：9 次 launch overhead + L2 cache 复用
```

**融合 4：GDN decode（mate 已实现）**

mate 的 `gdn_decode.py` 已经把以下操作融合进一个 tilelang kernel：
- QK L2 normalization
- Delta rule state update（`state = state + k^T × (v - state × k)`）
- Output projection

这是 mate 最有价值的部分，直接复用。

---

### 第三层：Wave-based 负载均衡（充分利用所有 SM）

**核心思想**：传统 kernel 的 tile 数量不一定是 SM 数的整数倍，导致最后一波 SM 利用率不足。Persistent kernel 的 wave 调度天然解决这个问题。

#### Tail Effect 示意

```
S5000 SM 数 = 64（假设）
传统 GEMM，tile 数 = 70：

Wave 1: [SM0][SM1]...[SM63]  ← 64 个 SM 全满
Wave 2: [SM0][SM1]...[SM5]   ← 只有 6 个 SM 在工作，58 个空闲
         ↑ 这一波的时间 ≈ Wave 1，但利用率只有 6/64 = 9.4%

Persistent kernel：
  每个 SM 处理 ceil(70/64) = 2 个 tile
  SM0~SM5 处理 2 个 tile，SM6~SM63 处理 1 个 tile
  没有空闲 SM，利用率 100%
```

对 Qwen3.5-397B 的 MoE（512 experts，top-10），每 decode step 有 10 个 expert GEMM，tile 数量很可能不整除 SM 数，wave 调度收益显著。

---

## 三、量化效果（实测 + 预估）

| 方案 | 状态 | ms/step | 吞吐 | 瓶颈 |
|------|------|---------|------|------|
| Eager baseline | ✓ 已实现 | 234 | 34 tok/s | CPU dispatch |
| MUSA Graph (decode only) | ✓ 已实现 | 59.7 | 134 tok/s | HBM + sampling overhead |
| **MUSA Graph (含 sampling)** | **✓ 已实现** | **51.8** | **154 tok/s** | **HBM 带宽** |
| Graph + RMSNorm-Linear 融合 | 待实现 | ~40-45 | ~180-200 tok/s | HBM 带宽 |
| Graph + 全部算子融合 | 待实现 | ~15-20 | ~400-530 tok/s（理论） | 接近硬件上限 |
| 硬件上限 | — | 2.4 | 420 tok/s | — |

> SPEC 目标 ≥ 150 tok/s 已达成（154.49 tok/s）。
> 下一目标：200+ tok/s（通过 RMSNorm+Linear 融合减少 HBM 读写）。

---

## 四、muserve 的实现路径

### 已有基础（已验证可用）

| 组件 | 来源 | 状态 |
|------|------|------|
| MUSA Graph capture 整个 decode step | torch_musa (pool + relaxed) | ✓ **已实现，134 tok/s** |
| GDN decode JITKernel 直接调用 | mate tilelang（绕过 cache） | ✓ 已实现，0.06ms/次 |
| MoE batched GEMM | mate `ragged_m_moe_gemm_8bit` | ✓ 已实现，2.3ms/层 |
| FP8 GEMM | `mate.gemm.gemm_fp8_nt_groupwise` | ✓ 已验证 |
| GDN prefill kernel | `mate.gdn_prefill.chunk_gated_delta_rule` | ✓ 已验证 |

### 需要新写的 kernel（按优先级排序）

| Kernel | 目的 | 预期收益 | 预估工程量 |
|--------|------|----------|-----------|
| RMSNorm + Linear fused（tilelang） | 减少 1 次 HBM 读写/层 | 134→~160 tok/s | 1~2 天 |
| MoE gate + top-10 routing fused | 减少 2 次 HBM 读写/层 | ~160→~180 tok/s | 1~2 天 |
| GDN 投影融合（QKV+A+B → 1 次 GEMM） | 减少 2 次 launch/层 | 小幅提升 | 0.5 天 |
| 单层全融合 kernel（tilelang） | 消除层内所有 HBM 中间读写 | ~200→~300 tok/s | 2~3 周 |

> **注意**：Persistent Kernel（原 Phase 4）优先级降低。MUSA Graph 已消除 dispatch overhead，
> 剩余瓶颈是 HBM 带宽。算子融合（减少 HBM 读写）比 persistent kernel（减少 launch）更有效。
> Persistent kernel 的价值在于可以利用 `T.sync_grid()` 做跨 SM 的 split-K reduction，
> 但这只在 batch 较大时有意义。

### 到 150 tok/s 的最短路径（当前优先）

```
1. 把 greedy_sample 纳入 Graph capture（消除 ~2ms Python dispatch）
2. RMSNorm + Linear 融合 kernel（减少 60×2 = 120 次 HBM 读写）
→ 预期：59.7ms → ~50ms → 160 tok/s
```

---

## 五、与 TileRT（GLM-5.1）的对比

TileRT 实现 400 tok/s 的核心和本文描述的架构完全一致：

| 设计点 | TileRT | muserve 目标 |
|--------|--------|-------------|
| Kernel launch 次数/step | 1 | 1（Phase 4） |
| 算子融合 | 整个 forward pass | 逐层融合（Phase 2A → Phase 4） |
| SM 利用率 | 100%（wave 调度） | 100%（persistent kernel） |
| 中间结果存储 | Register/Shared Memory | Register/Shared Memory |
| GPU 角色特化 | 不同 GPU 处理不同组件 | TP=8 均匀分配（后续可优化） |

**差异**：TileRT 针对 GLM-5.1（dense 模型）优化，muserve 针对 Qwen3.5-397B（MoE + GDN 混合架构）。MoE 的 expert routing 引入了额外的 wave 调度复杂度，但 512 experts 的细粒度也提供了更好的负载均衡机会。

**硬件差异**：GLM-5.1 在 H100 上跑 400 tok/s，H100 HBM 带宽 3.35 TB/s × 8 = 26.8 TB/s，是 S5000 的 3.7×。S5000 的硬件上限约 420 tok/s（Qwen3.5-397B A17B），与 H100 上 GLM-5.1 的 400 tok/s 在同一量级，原因是 Qwen3.5-397B 的 active 参数（17B）远少于 GLM-5.1 的总参数。
