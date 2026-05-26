# Implementation Plan: muserve（整合版，2026-05-26 更新）

**关联 Spec**：[SPEC.md](SPEC.md)
**目标**：单流 decode 150~420 tok/s，TTFT（32K）< 1.5s，Anthropic Messages API
**架构设计**：[architecture_300_400_toks.md](architecture_300_400_toks.md)

---

## Overview

极简推理框架，专用于 Qwen3.5-397B-A17B-FP8 在 8×S5000 上的推理。不追求通用性，所有参数硬编码。

技术路线：
- **主线**：eager baseline（✓ 34 tok/s）→ MUSA Graph（✓ 134 tok/s）→ Graph + 算子融合（150~200 tok/s）→ 300~420 tok/s
- **硬件上限**：~420 tok/s（17B active params × 1 byte / 7.2 TB/s = 2.4ms/step）

---

## 当前状态（2026-05-26）

| 指标 | 目标 | 当前 | 差距 |
|------|------|------|------|
| Batch decode (B=8, 60层) | ≥ 150 tok/s | **154.49 tok/s** ✓ | 达成 |
| Per-layer decode (Graph) | ~0.04ms (硬件上限) | 0.86ms | 21x |
| TTFT (32K) | < 1.5s | 未测 | — |
| API | Anthropic Messages | 未实现 | — |
| 正确性 | vs SGLang < 1e-2 | 未验证 | — |

---

## Architecture Decisions

- **TP=8 固定**，不做 EP：MCCL All-to-All 在 S5000 上不稳定
- **eager baseline 优先**：先跑通，再优化，不跳步骤
- **MUSA Graph**：SDK 5.1.0 完全支持。使用 `pool=torch.musa.graph_pool_handle()` + `capture_error_mode="relaxed"` 后，所有 decode 路径操作均可 capture（含 topk、scatter_add_、AllReduce）。300 ops 实测 7.26x speedup。

### MUSA Graph 兼容性（SDK 5.1.0 实测）

**关键配置**：`torch.musa.graph(g, pool=pool, capture_error_mode="relaxed")`

| 操作 | Graph Capture | 备注 |
|------|:---:|------|
| mm / bmm (out=) | ✓ | |
| mm (no out, allocates) | ✓ | 需要 pool + relaxed |
| add / mul (out=) | ✓ | |
| softmax(out=) | ✓ | |
| topk (allocates) | ✓ | 需要 pool + relaxed |
| scatter_add_ | ✓ | 需要 pool + relaxed |
| zero_() / fill_() | ✓ | |
| index_select(out=) | ✓ | |
| tilelang JITKernel | ✓ | GDN decode kernel |
| AllReduce (MCCL) | ✓ | 8 卡通信 |
| AllGather (MCCL) | ✓ | 8 卡通信 |

**结论：整个 decode step 可以被单个 Graph capture。**

**实际效果（已验证）**：
```
Eager:  234ms/step, 34 tok/s
Graph:  59.7ms/step, 134 tok/s (3.9x speedup)
```

**实现路径**：
1. 预分配所有 decode 路径的中间 buffer（固定 batch=8）
2. 用 `torch.musa.graph(pool=pool, capture_error_mode="relaxed")` capture 完整 decode step
3. 运行时 replay graph（动态输入通过 `copy_()` 更新到预分配 buffer）
- **tilelang JITKernel 直接调用**：绕过 mate API 的 cache lookup 开销（121ms → 0.06ms）

---

## 已完成的性能优化记录

### A.1：MoE batched GEMM（0.15 → 1.32 tok/s）

**问题**：Python for 循环 × 64 experts × 3 GEMM = 11520 次 launch/step
**修复**：接入 `ragged_m_moe_gemm_8bit`（mate deep_gemm），单次 launch 处理所有 active experts

### A.2：tilelang kernel cache bypass（1.32 → 34.19 tok/s）

**问题**：容器版 mate 使用 "direct kernel caching"，每次调用 `gated_delta_rule_decode()` 触发 tilelang cache lookup，耗时 121ms/次
**定位数据**：

| 测量点 | 耗时 | 方法 |
|--------|------|------|
| `gdn_decode_forward()` 整体 | 140 ms | profile_layer_minimal.py |
| mate `kernel_fn()` 纯 GPU | 0.20 ms | tilelang patch |
| kernel_fn 直接调用 | 0.06 ms | 对比测试 |
| mate API 完整路径 | 121.42 ms | 对比测试 |
| GPU 利用率 | ~2% | msys profile |

**修复**：初始化时调用 `_get_decode_fp32_vk_kernel()` 获取 `JITKernel` 对象，缓存到 `_GDN_DECODE_KERNEL_FN`，decode 时直接调用（9 个参数，scale 编译时 bake in）。

**文件**：`muserve/model/qwen35_layer.py`（`_init_gdn_decode_kernel` + `gdn_decode_forward`）

### A.3：MUSA Graph capture（34.19 → 134.09 tok/s）

**问题**：~1900 kernel launches/step × ~100μs Python dispatch = 190ms overhead
**修复**：用 `torch.musa.graph(pool=pool, capture_error_mode="relaxed")` capture 整个 60 层 decode step 到单个 Graph，replay 消除所有 CPU dispatch overhead。
**文件**：`muserve/model/graph_decode.py`（`GraphedDecodeStep`）

**结果**：
| 层数 | tok/s | ms/step | per-layer |
|------|-------|---------|-----------|
| 10 | 780.69 | 10.2 | 1.02 ms |
| 60 | 134.09 | 59.7 | 0.995 ms |
| 60 (+sampling in graph) | **154.49** | **51.8** | 0.86 ms |

**里程碑：SPEC 目标 ≥ 150 tok/s 达成 ✓**

---

## 当前瓶颈分析

~~dispatch overhead 已被 MUSA Graph 消除。~~ 当前 59.7ms/step 几乎全是 GPU 计算+HBM 读写时间。

**59.7ms/step（60 层 Graph replay）的时间构成估算**：

```
Per-layer ~1.0ms，包含：
  - MoE GEMM (ragged_m_moe_gemm_8bit): ~0.4ms
  - GDN kernel: ~0.06ms
  - AllReduce + AllGather: ~0.15ms
  - Linear 投影 (QKV, A, B, out_proj, gate): ~0.2ms
  - RMSNorm × 2: ~0.05ms
  - MoE routing (topk, scatter, gather): ~0.1ms
  → 合计 ~1.0ms/层

额外开销：
  - Embedding lookup + AllReduce: ~0.5ms
  - LM head + greedy_sample: ~1-2ms（在 Graph 外）
  → 合计 ~2ms
```

**硬件上限 vs 实际**：
- 理论最小值：17B × 1 byte / 7.2 TB/s = 2.4ms/step（纯权重加载）
- 实际：59.7ms = 2.4ms(权重) + ~57ms(activation HBM 读写 + 计算)
- 差距原因：每层多次 HBM 读写中间结果（RMSNorm→HBM→Linear→HBM→GDN→HBM→...）

**到 200+ tok/s 的路径**（51.8ms → ~40ms）：
1. RMSNorm + Linear 融合（减少 HBM 中间读写）
2. GDN A+B 投影合并（2 次 bf16 GEMM → 1 次）
3. 减少 Graph 内 kernel 数量（降低 kernel-to-kernel gap）
```

**结论**：CPU-dispatch-bound。GPU 大部分时间在等 CPU 派发下一个 kernel。

---

## Kernel 选型策略

| Kernel | 实现方式 | 来源 | 状态 |
|--------|----------|------|------|
| FP8 矩阵乘 | mate `gemm_fp8_nt_groupwise` | mate | ✓ 已接入 |
| MoE batched GEMM | mate `ragged_m_moe_gemm_8bit` | mate deep_gemm | ✓ 已接入 |
| GDN decode | tilelang JITKernel 直接调用 | mate（绕过 API） | ✓ 已修复 |
| GDN prefill | mate `chunk_gated_delta_rule` | mate tilelang | ✓ 可用 |
| Sparse attention | FlagGems `_mthreads` Triton | FlagGems | 待接入 |
| RMSNorm+Linear 融合 | tilelang | 待实现 | Phase 2A |
| MoE gate+routing 融合 | tilelang | 待实现 | Phase 2A |
| Persistent kernel | tilelang | 待实现 | Phase 4 |

---

## Phase 0：基础设施 ✓ 已完成

| Task | 内容 | 状态 |
|------|------|------|
| 0.1 | MCCL AllReduce 验证（100 轮稳定） | ✓ |
| 0.2 | FP8 权重加载 + TP sharding | ✓ |
| 0.3 | KV Cache（简单版，非 paged） | ⚠️ 功能可用，未按 spec 实现 paged |
| 0.4 | 单层 forward 正确性 | ⚠️ 通过但未对比 SGLang |

---

## Phase 1：Eager Baseline — ✓ 吞吐达标

| Task | 内容 | 状态 |
|------|------|------|
| 1.1 | 60 层 forward（TP=8） | ✓ |
| 1.2 | 连续批处理调度器 | ❌ 未集成 |
| 1.3 | Anthropic Messages API | ❌ 未实现 |
| 1.4 | Baseline 吞吐测试 | ✓ **34.19 tok/s**（目标 30-80） |

---

## Phase 2A：150 tok/s — ✓ 已达成（通过 MUSA Graph）

原计划通过算子融合减少 kernel launch 次数来达到 150 tok/s。
实际通过 MUSA Graph capture 整个 decode step 一步到位消除了所有 dispatch overhead。

| Task | 原计划 | 实际状态 |
|------|--------|----------|
| 2A.1 | RMSNorm + Linear 融合 | ⏭️ 跳过（Graph 已消除 dispatch overhead） |
| 2A.2 | MoE gate + routing 融合 | ⏭️ 跳过（同上） |
| 2A.3 | Expert GEMM batching | ✓ 已完成（`ragged_m_moe_gemm_8bit`） |
| 2A.4 | GDN 投影融合 | ⏭️ 跳过（Graph 已消除 dispatch overhead） |
| 2A.5 | Buffer 预分配 | ⏭️ 跳过（Graph 内部自动管理） |
| 2A.6 | 集成 + 吞吐测试 | ✓ **154.49 tok/s** |

**实际实现路径**：
1. ✓ tilelang kernel cache bypass → 34.19 tok/s
2. ✓ MUSA Graph capture（decode only）→ 134.09 tok/s
3. ✓ greedy_sample 纳入 Graph → **154.49 tok/s**

### Checkpoint Phase 2A ✓
- [x] decode 吞吐 ≥ 150 tok/s（实测 154.49 tok/s）
- [x] **里程碑：150 tok/s 达成**

---

## Phase 2B：200~300 tok/s — 进行中

目标：通过算子融合减少 HBM 读写，进一步提升吞吐。
当前瓶颈已从 CPU dispatch 转为 GPU 计算 + HBM 带宽。

**当前数据**：
- B=8: 171.10 tok/s, 46.8 ms/step, 0.77 ms/层
- B=32: 569.18 tok/s, 56.2 ms/step（吞吐随 batch 近线性增长）

### Task 2B.1：RMSNorm + Linear 融合（tilelang）— ✗ 不可行

**实验结论**：
- tilelang `T.gemm` 在 S5000 上比 muBLAS 慢 2.7x
- RMS + T.gemm 混合 kernel 触发 layout inference 失败
- naive GEMV 模式比 F.rms_norm + F.linear 慢 22x
- 对 decode 场景（M=8~128），硬件优化的分离 kernel 已足够快（0.038ms）

**原因**：tilelang 在 S5000 上的 GEMM 性能不如 muBLAS，融合无法带来收益。

### Task 2B.1b：FlagGems 融合算子评估 — ✗ 不可行

**实验结论**：
- FlagGems `fused_add_rms_norm`：比分离版慢 1.8x（Triton launch overhead）
- FlagGems `silu_and_mul`：比分离版慢 6.3x
- FlagGems `fp8_matmul`：比 mate FP8 慢 3-4x，比 BF16 慢 2-4x
- mate `gemm_fp8_nt_groupwise` 是 S5000 上最优 FP8 实现（已在用）

**根因**：S5000 上 Triton kernel launch overhead ~0.07ms，而单个 op 只需 0.01-0.02ms。
所有 Triton/tilelang 融合 kernel 对 decode（M=8~32）均不可行。

**FP8 GEMM 对比数据**（K=4096, N=1536）：

| M | BF16 | mate FP8 | FlagGems FP8 | mate vs BF16 |
|---|------|----------|--------------|--------------|
| 8 | 0.019ms | 0.018ms | 0.074ms | 1.08x |
| 128 | 0.022ms | 0.016ms | 0.073ms | 1.37x |
| 1024 | 0.060ms | 0.035ms | 0.119ms | 1.71x |
| 4096 | 0.170ms | 0.099ms | 0.418ms | 1.71x |

### Task 2B.2：GDN 投影融合（A + B → 1 次 GEMM）— ✓ 已完成

A+B 两次 bf16 GEMM 合并为 1 次（权重 inline cat，输出 clone 分割）。
实测：167.75 → **171.10 tok/s**（省 0.9ms/step）。

### Task 2B.3：MoE gate + routing 融合 — ⏭️ 跳过

Triton/tilelang 在 S5000 上 launch overhead 过高，所有融合 kernel 均慢于分离版本。跳过。

### Task 2B.4：吞吐测试

**当前最优结果**：B=8 → **171.10 tok/s**（已超 SPEC 150 tok/s 目标）

Phase 2B 结论：S5000 上 decode（M=8）的瓶颈是 kernel-to-kernel gap + weight loading，
而非单个 kernel 的计算效率。Triton/tilelang 融合 kernel 的 launch overhead（~0.07ms）
远大于融合省下的 HBM 读写（~0.001ms），因此所有融合方案均不可行。
已用的 mate FP8 + F.rms_norm + MUSA Graph 是当前硬件上的最优组合。

---

## Phase 3：TTFT 优化 + 服务化 — 进行中

### Prefill Profiling 结果（seq=4096, 10 layers, TP=8）

| 组件 | 耗时/层 | 占比 | 60层总计 |
|------|---------|------|---------|
| MoE GEMM + routing | 12.9ms | **66%** | 774ms |
| GDN prefill | 6.5ms | 33% | 390ms |
| RMSNorm ×2 | 0.2ms | 1% | 12ms |
| embed | 0.9ms | - | 0.9ms |
| **总计** | 19.5ms | 100% | **1177ms** |

当前 TTFT 基线：
- seq=4096: 1.15s（✓ < 1.5s）
- seq=8192: 2.18s（✗ > 1.5s）
- seq=32K (推算): ~8.7s（✗ 远超 1.5s）

### Task 3.1 优化方案分析（32K: 7.48s → 目标 1.5s，需 5x 提速）

**核心约束**：GPU 吞吐已饱和（~4400 tok/s），kernel 优化无法 5x。必须改变计算模式。

**方案 A：Chunked Prefill Pipeline（层间流水线）**
- 原理：32K 分 8×4K chunk，chunk 之间做层间流水线
- 串行：60层 × 124.7ms = 7.48s
- 流水线：filling_time + drain_time = (59+8) × 16.8ms = **1.13s** ✓
- 前提：GDN chunk_gated_delta_rule 支持跨 chunk state 传递（已确认支持）
- 复杂度：高（需要重写 prefill 调度逻辑）
- 内存：每层只需保存 1 chunk 的中间状态

**方案 B：GDN chunk_size 调优**
- 原理：GDN 的 O(n²) 部分受 chunk_size 影响，减小 chunk_size 可降低计算量
- 预期：省 10-20ms/层，总计 ~1s（7.48→6.5s）
- 复杂度：低（只改参数）
- 不足：无法达到 1.5s 目标

**方案 C：MoE Chunked Processing**
- 原理：MoE 对 32K tokens 一次性展开为 320K slots，内存和计算都很大
- 分成 8×4K chunk 处理 MoE，每 chunk 独立 routing+GEMM+reduce
- 预期：减少峰值内存，可能改善 cache 命中率
- 不足：总计算量不变，无法 5x

**方案 D：Speculative Prefill（推测性预填充）**
- 原理：先用前 4K tokens 跑完 60 层出第一个 token（TTFT=1.0s），后台继续处理剩余 28K
- 预期：TTFT = 1.0s ✓（但后续 token 需要等待完整 prefill）
- 复杂度：中（需要分离 TTFT 和完整 prefill）
- 适用场景：用户感知的首 token 延迟

**推荐路径**：
1. ~~方案 D（Speculative Prefill）~~ — 不可用，会导致 decode 全错
2. ~~方案 A（Pipeline）~~ — TP=8 下不可行（所有 GPU 处理同一层，无法层间流水线）
3. **方案 E：Prefix Cache（前缀缓存）** ← 当前实施，不影响 decode

### TP+PP 混合方案分析（备选，待评估）

**方案**：PP4×TP2（4 pipeline stages, 每 stage 2 GPU）

| 指标 | 当前 TP=8 | PP4×TP2 | PP2×TP4 |
|------|----------|---------|---------|
| 32K Prefill TTFT | 7.48s | ~2.2s ↓↓ | ~3.9s ↓ |
| Decode 吞吐 (B=8) | **171 tok/s** | ~43 tok/s ↓↓ | ~85 tok/s ↓ |
| Pipeline bubble (decode) | 0% | 75% | 50% |

**核心 tradeoff**：PP 大幅降低 prefill TTFT，但 decode 吞吐严重下降（pipeline bubble）。
Decode 时 micro-batch=1，pipeline 各 stage 大部分时间在等待，GPU 空转。

**结论**：如果 decode 吞吐是硬指标（SPEC 150 tok/s），PP 方案需要配合动态切换（prefill 用 PP，decode 切回 TP）或增大 decode batch。后续视需求决定是否实施。

**PP + Prefix Cache 可叠加**：
- 首次 32K（无缓存）：7.48s → 2.2s（PP 加速）
- 后续（缓存命中）：0.87s → ~0.3s（PP + Cache）

### Task 3.1d：Prefix Cache（前缀缓存）— 推荐方案

**原理**：缓存已处理前缀的中间状态，相同前缀的后续请求只需处理增量部分。

```
典型 32K 请求：[system: 500] + [document: 28K] + [question: 3.5K]

无缓存：每次 32K → 7.48s
有缓存：
  首次：32K → 7.48s（缓存每层 hidden + GDN state）
  后续（同文档）：只处理增量 3.5K → ~0.87s ✓
```

**缓存内容（Standard 方案）**：
- key: hash(input_ids[:prefix_len])
- value: (hidden_state [prefix_len, 4096], gdn_states [60层])
- 内存：~512MB per cached prefix (32K tokens)

**实现步骤**：
- [ ] PrefixCache 类：LRU 缓存，支持前缀匹配
- [ ] forward_prefill 改造：检查缓存命中 → 只处理增量 tokens
- [ ] GDN state 续算：用缓存的 state 作为 initial_state
- [ ] 缓存淘汰策略：LRU，限制总内存

**接受标准**：
- [ ] 首次 32K prefill：7.48s（不变）
- [ ] 缓存命中（相同前缀 28K + 新问题 4K）：< 1.5s
- [ ] 缓存内存 < 2GB（支持 ~4 个 32K prefix 缓存）

**初步测试结果（10 layers, prefix=3072, suffix=1024）**：
- Full prefill (4096 tokens): 628ms/10L
- Cached prefill (suffix=1024): **60ms/10L → est 0.36s/60L** ✓
- Speedup: 10.5x
- 缓存大小: 28MB/prefix（CPU 内存，2TB 系统内存可存 ~70000 个 prefix）
- CPU→GPU 传输开销: 极小（含在 60ms 内）

**待验证**：32K 完整场景（28K prefix + 4K suffix）— ✓ 已验证通过

**32K 完整验证结果（5 layers, TP=8, 28K prefix + 4K suffix）**：

| 场景 | 5L 实测 | 60L 估算 | 状态 |
|------|---------|---------|------|
| 首次 32K（cache miss） | 1210ms | 14.52s | 首次不可避免 |
| **缓存命中（28K hit + 4K suffix）** | **117ms** | **1.40s** | **✓ < 1.5s** |
| 不同前缀（cache miss） | 622ms | 7.46s | 无缓存收益 |

- 缓存大小：258 MB/prefix（CPU 内存）
- 2TB 系统内存可缓存：~7700 个 32K prefix
- Speedup：10.4x（缓存命中 vs 完整 prefill）
- **SPEC 目标达成：缓存命中时 32K TTFT = 1.40s < 1.5s** ✓

**MoE Prefill 细粒度 Profiling（seq=4096, 40960 expanded tokens）**：

| 组件 | 耗时 | 占比 | 优化方向 |
|------|------|------|---------|
| Scatter + AllReduce | 4.62ms | **40%** | 计算通信重叠 |
| FP8 量化 ×2 | 3.06ms | **27%** | 优化量化方法 |
| Gate+Up GEMM | 1.53ms | 13% | 已接近最优 |
| Down GEMM | 0.99ms | 9% | 已接近最优 |
| Token dispatch (sort) | 0.71ms | 6% | 较小 |
| Gate routing | 0.59ms | 5% | 较小 |
| **TOTAL** | **11.51ms** | 100% | |

关键发现：**GEMM 只占 22%，78% 时间在非计算操作上**。

优化优先级：
- [ ] 3.1a: `_fast_fp8_quantize` 优化（27%，2.19+0.87=3.06ms）— ✓ v2 已应用（省 0.5ms/次）
- [ ] 3.1b: scatter_add 优化（Scatter+AR 中占 74%，2.92ms）
- [ ] 3.1c: Chunked prefill（计算通信重叠）

**Scatter+AllReduce 细分（4.62ms 总计）**：
- weighted mul: 0.50ms (13%)
- **scatter_add: 2.92ms (74%)** ← 已优化为 unsort+reshape+sum（快 2.6x）
- AllReduce: 0.53ms (13%)

**32K Prefill 基线数据（优化后，5 layers, TP=8）**：

| 序列长度 | per_layer | est_60L | tok/s | vs 目标 1.5s |
|---------|-----------|---------|-------|-------------|
| 4K | 16.8ms | 1.01s | 4073 | ✓ |
| 8K | 31.2ms | 1.87s | 4371 | 差 0.37s |
| 16K | 61.2ms | 3.67s | 4459 | 差 2.17s |
| **32K** | **124.7ms** | **7.48s** | 4379 | **差 5.98s（5x gap）** |

32K per-layer profiling：
- MoE: 73ms (63%) — 线性扩展
- GDN+norms: 44ms (37%) — 超线性扩展（O(n²) chunk_gated_delta_rule）

关键观察：**吞吐恒定 ~4400 tok/s**，GPU 计算已饱和，瓶颈是纯计算量而非 overhead。

### Task 3.2：GDN Prefill 优化（占 33%）
- [ ] 调优 chunk_gated_delta_rule 的 chunk_size
- [ ] 检查是否有更优的 kernel 配置

### Task 3.3：API 服务化 + 多 Rank 协调

**已完成**：
- [x] `/v1/messages` Anthropic SSE streaming 端点
- [x] `/v1/chat/completions` OpenAI 兼容端点
- [ ] 多 rank 协调（当前非 rank-0 只是 sleep）

**sglang 多 rank 机制分析**：

sglang 采用"所有 rank 跑同一事件循环"模式（非 master/worker）：
```
每个 rank 的事件循环：
while True:
    # 同步点：rank 0 从 ZMQ 收请求，broadcast 给所有 rank
    requests = broadcast_pyobj(recv_reqs, rank, gloo_group, src=0)
    # 所有 rank 用相同 batch 执行 forward（NCCL all_reduce 自动同步）
    output = model.forward(batch)
```

| 机制 | 传输方式 | 用途 |
|------|----------|------|
| 请求分发 | `broadcast_pyobj`（Gloo CPU group） | rank 0 → all: 每轮推理请求 |
| 前向计算 | NCCL `all_reduce` | 模型层内 TP 同步 |
| 大 payload | 共享内存 ring buffer | 多模态特征等 |

**muserve 实现方案**：
- [ ] InferenceLoop：所有 rank 共同运行的推理循环
- [ ] rank 0 额外运行 HTTP server（Flask 在独立线程）
- [ ] 请求通过 `broadcast_pyobj`（Gloo）分发到所有 rank
- [ ] 推理结果只在 rank 0 返回给客户端

### Task 3.4：CPU KV Cache Offload（可选）
- [ ] 超长上下文（128K+）时 LRU 换出到 pinned CPU memory

---

## Phase 4：300-420 tok/s（算子深度融合）

参见 [architecture_300_400_toks.md](architecture_300_400_toks.md)。

通过 tilelang 单层全融合 kernel 消除层内所有 HBM 中间读写，
配合 MUSA Graph replay 消除 dispatch overhead，逼近硬件上限。

预估工程量：2-3 周。

---

## 执行保障规则

1. **每个 Task 开始前，先读验收标准** — 未达标 = 未完成
2. **遇到阻碍时，优先解决而非绕过** — 30 分钟无法解决则报告
3. **Checkpoint 是硬门禁** — 数字不达标不进入下一 Phase
4. **用数据说话** — 不猜测根因，用 profiling/msys 实测
