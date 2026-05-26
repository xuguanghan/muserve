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

### Task 2B.1：RMSNorm + Linear 融合（tilelang）

减少每层 2 次 HBM 中间读写，降低 per-layer 时间。

**接受标准**：
- [ ] `fused_rmsnorm_linear(x, norm_w, linear_w, scale)` → `[tokens, out_dim]`
- [ ] 与 unfused 版本误差 < 1e-2
- [ ] per-layer 时间从 0.77ms 降低

### Task 2B.2：GDN 投影融合（A + B → 1 次 GEMM）— ✓ 已完成

A+B 两次 bf16 GEMM 合并为 1 次（权重 inline cat，输出 clone 分割）。
实测：167.75 → **171.10 tok/s**（省 0.9ms/step）。

### Task 2B.3：MoE gate + routing 融合

gate_linear + softmax + topk → 1 次 kernel。

### Task 2B.4：吞吐测试

**接受标准**：
- [ ] B=8 吞吐 ≥ 200 tok/s
- [ ] B=32 吞吐 ≥ 700 tok/s

---

## Phase 3：TTFT 优化 + 服务化

### Task 3.1：Prefill 路径优化
- [ ] chunk_size=4096，32K TTFT < 1.5s
- [ ] GDN prefill kernel 正确处理跨 chunk state 传递

### Task 3.2：Anthropic Messages API
- [ ] `/v1/messages` + SSE streaming
- [ ] 连续批处理调度器集成

### Task 3.3：CPU KV Cache Offload（可选）
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
