# Spec: Qwen3.5-397B-FP8 极简推理框架（S5000 专用）

**版本**: 0.2  **日期**: 2026-05-25  **目标**: 单流 decode 150~200 tok/s（部分融合）→ 300~420 tok/s（persistent kernel）

---

## 一、Objective

### 背景

在 8×S5000（80GB HBM/卡，MUSA SDK 5.1.0）上推理 Qwen3.5-397B-A17B-FP8，
达到单流 decode **150~200 tok/s**，TTFT（32K 上下文）< **1.5s**。

**硬件规格（S5000 风冷版）**：
- FP8 算力 (Dense): 920 TFLOPS
- 显存容量: 80 GB HBM
- 显存带宽: 1.6 TB/s
- 卡间互联 MTLink (8 卡全互联): 784 GB/s

当前阻碍：
- SGLang 每个 decode step 需要 400+ 次 host kernel dispatch，overhead 主导延迟
- 没有针对 S5000 调优的 MoE dispatch 路径

已验证可用：
- MUSA Graph（SDK 5.1.0，需 `pool + capture_error_mode="relaxed"`，所有 decode 操作均可 capture）

### 解法

构建一个**极简专用推理框架**（目标 ~3000 行 Python），不追求普适性：
- 复用 mate 已验证的 kernel（GDN、flash_attn、fp8_gemm）
- 用 tilelang_musa 融合 decode 热路径，把每步 kernel launch 从 400+ 压到 ~30
- 利用 2TB CPU RAM 做 KV cache offload，支持长上下文和高并发

### 成功标准

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| 单流 decode 吞吐（部分融合） | ≥ 150 tok/s | `bench_decode.py --batch 1 --output-len 512` |
| 单流 decode 吞吐（persistent kernel） | ≥ 300 tok/s | 同上 |
| TTFT（32K 输入，batch=1） | **< 1.5s** | `bench_e2e.py --input-len 32768` |
| TTFT（4K 输入，batch=1） | < 300ms | `bench_e2e.py --input-len 4096` |
| 输出正确性 | logits 与 SGLang 参考实现误差 < 1e-2 | `test_correctness.py` |
| 服务启动时间（含 kernel 预热） | < 5 min | 计时从进程启动到第一个请求可响应 |

**TTFT 1.5s 可行性依据**：
32K prefill 理论 FLOPs = 2169 TFLOPS，S5000 FP8 实测约 296 TFLOPS/卡 × 8 = 2368 TFLOPS，
理论最小值 ~916ms。GDN recurrence（顺序计算）和 overhead 约 1.5× 系数 → 实际目标 **1.4~1.8s**。
需要 chunk_size ≥ 4096 且高效的 GDN prefill kernel（mate.chunk_gated_delta_rule 已验证可用）。

---

## 二、模型架构参数（硬编码，不做抽象）

```python
# Qwen3.5-397B-A17B-FP8 固定配置
NUM_LAYERS        = 60
HIDDEN_SIZE       = 4096
NUM_Q_HEADS       = 32
NUM_KV_HEADS      = 2          # 极度 GQA
HEAD_DIM          = 256
NUM_EXPERTS       = 512
NUM_EXPERTS_PER_TOK = 10       # top-10 routing
MOE_INTERMEDIATE  = 1024       # 每个 expert 的 intermediate size
# GatedDeltaNet (线性注意力) 参数
GDN_NUM_K_HEADS   = 16
GDN_NUM_V_HEADS   = 64
GDN_KEY_DIM       = 128
GDN_VALUE_DIM     = 128
GDN_CONV_KERNEL   = 4
VOCAB_SIZE        = 248320
MAX_SEQ_LEN       = 262144     # 256K
```

**内存估算**：
- 模型权重（FP8）：~379 GB（实测）
- 8 卡 HBM：640 GB → 权重全部放 GPU，剩余 ~261 GB 给 KV cache
- 2TB CPU RAM：KV cache offload，支持超长上下文或高并发

---

## 三、Tech Stack

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10 | 框架主体 |
| torch_musa | 2.9.0+1dc7872 | MUSA tensor ops |
| MUSA SDK | 5.1.0 | 硬件驱动 |
| mate | 0.2.1+mu436 | GDN/flash_attn/fp8_gemm kernel |
| tilelang_musa | 0.1.8+musa.3 | 融合 kernel 编写 |
| FastAPI + uvicorn | latest | HTTP 服务 |
| safetensors | latest | 权重加载 |

**API 协议优先级**：Anthropic Messages API（`/v1/messages`）优先，OpenAI-compatible 作为后续兼容层。

**不引入**：SGLang、vLLM、triton（除 tilelang 内部使用）、flashinfer

---

## 四、Project Structure

```
qwen35_s5000/                    ← 框架根目录
├── config.py                    ← 硬编码的模型/硬件配置，唯一配置文件
├── server.py                    ← FastAPI HTTP 服务（Anthropic Messages API 优先）
├── scheduler.py                 ← Continuous batching 调度器
├── model_runner.py              ← Forward pass 编排，管理 TP 进程
├── kv_cache.py                  ← Paged KV cache（GPU + CPU offload）
├── loader.py                    ← FP8 safetensors 权重加载 + TP sharding
│
├── model/
│   ├── qwen35_layer.py          ← 单层 forward（GDN + MoE + Attention）
│   └── qwen35_model.py          ← 完整模型 forward（60 层循环）
│
├── kernels/
│   ├── fused_decode.py          ← tilelang 融合 decode kernel（核心优化）
│   ├── moe_dispatch.py          ← MoE routing + expert GEMM（mate 封装）
│   ├── attention.py             ← flash_attn_with_kvcache 封装
│   └── gdn.py                   ← gated_delta_rule_decode 封装
│
├── bench/
│   ├── bench_decode.py          ← 单流 decode 吞吐测试
│   ├── bench_e2e.py             ← 端到端延迟测试
│   └── test_correctness.py      ← 与参考实现对比
│
└── scripts/
    ├── launch.sh                ← 启动命令（含 torchrun TP=8）
    └── warmup_kernels.py        ← 预编译所有 tilelang kernel
```

---

## 五、核心设计决策

### 5.1 并行策略：TP=8（固定）

8 卡 Tensor Parallel，不做 EP（Expert Parallel）。原因：
- EP 需要 All-to-All 通信，MCCL 在 S5000 上有已知问题
- TP=8 下每卡权重 ~47GB，KV cache ~30GB，显存够用
- 后续可以在 TP=8 基础上叠加 EP

### 5.2 tilelang 融合目标（decode 热路径）

不做完整 persistent kernel，只融合最高频的操作：

```
每层 decode step 的 kernel launch 数量对比：

未融合（eager）：
  RMSNorm(1) + QKV_proj(3) + GDN_decode(1) + Attn_proj(1) +
  MoE_gate(1) + Expert_GEMM×10(20) + AllReduce(2) + ...
  ≈ 40~50 launches/层 × 60 层 = 2400~3000 launches/step

融合目标（tilelang）：
  [RMSNorm + QKV_proj + GDN_decode] → 1 fused kernel
  [MoE_gate + top10_select]         → 1 fused kernel
  Expert_GEMM×10                    → 1 batched kernel（mate ragged_k）
  [Attn_proj + AllReduce]           → 1 kernel
  ≈ 6~8 launches/层 × 60 层 = 360~480 launches/step
```

**优先融合**（按收益排序）：
1. `RMSNorm + QKV_proj`：每层必跑，融合省 3 次 launch
2. `MoE gate + routing`：512 expert softmax + top-10 select，单独跑很贵
3. `Expert GEMM batching`：10 个 expert 合并成 1 次 batched GEMM

### 5.3 KV Cache：Paged + CPU Offload

```python
PAGE_SIZE = 16          # tokens per page
GPU_KV_PAGES = ...      # 由剩余 HBM 决定（~261GB / KV_size_per_page）
CPU_KV_PAGES = ...      # 2TB RAM 的一部分，用于 offload

# KV cache 大小估算（per layer, per page）：
# 2 × NUM_KV_HEADS × HEAD_DIM × PAGE_SIZE × sizeof(fp16)
# = 2 × 2 × 256 × 16 × 2 = 32KB per layer per page
# 60 layers: 60 × 32KB = 1.92MB per page
# GPU 261GB / 1.92MB ≈ 135,000 pages → 135,000 × 16 = 2.16M tokens on GPU
```

CPU offload 策略：decode 阶段 KV 全在 GPU；prefill 超长序列时 offload 到 CPU。

### 5.4 调度器：极简 Continuous Batching

```python
# 固定参数，不做动态调整
CHUNKED_PREFILL_SIZE = 2048   # 已验证有效
MAX_RUNNING_REQUESTS = 256
MAX_DECODE_BATCH     = 256    # decode batch size 上限
```

不实现：RadixAttention、prefix caching、LoRA、speculative decoding（Phase 1）

---

## 六、Commands

```bash
# 环境（在容器内）
docker exec -it sglang-musa5-dsv4 bash

# 预热所有 tilelang kernel（首次运行，约 5~10 min）
python qwen35_s5000/scripts/warmup_kernels.py \
    --model-path /data/models/Qwen_Qwen3.5-397B-A17B-FP8

# 启动服务（TP=8）
torchrun --nproc-per-node=8 qwen35_s5000/server.py \
    --model-path /data/models/Qwen_Qwen3.5-397B-A17B-FP8 \
    --port 8000

# 单流 decode 吞吐测试
python qwen35_s5000/bench/bench_decode.py \
    --base-url http://localhost:8000 \
    --batch 1 --input-len 128 --output-len 512 --rounds 20

# 端到端延迟测试
python qwen35_s5000/bench/bench_e2e.py \
    --base-url http://localhost:8000 \
    --input-len 4096 --output-len 256

# 正确性验证（对比 SGLang 参考输出）
python qwen35_s5000/bench/test_correctness.py \
    --model-path /data/models/Qwen_Qwen3.5-397B-A17B-FP8
```

---

## 七、Code Style

极简，无不必要抽象。示例：

```python
# kernels/gdn.py — 直接封装 mate，不加 class 层
import torch
import mate.gdn_decode as _gdn_dec

def gdn_decode(
    q: torch.Tensor,       # [B, 1, GDN_NUM_K_HEADS, GDN_KEY_DIM]
    k: torch.Tensor,
    v: torch.Tensor,       # [B, 1, GDN_NUM_V_HEADS, GDN_VALUE_DIM]
    state: torch.Tensor,   # [B, GDN_NUM_V_HEADS, GDN_VALUE_DIM, GDN_KEY_DIM]
    A_log: torch.Tensor,   # [GDN_NUM_V_HEADS] fp32
    a: torch.Tensor,       # [B, 1, GDN_NUM_V_HEADS]
    dt_bias: torch.Tensor, # [GDN_NUM_V_HEADS] fp32
    b: torch.Tensor,       # [B, 1, GDN_NUM_V_HEADS]
) -> tuple[torch.Tensor, torch.Tensor]:
    return _gdn_dec.gated_delta_rule_decode(q, k, v, state, A_log, a, dt_bias, b)
```

规范：
- 函数优于类，除非需要管理状态
- 类型注解必须有
- 注释只写 WHY，不写 WHAT
- 所有 tensor shape 在函数签名注释里标注
- 不用 `**kwargs` 传参，参数显式列出

---

## 八、Testing Strategy

| 测试类型 | 工具 | 位置 | 触发时机 |
|----------|------|------|----------|
| Kernel 正确性 | pytest | `bench/test_correctness.py` | 每次 kernel 修改后 |
| 吞吐 benchmark | 自定义脚本 | `bench/bench_decode.py` | 每次优化后 |
| 端到端 smoke test | curl | `bench/bench_e2e.py` | 服务启动后 |

不做单元测试覆盖率要求，只做关键路径的正确性验证。

---

## 九、Boundaries

**Always（必须做）**：
- 每次修改 kernel 后运行 `test_correctness.py`
- 所有 tensor 操作标注 shape
- tilelang kernel 首次编译后缓存到磁盘（`~/.cache/tilelang/`）

**Ask first（先确认）**：
- 修改 TP 并行策略（当前固定 TP=8）
- 引入新的 Python 依赖
- 修改 KV cache 的 page_size（影响所有 shape）
- 启用 CPU KV offload（需要测试 PCIe 带宽影响）

**Never（不做）**：
- 不做 MUSA Graph（SDK 5.1.0 broken，浪费时间）
- 不做多模态、LoRA、PP
- 不做通用模型支持（只支持 Qwen3.5-397B-FP8）
- 不在 decode 热路径里做 Python 级别的 per-token 逻辑

---

## 十、Open Questions（需要确认）

1. **MoE expert GEMM**：`mate.gemm.ragged_k_moe_gemm_8bit` 的 `ragged_tokens_info` 格式还未验证，需要看源码或联系摩尔线程确认。备选：用 10 次独立 `gemm_fp8_nt_groupwise` 调用。

2. **tilelang kernel 缓存**：首次编译 gdn_prefill 需要 ~30s（3 个 kernel）。是否在容器启动时 AOT 编译所有 shape？还是接受首次请求慢？

3. **2TB RAM 使用策略**：
   - 方案 A：只做 KV cache offload（实现简单，支持更多并发）
   - 方案 B：Expert weight prefetch（预测下一步 routing，提前把 expert 权重搬到 GPU）
   - 方案 B 更激进，但 512 expert 的 routing 预测准确率未知

4. **AllReduce 通信**：MCCL 在 S5000 上有已知问题（旧分支有 fallback patch）。TP=8 的 AllReduce 是否稳定？需要先验证。

5. **首个可测试里程碑**：是先做"能跑通但慢"（eager，无融合），还是直接做融合版本？建议先做 eager baseline 确认正确性，再做融合优化。

---

## 十一、实现阶段划分

### Phase 0：基础设施（1 周）
- 权重加载（FP8 safetensors → TP sharding）
- KV cache 分配器
- TP 进程组初始化（验证 MCCL AllReduce）
- 单层 forward 正确性验证

### Phase 1：Eager Baseline（1 周）
- 完整 60 层 forward（GDN + MoE + Attention）
- Continuous batching 调度器
- HTTP 服务
- **里程碑**：能跑通，测出 baseline 吞吐（预期 30~80 tok/s）

### Phase 2A：tilelang 部分融合（2~3 周）【主线】
- Fuse RMSNorm + QKV_proj（tilelang）
- Fuse MoE gate + top-10 routing（tilelang）
- Batched expert GEMM（mate ragged_k 或 tilelang batched kernel）
- **里程碑**：150~200 tok/s 单流

### Phase 2B：persistent kernel PoC（与 2A 并行，2~3 周）【探索线】
**目标**：验证 tilelang_musa 能否支撑单层完整 persistent kernel，为长期 300~420 tok/s 铺路

```
单层 persistent kernel 融合目标：
  [RMSNorm → QKV_proj → GDN_decode → RMSNorm → Attn → Attn_proj]
  [RMSNorm → MoE_gate → top10 → Expert_GEMM×10 → reduce]
  目标：单层 decode 从 ~40 launches → 2 launches（2 个 persistent kernel）
```

- 先做单层 decode 的 persistent kernel（不考虑 TP，单卡验证正确性）
- 验证 register pressure 和 shared memory 是否可行（S5000 架构限制）
- 如果单层可行，Phase 3 做完整 60 层

### Phase 3：KV Cache 优化 + Anthropic API（1 周）
- CPU KV offload（支持长上下文，chunk_size ≥ 4096）
- Anthropic Messages API 完整实现（`/v1/messages`，streaming SSE）
- **里程碑**：32K 上下文 TTFT < 1.5s，Anthropic API 可用

### Phase 4：persistent kernel 完整版（3~4 周，视 2B 结果决定）
- 全 60 层 persistent kernel
- TP=8 下的跨卡 AllReduce 融合进 kernel
- **里程碑**：300~420 tok/s 单流，接近 S5000 硬件上限
