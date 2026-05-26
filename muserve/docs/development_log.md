# Qwen3.5-397B-A17B-FP8 on MUSA (Moore Threads S4000) — 开发日志

## 项目概述

在 8 卡 Moore Threads S4000 (MUSA) GPU 上部署 Qwen3.5-397B-A17B-FP8 模型推理服务。
模型为混合架构：45 层 GatedDeltaNet (GDN) 线性注意力 + 15 层标准 GQA attention + MoE。

## 硬件环境

- GPU: 8 × Moore Threads S4000 (MTT S4000)
- 显存: ~80 GB/卡
- 互联: MCCL (Moore Threads Collective Communication Library)
- 软件栈: torch_musa, mate (FP8 GEMM + GDN kernels), tilelang

## 模型架构（从 config.json 确认）

```
总层数: 60
  - GDN 层 (45): [0,1,2,4,5,6,8,9,10,...] (非 4n+3 的层)
  - Attention 层 (15): [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59]
Hidden size: 4096
Vocab size: 248320
MoE: 512 experts, top-10 routing, intermediate=1024
GDN: 16 K-heads (dim=128), 64 V-heads (dim=128)
Attention: 32 Q-heads (dim=256), 2 KV-heads, output gate
TP: 8-way tensor parallelism
```

## Phase 0: 基础验证 ✅ 完成

| Task | 状态 | 结果 |
|------|------|------|
| 0.1 MCCL AllReduce | ✅ | 100 轮稳定，误差 0.00 |
| 0.2 权重加载 + TP sharding | ✅ | 406 tensors/层，64 experts/rank |
| 0.3 Paged KV Cache | ✅ | alloc/free/OOM/读写全部正确 |
| 0.4 单层 forward | ✅ | GDN + MoE + FP8 GEMM 全部通过 |

### 关键发现与修复

1. **weight_scale_inv 必须是 float32** — checkpoint 中存储为 bfloat16，`gemm_fp8_nt_groupwise` 要求 float32
2. **GDN reshape 后需要 .contiguous()** — tilelang kernel 对 stride 有要求
3. **in_proj_qkv 中 q/k/v 维度不等** — q=2048, k=2048, v=8192，不能简单 `//3` 切分
4. **out_proj 不能 TP 切分** — GDN 输出需要 AllGather 到全局 8192 维再做 out_proj
5. **A_log/dt_bias 需要按 TP rank 取切片** — 全局 64 heads，每卡只用 8 heads
6. **FP8 动态 per-block scaling** — 激活值超出 fp8e4m3 范围(±448)导致 NaN

## Phase 1: Eager Baseline ✅ 完成

| Task | 状态 | 结果 |
|------|------|------|
| 1.1 完整 60 层 forward | ✅ | Prefill + Decode 无 NaN，52.5 GB/卡 |
| 1.2 API Server | ✅ | OpenAI-compatible，SSE streaming |
| 1.3 端到端生成 | ✅ | 8 tokens 生成成功 |

### 性能指标（Eager Baseline，未优化）

```
权重加载: 106.8s (60 层，52.5 GB/卡)
Prefill (16 tokens): 18.3s (含首次 kernel JIT 编译)
Decode throughput: 0.15 tok/s (单流)
Decode latency: ~6.5s/token (60 层)
```

### 性能瓶颈分析

1. **MoE Python 循环** (占 decode 90%+): 每层 64 experts 串行 GEMM
2. **AllGather 延迟**: GDN 输出 AllGather 每层一次
3. **FP8 动态 scaling 开销**: 每次 GEMM 前计算 per-block scale
4. **无 KV Cache**: Attention 层每步重算（无历史 token 缓存）
5. **Kernel JIT**: tilelang 首次编译耗时大

## Phase 2: Kernel 优化 🔧 进行中

### Task 2.1: Batched MoE Routing ✅

文件: `muserve/kernels/batched_moe.py`

优化策略:
- 按 expert 分组 token，批量调用 GEMM
- 向量化权重累加（替代逐 token 循环）
- 预期提升: 3-5x decode 速度

### Task 2.2: FP8 GEMM 优化（规划中）

- 静态 calibration scale（替代动态 per-block）
- 预计算 x_scale 缓存
- 预期提升: 1.5-2x GEMM 速度

### Task 2.3: GDN AllGather 优化（规划中）

- AllGather 与 out_proj 计算重叠
- 或改为 out_proj 行切分 + AllReduce（减少通信量）

## Phase 3: 系统优化（规划中）

### 3.1 Continuous Batching
- 动态 batch 调度器
- Prefill/Decode 分离
- GDN state 管理（per-request）

### 3.2 KV Cache for Attention Layers
- 15 个 attention 层需要 KV cache
- Paged KV cache 已验证（Phase 0.3）
- 需要集成到 attention forward

### 3.3 Prefix Caching
- System prompt 共享
- GDN state 复用

## Phase 4: 生产就绪（规划中）

### 4.1 API 完整性
- Chat/Completions 端点 ✅
- Token streaming (SSE) ✅
- 多轮对话支持
- 并发请求处理

### 4.2 监控与部署
- Prometheus metrics
- 健康检查端点 ✅
- Docker 部署配置
- 负载均衡

## 代码结构

```
muserve/
├── __init__.py
├── config.py              # 模型配置常量
├── distributed.py         # MCCL 分布式通信
├── loader.py              # 权重加载 + TP sharding
├── server.py              # Flask API server
├── model/
│   ├── qwen35_layer.py    # 单层 forward (GDN + Attention + MoE)
│   └── qwen35_model.py    # 完整模型 forward
├── kernels/
│   ├── batched_moe.py     # Phase 2: 向量化 MoE
│   ├── fp8_matmul.py      # FP8 GEMM wrapper
│   ├── gdn.py             # GDN kernel wrapper
│   ├── attention.py       # Standard attention
│   ├── moe_dispatch.py    # MoE token dispatch
│   └── sparse_attn.py     # Sparse attention
├── memory/
│   └── paged_kv_cache.py  # Paged KV cache 分配器
├── scripts/
│   ├── test_single_layer.py  # Phase 0.4 验证
│   ├── test_forward.py       # Phase 1.1 验证
│   └── test_generate.py      # Phase 1.3 验证
└── docs/
    └── development_log.md    # 本文档
```

## TP Sharding 策略

| 组件 | 原始 shape | 切分方式 | 每卡 shape |
|------|-----------|---------|-----------|
| embed_tokens | [248320, 4096] | 行切分 | [31040, 4096] |
| in_proj_qkv | [12288, 4096] | 列切分 | [1536, 4096] |
| in_proj_a/b | [64, 4096] | 列切分 | [8, 4096] |
| in_proj_z | [8192, 4096] | 列切分 | [1024, 4096] |
| out_proj (GDN) | [4096, 8192] | 不切分 | [4096, 8192] |
| q_proj (Attn) | [16384, 4096] | 列切分 | [2048, 4096] |
| k_proj/v_proj | [512, 4096] | 不切分 | [512, 4096] |
| o_proj (Attn) | [4096, 8192] | 不切分 | [4096, 8192] |
| Expert weights | [1024, 4096] | 按 expert 分配 | 64 experts/卡 |
| gate (router) | [512, 4096] | 不切分 | [512, 4096] |

## 最终推理性能指标

### Eager Baseline (Phase 1, 当前)

| 指标 | 值 | 备注 |
|------|-----|------|
| 权重显存 | 52.5 GB/卡 | 60 层 FP8 + scale |
| Prefill latency | 18.3s (16 tok) | 含 JIT 编译 |
| Decode latency | 6.5s/token | 60 层串行 |
| Decode throughput | 0.15 tok/s | 单流 |
| TTFT (首 token) | ~18s | Prefill 时间 |

### 预期优化后 (Phase 2-3 完成后)

| 指标 | 预期值 | 优化来源 |
|------|--------|---------|
| Prefill latency | 2-3s (16 tok) | Kernel 缓存 + 优化 |
| Decode latency | 0.5-1s/token | Batched MoE + 静态 scale |
| Decode throughput | 1-2 tok/s | 单流 |
| Batch throughput | 5-10 tok/s | Continuous batching (batch=8) |
| TTFT | 2-3s | Kernel 预编译 |

## 已知问题

1. **进程退出 SIGABRT**: `destroy_distributed()` 后 MCCL 析构顺序问题，不影响正确性
2. **FlashAttention 不支持 head_dim=256**: MUSA 后端回退到 math attention
3. **Kernel JIT 首次编译慢**: tilelang 编译 ~15s，需要预热或缓存
4. **greedy_sample 跨卡同步**: 当前用 AllReduce 实现，可优化为单卡采样+broadcast
