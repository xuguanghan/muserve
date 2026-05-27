# Spec: muserve model 层 rebase 到 sglang qwen3_5.py

**版本**: 1.0  **日期**: 2026-05-27  
**状态**: 待确认

---

## 一、Objective

当前 muserve 推理正确性存在严重问题（prefill hidden norm 逐层爆炸至 ~3M，B=1 decode MoE crash），
且 bug 排查代价极高（代码是从头自写，无法对照已验证实现逐行比较）。

**策略转变**：从 sglang 的 `qwen3_5.py` 拷贝经过生产验证的完整实现作为基线，
再将 muserve 中已验证的 MUSA/mate 优化逐步叠加进去。
每一步都可以对比"优化前 vs 优化后"的 logits，确保优化不破坏正确性。

### 为什么这比继续 debug 当前代码更快

| 维度 | 继续 debug 当前代码 | Rebase 到 sglang |
|------|--------------------|--------------------|
| 正确性基线 | 未知（要从头证明） | sglang 已在 CUDA 上验证 |
| bug 定位 | 大海捞针（60层×多算子） | diff 对比，范围极小 |
| 优化移植 | 每次改动都可能引入新 bug | 单点替换，可逐步回归 |
| 工程成本 | 持续高（无参照物） | 一次性搬运 + 对齐 |

---

## 二、成功标准

| 标准 | 具体指标 | 验证方法 |
|------|----------|----------|
| 正确性 | prefill/decode logits 误差 < 1e-2 vs SGLang 参考 | `scripts/test_accuracy.py` |
| Decode 吞吐 | ≥ 150 tok/s（B=8，60层，MUSA Graph） | `bench_multiround_*.py` |
| TTFT | < 1.5s（32K context） | `bench_loadpup_prefix_cached.py` |
| 无回归 | 已通过的 GDN decode kernel、MoE batched GEMM 不退化 | 对比 ms/step |

---

## 三、范围界定

### 迁移内容（从 sglang qwen3_5.py 借鉴）

| 组件 | sglang 来源 | 迁移方式 |
|------|------------|---------|
| GDN forward 数学逻辑 | `Qwen3_5GatedDeltaNet.forward()` | 翻译为 muserve 风格（函数式，无 nn.Module） |
| Attention forward 数学逻辑 | `Qwen3_5AttentionDecoderLayer.forward()` | 同上 |
| MoE forward 数学逻辑 | `Qwen2MoeSparseMoeBlock.forward()` | 同上 |
| 权重 key 命名/sharding | `qwen3_5.py` load_weights | 对齐 loader.py 中的 key mapping |
| QK GemmaRMSNorm 实现 | `GemmaRMSNorm`（`1+w` 风格） | 对齐当前 `rms_norm()` 实现 |
| GDN conv1d + g/beta 公式 | `Qwen3_5GatedDeltaNet.forward()` | 重点：对齐 g 公式防止 ≤0 下溢 |

### 不引入的 sglang 专有层

- `RadixAttention`、`RadixLinearAttention`（sglang KV cache 抽象，muserve 自己管理）
- `FusedMoE`（Triton kernel，MUSA 无法运行）
- `LayerCommunicator`、`ForwardBatch`（sglang serving 框架层）
- `QKVParallelLinear` 等 vllm 风格并行层（muserve 用手动 col/row shard）

### 保持不变的 muserve 层

- `server.py`、`scheduler.py`、`inference_loop.py`（serving 逻辑）
- `loader.py`（FP8 safetensors 加载 + TP sharding）
- `kv_cache.py`、`memory/kv_cache.py`（KV cache 管理）
- `model/prefix_cache.py`（prefix cache）
- `model/graph_decode.py`（MUSA Graph capture）
- `kernels/`（所有 mate/tilelang kernel wrapper）

---

## 四、已验证优化的移植清单

以下优化已在当前 muserve 实测有效，rebase 后必须保留：

| 优化 | 当前实现位置 | 效果 | Rebase 后保留方式 |
|------|------------|------|-----------------|
| MUSA Graph capture | `model/graph_decode.py` | 34→134 tok/s | 保持不变，对新 forward 重新 capture |
| GDN tilelang JITKernel 直接调用 | `qwen35_layer.py:gdn_decode_forward()` | 121ms→0.06ms | 移植 `_init_gdn_decode_kernel()` + 直接调用 |
| MoE batched GEMM | `qwen35_layer.py:moe_forward()` | 11520 launch→1 | 移植 `ragged_m_moe_gemm_8bit` 路径 |
| MoE unsort+reshape+sum | `qwen35_layer.py:moe_forward()` | 2.6x vs scatter_add_ | 移植 `down_out[inverse_order].reshape(...).sum()` |
| GDN A+B 投影融合 | `qwen35_layer.py:gdn_decode_forward()` | 节省 0.9ms/step | 移植单次 GEMM + split 输出 |
| FP8 activation quantization | `qwen35_layer.py:_fast_fp8_quantize()` | ~0.5ms/MoE | 保持不变 |

---

## 五、关键 bug 根因（rebase 时需要对照修正的点）

当前代码已知的正确性问题，rebase 时通过对齐 sglang 来修复：

| Bug | 根因（已知/疑似） | sglang 参考位置 | 修复验证 |
|-----|-----------------|----------------|---------|
| Prefill hidden norm 逐层爆炸 | attention output gate 顺序错误 + GDN qkvz split 错配 | `Qwen3_5AttentionDecoderLayer.forward()` | layer 0 logits 对齐后逐层验证 |
| GDN g 公式 g=0 下溢 → NaN | `g = exp(-exp(A_log) * softplus(...))` 可能产生 g==0 | `Qwen3_5GatedDeltaNet.forward()` 中的 g 计算 | 对比 g tensor 分布 |
| B=1 decode MoE crash | `ragged_moe_gemm_8bit` 对齐问题 | sglang MoE dispatch 逻辑 | B=1 eager decode 不 crash |
| conv1d 未按 TP 切分 | conv1d weight 在 prefill 中未按 rank shard | `_override_weight_loader` + `mamba_v2_sharded_weight_loader` | shape 对齐检查 |

---

## 六、实施计划（4个阶段，门控推进）

### Phase 1：建立 sglang 参考输出（Gate：layer 0 logits 误差 < 1e-2）

**目标**：在 S5000 上运行 sglang qwen3_5.py，取得 layer-by-layer logits 作为 ground truth。

**任务**：
- [ ] 确认 sglang 在 S5000/MUSA 上可以 eager 运行（即使很慢）
- [ ] 写 `scripts/capture_sglang_reference.py`：dump 每层 hidden state 到文件
- [ ] 如 sglang 无法在 MUSA 上运行，改用：在 CUDA 机器上用相同权重取 reference

**验收**：reference logits 文件存在，可复现

---

### Phase 2：复制 sglang 实现到 muserve，对齐正确性（Gate：全 60 层 logits < 1e-2）

**目标**：在 muserve 框架内运行 sglang 的数学逻辑（无任何 muserve 优化），验证正确性。

**任务**：
- [ ] 新建 `muserve/model/qwen35_layer_sglang.py`，复制 sglang GDN/Attn/MoE forward 数学逻辑
  - 去除 sglang 专有层（RadixAttention 等），替换为 muserve 的 mate kernel 调用
  - 保留 sglang 的数学顺序（norm → proj → gdn/attn → residual → norm → moe → residual）
- [ ] 更新 `muserve/model/qwen35_model.py`，使用新 layer 实现
- [ ] 运行 `scripts/test_accuracy.py`，逐层对比 vs reference
- [ ] 修复所有 logits 误差 > 1e-2 的层

**验收**：60 层全部 logits 误差 < 1e-2（prefill + decode B=1 eager）

---

### Phase 3：逐步叠加已验证优化（Gate：每个优化后 logits 不退化）

**目标**：在正确性已验证的基线上，逐一移植 muserve 的性能优化。

**顺序**（从风险最低到最高）：

1. **FP8 activation quantization**（纯量化，logits 误差应 < 1e-3）
2. **MoE unsort+reshape+sum**（数学等价，替换 scatter_add_）
3. **MoE batched GEMM**（`ragged_m_moe_gemm_8bit`，验证 B=1 不 crash）
4. **GDN A+B 投影融合**（单 GEMM split，验证 shape 对齐）
5. **GDN tilelang JITKernel 直接调用**（验证与 mate API 输出一致）
6. **MUSA Graph capture**（整体 capture，验证 graph replay logits == eager logits）

每步任务：
- [ ] 应用优化
- [ ] 运行 `scripts/test_accuracy.py`（logits 对比）
- [ ] 运行 `bench_multiround_*.py`（ms/step 对比，确认有收益）

**验收**：所有优化叠加后，logits 误差 < 1e-2，decode 吞吐 ≥ 150 tok/s

---

### Phase 4：端到端 serving 验证（Gate：TTFT < 1.5s，API 正常）

**目标**：确认完整 serving 路径（prefill → decode → API 返回）正确。

**任务**：
- [ ] 修复 InferenceLoop shutdown hang（rank 0 broadcast shutdown 信号）
- [ ] 修复 B=1 eager decode MoE crash（如 Phase 3 未覆盖）
- [ ] 运行 `bench_loadpup_prefix_cached.py`，验证 TTFT < 1.5s（32K）
- [ ] 运行 `bench_multiround_anthropic.py`，验证 API 端到端

**验收**：完整对话流程可跑通，TTFT < 1.5s

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| sglang 无法在 MUSA 上运行，无法取 reference | 高 | 高 | 在有 CUDA 的机器上取 reference，权重相同即可 |
| Phase 2 中 sglang 数学逻辑翻译出错（仍有 bug） | 中 | 中 | 逐层对比，layer 0 先过再推进 |
| Phase 3 中某个优化破坏正确性 | 中 | 低 | 每步单独验证，可随时回滚 |
| mate tilelang kernel 在 rebase 后参数不对齐 | 低 | 高 | Phase 3 中单独验证 GDN kernel 输出 |

---

## 八、不在本 spec 范围内

- 300~420 tok/s 的算子融合（tilelang 单层全融合 kernel）：属于 architecture_300_400_toks.md Phase 4，本 spec 完成后继续
- EP（Expert Parallelism）：设计约束，不做
- 多 batch serving / continuous batching：不在当前目标内

---

## 九、Open Questions（需要确认）

1. **sglang 能否在 S5000/MUSA 上 eager 运行？** 这决定 Phase 1 的参考来源
2. **`scripts/test_accuracy.py` 当前是否已实现？** 还是需要新写
3. **rebase 后的文件命名**：直接替换 `qwen35_layer.py`，还是新建 `qwen35_layer_v2.py` 先并行？
4. **conv1d weight 在 loader.py 中是否已按 TP 正确 shard？** 这是 prefill NaN 的疑似根因之一
