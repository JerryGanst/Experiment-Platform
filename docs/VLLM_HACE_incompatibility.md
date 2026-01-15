# 为什么 HACE 实验不能使用 VLLM

## 结论

**HACE 的头级差异化（head-level differentiation）机制与 VLLM 不兼容**。必须使用 HuggingFace transformers 进行推理。

---

## 背景

### HACE 的核心机制

HACE（Hierarchical Adaptive Cache Eviction）通过两个层级进行 KV Cache 预算分配：

1. **层级分配（Layer-level）**：不同 transformer 层获得不同的 cache 预算
2. **头级分配（Head-level）**：同一层内，不同 attention head 根据熵值获得不同权重

头级分配的关键是：**在 attention 计算过程中实时计算每个 head 的熵值**，然后用熵值调整 token 选择的分数权重。

### 实现方式

我们通过 **monkey patching** 修改了 Qwen2 的 attention 前向传播函数（`modify_qwen2.py`）：

```python
# modify_qwen2.py 中的关键代码
def qwen2_attn_forward_cake(self, ...):
    # 计算 attention weights
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # 计算每个 head 的熵
    attn_dist = attn_weights.mean(dim=2)  # [bsz, num_heads, kv_len]
    log_attn = torch.log(attn_dist + 1e-10)
    head_entropy = -(attn_dist * log_attn).sum(dim=-1)  # [num_heads]

    # 传递给 cache 管理器
    past_key_value.update_score(pref_score, hh_score, head_entropy)
```

这个修改后的函数：
1. 在每次 attention 计算时获取 attention weights
2. 计算每个 head 的熵值（attention 分布的分散程度）
3. 将熵值传递给 KV Cache 管理器用于后续的 token 选择

---

## 问题：VLLM 不兼容

### VLLM 的架构

VLLM 是一个高性能推理引擎，它**不使用 HuggingFace transformers 的前向传播代码**：

```
HuggingFace transformers:
  model.forward() → Qwen2Attention.forward() → 我们的 monkey patch 生效 ✓

VLLM:
  LLM.generate() → VLLM 内部 attention kernel → 完全绕过 transformers 代码 ✗
```

### 验证过程

我们在 GCP 服务器上进行了实验验证：

| 实验 | head_mode | 分数 |
|------|-----------|------|
| L1 baseline | 无 | 10.52 |
| LH1 | high_entropy | 10.52 |

**两个实验分数完全相同**，说明 `head_mode` 设置被完全忽略。

通过代码检查确认：
```bash
# modify_qwen2.py 中计算 head_entropy 的代码
head_entropy = -(attn_dist * log_attn).sum(dim=-1)
past_key_value.update_score(pref_score, hh_score, head_entropy)

# cake_cache.py 中使用 head_entropy 的条件
if head_entropy is not None and head_mode in ("high_entropy", "low_entropy"):
    # 应用熵加权...
```

在 VLLM 模式下：
- `modify_qwen2.py` 的代码**从未被调用**
- `head_entropies` 列表始终为空
- 条件 `if head_entropy is not None` 永远失败
- 所有 head_mode 设置被忽略

---

## 技术原因深入分析

### 1. Attention 实现路径不同

**HuggingFace transformers**:
```
Qwen2ForCausalLM.forward()
  → Qwen2Model.forward()
    → Qwen2DecoderLayer.forward()
      → Qwen2Attention.forward()  ← 我们的 monkey patch 在这里
```

**VLLM**:
```
LLM.generate()
  → vllm.engine.LLMEngine
    → vllm.worker.Worker
      → vllm.attention.Attention (自己的实现)
        → Flash Attention / Triton kernels  ← 完全不同的代码路径
```

### 2. 无法获取中间 Attention Weights

VLLM 使用高度优化的 CUDA/Triton kernels（如 Flash Attention），这些 kernels：
- **不返回完整的 attention weights**（为了节省显存）
- 直接计算 softmax(QK^T)V，中间结果不保存
- 无法在计算过程中提取熵值

### 3. 修改 VLLM 的代价

要让 HACE 在 VLLM 上工作，需要：
1. Fork VLLM 源码
2. 修改 attention kernel 实现，在计算过程中提取 attention weights
3. 添加熵计算逻辑
4. 将熵值传递给外部的 cache 管理器

这会：
- 破坏 VLLM 的内存优化（Flash Attention 不保存 attention weights 是设计选择）
- 降低推理性能
- 增加大量维护成本

---

## 解决方案

### 当前方案：使用 HuggingFace transformers

使用原始的 HF 推理脚本，HACE 的头级差异化逻辑可以正常工作：

```bash
# 运行实验
python3 src/third_party/cakekv-main/cakekv-main/experiments/LongBench/pred_cake.py \
    --model qwen2.5-7b-instruct \
    --compress \
    --cache_size 256 \
    --window_size 32 \
    --pred_name "hace_LH1_high_entropy"
```

### 性能对比

| 推理引擎 | HACE 兼容 | 相对速度 | 显存效率 |
|----------|-----------|----------|----------|
| HuggingFace | ✓ | 1x (基准) | 较低 |
| VLLM | ✗ | 3-5x | 较高 |

虽然 HF 较慢，但：
1. **实验正确性比速度更重要**
2. 单次实验 200 样本，HF 也能在合理时间内完成
3. 验证方法有效后，可以考虑工程优化

### 未来可能的优化方向

如果实验验证 HACE 头级策略有效，后续可考虑：

1. **静态熵分析**：离线预计算模型的 head 熵分布特性，运行时使用固定权重
2. **采样估计**：在少量样本上用 HF 计算熵，然后在 VLLM 推理时使用预计算的权重
3. **VLLM 插件开发**：开发自定义 attention 后端（工程量大）

---

## 总结

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| VLLM 下 head_mode 无效 | VLLM 不调用 HF 的 attention 代码 | 使用 HF transformers |
| 无法获取 attention weights | VLLM 的 Flash Attention 不返回中间结果 | 使用 HF 的 SDPA/eager attention |
| 熵值无法计算 | 依赖 attention weights | 使用 HF，或预计算静态权重 |

**结论**：当前阶段必须使用 HuggingFace transformers 进行 HACE 实验，以确保头级差异化逻辑正确生效。
