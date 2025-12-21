# VLLM 部署指南

本文档介绍如何在 Experiment-Platform 中使用 VLLM 高性能推理引擎。

## ⚠️ 重要工程约束

在开始之前，请仔细阅读以下关键限制：

1. **显存共存问题**: HF 模型（用于 Warmup）和 VLLM 引擎**无法在同一进程内同时存在**
2. **Prefix Caching 必须禁用**: CAKE/AdaKV 对不同请求应用不同策略，会导致缓存命中错误
3. **仅支持 TP=1**: 多卡张量并行下的 Attention 数据聚合逻辑尚未实现
4. **Token-to-Block 对齐**: CAKE/AdaKV 的 Token 级预算会被对齐到 VLLM 的 Block 级
5. **Attention 无法直接获取**: VLLM 的 fused kernel 不导出 attention scores，必须用 HF 外部预热



## 目录

1. [概述](#概述)
2. [安装要求](#安装要求)
3. [配置说明](#配置说明)
4. [使用方式](#使用方式)
5. [串行执行管道](#串行执行管道)
6. [注意力数据收集](#注意力数据收集)
7. [CAKE/AdaKV 策略集成](#cakeadakv-策略集成)
8. [工程约束详解](#工程约束详解)
9. [常见问题](#常见问题)

## 概述

VLLM 是一个高性能的 LLM 推理引擎，支持 PagedAttention、Continuous Batching 等优化技术。本平台支持两种后端：

- **HuggingFace (hf)**: 默认后端，基于 transformers 库，支持 `output_attentions=True`
- **VLLM (vllm)**: 高性能后端，适合生产部署，吞吐量更高

### 后端对比

| 特性 | HuggingFace | VLLM |
|------|-------------|------|
| 吞吐量 | 中等 | 高 |
| 延迟 | 较高 | 较低 |
| 注意力输出 | 原生支持 | 需要外部收集 |
| KV Cache 控制 | 有限 | PagedAttention |
| 张量并行 | 需要 accelerate | 原生支持 |
| 部署模式 | 仅进程内 | 进程内/Server |

## 安装要求

### 硬件要求

- NVIDIA GPU（推荐 RTX 3090/4090 或更高）
- CUDA 11.8 或更高版本
- 至少 24GB 显存（运行 7B 模型）

### 软件要求

```bash
# 基础依赖（已在 requirements.txt 中）
pip install torch>=2.0.0 transformers>=4.43.0

# VLLM 安装（手动安装）
pip install vllm>=0.4.0

# 可选：xformers（提升性能）
pip install xformers>=0.0.23

# 可选：Ray（分布式推理）
pip install ray>=2.9.0
```

### 验证安装

```python
from hace_core.models.model_loader import detect_available_backends

backends = detect_available_backends()
print(backends)
# {'huggingface': True, 'vllm': True}
```

## 配置说明

### 基础配置

在 `hace_core/config.py` 中配置推理后端：

```python
MODEL_CONFIG = {
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
    "precision": "fp16",
    "inference_backend": "vllm",  # 切换到 VLLM
}
```

### VLLM 专用配置

```python
VLLM_CONFIG = {
    # 引擎模式
    "mode": "inprocess",  # "inprocess" 或 "server"

    # 多进程启动方式（重要）：CUDA 场景推荐使用 spawn，避免 fork 后 CUDA 复初始化报错
    # 等价环境变量：VLLM_WORKER_MULTIPROC_METHOD=spawn
    "worker_multiproc_method": "spawn",

    # 注意力后端（可选）：默认 FLASH_ATTN 在部分新 GPU/驱动组合下可能不兼容，可切到 TRITON_ATTN
    # 等价环境变量：VLLM_ATTENTION_BACKEND=TRITON_ATTN
    # 可选值示例：FLASH_ATTN / TRITON_ATTN / FLASHINFER / ...
    "attention_backend": "TRITON_ATTN",

    # 进程内模式配置
    "tensor_parallel_size": 1,      # 张量并行（多GPU）
    "pipeline_parallel_size": 1,    # 流水线并行
    "gpu_memory_utilization": 0.90, # GPU 显存利用率
    "max_model_len": 4096,          # 最大上下文长度
    "enforce_eager": False,         # 禁用 CUDA Graph
    "trust_remote_code": True,

    # KV Cache 配置
    "block_size": 16,               # PagedAttention 块大小
    "enable_prefix_caching": False,

    # Server 模式配置
    "server_url": "http://localhost:8000",
    "timeout": 300,

    # 采样参数
    "sampling_params": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    },

    # 注意力收集配置
    "attention_collection": {
        "enabled": False,
        "mode": "external_warmup",
        "warmup_samples": 10,
    },
}
```

### 多 GPU 配置

```python
VLLM_CONFIG = {
    "tensor_parallel_size": 2,  # 使用 2 个 GPU
    "gpu_memory_utilization": 0.95,
}
```

## 使用方式

### 方式一：使用推理后端抽象

```python
from hace_core.models.inference_backend import create_inference_backend, GenerationConfig
from hace_core.config import MODEL_CONFIG, VLLM_CONFIG

# 切换到 VLLM
MODEL_CONFIG["inference_backend"] = "vllm"

# 创建后端
backend = create_inference_backend(MODEL_CONFIG, VLLM_CONFIG)
backend.initialize()

# 生成文本
config = GenerationConfig(max_new_tokens=256, temperature=0.0)
output = backend.generate("What is the capital of France?", config)
print(output.text)

# 清理
backend.cleanup()
```

### 方式二：使用 model_loader

```python
from hace_core.models.model_loader import load_inference_backend
from hace_core.config import MODEL_CONFIG, VLLM_CONFIG

MODEL_CONFIG["inference_backend"] = "vllm"
backend = load_inference_backend(MODEL_CONFIG, VLLM_CONFIG)

# 使用 backend...
```

### 方式三：运行实验脚本

```bash
# 使用 HuggingFace 后端（默认）
python scripts/run_experiments.py --experiment baseline --backend hf

# 使用 VLLM 后端
python scripts/run_experiments.py --experiment baseline --backend vllm

# VLLM 多 GPU
python scripts/run_experiments.py --experiment cake --backend vllm --vllm-tp 2
```

### 方式四：Server 模式

1. 启动 VLLM Server：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8000 \
    --tensor-parallel-size 1
```

2. 配置 Server 模式：

```python
VLLM_CONFIG = {
    "mode": "server",
    "server_url": "http://localhost:8000",
}
```

3. 使用：

```python
backend = create_inference_backend(MODEL_CONFIG, VLLM_CONFIG)
backend.initialize()
output = backend.generate("Hello!")
```

## 注意力数据收集

VLLM 默认不返回注意力权重，但 CAKE/AdaKV 策略需要注意力数据。平台提供两种解决方案：

### 方案一：外部预热（推荐）

使用 HuggingFace 模型收集注意力数据，然后在 VLLM 中使用优化后的 KV 预算：

```python
from hace_core.models.attention_collector import (
    AttentionCollector,
    AttentionCollectionConfig,
    VLLMAttentionAdapter,
)
from hace_core.config import MODEL_CONFIG

# 1. 配置注意力收集
config = AttentionCollectionConfig(
    mode="external_warmup",
    warmup_samples=10,
    cache_file="./cache/attention_data.pkl",
)

# 2. 收集注意力数据
collector = AttentionCollector(config)
sample_prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    # ... 更多样本
]
attention_data = collector.collect_from_hf_model(MODEL_CONFIG, sample_prompts)

# 3. 计算 KV 预算
adapter = VLLMAttentionAdapter(attention_data)
layer_budgets = adapter.compute_layer_budgets(total_cache_size=4096)
head_budgets = adapter.compute_head_budgets(layer_budgets)

# 4. 导出 VLLM KV 配置
vllm_kv_config = adapter.to_vllm_kv_config(
    total_cache_size=4096,
    block_size=16
)
print(vllm_kv_config)
```

### 方案二：缓存复用

收集一次注意力数据后缓存，后续直接加载：

```python
# 首次运行：收集并缓存
config = AttentionCollectionConfig(
    cache_file="./cache/attention_data.pkl",
    use_cache=True,
)
collector = AttentionCollector(config)
attention_data = collector.collect_from_hf_model(MODEL_CONFIG, sample_prompts)

# 后续运行：直接加载缓存
from hace_core.models.attention_collector import AttentionData
attention_data = AttentionData.load("./cache/attention_data.pkl")
```

## CAKE/AdaKV 策略集成

### 混合推理管道

当使用 VLLM 且需要 CAKE/AdaKV 优化时：

```python
from hace_core.models.model_loader import create_hybrid_inference_pipeline
from hace_core.config import MODEL_CONFIG, VLLM_CONFIG

MODEL_CONFIG["inference_backend"] = "vllm"
VLLM_CONFIG["attention_collection"]["enabled"] = True

# 创建混合管道
pipeline = create_hybrid_inference_pipeline(
    MODEL_CONFIG,
    VLLM_CONFIG,
    optimization_method="cake"
)

# 使用注意力收集器
if pipeline["attention_collector"]:
    attention_data = pipeline["attention_collector"].collect_from_hf_model(
        MODEL_CONFIG,
        sample_prompts
    )
    # 计算预算...

# 使用 VLLM 进行推理
backend = pipeline["inference_backend"]
backend.initialize()
output = backend.generate("Your prompt here")
```

### 预算映射到 VLLM

将 CAKE/AdaKV 预算映射到 VLLM PagedAttention 块：

```python
adapter = VLLMAttentionAdapter(attention_data)
kv_config = adapter.to_vllm_kv_config(
    total_cache_size=4096,
    block_size=16
)

# kv_config 包含：
# - layer_budgets: 每层 token 预算
# - head_budgets: 每层每头 token 预算
# - layer_blocks: 每层 block 数量
```

## 串行执行管道

由于显存共存问题，必须使用串行执行策略。`SerialInferencePipeline` 封装了正确的执行顺序：

```python
from hace_core.models.vllm_integration import SerialInferencePipeline
from hace_core.config import MODEL_CONFIG, VLLM_CONFIG

# 创建管道
pipeline = SerialInferencePipeline(
    model_config=MODEL_CONFIG,
    vllm_config=VLLM_CONFIG,
    allocator_config={"total_cache_size": 4096}
)

# 阶段 1：收集注意力（加载 HF 模型，完成后释放）
sample_prompts = ["What is AI?", "Explain transformers."]
attention_data = pipeline.phase1_collect_attention(
    sample_prompts,
    cache_file="./cache/attn.pkl"  # 可选：缓存到文件
)

# 阶段 2：计算预算
request_allocator = pipeline.phase2_compute_budgets()

# 阶段 3：初始化 VLLM（此时 HF 显存已释放）
vllm_backend = pipeline.phase3_initialize_vllm()

# 执行推理
output = pipeline.generate(["Your question here"], request_id="req-001")

# 清理
pipeline.cleanup()
```

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    SerialInferencePipeline                   │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: HF Model Load → Attention Collection → Memory Free │
│  Phase 2: Budget Computation → Block Alignment               │
│  Phase 3: VLLM Engine Init (safe, HF memory released)       │
│  Generate: Request-Scoped Context → VLLM Inference          │
└─────────────────────────────────────────────────────────────┘
```

## 工程约束详解

### 1. 显存生命周期管理

**问题**: VLLM 默认预占 GPU 90%+ 显存，HF 模型无法同时加载。

**解决方案**: 串行加载策略
```python
from hace_core.models.vllm_integration import MemoryManager

# 查看当前显存状态
info = MemoryManager.get_gpu_memory_info()
print(f"GPU 0: {info[0]['free_gb']:.2f} GB free")

# 强制清理（在 Phase 1 完成后自动调用）
MemoryManager.force_cleanup()
```

### 2. Token-to-Block 预算对齐

**问题**: CAKE/AdaKV 计算 Token 级预算，VLLM 使用 Block 级管理（默认 16 tokens/block）。

**解决方案**: `BudgetAligner` 智能对齐
```python
from hace_core.models.vllm_integration import BudgetAligner

aligner = BudgetAligner(block_size=16)

# 按重要性分配余数（推荐）
aligned = aligner.align_layer_budgets(
    layer_token_budgets=[100, 150, 80, 120],
    priority_scores=[0.9, 0.7, 0.3, 0.5],  # 层重要性
    rounding_strategy="importance_weighted"
)
# 结果: [6, 9, 5, 8] blocks (100/16≈6, 余数分给高优先级层)
```

### 3. VLLM 特性冲突

**必须禁用的特性**:

| 特性 | 原因 |
|------|------|
| `enable_prefix_caching` | CAKE 对不同请求可能应用不同策略，缓存会命中错误 |
| `tensor_parallel_size > 1` | Attention 数据聚合逻辑未实现 |
| `pipeline_parallel_size > 1` | Layer 级统计不完整 |

```python
from hace_core.models.vllm_integration import check_vllm_compatibility, get_safe_vllm_config

# 检查兼容性
report = check_vllm_compatibility(VLLM_CONFIG)
if not report.is_compatible:
    print("Errors:", report.errors)
    print("Recommendations:", report.recommendations)

# 自动获取安全配置
safe_config = get_safe_vllm_config(VLLM_CONFIG)
```

### 4. 请求级分配器

**问题**: VLLM Server 是高并发异步的，共享状态会导致数据污染。

**解决方案**: 每请求独立上下文
```python
from hace_core.models.vllm_integration import RequestScopedAllocator

allocator = RequestScopedAllocator(
    attention_data=attention_data,
    block_size=16
)

# 为每个请求创建独立上下文
ctx1 = allocator.create_context("req-001", input_length=512)
ctx2 = allocator.create_context("req-002", input_length=1024)

print(ctx1.aligned_budgets.total_tokens_aligned)
```

### 5. Tokenizer 对齐验证

**问题**: HF 和 VLLM Tokenizer 在 special tokens 处理上可能有差异。

```python
from hace_core.models.vllm_integration import TokenizerAlignmentChecker

# 检查对齐
is_aligned, mismatches = TokenizerAlignmentChecker.check_alignment(
    hf_tokenizer,
    vllm_tokenizer
)

if not is_aligned:
    print("WARNING: Tokenizer mismatch detected!")
    for m in mismatches:
        print(f"  - {m}")
```

## 常见问题

### 1) `Cannot re-initialize CUDA in forked subprocess`

vLLM 0.12+ 默认可能用 `fork` 启动子进程，在父进程触发过 CUDA 初始化后就会报这个错。解决：

- 设置环境变量：`VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 或在代码里把 `VLLM_CONFIG["worker_multiproc_method"] = "spawn"`

### 2) `cudaErrorUnsupportedPtxVersion` / `unsupported toolchain`

通常是默认 `FLASH_ATTN` 后端与当前 GPU/驱动/二进制不兼容导致。可切到 Triton 后端：

- 设置环境变量：`VLLM_ATTENTION_BACKEND=TRITON_ATTN`
- 或在代码里把 `VLLM_CONFIG["attention_backend"] = "TRITON_ATTN"`

### Q: VLLM 安装失败

A: VLLM 对 CUDA 版本有严格要求。请确保：
- CUDA 版本 >= 11.8
- PyTorch 与 CUDA 版本匹配
- 参考 [VLLM 官方安装指南](https://docs.vllm.ai/en/latest/getting_started/installation.html)

### Q: 显存不足

A: 尝试以下方案：
1. 降低 `gpu_memory_utilization`
2. 减小 `max_model_len`
3. 使用量化模型 (`quantization: "awq"`)
4. 使用多 GPU 张量并行

### Q: VLLM 与 CAKE 如何配合？

A: 由于 VLLM 不支持直接输出注意力，采用"外部预热"模式：
1. 使用 HuggingFace 模型收集注意力数据（离线或冷启动）
2. 使用 UnifiedCakeAdaKVAllocator 计算预算
3. 将预算配置应用到 VLLM 推理

### Q: Server 模式连接失败

A: 检查以下项：
1. VLLM Server 是否正常运行
2. 端口是否正确
3. 防火墙是否允许连接
4. 尝试直接访问 `http://localhost:8000/health`

### Q: 如何切换回 HuggingFace？

A: 修改配置：

```python
MODEL_CONFIG["inference_backend"] = "hf"
```

或使用命令行参数：

```bash
python scripts/run_experiments.py --backend hf
```

## 参考资源

- [VLLM 官方文档](https://docs.vllm.ai/)
- [VLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
