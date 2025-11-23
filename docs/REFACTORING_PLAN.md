# 项目结构重构计划

本文档记录项目结构的当前问题和建议的重构方案。

## 当前问题概览

### 1. 双核心库问题

项目存在两个"核心"模块，职责划分不清：

| 模块 | 位置 | 职责 | 问题 |
|------|------|------|------|
| `hace_core` | `/hace_core/` | 配置、模型加载、数据集、VLLM集成 | 主包，命名清晰 |
| `core_code` | `/src/core_code/` | 统一分配器、策略选择、内存管理 | 第二核心，命名模糊 |

**影响**：
- 开发者困惑：不知道新功能应该放在哪里
- 导入混乱：部分文件用 `hace_core.`，部分用 `src.core_code.`
- 潜在重复：两处都有类似功能的实现

### 2. 数据加载重复

| 文件 | 功能 |
|------|------|
| `hace_core/data/dataset_loader.py` | HuggingFace datasets 加载 |
| `src/common/data_loader.py` | 本地文件数据加载 |

**问题**：功能重叠，应合并为统一接口。

### 3. 内存管理类名称相似

| 类 | 位置 | 用途 |
|----|------|------|
| `MemoryManager` | `hace_core/models/vllm_integration.py` | GPU 显存生命周期管理 |
| `UnifiedMemoryManager` | `src/core_code/memory_manager.py` | KV Cache 锚点协调 |

**结论**：虽然名称相似，但用途完全不同，无需合并。建议重命名以区分：
- `GPUMemoryManager` (VLLM专用)
- `KVCacheMemoryManager` (算法专用)

### 4. 散落的方法实现

```
src/
├── methods/
│   ├── cake/         # CAKE 方法
│   ├── h2o/          # H2O 方法
│   └── full_cache/   # Full Cache 基线
├── cake_runner/      # CAKE 运行器（应与 cake/ 合并？）
└── monitoring/       # 监控工具
```

---

## 建议的目标结构

```
Experiment-Platform/
├── hace/                           # 统一主包（从 hace_core 重命名）
│   ├── __init__.py
│   ├── config.py                   # 配置
│   │
│   ├── core/                       # 核心算法（从 src/core_code 迁入）
│   │   ├── __init__.py
│   │   ├── allocator.py            # UnifiedCakeAdaKVAllocator
│   │   ├── kv_memory.py            # KV Cache 内存管理
│   │   ├── strategy.py             # 策略选择
│   │   ├── normalizer.py           # 指标归一化
│   │   └── integration.py          # 集成框架
│   │
│   ├── models/                     # 模型相关
│   │   ├── __init__.py
│   │   ├── loader.py               # 模型加载
│   │   ├── converters/             # CAKE/H2O 转换器
│   │   │   ├── cake.py
│   │   │   └── h2o.py
│   │   └── backends/               # 推理后端
│   │       ├── __init__.py
│   │       ├── base.py             # BaseInferenceBackend
│   │       ├── huggingface.py      # HuggingFaceBackend
│   │       └── vllm/               # VLLM 相关
│   │           ├── __init__.py
│   │           ├── backend.py      # VLLMBackend
│   │           ├── attention.py    # AttentionCollector
│   │           └── memory.py       # GPU MemoryManager
│   │
│   ├── data/                       # 数据加载（合并两个 loader）
│   │   ├── __init__.py
│   │   ├── loader.py               # 统一数据加载接口
│   │   ├── hf_datasets.py          # HuggingFace datasets
│   │   └── local_files.py          # 本地文件加载
│   │
│   ├── methods/                    # KV 优化方法（从 src/methods 迁入）
│   │   ├── __init__.py
│   │   ├── base.py                 # 基类
│   │   ├── cake/
│   │   ├── h2o/
│   │   └── full_cache/
│   │
│   └── utils/                      # 工具
│       ├── __init__.py
│       └── monitoring.py           # 监控
│
├── scripts/                        # 运行脚本
│   ├── run_experiments.py
│   └── download.py
│
├── evaluation/                     # 评估（保持独立）
│   ├── baselines/
│   └── experiments/
│
├── tests/                          # 测试
│   ├── unit/
│   ├── integration/
│   └── baselines/
│
├── configs/                        # 配置文件
├── data/                           # 数据目录
├── docs/                           # 文档
├── models/                         # 本地模型缓存
└── runs/                           # 运行输出
```

---

## 重构步骤

### Phase 1: 准备工作（低风险）

- [x] 清理根目录杂乱文件
- [x] 修复移动文件的路径引用
- [ ] 创建兼容层（alias imports）
- [ ] 添加 deprecation warnings

### Phase 2: 合并数据加载（中风险）

```python
# 目标：hace/data/loader.py

def load_dataset(source: str, **kwargs):
    """统一数据加载接口

    Args:
        source: 数据源
            - "hf://dataset_name" -> HuggingFace
            - "local://path/to/file" -> 本地文件
            - "dataset_name" -> 自动检测
    """
    if source.startswith("hf://"):
        return load_from_huggingface(source[5:], **kwargs)
    elif source.startswith("local://"):
        return load_from_local(source[8:], **kwargs)
    else:
        return auto_detect_and_load(source, **kwargs)
```

### Phase 3: 迁移 core_code（高风险）

1. 在 `hace/core/` 创建新模块
2. 添加 `src/core_code/` 的 alias imports
3. 逐步迁移调用方
4. 添加 deprecation warnings
5. 最终移除 `src/core_code/`

### Phase 4: 迁移 methods（中风险）

1. 将 `src/methods/` 移动到 `hace/methods/`
2. 合并 `src/cake_runner/` 到 `hace/methods/cake/`
3. 更新所有导入路径

### Phase 5: 清理（低风险）

1. 移除空目录
2. 更新文档
3. 添加迁移指南

---

## 兼容性策略

为避免破坏现有代码，使用 alias imports 保持向后兼容：

```python
# src/core_code/__init__.py (过渡期)

import warnings
from hace.core.allocator import UnifiedCakeAdaKVAllocator
from hace.core.strategy import StrategySelector
# ... 其他导出

warnings.warn(
    "src.core_code is deprecated, use hace.core instead",
    DeprecationWarning,
    stacklevel=2
)
```

---

## 优先级建议

| 优先级 | 任务 | 原因 |
|--------|------|------|
| **P0** | 保持现状稳定 | 当前代码可运行，不急于重构 |
| **P1** | 合并数据加载 | 功能重复最明显，影响范围小 |
| **P2** | 重命名 MemoryManager | 避免混淆，简单改动 |
| **P3** | 迁移 core_code | 工作量大，需要充分测试 |
| **P4** | 迁移 methods | 依赖 P3 完成 |

---

## 注意事项

1. **不要急于重构**：当前结构虽然混乱，但功能正常
2. **逐步迁移**：每次只改一小部分，确保测试通过
3. **保持兼容**：使用 alias imports 避免破坏现有代码
4. **文档先行**：在重构前更新相关文档

---

## 相关文件

- 当前结构分析：本文档
- VLLM 部署指南：`docs/VLLM_DEPLOYMENT.md`
- 清理报告归档：`docs/archive/`
