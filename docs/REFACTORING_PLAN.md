# 项目结构重构计划

本文档记录项目结构重构的完成状态。

## 重构完成状态

| Phase | 任务 | 状态 |
|-------|------|------|
| Phase 1 | 清理根目录 | ✅ 完成 |
| Phase 2 | 合并数据加载模块 | ✅ 完成 |
| Phase 3 | 迁移 core_code → hace_core/core | ✅ 完成 |
| Phase 4 | 迁移 methods → hace_core/methods | ✅ 完成 |
| Phase 5 | 清理废弃文件 | ✅ 完成 |

---

## 最终项目结构

```
Experiment-Platform/
├── hace_core/                    # 主核心库
│   ├── __init__.py
│   ├── config.py                 # 配置
│   ├── algorithms.py             # 基础算法
│   │
│   ├── core/                     # CAKE-AdaKV 核心算法
│   │   ├── __init__.py
│   │   ├── unified_allocator.py  # 统一分配器
│   │   ├── indicator_normalizer.py
│   │   ├── strategy_selector.py
│   │   ├── memory_manager.py     # KV Cache 内存管理
│   │   ├── integration_framework.py
│   │   └── launcher.py
│   │
│   ├── data/                     # 统一数据加载
│   │   ├── __init__.py
│   │   ├── unified_loader.py     # 统一接口 (hf://, local://)
│   │   └── dataset_loader.py     # HuggingFace 数据集
│   │
│   ├── models/                   # 模型相关
│   │   ├── __init__.py
│   │   ├── model_loader.py       # 模型加载
│   │   ├── inference_backend.py  # 推理后端 (HF/VLLM)
│   │   ├── attention_collector.py
│   │   ├── vllm_integration.py   # VLLM 集成工具
│   │   ├── cake_converter.py
│   │   ├── h2o_converter.py
│   │   └── cake_subprocess_adapter.py
│   │
│   ├── methods/                  # KV 优化方法
│   │   ├── __init__.py
│   │   ├── cake/                 # CAKE 方法 + 运行器
│   │   ├── h2o/                  # H2O 方法
│   │   └── full_cache/           # Full Cache 基线
│   │
│   └── utils/                    # 工具
│       ├── __init__.py
│       ├── unified_monitor.py    # 监控
│       └── plotter.py            # 绘图
│
├── src/                          # 第三方代码
│   └── third_party/
│       └── cakekv-main/          # 原始 CAKE 参考实现
│
├── scripts/                      # 运行脚本
├── evaluation/                   # 评估脚本
├── tests/                        # 测试
├── configs/                      # 配置文件
├── data/                         # 数据目录
├── docs/                         # 文档
├── models/                       # 本地模型缓存
└── runs/                         # 运行输出
```

---

## 导入路径

### 数据加载

```python
from hace_core.data import load_data

# 自动检测数据源
data = load_data("hotpotqa")

# HuggingFace
data = load_data("hf://THUDM/LongBench", subset="hotpotqa")

# 本地文件
data = load_data("local://data/custom.jsonl")
```

### 核心分配器

```python
from hace_core.core import (
    UnifiedCakeAdaKVAllocator,
    UnifiedCacheConfig,
    CakeAdaKVIntegration,
    IntegrationConfig,
)
```

### 模型加载

```python
from hace_core.models import load_model_and_tokenizer
from hace_core.models.inference_backend import (
    HuggingFaceBackend,
    VLLMBackend,
)
```

### 监控工具

```python
from hace_core.utils import UnifiedMonitor
```

---

## 已删除的目录

以下目录已被迁移到 `hace_core/` 并删除：

- `src/core_code/` → `hace_core/core/`
- `src/common/` → `hace_core/data/`
- `src/methods/` → `hace_core/methods/`
- `src/cake_runner/` → `hace_core/methods/cake/`
- `src/monitoring/` → `hace_core/utils/`

Git 历史中保留了完整的迁移记录。

---

## 相关文档

- VLLM 部署指南: `docs/VLLM_DEPLOYMENT.md`
- 清理报告归档: `docs/archive/`
