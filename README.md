# HACE KV 缓存优化实验平台

该仓库实现了一个用于研究 **Hierarchical-Head Attention Cache Enhancement (HACE)** 的实验平台。HACE 结合层级压缩与注意力头级压缩两种策略，旨在在保持生成质量的同时显著降低 KV 缓存占用。仓库中还包含多种基线方法，便于与 HACE 进行系统对比。

## 项目结构

```text
hace-kv-optimization/
├── main.py                      # 主入口：运行 HACE 与各基线的实验
├── config.yaml                  # 实验配置
├── run_experiments.sh           # 批量运行脚本
│
├── hace_core/                   # HACE 算法实现
│   ├── __init__.py
│   ├── cache.py                 # HACE 缓存逻辑
│   ├── models/                  # 模型修改与适配
│   └── algorithms.py            # HACE 核心算法
│
├── baselines/                   # 基线方法实现
│   ├── __init__.py
│   ├── base_method.py           # 基类
│   ├── full_cache/              # 纯缓存对照组
│   │   └── full_cache.py
│   ├── cake/                    # CAKE baseline 及其适配器
│   │   ├── cakekv-main/         # 原始 CAKE 代码
│   │   └── cake_wrapper.py
│   ├── ada_kv/                  # Ada-KV baseline
│   │   └── ada_kv.py
│   └── streaming_llm/           # StreamingLLM baseline
│       └── streaming_llm.py
│
├── experiments/                 # 实验运行框架
│   ├── runner.py                # 统一运行入口
│   ├── datasets.py              # 数据集加载
│   ├── metrics.py               # 评估指标
│   └── benchmarks.py            # 基准测试脚本
│
├── analysis/                    # 结果分析与可视化
│   ├── plotter.py               # 绘制对比图表
│   ├── tables.py                # 生成结果表格
│   └── statistical_tests.py     # 统计显著性检验
│
└── results/                     # 实验结果输出
    ├── raw/                     # 原始数据
    ├── figures/                 # 图表
    └── tables/                  # 表格
```

## 快速开始

1. 安装依赖并准备环境：
   ```bash
   bash setup_env.sh
   ```
2. 运行一次示例实验：
   ```bash
   ./run_experiments.sh
   ```
   或者直接执行：
   ```bash
   python main.py --config config.yaml
   ```

实验结果将被保存到 `results/` 目录，可使用 `analysis/` 下的脚本进行可视化与统计分析。

