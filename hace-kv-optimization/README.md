c# H2O KV缓存优化实验

这个项目实现了H2O KV缓存优化技术的实验框架，用于评估这种技术对大型语言模型(LLM)性能的影响。

## 项目结构

```
h2o_experiment/
├── main.py                    # 基线实验脚本
├── h2o_main.py                # H2O实验脚本
├── run_comparison.py          # 基线和H2O对比脚本
├── run_experiment.py          # 统一命令行工具
├── config.py                  # 配置文件
├── models/                    # 模型相关代码
│   ├── model_loader.py        # 模型加载模块
│   └── h2o_converter.py       # H2O转换器模块
├── data/                      # 数据相关代码
│   └── dataset_loader.py      # 数据集加载模块
├── metrics/                   # 指标相关代码
│   └── metrics_collector.py   # 指标收集模块
├── utils/                     # 工具代码
│   ├── gpu_monitor.py         # GPU监控工具
│   └── visualization.py       # 结果可视化工具（待实现）
├── results/                   # 结果存储目录
├── logs/                      # 日志目录
└── requirements.txt           # 依赖项
```

## 什么是H2O优化?

H2O是一种针对大型语言模型(LLM)生成过程中KV缓存的优化策略。它结合了两种缓存机制：

1. **Heavy Hitter (H)**：保留接收注意力最多的tokens（重要token）
2. **Recent (O)**：保留最近生成的tokens（最近token）

通过只保留这两类最有价值的token的缓存，H2O可以在保持生成质量的同时显著减少内存使用。

## 实验阶段

实验分为以下阶段：

1. **基线测试**：使用标准KV缓存策略，测量性能基准
2. **H2O优化测试**：使用H2O KV缓存策略，比较与基线的差异
3. **对比分析**：生成详细的对比报告，包含性能指标和可视化结果

## 如何使用

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用统一命令行工具

我们提供了一个统一的命令行工具`run_experiment.py`来方便地运行不同类型的实验：

```bash
# 查看帮助
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_experiment

# 运行基线实验
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_experiment baseline --dataset mmlu --language english

# 运行H2O实验
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_experiment h2o --dataset mmlu --h2o_ratios 0.1,0.2

# 运行对比实验（自动运行基线和H2O实验并生成报告）
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_experiment compare --dataset mmlu --kv_lengths 512,1024 --batch_sizes 1,8
```

### 运行单独的实验脚本

#### 基线实验

```bash
# 英文MMLU数据集基线实验
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.main --language english --dataset mmlu --output_dir ./results/mmlu

# 中文C-Eval数据集基线实验
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.main --language chinese --dataset ceval --output_dir ./results/ceval

# 自定义KV缓存长度和批处理大小
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.main --dataset mmlu --kv_cache_lengths 512,1024 --batch_sizes 1,8
```

#### H2O优化实验

```bash
# 英文MMLU数据集H2O实验
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.h2o_main --language english --dataset mmlu --output_dir ./results/mmlu_h2o

# 指定H2O比率
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.h2o_main --dataset mmlu --h2o_ratios 0.05,0.1,0.2
```

#### 比较实验

```bash
# 运行基线和H2O实验并生成对比报告
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_comparison --dataset mmlu --language english

# 使用已有的实验结果生成报告
python -m Research Framework for Optimizing Head-level KV Cache Based on CAKE.run_comparison --skip_baseline --skip_h2o --baseline_csv ./results/baseline_results.csv --h2o_csv ./results/h2o_results.csv
```

## 监控的指标

实验会收集以下关键指标：

- **TTFT (Time To First Token)**：生成第一个令牌所需的时间(ms)
- **TPOT (Time Per Output Token)**：平均每个令牌的生成时间(ms)
- **TPS (Tokens Per Second)**：每秒生成的令牌数（吞吐量）
- **峰值GPU内存**：生成过程中的最大GPU内存使用量(MB)
- **平均GPU内存**：生成过程中的平均GPU内存使用量(MB)

## 配置参数

可以通过修改`config.py`文件来调整实验参数：

```python
# 模型配置
MODEL_CONFIG = {
    "model_name_or_path": "facebook/opt-1.3b",  # 可以根据实际使用的模型更改
    "precision": "fp16",  # 或 "bf16", "int8" 等
    "device": "cuda"
}

# 实验配置
EXPERIMENT_CONFIG = {
    "kv_cache_lengths": [128, 256, 512, 1024, 2048],
    "batch_sizes": [1, 4, 8, 16],
    "repeat_times": 3,  # 每个配置重复实验的次数
    "random_seed": 42,
    "max_new_tokens": 128,  # 每次生成的最大令牌数
    "h2o_ratios": [0.05, 0.1, 0.2, 0.3]  # heavy_ratio和recent_ratio的可能值
}
```

## 支持的模型类型

目前支持以下模型类型的H2O优化：

- **OPT**: Facebook的OPT模型系列
- **LLaMA**: Meta的LLaMA模型系列
- **GPT-NeoX**: EleutherAI的GPT-NeoX模型系列

## H2O优化的实现

H2O优化基于对模型注意力机制的修改，主要通过以下方式实现：

1. **重要性跟踪**：在前向传播过程中跟踪每个token接收的注意力权重总和
2. **内存管理**：当KV缓存达到阈值时，保留固定比例的重要token和最近token
3. **缓存更新**：动态维护一个重要性排序，确保关键信息不被丢弃

## 结果分析

实验结果将以以下形式呈现：

1. **CSV文件**：包含所有性能指标的详细数据
2. **HTML报告**：包含对比图表和性能改进分析
3. **可视化图表**：展示不同配置下的性能指标变化

## 未来工作

- 实现更多模型架构的H2O支持
- 添加更多评估数据集和指标
- 实现H2O缓存行为的可视化
- 支持不同heavy_ratio和recent_ratio组合的消融实验
- 支持混合精度和量化模型的测试 