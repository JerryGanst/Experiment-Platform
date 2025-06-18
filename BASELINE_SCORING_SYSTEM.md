# 基线评分系统 - Full KV 100分基准

## 概述

根据您的研究报告要求，我们已经实现了一套完整的基线评分系统，严格按照HACE论文的评分标准，以**Full KV模式作为100分基准**，实现相对评分机制。

## 系统架构

### 1. 核心组件

- **`eval_utils.py`**: 评分工具模块，实现基线管理和相对评分
- **`baseline_fullkv.json`**: 基线分数存储文件
- **修改后的实验脚本**: 支持基线建立和相对评分

### 2. 评分流程

```
1. 运行Full KV实验 → 建立基线 (100分)
2. 运行其他策略实验 → 计算相对分数
3. 生成对比报告 → 百分制显示结果
```

## 使用方法

### 快速开始

```bash
# 完整流程 (推荐)
run_complete_baseline_comparison.bat

# 或分步执行
run_fullkv_baseline.bat      # 建立基线
run_cake_experiments.bat     # 运行CAKE实验
```

### 手动运行

```bash
# 1. 建立Full KV基线
python hace-kv-optimization/baselines/fullkvcache_main.py \
    --enable_scoring --is_baseline_run \
    --datasets hotpotqa,multi_news

# 2. 运行CAKE实验
python hace-kv-optimization/baselines/cake_main.py \
    --enable_scoring \
    --datasets hotpotqa,multi_news
```

## 评分标准

### 数据集映射 (严格按照研究报告要求)

| 数据集 | 评分函数 | 说明 |
|--------|----------|------|
| hotpotqa | qa_f1_score | QA任务，多跳推理问答，基于F1分数 |
| multi_news | rouge_score | 摘要任务，新闻摘要，基于ROUGE-L F1 |

### 相对评分计算

```python
相对分数 = (策略得分 / Full KV基线得分) × 100
```

- **Full KV**: 恒定100分
- **其他策略**: 相对百分比 (如95分表示保留95%性能)

## 输出示例

### 基线报告
```
==================================================
策略: Full KV (基线)
==================================================
平均相对分数: 100.00/100
平均原始分数: 0.5410
评测数据集数量: 3
数据集列表: mmlu, narrativeqa, qasper

详细分数:
  hotpotqa: 100.00/100 (原始: 0.5533)
  multi_news: 100.00/100 (原始: 0.4396)
==================================================
```

### CAKE相对评分报告
```
==================================================
策略: CAKE
==================================================
平均相对分数: 90.00/100
平均原始分数: 0.4869
评测数据集数量: 3
数据集列表: mmlu, narrativeqa, qasper

详细分数:
  hotpotqa: 92.00/100 (原始: 0.5090)
  multi_news: 95.00/100 (原始: 0.4176)
==================================================
```

## 性能等级划分

| 相对分数范围 | 等级 | 说明 |
|-------------|------|------|
| ≥95分 | 🟢 优秀 | 几乎无性能损失 |
| 90-94分 | 🟡 良好 | 轻微性能损失 |
| 80-89分 | 🟠 一般 | 中等性能损失 |
| <80分 | 🔴 较差 | 显著性能损失 |

## 文件结构

```
Experiment-Platform/
├── hace-kv-optimization/
│   ├── eval_utils.py                    # 评分工具模块
│   ├── baseline_fullkv.json            # 基线分数文件
│   └── baselines/
│       ├── fullkvcache_main.py         # Full KV实验脚本
│       └── cake_main.py                # CAKE实验脚本
├── run_fullkv_baseline.bat             # 建立基线脚本
├── run_cake_experiments.bat            # CAKE实验脚本
├── run_complete_baseline_comparison.bat # 完整流程脚本
└── test_baseline_system.py             # 基线评分系统测试
```

## 测试验证

```bash
# 测试基线评分系统
python debug_eval_utils.py
```

## 技术特性

### 1. 鲁棒性
- 异常处理和错误恢复
- 基线文件自动管理
- 多次重复实验支持

### 2. 可扩展性
- 支持新数据集添加
- 支持新评分函数
- 支持新优化策略

### 3. 一致性
- 统一的评分标准
- 相同的随机种子
- 标准化的输出格式

## 配置说明

### 实验参数
- **模型**: NousResearch/Llama-2-7b-hf (与baseline一致)
- **数据集**: hotpotqa, multi_news (按研究报告要求)
- **KV Cache长度**: 512, 1024, 2048
- **批处理大小**: 1
- **最大新令牌**: 100

### CAKE特定参数
- **分配策略**: uniform, adaptive
- **缓存预算**: 0.7, 0.8, 0.9

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖已安装
2. **基线文件不存在**: 先运行Full KV基线实验
3. **评分为0**: 检查数据集配置和评分函数

### 调试模式

```bash
# 启用详细日志
--log_level DEBUG

# 启用CUDA调试
set CUDA_LAUNCH_BLOCKING=1
```

## 结果解读

### 相对分数含义
- **100分**: 与Full KV性能完全一致
- **95分**: 保留95%性能，损失5%
- **90分**: 保留90%性能，损失10%

### 综合评估
- 关注平均相对分数
- 考虑各数据集表现差异
- 结合效率指标(内存、速度)进行综合判断

---

**注意**: 该评分系统严格按照您的研究报告要求实现，确保与HACE论文的评分标准一致，为KV缓存优化策略提供公平、量化的性能评估。 