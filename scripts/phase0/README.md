# Phase 0: 假设验证实验框架

## 概述

本框架用于验证核心研究假设：

> **H1**: 最优 KV cache 驱逐策略随压缩率变化 - 在高压缩时表现最好的策略，在低压缩时可能不是最优的。

通过双因素方差分析 (Two-Way ANOVA) 检验 **策略 × 压缩率** 的交互效应。

## 实验设计

| 因子 | 水平数 | 取值 |
|------|--------|------|
| 策略 (Strategy) | 3 | high_entropy, low_entropy, uniform |
| 压缩率 (Cache Size) | 3 | 128, 512, 2048 tokens |
| 数据集 (Dataset) | 2 | qmsum (摘要), hotpotqa (QA) |
| 重复 (Seeds) | 3 | 42, 123, 456 |

**总计**: 3 × 3 × 2 × 3 = **54 次实验**

## 快速开始

```bash
# 1. 查看状态
python pre-experiment/main.py status

# 2. 干运行 (预览命令)
python pre-experiment/main.py run --dry-run

# 3. 运行实验 (支持断点续跑)
python pre-experiment/main.py run --device 0

# 4. 仅分析 (已有结果时)
python pre-experiment/main.py analyze

# 5. 完整流程 (实验 + 分析 + 绘图)
python pre-experiment/main.py all --device 0
```

## 目录结构

```
pre-experiment/
├── config/
│   └── phase0_config.py      # 配置: 策略/压缩率/阈值
│
├── runners/
│   ├── checkpoint_manager.py # 断点续跑管理
│   ├── experiment_runner.py  # 单次实验执行
│   └── batch_runner.py       # 批量执行器
│
├── parsers/
│   ├── eval_parser.py        # 解析 eval.py 输出
│   └── result_aggregator.py  # 聚合为 CSV
│
├── analysis/
│   ├── two_way_anova.py      # 双因素方差分析
│   ├── effect_size.py        # 效应量计算
│   └── decision_generator.py # 自动生成决策
│
├── visualization/
│   └── interaction_plot.py   # 交互效应图
│
├── main.py                   # CLI 入口
│
└── results/                  # 输出
    ├── checkpoints/progress.json  # 断点状态
    ├── raw/                       # 原始预测
    ├── raw_results.csv            # 聚合结果
    ├── anova_results.json         # ANOVA 输出
    ├── interaction_plot.png       # 可视化
    └── decision.md                # Go/No-Go 决策
```

## 判定标准

### 统计显著性
- **检验**: Two-Way ANOVA 交互项
- **阈值**: p < 0.05

### 实际显著性 (效应量)
- **指标**: η² (eta-squared)
- **阈值**: > 0.02 (2%)

### 决策树

```
Phase 0 Results
├── 交互显著 (p < 0.05) 且 效应量 > 2%
│   └── ✅ PROCEED: 继续 Phase 1-4
│
├── 交互显著但效应量 < 2%
│   └── ⚠️ RECONSIDER: 效应太小
│
├── 交互不显著 (p ≥ 0.05)
│   └── ❌ ABANDON: 放弃当前方向
│
└── 混合结果 (部分数据集显著)
    └── 🔄 REFINE: 聚焦于有效应的任务类型
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `raw_results.csv` | 54 行结果，包含策略、压缩率、数据集、种子、分数 |
| `anova_results.json` | F 统计量、p 值、效应量 |
| `interaction_plot.png` | 策略 × 压缩率 交互可视化 |
| `decision.md` | **最终决策文档** - Go/No-Go 判定及建议 |

## 断点续跑

框架支持自动断点续跑：

1. 每完成一个实验立即保存到 `checkpoints/progress.json`
2. 重新运行时自动跳过已完成的实验
3. 失败的实验最多重试 2 次

```bash
# 查看当前进度
python pre-experiment/main.py status

# 继续运行
python pre-experiment/main.py run --device 0

# 强制从头开始
python pre-experiment/main.py run --no-resume --device 0

# 重置所有进度
python pre-experiment/main.py reset
```

## 预计时间

- 单次实验: ~10-15 分钟 (A100)
- 54 次实验: ~10-14 小时
- 分析 + 绘图: < 5 分钟

## 依赖

- Python 3.8+
- numpy, pandas, scipy
- matplotlib
- filelock

无需 statsmodels (ANOVA 手动实现)。

## 常见问题

### Q: 如何修改实验参数？
编辑 `config/phase0_config.py`:
```python
STRATEGIES = ["high_entropy", "low_entropy", "uniform"]
CACHE_SIZES = [128, 512, 2048]
DATASETS = ["qmsum", "hotpotqa"]
```

### Q: 如何添加新数据集？
1. 在 `DATASETS` 列表中添加数据集名
2. 在 `DATASET_CONFIG` 中配置 metric 和 max_new_tokens

### Q: 实验失败怎么办？
- 失败的实验会自动重试 (最多 2 次)
- 查看 `results/phase0.log` 获取详细错误信息
- 使用 `status` 命令查看失败数量

---

*Phase 0 Hypothesis Validation Framework v1.0*
