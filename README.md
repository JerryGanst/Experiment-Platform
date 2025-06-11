# 🎯 CAKE实验平台 - 基线比较评分系统

这是一个专门用于运行CAKE (Cascading and Adaptive KV Cache Eviction) 实验的平台，实现了基于Full KV基线的相对评分系统，严格按照研究报告要求进行实验评估。

## 📊 核心特性

- **严格基线比较**: Full KV cache作为100分基准，其他策略相对评分
- **标准化数据集**: hotpotqa (QA任务) + multi_news (摘要任务)
- **精确评分指标**: qa_f1_score + rouge_score
- **自动化实验流程**: 一键建立基线 → 运行实验 → 生成报告

## 🏗️ 项目结构

```
Experiment-Platform/
├── README.md                             # 项目说明文档
├── BASELINE_SCORING_SYSTEM.md           # 基线评分系统详细说明
├── SYSTEM_READY.md                      # 系统就绪状态
├── shuoming.md                          # 研究报告规范
├── test_baseline_system.py              # 基线评分系统测试
│
├── run_fullkv_baseline.bat              # 建立Full KV基线
├── run_cake_experiments.bat             # 运行CAKE实验
├── run_complete_baseline_comparison.bat # 完整流程执行
│
└── hace-kv-optimization/                # 核心实验代码
    ├── eval_utils.py                    # 评分工具模块
    ├── baseline_fullkv.json             # 基线分数存储
    ├── baselines/                       # 基线方法实现
    │   ├── fullkvcache_main.py          # Full KV缓存主程序
    │   ├── cake_main.py                 # CAKE主程序
    │   └── ...
    ├── cakekv-main/                     # CAKE算法实现
    ├── data/                            # 数据集目录
    └── ...
```

## 🚀 快速开始

### 1. 测试系统状态

```bash
# 验证基线评分系统是否正常工作
python test_baseline_system.py
```

### 2. 运行完整实验

```bash
# 一键运行完整流程 (推荐)
run_complete_baseline_comparison.bat
```

### 3. 分步执行

```bash
# 步骤1: 建立Full KV基线 (100分标准)
run_fullkv_baseline.bat

# 步骤2: 运行CAKE实验 (相对评分)
run_cake_experiments.bat
```

## 📈 实验配置

### 🎯 数据集配置
- **hotpotqa**: 多跳推理QA任务，使用 `qa_f1_score` 评分
- **multi_news**: 新闻摘要任务，使用 `rouge_score` 评分

### 🤖 模型配置
- **基础模型**: NousResearch/Llama-2-7b-hf
- **KV Cache长度**: 128, 256, 512, 1024, 2048 tokens
- **批处理大小**: 1
- **最大新令牌**: 100

### 🔧 CAKE参数
- **分配策略**: uniform（均匀分配）, adaptive（自适应分配）
- **缓存预算**: 0.7, 0.8, 0.9（相对完整缓存的比例）
- **窗口大小**: 32

## 📊 结果示例

### Full KV基线报告 (100分标准)
```
策略: Full KV (基线)
平均相对分数: 100.00/100
平均原始分数: 0.4964
数据集列表: hotpotqa, multi_news

详细分数:
  hotpotqa: 100.00/100 (原始: 0.5533)
  multi_news: 100.00/100 (原始: 0.4396)
```

### CAKE相对评分报告
```
策略: CAKE
平均相对分数: 93.50/100
平均原始分数: 0.4633
数据集列表: hotpotqa, multi_news

详细分数:
  hotpotqa: 92.00/100 (原始: 0.5090)
  multi_news: 95.00/100 (原始: 0.4176)
```

## 🎯 评分系统核心原理

### 基线建立
1. 使用Full KV cache运行所有数据集
2. 记录每个数据集的原始分数到 `baseline_fullkv.json`
3. Full KV cache 恒定为100分基准

### 相对评分计算
```python
相对分数 = (策略原始分数 / Full KV基线分数) × 100
```

### 报告生成
- **平均相对分数**: 所有数据集相对分数的平均值
- **数据集详情**: 每个数据集的相对分数和原始分数
- **性能分类**: 🟢优秀(≥95) 🟡良好(90-94) 🟠一般(80-89) 🔴较差(<80)

## 📋 依赖要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- Datasets
- Evaluate
- CUDA (推荐)

## 🆘 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少batch_size
   - 降低KV cache长度
   - 使用CPU模式

2. **数据集加载失败**
   - 检查 `hace-kv-optimization/data/` 目录
   - 运行 `download_longbench.py` 重新下载

3. **基线文件缺失**
   - 先运行 `run_fullkv_baseline.bat` 建立基线
   - 检查 `baseline_fullkv.json` 是否存在

## 📚 详细文档

- [基线评分系统详细说明](BASELINE_SCORING_SYSTEM.md)
- [系统就绪状态](SYSTEM_READY.md)
- [研究报告规范](shuoming.md)

## 🏆 系统状态

✅ **配置正确**: hotpotqa + multi_news  
✅ **评分验证**: qa_f1_score + rouge_score  
✅ **基线建立**: baseline_fullkv.json  
✅ **测试通过**: 所有功能正常  
✅ **文档完整**: 使用说明齐全  

---

**系统已准备就绪，可以开始运行CAKE实验！** 🎉

