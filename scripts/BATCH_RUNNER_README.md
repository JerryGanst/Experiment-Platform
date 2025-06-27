# Document QA Baseline Batch Runner 使用指南

## 概述

本目录提供了两个批量运行脚本，用于一次性执行三个Document QA基线实验，每个实验重复3次。

## 脚本文件

### 1. `run_all_docqa_baselines_batch.bat` (Windows批处理)
- 兼容性：Windows 系统
- 功能：基础批量运行功能
- 适用场景：简单的批量执行需求

### 2. `run_all_docqa_baselines_batch.ps1` (PowerShell)
- 兼容性：Windows PowerShell / PowerShell Core
- 功能：增强型批量运行，包含更好的错误处理和进度显示
- 适用场景：需要详细进度监控和错误处理的场景

## 实验配置

### 数据集和Token配置
| 数据集 | Max New Tokens | CAKE匹配 | 输出目录 |
|--------|----------------|----------|----------|
| Qasper | 128 | ✅ | `./runs/qasper_baseline_fullkv` |
| MultifieldQA English | 64 | ✅ | `./runs/multifieldqa_en_baseline_fullkv` |
| HotpotQA | 32 | ✅ | `./runs/hotpotqa_baseline_fullkv` |

### 实验参数
- **重复次数**: 3次 (每个数据集)
- **总实验数**: 9个 (3个数据集 × 3次重复)
- **KV Cache长度**: 128, 1024
- **批处理大小**: 1
- **评分**: 启用 (`--enable_scoring`)

## 使用方法

### 前提条件

1. **设置环境变量**：
   ```batch
   # Windows Command Prompt
   set HACE_MODEL_PATH=your_model_path_or_huggingface_id
   
   # PowerShell
   $env:HACE_MODEL_PATH = "your_model_path_or_huggingface_id"
   ```

2. **示例模型路径**：
   ```batch
   # Hugging Face Hub
   set HACE_MODEL_PATH=microsoft/DialoGPT-medium
   
   # 本地模型路径
   set HACE_MODEL_PATH=D:\AI\models\mistral-7b-instruct-v0.3
   ```

### 运行方式

#### 方式1: Windows批处理 (推荐用于简单执行)
```batch
# 进入scripts目录
cd scripts

# 运行批处理脚本
run_all_docqa_baselines_batch.bat
```

#### 方式2: PowerShell (推荐用于详细监控)
```powershell
# 进入scripts目录
cd scripts

# 运行PowerShell脚本 (默认3次重复)
.\run_all_docqa_baselines_batch.ps1

# 自定义重复次数
.\run_all_docqa_baselines_batch.ps1 -Repetitions 5

# 跳过目录重置
.\run_all_docqa_baselines_batch.ps1 -SkipReset

# 指定模型路径
.\run_all_docqa_baselines_batch.ps1 -ModelPath "microsoft/DialoGPT-medium"
```

### PowerShell脚本参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `-ModelPath` | String | `$env:HACE_MODEL_PATH` | 模型路径或Hugging Face ID |
| `-Repetitions` | Int | 3 | 每个数据集的重复次数 |
| `-SkipReset` | Switch | False | 跳过目录重置步骤 |

## 执行流程

### 步骤1: 重置基线目录
- 删除现有的基线输出目录
- 创建新的空目录
- 确保干净的实验环境

### 步骤2: 批量运行实验
- 按轮次执行实验 (第1轮、第2轮、第3轮)
- 每轮运行所有3个数据集
- 实时显示进度和状态

## 输出和结果

### 目录结构
```
runs/
├── qasper_baseline_fullkv/
│   ├── results_summary.csv
│   ├── detailed_results.csv
│   └── experiment_log.txt
├── multifieldqa_en_baseline_fullkv/
│   ├── results_summary.csv
│   ├── detailed_results.csv
│   └── experiment_log.txt
└── hotpotqa_baseline_fullkv/
    ├── results_summary.csv
    ├── detailed_results.csv
    └── experiment_log.txt
```

### 结果文件说明
- `results_summary.csv`: 汇总结果
- `detailed_results.csv`: 详细结果数据
- `experiment_log.txt`: 实验日志

## 监控和错误处理

### PowerShell版本特性
- ✅ 彩色输出和进度显示
- ✅ 实时时间统计
- ✅ 错误处理和恢复选项
- ✅ 实验统计报告
- ✅ 自定义参数支持

### 错误处理
- 自动检测实验失败
- 提供继续或中止选项
- 记录失败统计信息
- 生成最终报告

## 实验时间估算

基于历史数据的时间估算：

| 数据集 | 单次实验时间 | 3次重复总时间 |
|--------|-------------|-------------|
| Qasper | ~15-20分钟 | ~45-60分钟 |
| MultifieldQA | ~10-15分钟 | ~30-45分钟 |
| HotpotQA | ~8-12分钟 | ~24-36分钟 |
| **总计** | **~33-47分钟** | **~99-141分钟** |

> **注意**: 实际时间取决于硬件性能、模型大小和网络速度。

## 故障排除

### 常见问题

1. **HACE_MODEL_PATH 未设置**
   ```
   [ERROR] HACE_MODEL_PATH is not set. Please set it first.
   ```
   **解决方案**: 设置环境变量 `HACE_MODEL_PATH`

2. **模型路径不存在**
   ```
   [ERROR] Model path does not exist
   ```
   **解决方案**: 检查模型路径是否正确，或使用Hugging Face ID

3. **Python环境问题**
   ```
   [ERROR] Python is not recognized
   ```
   **解决方案**: 确保Python已安装并添加到PATH

4. **内存不足**
   ```
   [ERROR] CUDA out of memory
   ```
   **解决方案**: 
   - 关闭其他GPU进程
   - 减少批处理大小
   - 使用更小的模型

### 中断和恢复

如果实验被中断：
1. 记录已完成的实验
2. 使用 `-SkipReset` 参数继续
3. 或者删除对应的输出目录重新开始

## 下一步操作

实验完成后的建议步骤：

1. **查看结果**：
   ```bash
   # 查看汇总结果
   head runs/*/results_summary.csv
   ```

2. **运行CAKE实验**：
   ```bash
   # 运行对应的CAKE实验进行对比
   python src/methods/cake/cake_main.py --datasets qasper,multifieldqa_en,hotpotqa
   ```

3. **生成对比报告**：
   ```bash
   # 生成性能对比分析
   python evaluation/analysis/generate_report.py
   ```

## 技术支持

如有问题或需要帮助，请：
1. 检查日志文件 `experiment_log.txt`
2. 运行 `python check_model_config.py` 检查配置
3. 查看项目根目录的 `README.md` 文档 