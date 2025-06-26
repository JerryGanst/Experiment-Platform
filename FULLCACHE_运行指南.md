# FullCache 实验运行指南

## 🎯 修复内容总结

### 主要修复：
1. **修复硬编码路径问题** - `hace_core/config.py` 和批处理文件
2. **修复评分系统** - `evaluation/baselines/fullkvcache_main.py`
   - ✅ Ground Truth提取正确（从空字符串到`"Other location"`）
   - ✅ 评分模块导入路径修复
   - ✅ `classification_score`函数`all_classes`参数传递
   - ✅ 移除困惑的基线运行提示信息
3. **添加便捷运行脚本**（现已支持便携式路径）
4. **包含完整实验结果**

## 🚀 完整运行命令

### 方法1：使用便携式批处理文件（推荐）

#### 首次设置：
1. 复制示例配置文件：
   ```bash
   copy setup_env.bat.example setup_env.bat
   ```

2. 编辑 `setup_env.bat` 文件，设置您的模型路径：
   ```batch
   set HACE_MODEL_PATH=您的模型路径\7B-Instruct-v0.3
   ```

3. 运行环境设置（可选）：
   ```bash
   .\setup_env.bat
   ```

#### 运行实验：
```bash
# 运行带评分的fullcache实验
.\run_fullcache_with_scoring.bat

# 或者运行不带评分的基础实验
.\run_fullcache.bat
```

> **注意**: 如果未设置 `HACE_MODEL_PATH` 环境变量，脚本会提示您输入模型路径。

### 方法2：设置环境变量后运行

```bash
# PowerShell环境下设置环境变量
$env:HACE_MODEL_PATH = "您的模型路径\7B-Instruct-v0.3"

# 或者在CMD中设置
set HACE_MODEL_PATH=您的模型路径\7B-Instruct-v0.3

# 然后运行脚本
.\run_fullcache_with_scoring.bat
```

### 方法3：直接使用Python命令

```bash
# 运行带真正评分的fullcache实验
python evaluation/baselines/fullkvcache_main.py --model_name "您的模型路径\7B-Instruct-v0.3" --datasets trec --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 50 --enable_scoring
```

### 方法4：测试其他数据集

```bash
# 测试不同数据集
python evaluation/baselines/fullkvcache_main.py --model_name "您的模型路径\7B-Instruct-v0.3" --datasets hotpotqa --kv_cache_lengths 128 --batch_sizes 1 --repetitions 1 --max_new_tokens 100 --enable_scoring

# 测试不同KV缓存长度
python evaluation/baselines/fullkvcache_main.py --model_name "您的模型路径\7B-Instruct-v0.3" --datasets trec --kv_cache_lengths 64,128,256 --batch_sizes 1 --repetitions 1 --max_new_tokens 50 --enable_scoring
```

## 📊 实验结果

### 最新实验结果位置：
```
runs/fullkvcache_run_20250626_184143/
├── all_fullkvcache_experiments_summary.csv
├── all_fullkvcache_experiments_summary.json
└── ds_trec_kv128_bs1_rep0/
    ├── evaluation_results_fullkvcache_trec_kv128_bs1_rep0_20250626_184143.json
    └── fullkvcache_metrics_fullkvcache_trec_kv128_bs1_rep0_20250626_184143.json
```

### 评分结果示例：
- **生成文本**: `"General Knowledge\n\nThe first golf course in the United States was located in South Carolina..."`
- **标准答案**: `"Other location"`
- **评分结果**: `0.0` (正确，因为生成文本未包含正确分类标签)

## 🔧 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model_name` | 模型路径 | `"C:\Users\Administrator\mistral_models\7B-Instruct-v0.3"` |
| `--datasets` | 数据集名称 | `trec`, `hotpotqa`, `multi_news` |
| `--kv_cache_lengths` | KV缓存长度 | `128`, `64,128,256` |
| `--batch_sizes` | 批处理大小 | `1`, `1,2,4` |
| `--repetitions` | 重复次数 | `1`, `3` |
| `--max_new_tokens` | 最大生成token数 | `50`, `100` |
| `--enable_scoring` | 启用评分 | 添加此参数启用 |

## 🎯 评分功能

### 支持的评分类型：
- **分类任务** (TREC): `classification_score`
- **问答任务** (HotpotQA): `qa_f1_score`  
- **摘要任务** (Multi-News): `rouge_score`

### 评分结果文件：
- `evaluation_results_*.json`: 包含生成文本、标准答案、评分结果
- `fullkvcache_metrics_*.json`: 包含性能指标（吞吐量、延迟等）

## 🐛 故障排除

### 常见问题：
1. **模块导入错误**: 确保设置了`PYTHONPATH`环境变量
2. **模型路径错误**: 确保模型文件存在于指定路径
3. **评分模块错误**: 评分依赖已自动修复，无需手动安装

### 验证安装：
```bash
# 测试评分功能
python test_scoring_fix.py
```

## 📈 性能指标

实验会自动记录：
- **TTFT** (Time To First Token): 首token生成时间
- **TPOT** (Time Per Output Token): 平均每token生成时间  
- **吞吐量**: tokens/秒
- **GPU显存使用**: 峰值显存占用
- **生成质量**: 评分结果

## 🎉 成功标志

运行成功时会看到：
```
✅ 评分模块加载成功
✅ 使用处理后的reference: Other location
✅ 评分详情: 函数=classification_score, 生成文本='...', 标准答案='Other location', 分数=0.0
✅ Evaluation score: 0.0000
```

现在你的fullcache实验系统完全正常工作，支持真正的评分功能！🚀 