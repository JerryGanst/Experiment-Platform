# 🎯 KV缓存优化实验平台

这是一个专门用于KV缓存优化方法研究的实验平台，支持基线对比、CAKE优化、H2O优化等多种方法的性能评估。

## 📊 核心特性

- **统一实验入口**: 一个脚本管理所有实验类型
- **完整评估流程**: 基线 → 优化方法 → 对比分析 → 报告生成
- **模块化设计**: 基线执行、方法实现、分析工具分离
- **标准化评估**: 统一的评分指标和分析流程

## 🏗️ 项目结构

```
Experiment-Platform/
├── README.md                    # 项目说明文档
├── run_experiments.py           # 🚀 统一实验入口脚本
├── configs/                     # 配置文件
├── data/                        # 数据集
├── runs/                        # 实验结果
│
├── evaluation/                  # 📊 统一评估模块
│   ├── baselines/              # 基线方法执行
│   │   ├── fullkvcache_main.py # 完整KV缓存基线
│   │   ├── baseline_main.py    # 标准基线
│   │   └── data/               # 基线测试数据
│   ├── analysis/               # 实验数据分析
│   │   ├── statistical_tests.py # 统计分析器
│   │   ├── tables.py           # 表格生成
│   │   └── generate_report.py  # 报告生成
│   └── experiments/            # 实验管理协调
│       ├── run_experiment.py   # 主实验执行器
│       └── run_comparison.py   # 对比分析器
│
├── src/                        # 源代码模块
│   ├── methods/                # KV缓存优化方法
│   │   ├── cake/              # CAKE层级感知优化
│   │   ├── h2o/               # H2O Heavy-Hitter优化
│   │   └── full_cache/        # 完整缓存相关
│   ├── monitoring/            # 监控和可视化
│   │   ├── unified_monitor.py # 统一监控器
│   │   └── plotter.py         # 图表生成
│   └── common/                # 通用工具
│
└── hace_core/                  # 核心模块
    ├── models/                 # 模型加载器
    ├── data/                   # 数据加载器
    └── config.py               # 配置文件
```

## 🚀 快速开始

### 1. 系统要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA (推荐)

### 2. 安装依赖

```bash
pip install -r evaluation/requirements.txt
```

### 3. 模型路径配置

系统支持多种模型路径配置方式，优先级如下（Linux/SSH 环境）：

#### 方式1: 环境变量（推荐）
```bash
# 设置模型路径环境变量
export CORECODE_MODEL_PATH="/mnt/models/mistral-7b-instruct-v0.3"
export HACE_MODEL_PATH="$CORECODE_MODEL_PATH"

# 可写入 ~/.bashrc 或 .env.corecode 后 source
echo 'export CORECODE_MODEL_PATH="/mnt/models/mistral-7b-instruct-v0.3"' >> ~/.bashrc
```

#### 方式2: 项目本地模型目录
```bash
# 在项目根目录创建 models 文件夹
mkdir models
# 将模型文件放入 models/mistral-7b-instruct-v0.3/
```

#### 方式3: 用户主目录
```bash
# 在用户主目录创建 models 文件夹
mkdir ~/models
# 将模型文件放入 ~/models/mistral-7b-instruct-v0.3/
```

⚠️ **重要**: 
- 仓库现已不再包含任何 `.bat` / Windows 工具，默认假设通过 SSH 连接 Linux 服务器
- 必须显式配置 `CORECODE_MODEL_PATH` 和/或 `HACE_MODEL_PATH`，不提供自动下载兜底方案

### 4. 数据路径配置

远端 GPU 服务器通常把数据集挂载在共享目录。通过设置 `DATASETS_ROOT` 即可让所有脚本自动找到 `.jsonl` 数据：

```bash
export DATASETS_ROOT="/mnt/datasets/corecode"
# 测试：python - <<'PY'
from src.common.data_loader import find_data_file
print(find_data_file("hotpotqa"))
# PY
```

详见 `docs/REMOTE_GPU_SETUP.md` 获取完整的 SSH/远端部署指引。

### 4. 运行实验

#### 🎯 完整评估流程 (推荐)

```bash
# 运行完整评估：基线 → CAKE → 对比分析
python run_experiments.py --experiment full_evaluation --tag production

# 使用自定义参数
python run_experiments.py \
    --experiment full_evaluation \
    --datasets hotpotqa,multi_news \
    --kv_lengths 128,1024 \
    --tag my_experiment
```

#### 🔵 只运行基线实验

```bash
# 运行基线实验
python run_experiments.py --experiment baseline --tag baseline_test

# 指定KV长度
python run_experiments.py \
    --experiment baseline \
    --kv_lengths 128,1024 \
    --datasets hotpotqa
```

#### 🟢 只运行CAKE实验

```bash
# 运行CAKE实验
python run_experiments.py --experiment cake --tag cake_test

# 自定义CAKE参数
python run_experiments.py \
    --experiment cake \
    --allocation_strategies adaptive \
    --cache_budgets 0.5,0.7,0.9
```

#### 📊 只运行对比分析

```bash
# 分析已有的实验结果
python run_experiments.py --experiment comparison --tag analysis
```

#### 🔍 干运行 (查看将要执行的命令)

```bash
# 查看完整评估流程将要执行的命令
python run_experiments.py --experiment full_evaluation --dry-run
```

## 📈 实验配置

### 🎯 推荐配置

- **数据集**: hotpotqa, multi_news
- **KV长度**: 128, 1024 tokens (限制范围)
- **批处理大小**: 1
- **CAKE缓存预算**: 0.7
- **分配策略**: adaptive

### 🤖 模型配置

- **基础模型**: NousResearch/Llama-2-7b-hf
- **精度**: float16
- **最大新令牌**: 100

## 🧪 LongBench 基线对齐（CAKE）

为保证 LongBench 评测与 CAKE 官方流程一致，基线评测做了如下对齐：

- **Prompt 模板**：使用 CAKE 的 `dataset2prompt.json`（包含 `{context}` + `{input}`）
- **中间截断**：超长 prompt 采用“前半 + 后半”的中间截断
- **Chat 模板**：对 Llama2/Llama3/Mistral/Qwen 使用对应聊天格式；对 `trec/triviaqa/samsum/lsht/lcc/repobench-p` 跳过 chat 包装
- **样本保留**：评测样本保留 `original_sample`，便于复现原始字段
- **解码与长度**：使用贪心解码（`do_sample=False`），并按 `dataset2maxlen.json` 设置每任务 `max_new_tokens`
- **可选开关**：`--no_cake_prompting` / `--no_cake_maxlen` / `--no_greedy_decode` 可关闭对应对齐策略

对应实现位置：

- `evaluation/baselines/baseline_main.py`
- `hace_core/data/dataset_loader.py`

### 📋 完整参数列表

```bash
python run_experiments.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment` | `full_evaluation` | 实验类型: baseline/cake/h2o/comparison/full_evaluation |
| `--tag` | `default` | 运行标签，用于区分不同实验 |
| `--datasets` | `hotpotqa,multi_news` | 数据集列表，逗号分隔 |
| `--kv_lengths` | `128,1024` | KV缓存长度列表 (推荐: 128,1024) |
| `--batch_sizes` | `1` | 批处理大小列表 |
| `--allocation_strategies` | `adaptive` | CAKE分配策略 |
| `--cache_budgets` | `0.7` | CAKE缓存预算 |
| `--dry-run` | False | 仅显示将要运行的命令 |

## 📊 结果解读

### 实验结果目录结构

```
runs/20250625_172812_production/
├── baseline_results/           # 基线实验结果
├── cake_results/              # CAKE实验结果
└── comparison_report/         # 对比分析报告
    ├── visualizations/        # 图表文件
    ├── statistical_analysis/  # 统计分析
    └── summary_tables_*.json  # 汇总表格
```

### 关键指标

- **TTFT** (Time to First Token): 首个令牌生成时间
- **TPOT** (Time per Output Token): 每个输出令牌时间
- **内存使用**: GPU内存占用
- **准确性**: 任务特定评分 (F1/ROUGE)

## 🔧 高级用法

### 自定义实验流程

```bash
# 1. 先运行基线建立参考
python run_experiments.py --experiment baseline --tag reference

# 2. 运行多种CAKE配置
python run_experiments.py \
    --experiment cake \
    --cache_budgets 0.5,0.6,0.7,0.8,0.9 \
    --tag cake_sweep

# 3. 生成对比报告
python run_experiments.py --experiment comparison --tag final_report
```

### 批量实验

```bash
# 在不同数据集上运行实验
for dataset in hotpotqa multi_news; do
    python run_experiments.py \
        --experiment full_evaluation \
        --datasets $dataset \
        --tag ${dataset}_experiment
done
```


## HACE 注意力权重存储与 Glue 实验

### 注意力权重存储（HACE 方法）
- 存储实现：`hace_core/models/attention_collector.py`（`AttentionData.save`）
- 保存格式：`.pkl/.pickle`（推荐）或 `.json`（体积大、慢）
- 触发条件：在 `hace_core/config.py` 的 `VLLM_CONFIG["attention_collection"]["cache_attention_file"]` 指定路径；否则不会落盘
- 数据结构：`attention_weights_list`（list of `[batch, heads, seq, seq]` 数组）、`num_layers`、`num_heads`、`seq_length`、`model_name`、`collection_mode`（可选 `layer_statistics/head_statistics`）

### 实验命令（示例）

#### 1) 采样收集注意力权重（小样本）
```bash
python - <<'PY'
from hace_core.models.attention_collector import AttentionCollector, AttentionCollectionConfig
from hace_core import config

sample_prompts = [
    "Summarize the meeting in one sentence.",
    "What is the main decision?",
]
cfg = AttentionCollectionConfig(
    warmup_samples=2,
    max_seq_length=1024,
    cache_file="/cloud/cloud-ssd1/Experiment-Platform/runs/attn_cache.pkl",
    use_cache=False,
    precision="fp16",
)
collector = AttentionCollector(cfg)
data = collector.collect_from_hf_model(
    {"model_name_or_path": config.EXPERIMENT_CONFIG["model_name_or_path"]},
    sample_prompts,
)
data.save(cfg.cache_file)
print("saved", cfg.cache_file)
PY
```
> 说明：这一步会加载 HF 模型并占用显存，建议在 GPU 空闲时运行。

#### 2) Glue/统一分配器（HACE）计算预算
```bash
python - <<'PY'
from hace_core.models.attention_collector import AttentionData
from hace_core.core import CakeAdaKVIntegration, IntegrationConfig

data = AttentionData.load("/cloud/cloud-ssd1/Experiment-Platform/runs/attn_cache.pkl")
integration = CakeAdaKVIntegration(IntegrationConfig(total_cache_size=4096, enable_monitoring=True))
layer_budgets, head_budgets = integration.optimize_cache(data.attention_weights_list)

print("layers", layer_budgets)
print("heads[0]", head_budgets[0][:8])
PY
```

### 注意力熵方向验证实验（建议方案）
- **数据**：`qmsum/gov_report/multi_news` 各 50 条样本，`max_seq_length=1024`
- **信号**：注意力熵 `H_attn`（归一化 `H/log L`）与预测熵/置信度 `H_pred`
- **方向 A/B**：高熵保留 vs 低熵保留，固定相同预算（如 60%）
- **评价**：ROUGE-L/PPL + 实际耗时/预算 + 相关性 `corr(H_attn, delta_LL)`
  - `delta_LL = LL_full - LL_layer` 用来衡量“继续计算的边际收益”
  - 若 `corr(H_attn, delta_LL) > 0` → 高熵更应保留；反之低熵更应保留


## 🔬 HACE pref_mode 实验

### 实验目的

对比不同的注意力预算分配策略：

| 模式 | 公式 | 含义 |
|------|------|------|
| `normal` | H^(+) * V^(+) | 高熵高方差 → 高预算（原设计） |
| `reverse_disp` | (1/H)^(+) * V^(+) | 低熵 → 高预算（导师建议） |
| `reverse` | (1/H)^(+) * (1/V)^(+) | 低熵低方差 → 高预算（更激进） |

### 运行实验

```bash
# SSH连接服务器
ssh -p 23 root@117.50.34.209

# 激活环境
cd /cloud/cloud-ssd1/Experiment-Platform
source /usr/local/miniconda3/bin/activate py312

# 运行 reverse_disp 模式
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
    --pref_mode reverse_disp \
    --pred_name hace_reverse_disp_128 \
    --device 0

# 运行 normal 模式
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --pred_name hace_normal_128 \
    --device 0

# 后台运行（推荐）
nohup python scripts/run_hace_qmsum.py \
    --pref_mode reverse_disp \
    --pred_name hace_reverse_disp_128 \
    --device 0 > logs/reverse_disp.log 2>&1 &
```

### 监控实验进度

```bash
# 使用监控脚本
bash scripts/monitor_experiment.sh

# 手动检查进度
wc -l results_clean/*/qmsum.jsonl

# 查看GPU状态
nvidia-smi

# 查看运行中的进程
ps aux | grep run_hace

# 查看日志
tail -f logs/reverse_disp.log
```

### 评估结果

```bash
# 计算ROUGE分数
python scripts/eval_rouge.py results_clean/hace_normal_128/qmsum.jsonl
python scripts/eval_rouge.py results_clean/hace_reverse_disp_128/qmsum.jsonl

# 对比结果
python scripts/compare_hace_results.py
```

### 结果解读

- 如果 `reverse_disp` ROUGE > `normal` → 导师对了，低熵层应该分配更多预算
- 如果 `normal` > `reverse_disp` → 原设计正确，高熵层需要更多预算
- 如果差不多 → 熵信号可能本身不够好

---

## 🔬 HACE 头级自适应优化 (Ada-KV 集成)

### 代码位置

**重要**: HACE 使用的 CAKE 代码位于 `src/third_party/cakekv-main/cakekv-main/`。修改此目录会影响所有使用 CAKE 的实验。

关键文件：
- `cake/cake_cache.py` - KV 缓存管理和预算分配
- `cake/model/modify_qwen2.py` - 注意力前向计算和分数计算
- `experiments/LongBench/pred_cake.py` - 实验运行入口

### 头级优化模式

| 模式 | 环境变量 | 算法 | 含义 |
|------|----------|------|------|
| `adakv` | `HACE_HEAD_MODE=adakv` | Ada-KV counting | 论文原版：全局 top-B 计数分配 |
| `lh1` | `HACE_HEAD_MODE=lh1` | 浓度反比 | 低浓度 head → 高预算 (我们的假设) |
| `lh2` | `HACE_HEAD_MODE=lh2` | 浓度正比 | 高浓度 head → 高预算 (导师假设) |
| (空) | `HACE_HEAD_MODE=` | 均匀分配 | 原始 CAKE 行为 |

### Ada-KV 算法原理 (论文: arXiv:2407.11550)

```
Algorithm 2 (Ada-KV):
1. 拼接所有 head 的 attention weights
2. 全局选取 top-B 个最大权重
3. 统计每个 head 在 top-B 中出现的频率
4. 按频率比例分配预算
5. 用 α=0.5 混合：budget = α × adaptive + (1-α) × uniform
```

### 运行头级实验

```bash
# SSH 连接服务器
ssh -p 23 root@117.50.34.209
cd /cloud/cloud-ssd1/Experiment-Platform
source /usr/local/miniconda3/bin/activate py312

# 方式1: 使用实验脚本
nohup bash scripts/run_head_level_experiments.sh > logs/head_experiments.log 2>&1 &

# 方式2: 手动运行单个配置
CUDA_VISIBLE_DEVICES=0 python scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --head_mode adakv \
    --pred_name hace_adakv_128 \
    --device 0
```

### 同步代码到服务器

```bash
rsync -avz --progress \
    -e "ssh -p 23" \
    --include='src/third_party/cakekv-main/cakekv-main/cake/cake_cache.py' \
    --include='src/third_party/cakekv-main/cakekv-main/cake/model/modify_qwen2.py' \
    --include='src/third_party/cakekv-main/cakekv-main/experiments/LongBench/pred_cake.py' \
    --include='scripts/run_hace_qmsum.py' \
    --include='scripts/run_head_level_experiments.sh' \
    --include='hace_core/models/cake_converter.py' \
    --include='hace_core/models/cake_subprocess_adapter.py' \
    --include='*/' \
    --exclude='*' \
    ./ root@117.50.34.209:/cloud/cloud-ssd1/Experiment-Platform/
```

---

## 🆘 故障排除

### 常见问题

1. **脚本文件不存在**
   ```
   ❌ 基线脚本不存在: evaluation/baselines/fullkvcache_main.py
   ```
   - 检查文件路径是否正确
   - 确认目录结构完整

2. **GPU内存不足**
   - 使用较小的KV长度: `--kv_lengths 128`
   - 确保batch_size为1

3. **导入模块失败**
   - 检查Python路径配置
   - 确认hace_core模块可访问

### 调试技巧

```bash
# 使用dry-run查看将要执行的命令
python run_experiments.py --experiment baseline --dry-run

# 查看详细帮助
python run_experiments.py --help

# 运行单个数据集进行快速测试
python run_experiments.py \
    --experiment baseline \
    --datasets hotpotqa \
    --kv_lengths 128 \
    --tag debug
```

## 📚 模块说明

### evaluation/ - 统一评估模块
- **baselines/**: 基线方法的执行和测试，为优化方法提供性能对比的参考标准
- **analysis/**: 专门用于分析基线和优化方法的实验结果，包含统计分析、表格生成、报告生成
- **experiments/**: 负责协调基线和优化方法的实验执行，连接基线执行、优化方法和结果分析

### src/ - 源代码模块
- **methods/**: 各种KV缓存优化技术的实现
- **monitoring/**: 监控和可视化功能，与监控功能统一
- **common/**: 通用工具和数据加载器

### hace_core/ - 核心模块
- **models/**: 模型加载和配置
- **data/**: 数据集加载和处理
- **config.py**: 全局配置管理

## 🏆 系统状态

✅ **目录结构**: 模块化设计完成  
✅ **统一入口**: run_experiments.py就绪  
✅ **KV长度限制**: 128/1024 tokens  
✅ **评估流程**: 基线→优化→分析完整  
✅ **文档更新**: 使用说明齐全  

---

**系统已准备就绪，开始你的KV缓存优化研究！** 🎉
