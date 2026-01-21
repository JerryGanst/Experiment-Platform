# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV Cache optimization research platform for LLM inference. Implements HACE (Hierarchical Adaptive Cache Eviction) and CAKE algorithms for Qwen, LLaMA, and Mistral models. Designed for remote GPU execution.

## Essential Commands

```bash
# Run unified experiments
python scripts/experiments/run_experiments.py --experiment full_evaluation --tag production
python scripts/experiments/run_experiments.py --experiment baseline --datasets hotpotqa --dry-run

# HACE experiments (uses HuggingFace, not VLLM)
python scripts/experiments/run_hace.py --pref_mode reverse_disp --pred_name hace_test --device 0

# Full HACE comparison (3 modes)
./scripts/shell/run_hace_experiments.sh

# Evaluation
python vendor/cake/longbench/eval.py \
    --model qwen2.5-7b-instruct --dir_path results_clean/hace_test

# Tests
pytest tests/
```

## Architecture

- **src/hace/**: Core framework (formerly hace_core/)
  - `config.py` - Central configuration (MODEL_CONFIG, EXPERIMENT_CONFIG, DATASET_CONFIG)
  - `models/` - Model loading, attention collection, CAKE/H2O converters
  - `core/` - Unified allocator, strategy selector, integration framework
  - `methods/base.py` - Base class for KV cache methods

- **vendor/cake/**: CAKE implementation (扁平化的第三方代码)
  - `cake_cache.py` - KV cache management and budget allocation
  - `model/modify_qwen2.py` - Attention patching with HACE head-level entropy
  - `longbench/pred_cake.py` - Experiment entry point
  - `longbench/eval.py` - Evaluation script

- **config/**: 统一配置目录
  - `models/` - model2path.json, model2maxlen.json, model2tau.json
  - `datasets/` - dataset2prompt.json, dataset2maxlen.json
  - `experiments/` - cake_grid.csv, kv_lengths.yaml

- **scripts/**: 脚本 (分类组织)
  - `experiments/` - run_hace.py, run_fullkv.py, run_experiments.py
  - `phase0/` - Phase 0 验证脚本
  - `maintenance/` - 维护脚本
  - `diagnostics/` - 诊断脚本
  - `shell/` - Shell 脚本

- **evaluation/**: Baseline evaluation framework with statistical analysis
- **data/**: 37 LongBench datasets (*.jsonl)

## Critical Constraints

**HACE requires HuggingFace, not VLLM**: HACE's head-level differentiation needs attention weights during inference. VLLM's Flash Attention kernels don't expose intermediate attention weights. See `docs/VLLM_HACE_incompatibility.md`.

## Dependency Versions (CRITICAL)

The CAKE codebase has specific version requirements due to transformers API changes:

```
# Verified working combination (2025-01-14)
transformers==4.45.0
torch==2.11.0.dev20260113+cu128  # PyTorch Nightly for Blackwell GPU (sm_120)
```

**Known compatibility issues fixed:**
- transformers >= 4.46: `DynamicCache.key_cache` structure changed from `List[Tensor]` to `List[List[Tensor]]`
- transformers >= 4.57: New `layers` API in DynamicCache
- Blackwell GPU (RTX PRO 6000, sm_120): Requires PyTorch Nightly with cu128

**If you get garbage output (000000...)**: Check transformers version compatibility with CAKE's attention patching code.

## HACE Configuration

Environment variables:
- `HACE_PREF_MODE`: normal | reverse_disp | reverse | uniform
- `HACE_HEAD_MODE`: (empty) | high_entropy | low_entropy | budget_realloc | adakv

Preference modes:
- `normal`: H^(+) * V^(+) - High entropy → high budget
- `reverse_disp`: (1/H)^(+) * V^(+) - Low entropy → high budget
- `reverse`: (1/H)^(+) * (1/V)^(+) - Low entropy + low variance → high budget

### Known Issues (2025-01-21 Updated)

#### Zero-Padding Problem (Critical)

**Variable budget 方案在 CAKE 框架下都有性能问题**，包括 `budget_realloc` 和 `adakv`。

**根本原因**：
1. Variable budget 产生不同长度的 KV cache（如 [152, 134, 178, 140]）
2. CAKE 框架要求所有 head 的 KV cache 长度相同
3. 因此需要 padding 到 max_budget
4. **Padding 使用零向量**（`vendor/cake/cake_cache.py` 第 593-597 行），会干扰注意力计算

**实际 AdaKV 实验数据** (2025-01-16, Qwen2.5-7B, alpha=0.8)：

从 `logs/smoke_adakv_alpha08.log` 观察到的 budget 分布：
```
[153, 81, 263, 107], range=182, max_padding=69.2%
[161, 191, 371, 297], range=210, max_padding=56.6%
[118, 91, 223, 92], range=132, max_padding=59.2%
```

| 样本 | Budget 分布 | Range | 最大 Padding 比例 |
|-----|------------|-------|-----------------|
| 1 | [153, 81, 263, 107] | 182 | **69.2%** (head 1) |
| 2 | [161, 191, 371, 297] | 210 | **56.6%** (head 0) |
| 3 | [118, 91, 223, 92] | 132 | **59.2%** (head 1) |

**结论**：即使 alpha=0.8（80% uniform + 20% adaptive），padding 比例仍高达 **60-70%**，这解释了为什么 AdaKV (14.55%) 显著低于 uniform (21.75%)。

**重要澄清**：Qwen 在 `budget_realloc` 上"能用"**不是因为有 mask fix**，而是因为熵范围小（~0.75），budget variance 相对较小。但 AdaKV 的 dispersion-based 算法产生更大的 variance。

#### 模型兼容性矩阵

| Model | budget_realloc | adakv | high_entropy | low_entropy | uniform |
|-------|---------------|-------|--------------|-------------|---------|
| Qwen2.5-7B | ⚠️ 有损 | ⚠️ 有损 | ✅ Works | ✅ Works | ✅ Works |
| Mistral-7B | ❌ Garbage | ❌ Garbage | ✅ Works | ✅ Works | ✅ Works |

- ⚠️ 有损：能运行但性能下降（如 AdaKV 14.55% vs uniform 21.75%）
- ❌ Garbage：输出乱码

#### 解决方案选项

| 方案 | 工作量 | 效果预期 | 代码位置 |
|-----|--------|---------|---------|
| 1. 放弃 variable budget | 0 | 无 | - |
| 2. 调高 alpha（增加 uniform 比例） | 低 | 折中 | `cake_cache.py` L524 |
| 3. 实现 attention mask | 高 | 最优 | 需新增 |

**当前 AdaKV alpha 设置**：`alpha=0.8`（80% uniform + 20% adaptive），已经很保守但仍有性能损失。

**Workaround**: 对于需要可靠性能的实验，使用 `high_entropy`、`low_entropy` 或 `uniform` 模式。

### Experiment Quality Control (Critical Lesson)

**永远先检查输出内容，不仅仅是日志配置！**

测试实验时的检查清单：
1. ✅ 检查日志确认配置生效 (如 `[HACE] Budget realloc enabled`)
2. ✅ **检查输出内容是否是有意义的文本**（不是乱码/重复字符）
3. ✅ 运行评估脚本验证得分合理
4. ✅ 通过后再跑全量

**教训**: 2025-01-15 因只检查日志未检查输出，浪费了 200 样本的全量实验时间。

## Sampling Configuration (Phase 0)

Phase 0 experiments use sampling decoding to introduce randomness for statistical validity:

```python
# run_hace_qmsum.py generation settings
do_sample=True        # Enable sampling (not greedy)
temperature=0.7       # Moderate randomness (industry default)
top_p=0.95            # Nucleus sampling
--seed <int>          # Random seed for reproducibility
```

- **Seed controls randomness**: Same seed → identical output; different seeds → varied outputs
- **Temperature=0.7**: Conservative value that maintains quality while introducing variance
- **Purpose**: Enable ANOVA to estimate within-group variance across 3 repeated runs

## Remote Execution

Primary server: `ssh -p 23 root@117.50.241.187`
- **GPU**: 双 RTX 4090 (24GB x 2)
- **Project root**: `/cloud/cloud-ssd1/Experiment-Platform`
- **Global models**: `/model` (symlinked to project)
- **Environment**: `source /usr/local/miniconda3/bin/activate py310`
- **Logs**: `logs/` 目录（实验日志按实验名命名）

```bash
# Sync code to server
rsync -avz -e "ssh -p 23" ./ root@117.50.241.187:/cloud/cloud-ssd1/Experiment-Platform/

# Monitor experiments
bash scripts/shell/monitor_experiment.sh
nvidia-smi
wc -l results_clean/*/qmsum.jsonl
```

## Model Path Configuration

Set via environment variable (preferred):
```bash
export CORECODE_MODEL_PATH="/mnt/models/model-name"
export HACE_MODEL_PATH="$CORECODE_MODEL_PATH"
```

Or place models in: `./models/`, `~/models/`, or configure in `config/models/model2path.json`

## Key Metrics

- ROUGE-1/2/L for summarization tasks
- F1 for QA tasks
- TTFT (Time to First Token), TPOT (Time per Output Token)
- GPU memory usage

---

## Research Methodology

### Core Principle
**Validate hypotheses before building contributions upon them.** Any observation that forms the basis of a research claim must be rigorously verified with statistical testing before proceeding.

### Current Core Hypothesis (Under Validation)
> "The optimal KV cache eviction strategy varies with compression rate - a strategy that works best at high compression may not be optimal at low compression."

**Status**: UNVALIDATED. Current experiments only tested at cache_size=128, showing ~0.3% difference between strategies (within statistical noise).

---

## Phase 0: Hypothesis Validation Protocol

### Research Hypotheses
- **H0 (Null)**: Optimal strategy is independent of compression rate. Strategy rankings remain consistent across different cache sizes.
- **H1 (Alternative)**: Significant interaction effect exists between strategy and compression rate. Some strategies become relatively better/worse as compression changes.

### Experiment Design (Minimal Viable Validation)

| Factor | Levels | Values |
|--------|--------|--------|
| Strategy | 3 | high_entropy, low_entropy, uniform |
| Cache Size | 3 | 128, 512, 2048 tokens |
| Dataset | 2 | qmsum (summarization), hotpotqa (QA) |
| Repeats | 3 | Different random seeds |

**Total runs**: 3 × 3 × 2 × 3 = **54 experiments**
**Samples per run**: 100 (sufficient for statistical power)
**Estimated time**: 1-2 days on A100

### Control Variables (MUST BE FIXED)
- Model: Qwen2.5-7B-Instruct
- Window size: 32 tokens (or proportional: window = cache_size / 4)
- Decoding: greedy (temperature=0)
- Max new tokens: dataset-specific (qmsum=512, hotpotqa=128)
- Prompt template: CAKE standard templates

### Judgment Criteria

#### Statistical Significance
- **Test**: Two-way ANOVA (strategy × cache_size interaction term)
- **Threshold**: p < 0.05 for interaction effect

#### Practical Significance (Effect Size)
- **Minimum meaningful effect**: |Δ ROUGE| > 2% or |Δ F1| > 2%
- **Rationale**: Smaller differences are within typical variance and not worth optimizing

### Decision Tree

```
Phase 0 Results
├── Interaction SIGNIFICANT (p < 0.05) AND Effect Size > 2%
│   └── PROCEED to Phase 1-4: Core hypothesis validated
│
├── Interaction SIGNIFICANT but Effect Size < 2%
│   └── RECONSIDER: Effect exists but too small for practical impact
│       → Consider: Is this worth a paper? Pivot to other research questions?
│
├── Interaction NOT SIGNIFICANT (p >= 0.05)
│   └── ABANDON current direction
│       → Document negative result, pivot to alternative hypotheses
│
└── Mixed results (significant for some datasets, not others)
    └── REFINE: Focus on task types where effect exists
        → Narrow research scope to specific domains
```

### Output Specification

**Directory**: `runs/phase0_validation/`

**Required outputs**:
| File | Description |
|------|-------------|
| `raw_results.csv` | All 54 runs with strategy, cache_size, dataset, seed, ROUGE/F1 scores |
| `anova_results.json` | Two-way ANOVA output: F-statistics, p-values, effect sizes |
| `interaction_plot.png` | Strategy × Cache Size interaction visualization |
| `decision.md` | Go/No-Go decision with justification |

### Expected Interaction Plot (If H1 is True)

```
ROUGE-L Score
  │
  │     ╭── high_entropy
  │    ╱ ╲
  │   ╱   ╲__
  │  ╱        ╲
  │ ╱          ╲── low_entropy
  │/            ╲
  │──────────────╲── uniform
  └────────────────────────
     128    512    2048   (cache size)

  Lines crossing = Interaction effect exists
```

### If H0 is True (No Interaction)

```
ROUGE-L Score
  │
  │  ────────────── high_entropy (best at all sizes)
  │  ────────────── low_entropy
  │  ────────────── uniform
  │
  └────────────────────────
     128    512    2048   (cache size)

  Parallel lines = No interaction, one strategy always best
```

---

## Phase 0 Validation Commands

### Pre-flight Test Strategy (CRITICAL)

**永远先跑测试实验！** 在全量实验前，先用 20 个样本验证配置正确性。

设置环境变量限制样本数：
```bash
export HACE_MAX_SAMPLES=20
```

测试实验的检查清单：
1. 运行 20 样本测试
2. 检查输出文件是否正确生成
3. 检查日志确认 head_mode 是否正确识别 (如 `[HACE] Head-level adaptive mode enabled: budget_realloc`)
4. 确认评估脚本能正确解析结果
5. **通过后**再跑全量 (去掉 HACE_MAX_SAMPLES 或设为 0)

这可以避免 "配置不生效但全量已跑完" 的浪费时间问题。

### Quick Single-Condition Test
```bash
# Test one configuration to verify pipeline works
python scripts/experiments/run_hace.py \
    --model qwen2.5-7b-instruct \
    --dataset qmsum \
    --cache_size 512 \
    --window_size 32 \
    --pref_mode normal \
    --head_mode high_entropy \
    --pred_name phase0_test_high_512 \
    --device 0

# Evaluate
python vendor/cake/longbench/eval.py \
    --model qwen2.5-7b-instruct \
    --dir_path results_clean/phase0_test_high_512
```

### Full Phase 0 Sweep
```bash
# Run all 54 conditions systematically
# Use run_phase0_validation.py when created, or manual loop:

for cache in 128 512 2048; do
    for mode in high_entropy low_entropy uniform; do
        for dataset in qmsum hotpotqa; do
            for seed in 42 123 456; do
                python scripts/experiments/run_hace.py \
                    --cache_size $cache \
                    --head_mode $mode \
                    --dataset $dataset \
                    --seed $seed \
                    --pred_name phase0_${mode}_${cache}_${dataset}_s${seed}
            done
        done
    done
done
```

### Statistical Analysis
```python
# Use existing StatisticsAnalyzer
from evaluation.analysis.statistical_tests import StatisticsAnalyzer

analyzer = StatisticsAnalyzer()

# Load Phase 0 results
import pandas as pd
df = pd.read_csv("runs/phase0_validation/raw_results.csv")

# Two-way ANOVA for interaction
import scipy.stats as stats
# Perform analysis and save to anova_results.json
```

---

## Phase 1-4: Full Research Plan (ONLY AFTER Phase 0 PASSES)

> **IMPORTANT**: Do not proceed to Phase 1-4 until Phase 0 validates the core hypothesis.

### Phase 1: Collect Interaction Data
- Systematic data collection across strategies and compression rates
- 10 datasets (5 summarization, 5 QA)
- 5 cache sizes: 64, 128, 256, 512, 1024 tokens

### Phase 2: Characterize Interaction Patterns
- Which task types show strongest interaction?
- At what compression threshold does the optimal strategy change?
- Can we predict optimal strategy from task characteristics?

### Phase 3: Develop Adaptive Strategy
- Design automatic strategy selection based on runtime signals
- Implement as extension to CAKE framework
- Validate on held-out datasets

### Phase 4: Paper & Contribution
- Frame contribution around validated interaction effect
- Compare against static strategy baselines
- Ablation studies on adaptive mechanism

---

## Time Budget

| Phase | Duration | Exit Criteria |
|-------|----------|---------------|
| Phase 0 | Max 1 week | Hypothesis validated OR abandoned |
| Phase 1 | 2 weeks | Complete interaction dataset |
| Phase 2 | 1 week | Interaction patterns characterized |
| Phase 3 | 2 weeks | Adaptive strategy implemented |
| Phase 4 | 2 weeks | Paper draft completed |

**Critical**: If Phase 0 fails (H0 not rejected), stop immediately. Do not invest in Phases 1-4 for a non-existent effect.

---

## GPU 使用规则 (双卡机器)

**不要使用张量并行分摊，一张卡跑一个实验**：

```bash
# 正确: 各卡独立跑实验
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/run_hace.py --device 0 --pred_name exp1 &
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/run_hace.py --device 0 --pred_name exp2 &
wait

# 错误: 不要这样 (会尝试张量并行)
CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/run_hace.py
```

这样可以同时运行两个独立实验，最大化 GPU 利用率。
