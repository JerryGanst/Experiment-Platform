#!/bin/bash
# HACE Head-Level Experiments
# 实验矩阵：测试头级熵分配策略

set -e

cd /cloud/cloud-ssd1/Experiment-Platform

# Activate environment
source /usr/local/miniconda3/bin/activate py312

# Common settings
MODEL="qwen2.5-7b-instruct"
DATASET="qmsum"
CACHE_SIZE=128
WINDOW_SIZE=32

echo "=========================================="
echo "HACE Head-Level Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Cache Size: $CACHE_SIZE"
echo "Window Size: $WINDOW_SIZE"
echo ""

# Experiment matrix:
# | 实验 | 层级策略 (pref_mode) | 头级策略 (head_mode) | 假设 |
# |------|---------------------|---------------------|------|
# | LH-A | normal              | high_entropy        | 高熵头注意力分散，需要更多token |
# | LH-B | normal              | low_entropy         | 低熵头更重要（导师建议） |
# | LH-C | reverse_disp        | high_entropy        | 混合测试 |
# | LH-D | reverse_disp        | low_entropy         | 双低熵优先 |

# Baseline: L1 - CAKE original (normal, no head alloc)
echo "[1/5] Running Baseline L1 (CAKE original: normal, uniform heads)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "" \
    --pred_name "hace_L1_normal_uniform" \
    2>&1 | tee logs/hace_L1.log

# LH-A: normal + high_entropy
echo ""
echo "[2/5] Running LH-A (normal + high_entropy heads)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "high_entropy" \
    --pred_name "hace_LH_A_normal_highent" \
    2>&1 | tee logs/hace_LH_A.log

# LH-B: normal + low_entropy (导师建议)
echo ""
echo "[3/5] Running LH-B (normal + low_entropy heads - 导师建议)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "low_entropy" \
    --pred_name "hace_LH_B_normal_lowent" \
    2>&1 | tee logs/hace_LH_B.log

# LH-C: reverse_disp + high_entropy
echo ""
echo "[4/5] Running LH-C (reverse_disp + high_entropy heads)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "reverse_disp" \
    --head_mode "high_entropy" \
    --pred_name "hace_LH_C_revdisp_highent" \
    2>&1 | tee logs/hace_LH_C.log

# LH-D: reverse_disp + low_entropy (双低熵优先)
echo ""
echo "[5/5] Running LH-D (reverse_disp + low_entropy heads - 双低熵)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "reverse_disp" \
    --head_mode "low_entropy" \
    --pred_name "hace_LH_D_revdisp_lowent" \
    2>&1 | tee logs/hace_LH_D.log

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results are in results_clean/:"
ls -la results_clean/ | grep hace_L

echo ""
echo "Run evaluation with:"
echo "  python3 scripts/eval_hace_results.py"
