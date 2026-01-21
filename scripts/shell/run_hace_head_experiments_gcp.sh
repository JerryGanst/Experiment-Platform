#!/bin/bash
# HACE Head-Level Experiments (GCP version)

set -e

cd /mnt/disks/experiment/Experiment-Platform

# Activate environment
source /mnt/disks/experiment/my_env/bin/activate

# Create logs directory
mkdir -p logs

# Common settings
MODEL="qwen2.5-7b-instruct"
DATASET="qmsum"
CACHE_SIZE=128
WINDOW_SIZE=32

echo "==========================================" 
echo "HACE Head-Level Experiments (GCP)"
echo "==========================================" 
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Cache Size: $CACHE_SIZE"
echo "Window Size: $WINDOW_SIZE"
echo ""

# LH-A: normal + high_entropy
echo "[1/4] Running LH-A (normal + high_entropy heads)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "high_entropy" \
    --pred_name "hace_LH_A_normal_highent" \
    2>&1 | tee logs/hace_LH_A.log

# LH-B: normal + low_entropy
echo ""
echo "[2/4] Running LH-B (normal + low_entropy heads)..."
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
echo "[3/4] Running LH-C (reverse_disp + high_entropy heads)..."
python3 scripts/run_hace_qmsum.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "reverse_disp" \
    --head_mode "high_entropy" \
    --pred_name "hace_LH_C_revdisp_highent" \
    2>&1 | tee logs/hace_LH_C.log

# LH-D: reverse_disp + low_entropy
echo ""
echo "[4/4] Running LH-D (reverse_disp + low_entropy heads)..."
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
ls -la results_clean/ | grep hace_LH
