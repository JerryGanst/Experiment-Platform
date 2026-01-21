#!/bin/bash
# HACE Preference Mode Comparison Experiments
# This script runs HACE with different preference modes on QMSum

set -e  # Exit on error

# Configuration
DEVICE=0
MODEL="qwen2.5-7b-instruct"
CACHE_SIZE=128
WINDOW_SIZE=32
DATASET="qmsum"

echo "=============================================="
echo "HACE Preference Mode Comparison Experiments"
echo "=============================================="
echo "Device: GPU ${DEVICE}"
echo "Model: ${MODEL}"
echo "Cache Size: ${CACHE_SIZE}"
echo "Window Size: ${WINDOW_SIZE}"
echo "Dataset: ${DATASET}"
echo "=============================================="
echo ""

# Experiment 1: Normal mode (baseline - high entropy gets more budget)
echo "[1/3] Running NORMAL mode (high entropy → high budget)..."
CUDA_VISIBLE_DEVICES=${DEVICE} python scripts/run_hace_qmsum.py \
  --model ${MODEL} \
  --cache_size ${CACHE_SIZE} \
  --window_size ${WINDOW_SIZE} \
  --pref_mode normal \
  --pred_name hace_normal_128 \
  --device 0 \
  --dataset ${DATASET}

echo ""
echo "[1/3] Evaluating NORMAL mode..."
python src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py \
  --model ${MODEL} \
  --dir_path results_clean/hace_normal_128

echo ""
echo "=============================================="
echo ""

# Experiment 2: Reverse_disp mode (advisor's suggestion - low entropy gets more budget)
echo "[2/3] Running REVERSE_DISP mode (low entropy → high budget)..."
CUDA_VISIBLE_DEVICES=${DEVICE} python scripts/run_hace_qmsum.py \
  --model ${MODEL} \
  --cache_size ${CACHE_SIZE} \
  --window_size ${WINDOW_SIZE} \
  --pref_mode reverse_disp \
  --pred_name hace_reverse_disp_128 \
  --device 0 \
  --dataset ${DATASET}

echo ""
echo "[2/3] Evaluating REVERSE_DISP mode..."
python src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py \
  --model ${MODEL} \
  --dir_path results_clean/hace_reverse_disp_128

echo ""
echo "=============================================="
echo ""

# Experiment 3: Reverse mode (both entropy and variance reversed)
echo "[3/3] Running REVERSE mode (low entropy + low variance → high budget)..."
CUDA_VISIBLE_DEVICES=${DEVICE} python scripts/run_hace_qmsum.py \
  --model ${MODEL} \
  --cache_size ${CACHE_SIZE} \
  --window_size ${WINDOW_SIZE} \
  --pref_mode reverse \
  --pred_name hace_reverse_128 \
  --device 0 \
  --dataset ${DATASET}

echo ""
echo "[3/3] Evaluating REVERSE mode..."
python src/third_party/cakekv-main/cakekv-main/experiments/LongBench/eval.py \
  --model ${MODEL} \
  --dir_path results_clean/hace_reverse_128

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results Summary:"
echo "----------------"

# Extract and display results
for mode in normal reverse_disp reverse; do
  result_file="results_clean/hace_${mode}_128/result.json"
  if [ -f "$result_file" ]; then
    echo ""
    echo "${mode} mode:"
    cat "$result_file"
  fi
done

echo ""
echo "=============================================="
echo "Detailed results saved in:"
echo "  - results_clean/hace_normal_128/"
echo "  - results_clean/hace_reverse_disp_128/"
echo "  - results_clean/hace_reverse_128/"
echo "=============================================="
