#!/bin/bash
# HACE Head-Level Experiments with VLLM
# RTX PRO 6000 Blackwell (97GB VRAM)

set -e

cd /mnt/disks/experiment/Experiment-Platform

# Activate environment
source /mnt/disks/experiment/my_env/bin/activate

# IMPORTANT: Use TRITON attention backend for Blackwell GPU compatibility
# Flash Attention PTX kernels are not compatible with sm_120 (Blackwell)
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
echo "Using VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"

# Install/upgrade vllm if needed
pip install vllm --quiet 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Common settings
MODEL_PATH="/mnt/disks/experiment/models/Qwen2.5-7B-Instruct"
DATASET="qmsum"
CACHE_SIZE=128
WINDOW_SIZE=32
MAX_MODEL_LEN=32768
GPU_UTIL=0.90
NUM_SAMPLES=200

echo "=========================================="
echo "HACE Head-Level Experiments (VLLM)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Cache Size: $CACHE_SIZE"
echo "Window Size: $WINDOW_SIZE"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "GPU Utilization: $GPU_UTIL"
echo ""

# LH-A: normal + high_entropy
echo "[1/4] Running LH-A (normal + high_entropy heads)..."
python3 scripts/run_hace_vllm.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "high_entropy" \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_UTIL \
    --num_samples $NUM_SAMPLES \
    --pred_name "hace_LH_A_normal_highent" \
    2>&1 | tee logs/hace_LH_A_vllm.log

echo ""
echo "[2/4] Running LH-B (normal + low_entropy heads)..."
python3 scripts/run_hace_vllm.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "normal" \
    --head_mode "low_entropy" \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_UTIL \
    --num_samples $NUM_SAMPLES \
    --pred_name "hace_LH_B_normal_lowent" \
    2>&1 | tee logs/hace_LH_B_vllm.log

echo ""
echo "[3/4] Running LH-C (reverse_disp + high_entropy heads)..."
python3 scripts/run_hace_vllm.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "reverse_disp" \
    --head_mode "high_entropy" \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_UTIL \
    --num_samples $NUM_SAMPLES \
    --pred_name "hace_LH_C_revdisp_highent" \
    2>&1 | tee logs/hace_LH_C_vllm.log

echo ""
echo "[4/4] Running LH-D (reverse_disp + low_entropy heads)..."
python3 scripts/run_hace_vllm.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --cache_size $CACHE_SIZE \
    --window_size $WINDOW_SIZE \
    --pref_mode "reverse_disp" \
    --head_mode "low_entropy" \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_UTIL \
    --num_samples $NUM_SAMPLES \
    --pred_name "hace_LH_D_revdisp_lowent" \
    2>&1 | tee logs/hace_LH_D_vllm.log

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results in results_clean/:"
ls -la results_clean/ | grep hace_LH
