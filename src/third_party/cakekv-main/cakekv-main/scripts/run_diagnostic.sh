#!/bin/bash
# run_diagnostic.sh
# HACE Token Retention Diagnostic: Compare Baseline vs Ada-KV

set -e

cd /mnt/disks/experiment/Experiment-Platform/src/third_party/cakekv-main/cakekv-main
source /mnt/disks/experiment/my_env/bin/activate

export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
MAX_SAMPLES=${1:-5}  # Default: 5 samples for quick diagnostic
CACHE_SIZE=${2:-256}
WINDOW_SIZE=${3:-32}
MODEL="qwen2.5-7b-instruct"

OUTPUT_DIR="diagnostic_results"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "HACE Diagnostic: Baseline vs Ada Comparison"
echo "=============================================="
echo "Max samples: $MAX_SAMPLES"
echo "Cache size: $CACHE_SIZE"
echo "Window size: $WINDOW_SIZE"
echo ""

# === 1. Run Baseline (no head-adaptive), record diagnostic ===
echo "[1/3] Running Baseline with diagnostic..."
HACE_DIAGNOSTIC=1 \
HACE_PREF_MODE=normal \
HACE_HEAD_MODE="" \
python3 experiments/LongBench/pred_cake.py \
    --model "$MODEL" \
    --compress \
    --cache_size "$CACHE_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --pred_name "diagnostic_baseline" \
    --device 0 \
    --max_samples "$MAX_SAMPLES" 2>&1 | tee "$OUTPUT_DIR/baseline_log.txt"

# Save diagnostic results
python3 -c "
from cake.model.modify_qwen2 import DIAGNOSTIC
DIAGNOSTIC.save('$OUTPUT_DIR/diagnostic_baseline.json')
print(f'Baseline stats: {DIAGNOSTIC.get_stats()}')
"

# === 2. Run Ada (high_entropy mode), record diagnostic ===
echo ""
echo "[2/3] Running Ada (high_entropy) with diagnostic..."
HACE_DIAGNOSTIC=1 \
HACE_PREF_MODE=normal \
HACE_HEAD_MODE=high_entropy \
python3 experiments/LongBench/pred_cake.py \
    --model "$MODEL" \
    --compress \
    --cache_size "$CACHE_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --pred_name "diagnostic_ada_high" \
    --device 0 \
    --max_samples "$MAX_SAMPLES" \
    --use_head_adaptive 2>&1 | tee "$OUTPUT_DIR/ada_high_log.txt"

python3 -c "
from cake.model.modify_qwen2 import DIAGNOSTIC
DIAGNOSTIC.save('$OUTPUT_DIR/diagnostic_ada_high.json')
print(f'Ada (high_entropy) stats: {DIAGNOSTIC.get_stats()}')
"

# === 3. Generate visualizations ===
echo ""
echo "[3/3] Generating visualizations..."

# Layer 0 analysis
python3 diagnostic_visualize.py \
    --baseline "$OUTPUT_DIR/diagnostic_baseline.json" \
    --ada "$OUTPUT_DIR/diagnostic_ada_high.json" \
    --sample 0 \
    --layer 0 \
    --output "$OUTPUT_DIR/diagnostic_layer0.png"

# Mid-layer analysis (layer 15 for a 28-layer model)
python3 diagnostic_visualize.py \
    --baseline "$OUTPUT_DIR/diagnostic_baseline.json" \
    --ada "$OUTPUT_DIR/diagnostic_ada_high.json" \
    --sample 0 \
    --layer 15 \
    --output "$OUTPUT_DIR/diagnostic_layer15.png"

# Cross-layer analysis
python3 diagnostic_visualize.py \
    --baseline "$OUTPUT_DIR/diagnostic_baseline.json" \
    --ada "$OUTPUT_DIR/diagnostic_ada_high.json" \
    --sample 0 \
    --all-layers \
    --output "$OUTPUT_DIR/diagnostic_all_layers.png"

echo ""
echo "=============================================="
echo "Diagnostic Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
echo "  - diagnostic_baseline.json"
echo "  - diagnostic_ada_high.json"
echo "  - diagnostic_layer0.png"
echo "  - diagnostic_layer15.png"
echo "  - diagnostic_all_layers.png"
echo ""
echo "To analyze specific answer positions, run:"
echo "  python3 diagnostic_visualize.py \\"
echo "    --baseline $OUTPUT_DIR/diagnostic_baseline.json \\"
echo "    --ada $OUTPUT_DIR/diagnostic_ada_high.json \\"
echo "    --answer-positions '100,101,102,103'"
