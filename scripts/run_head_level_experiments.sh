#!/bin/bash
# Run HACE head-level adaptive experiments
# Usage: nohup bash scripts/run_head_level_experiments.sh > logs/head_experiments.log 2>&1 &

cd /cloud/cloud-ssd1/Experiment-Platform
source /usr/local/miniconda3/bin/activate py312

echo "=============================================="
echo "HACE Head-Level Adaptive Experiments"
echo "Time: $(date)"
echo "=============================================="

# Experiment matrix:
# 1. Baseline: CAKE without head-level (head_mode="")
# 2. LH1: Low concentration heads get more budget (our hypothesis)
# 3. LH2: High concentration heads get more budget (advisor's hypothesis)

mkdir -p logs

# Experiment 1: Baseline (no head-level)
echo ""
echo "[1/3] Running baseline (no head-level optimization)..."
rm -rf results_clean/hace_baseline_head
python3 scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --head_mode "" \
    --cache_size 128 \
    --pred_name hace_baseline_head \
    --device 0

echo "Baseline completed at $(date)"
echo "Sample count: $(wc -l < results_clean/hace_baseline_head/qmsum.jsonl 2>/dev/null || echo 0)"

# Experiment 2: LH1 (low concentration -> high budget)
echo ""
echo "[2/3] Running LH1 (low concentration -> high budget)..."
rm -rf results_clean/hace_lh1
python3 scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --head_mode lh1 \
    --cache_size 128 \
    --pred_name hace_lh1 \
    --device 0

echo "LH1 completed at $(date)"
echo "Sample count: $(wc -l < results_clean/hace_lh1/qmsum.jsonl 2>/dev/null || echo 0)"

# Experiment 3: LH2 (high concentration -> high budget)
echo ""
echo "[3/3] Running LH2 (high concentration -> high budget)..."
rm -rf results_clean/hace_lh2
python3 scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --head_mode lh2 \
    --cache_size 128 \
    --pred_name hace_lh2 \
    --device 0

echo "LH2 completed at $(date)"
echo "Sample count: $(wc -l < results_clean/hace_lh2/qmsum.jsonl 2>/dev/null || echo 0)"

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Time: $(date)"
echo "=============================================="

# Calculate ROUGE scores
echo ""
echo "Calculating ROUGE scores..."
python3 -c "
import json
from pathlib import Path
from rouge_score import rouge_scorer

def calc_rouge(pred_path):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with open(pred_path) as f:
        for line in f:
            data = json.loads(line)
            pred = data['pred']
            refs = data['answers']
            if isinstance(refs, str):
                refs = [refs]

            best_scores = None
            for ref in refs:
                s = scorer.score(ref, pred)
                if best_scores is None or s['rouge1'].fmeasure > best_scores['rouge1'].fmeasure:
                    best_scores = s

            for k in scores:
                scores[k].append(best_scores[k].fmeasure)

    return {k: sum(v)/len(v)*100 for k, v in scores.items()}

print('='*60)
print('HACE Head-Level Experiments - ROUGE Scores')
print('='*60)

results = {}
for name in ['hace_baseline_head', 'hace_lh1', 'hace_lh2']:
    path = Path(f'results_clean/{name}/qmsum.jsonl')
    if path.exists():
        scores = calc_rouge(path)
        results[name] = scores
        print(f'{name}:')
        print(f'  ROUGE-1: {scores[\"rouge1\"]:.2f}')
        print(f'  ROUGE-2: {scores[\"rouge2\"]:.2f}')
        print(f'  ROUGE-L: {scores[\"rougeL\"]:.2f}')
        print()

# Print comparison
if len(results) == 3:
    print('='*60)
    print('Comparison Summary:')
    print('='*60)
    baseline = results['hace_baseline_head']
    lh1 = results['hace_lh1']
    lh2 = results['hace_lh2']

    print(f'LH1 vs Baseline:')
    for m in ['rouge1', 'rouge2', 'rougeL']:
        diff = lh1[m] - baseline[m]
        print(f'  {m}: {diff:+.2f}')

    print(f'LH2 vs Baseline:')
    for m in ['rouge1', 'rouge2', 'rougeL']:
        diff = lh2[m] - baseline[m]
        print(f'  {m}: {diff:+.2f}')

    print(f'LH1 vs LH2:')
    for m in ['rouge1', 'rouge2', 'rougeL']:
        diff = lh1[m] - lh2[m]
        print(f'  {m}: {diff:+.2f}')
"

echo "Done!"
