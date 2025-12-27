#!/bin/bash
# Run both HACE experiments sequentially
# Usage: nohup bash scripts/run_both_experiments.sh > logs/both_experiments.log 2>&1 &

cd /cloud/cloud-ssd1/Experiment-Platform
source /usr/local/miniconda3/bin/activate py312

echo "=========================================="
echo "Starting HACE Experiment Suite"
echo "Time: $(date)"
echo "=========================================="

# Run reverse_disp mode
echo ""
echo "[1/2] Running reverse_disp mode..."
rm -rf results_clean/hace_reverse_disp_final
python3 scripts/run_hace_qmsum.py \
    --pref_mode reverse_disp \
    --pred_name hace_reverse_disp_final \
    --device 0

echo "reverse_disp completed at $(date)"
echo "Sample count: $(wc -l < results_clean/hace_reverse_disp_final/qmsum.jsonl)"

# Run normal mode
echo ""
echo "[2/2] Running normal mode..."
rm -rf results_clean/hace_normal_final
python3 scripts/run_hace_qmsum.py \
    --pref_mode normal \
    --pred_name hace_normal_final \
    --device 0

echo "normal completed at $(date)"
echo "Sample count: $(wc -l < results_clean/hace_normal_final/qmsum.jsonl)"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Time: $(date)"
echo "=========================================="

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

            # Take max score across references
            best_scores = None
            for ref in refs:
                s = scorer.score(ref, pred)
                if best_scores is None or s['rouge1'].fmeasure > best_scores['rouge1'].fmeasure:
                    best_scores = s

            for k in scores:
                scores[k].append(best_scores[k].fmeasure)

    return {k: sum(v)/len(v)*100 for k, v in scores.items()}

print('='*60)
print('ROUGE Scores Comparison')
print('='*60)

for name in ['hace_reverse_disp_final', 'hace_normal_final']:
    path = Path(f'results_clean/{name}/qmsum.jsonl')
    if path.exists():
        scores = calc_rouge(path)
        print(f'{name}:')
        print(f'  ROUGE-1: {scores[\"rouge1\"]:.2f}')
        print(f'  ROUGE-2: {scores[\"rouge2\"]:.2f}')
        print(f'  ROUGE-L: {scores[\"rougeL\"]:.2f}')
        print()
"

echo "Done!"
