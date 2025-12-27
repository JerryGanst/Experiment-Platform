#!/usr/bin/env python3
"""Compare HACE experiment results between different pref_modes."""
import json
import sys
from pathlib import Path

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge_score...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
    from rouge_score import rouge_scorer


def evaluate_predictions(jsonl_path):
    """Calculate ROUGE scores for predictions."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    predictions = []
    answers = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data['pred'])
            # Take the first answer as reference
            answers.append(data['answers'][0] if data['answers'] else "")

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ans in zip(predictions, answers):
        scores = scorer.score(ans, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': round(100 * sum(rouge1_scores) / len(rouge1_scores), 2),
        'rouge2': round(100 * sum(rouge2_scores) / len(rouge2_scores), 2),
        'rougeL': round(100 * sum(rougeL_scores) / len(rougeL_scores), 2),
        'num_samples': len(predictions)
    }


def main():
    results_dir = Path(__file__).parent.parent / "results_clean"

    experiments = {
        "normal (高熵→高预算)": results_dir / "hace_normal_128" / "qmsum.jsonl",
        "reverse_disp (低熵→高预算)": results_dir / "hace_reverse_disp_128" / "qmsum.jsonl",
    }

    print("=" * 80)
    print("HACE Preference Mode Comparison on QMSum")
    print("=" * 80)
    print()

    results = {}
    for mode_name, path in experiments.items():
        if not path.exists():
            print(f"⚠️  {mode_name}: 文件不存在 ({path})")
            continue

        print(f"评估 {mode_name}...")
        scores = evaluate_predictions(path)
        results[mode_name] = scores

        print(f"  样本数: {scores['num_samples']}")
        print(f"  ROUGE-1: {scores['rouge1']}")
        print(f"  ROUGE-2: {scores['rouge2']}")
        print(f"  ROUGE-L: {scores['rougeL']}")
        print()

    # Compare results
    if len(results) == 2:
        modes = list(results.keys())
        mode1, mode2 = modes[0], modes[1]

        print("=" * 80)
        print("对比分析")
        print("=" * 80)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            diff = results[mode2][metric] - results[mode1][metric]
            winner = mode2 if diff > 0 else mode1

            print(f"\n{metric.upper()}:")
            print(f"  {mode1}: {results[mode1][metric]:.2f}")
            print(f"  {mode2}: {results[mode2][metric]:.2f}")
            print(f"  差异: {abs(diff):.2f} {'↑' if diff > 0 else '↓'}")
            print(f"  {'✓' if diff > 0 else '✗'} {winner} 表现更好")

        print("\n" + "=" * 80)
        print("结论")
        print("=" * 80)

        avg_normal = sum([results[mode1]['rouge1'], results[mode1]['rouge2'], results[mode1]['rougeL']]) / 3
        avg_reverse = sum([results[mode2]['rouge1'], results[mode2]['rouge2'], results[mode2]['rougeL']]) / 3

        if avg_reverse > avg_normal:
            print(f"\n✓ {mode2} 的平均ROUGE分数更高 ({avg_reverse:.2f} vs {avg_normal:.2f})")
            print("这意味着：低注意力熵的层应该获得更多计算预算")
            print("导师的观点是正确的！")
        else:
            print(f"\n✓ {mode1} 的平均ROUGE分数更高 ({avg_normal:.2f} vs {avg_reverse:.2f})")
            print("这意味着：高注意力熵的层应该获得更多计算预算")
            print("原始设计是正确的。")


if __name__ == "__main__":
    main()
