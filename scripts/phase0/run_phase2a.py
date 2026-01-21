#!/usr/bin/env python3
"""
Phase 2A: 快速验证交互效应存在性

仅运行 9 个实验 (3 策略 × 3 压缩率 × 1 数据集 × 1 种子)
跑完后立即进行 ANOVA 分析，判断是否存在交互效应
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from itertools import product

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.phase0_config import (
    MODEL, WINDOW_SIZE, PREF_MODE,
    RAW_RESULTS_DIR, CHECKPOINTS_DIR,
    get_experiment_id, get_head_mode,
    ensure_directories, PROJECT_ROOT
)

# Phase 2A 配置: 最小化验证集
STRATEGIES = ["high_entropy", "low_entropy", "uniform"]
CACHE_SIZES = [128, 512, 2048]
DATASET = "qmsum"
SEED = 42

CHECKPOINT_FILE = CHECKPOINTS_DIR / "phase2a_progress.json"


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "start_time": None}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_single_experiment(strategy, cache_size, device=0):
    """Run a single experiment."""
    exp_id = get_experiment_id(strategy, cache_size, DATASET, SEED)
    head_mode = get_head_mode(strategy)

    # Experiment-specific output directory
    exp_output_dir = RAW_RESULTS_DIR / exp_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "experiments" / "run_hace.py"),
        "--model", MODEL,
        "--dataset", DATASET,
        "--cache_size", str(cache_size),
        "--window_size", str(WINDOW_SIZE),
        "--pref_mode", PREF_MODE,
        "--pred_name", f"phase2a_{exp_id}",
        "--device", str(device),
        "--output_dir", str(exp_output_dir),
        "--seed", str(SEED),
    ]

    if head_mode:
        cmd.extend(["--head_mode", head_mode])

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {exp_id}")
    print(f"  Strategy: {strategy} (head_mode={head_mode or 'disabled'})")
    print(f"  Cache size: {cache_size}")
    print(f"  Output: {exp_output_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=7200)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {exp_id} completed in {elapsed/60:.1f} min")
            return True, elapsed
        else:
            print(f"\n✗ {exp_id} failed with code {result.returncode}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        print(f"\n✗ {exp_id} timed out")
        return False, 7200
    except Exception as e:
        print(f"\n✗ {exp_id} error: {e}")
        return False, time.time() - start_time


def compute_rouge_for_experiment(exp_id):
    """Compute ROUGE-L score for a single experiment."""
    from rouge import Rouge

    exp_dir = RAW_RESULTS_DIR / exp_id
    result_file = exp_dir / f"{DATASET}.jsonl"

    if not result_file.exists():
        return None

    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]

    rouge = Rouge()
    scores = []

    for r in results:
        pred = r['pred']
        ref = r['answers'][0] if r['answers'] else ""

        if not pred.strip() or not ref.strip():
            continue

        try:
            score = rouge.get_scores(pred, ref)[0]
            scores.append(score['rouge-l']['f'])
        except:
            continue

    return sum(scores) / len(scores) if scores else None


def run_anova_analysis():
    """Run preliminary ANOVA analysis on Phase 2A results."""
    import numpy as np

    print("\n" + "="*60)
    print("ANOVA 初步分析")
    print("="*60)

    # Collect results
    data = []
    for strategy in STRATEGIES:
        for cache_size in CACHE_SIZES:
            exp_id = get_experiment_id(strategy, cache_size, DATASET, SEED)
            rouge_l = compute_rouge_for_experiment(exp_id)

            if rouge_l is not None:
                data.append({
                    'strategy': strategy,
                    'cache_size': cache_size,
                    'rouge_l': rouge_l * 100  # Convert to percentage
                })
                print(f"  {exp_id}: ROUGE-L = {rouge_l*100:.2f}%")

    if len(data) < 9:
        print(f"\n⚠️ 只收集到 {len(data)}/9 个实验结果，无法进行完整分析")
        return None

    # Create a simple 3x3 matrix
    print("\n=== ROUGE-L 矩阵 ===")
    print(f"{'Strategy':<15} | cache=128 | cache=512 | cache=2048")
    print("-" * 55)

    for strategy in STRATEGIES:
        row = [d['rouge_l'] for d in data if d['strategy'] == strategy]
        row_str = " | ".join([f"{r:>9.2f}" for r in row])
        print(f"{strategy:<15} | {row_str}")

    # Simple interaction effect check
    print("\n=== 交互效应初步判断 ===")

    # Check if strategy rankings change across cache sizes
    rankings = {}
    for cache_size in CACHE_SIZES:
        subset = [(d['strategy'], d['rouge_l']) for d in data if d['cache_size'] == cache_size]
        subset.sort(key=lambda x: x[1], reverse=True)
        rankings[cache_size] = [s[0] for s in subset]
        print(f"cache_size={cache_size}: {' > '.join(rankings[cache_size])}")

    # Check for rank reversal
    rank_consistent = rankings[128] == rankings[512] == rankings[2048]

    if rank_consistent:
        print("\n❌ 无交互效应迹象: 策略排名在所有压缩率下保持一致")
        print("   建议: 重新考虑研究方向")
    else:
        print("\n✅ 可能存在交互效应: 策略排名随压缩率变化")
        print("   建议: 继续完整的 Phase 0 验证 (54个实验)")

    return not rank_consistent


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2A: Quick interaction effect validation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis")
    args = parser.parse_args()

    ensure_directories()

    if args.analyze_only:
        run_anova_analysis()
        return

    if args.reset and CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint reset.")

    checkpoint = load_checkpoint()
    if checkpoint["start_time"] is None:
        checkpoint["start_time"] = datetime.now().isoformat()

    experiments = list(product(STRATEGIES, CACHE_SIZES))

    print("\n" + "#"*60)
    print("Phase 2A: 快速验证交互效应存在性")
    print(f"  实验数: {len(experiments)}")
    print(f"  预估时间: 5-6小时")
    print(f"  已完成: {len(checkpoint['completed'])}")
    print("#"*60 + "\n")

    for strategy, cache_size in experiments:
        exp_id = get_experiment_id(strategy, cache_size, DATASET, SEED)

        if exp_id in checkpoint["completed"]:
            print(f"跳过 {exp_id} (已完成)")
            continue

        success, elapsed = run_single_experiment(strategy, cache_size, args.device)

        if success:
            checkpoint["completed"].append(exp_id)
        else:
            checkpoint["failed"].append({
                "exp_id": exp_id,
                "time": datetime.now().isoformat()
            })

        save_checkpoint(checkpoint)

    # Run analysis
    run_anova_analysis()

    print(f"\n{'='*60}")
    print("Phase 2A 完成")
    print(f"  成功: {len(checkpoint['completed'])}/{len(experiments)}")
    print(f"  失败: {len(checkpoint['failed'])}")
    print("="*60)


if __name__ == "__main__":
    main()
