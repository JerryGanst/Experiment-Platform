#!/usr/bin/env python3
"""
Phase 0 Hypothesis Validation Experiment Runner

Runs all experiments in the Stage B/C matrix:
- 3 strategies × 3 cache_sizes × 3 seeds = 27 experiments per dataset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root and pre-experiment to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.phase0_config import (
    STRATEGIES, CACHE_SIZES, DATASETS, SEEDS,
    MODEL, WINDOW_SIZE, PREF_MODE,
    CHECKPOINTS_DIR, RAW_RESULTS_DIR, CHECKPOINT_FILE,
    get_experiment_id, get_pred_name, get_head_mode,
    ensure_directories, PROJECT_ROOT
)


def load_checkpoint():
    """Load experiment progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "start_time": None}


def save_checkpoint(checkpoint):
    """Save experiment progress checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_single_experiment(strategy, cache_size, dataset, seed, device=0):
    """Run a single experiment configuration."""
    exp_id = get_experiment_id(strategy, cache_size, dataset, seed)
    pred_name = get_pred_name(strategy, cache_size, dataset, seed)
    head_mode = get_head_mode(strategy)

    # Each experiment gets its own output directory to prevent file conflicts
    exp_output_dir = RAW_RESULTS_DIR / exp_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = exp_output_dir / f"{dataset}.jsonl"

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "experiments" / "run_hace.py"),
        "--model", MODEL,
        "--dataset", dataset,
        "--cache_size", str(cache_size),
        "--window_size", str(WINDOW_SIZE),
        "--pref_mode", PREF_MODE,
        "--pred_name", pred_name,
        "--device", str(device),
        "--output_dir", str(exp_output_dir),  # Use experiment-specific output dir
        "--seed", str(seed),
    ]

    # Add head_mode if not uniform
    if head_mode:
        cmd.extend(["--head_mode", head_mode])

    print(f"\n{'='*60}")
    print(f"Running: {exp_id}")
    print(f"  Strategy: {strategy} (head_mode={head_mode or 'disabled'})")
    print(f"  Cache size: {cache_size}")
    print(f"  Dataset: {dataset}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=3600  # 1 hour timeout per experiment
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {exp_id} completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n✗ {exp_id} failed with return code {result.returncode}")
            return False, elapsed

    except subprocess.TimeoutExpired:
        print(f"\n✗ {exp_id} timed out after 1 hour")
        return False, 3600
    except Exception as e:
        print(f"\n✗ {exp_id} failed with error: {e}")
        return False, time.time() - start_time


def run_stage(datasets_to_run, device=0):
    """Run all experiments for specified datasets."""
    ensure_directories()
    checkpoint = load_checkpoint()

    if checkpoint["start_time"] is None:
        checkpoint["start_time"] = datetime.now().isoformat()

    total_experiments = len(STRATEGIES) * len(CACHE_SIZES) * len(datasets_to_run) * len(SEEDS)
    completed_count = len(checkpoint["completed"])

    print(f"\n{'#'*60}")
    print(f"Phase 0 Experiment Runner")
    print(f"  Strategies: {STRATEGIES}")
    print(f"  Cache sizes: {CACHE_SIZES}")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {completed_count}")
    print(f"  Remaining: {total_experiments - completed_count}")
    print(f"{'#'*60}\n")

    for strategy in STRATEGIES:
        for cache_size in CACHE_SIZES:
            for dataset in datasets_to_run:
                for seed in SEEDS:
                    exp_id = get_experiment_id(strategy, cache_size, dataset, seed)

                    # Skip if already completed
                    if exp_id in checkpoint["completed"]:
                        print(f"Skipping {exp_id} (already completed)")
                        continue

                    # Run experiment
                    success, elapsed = run_single_experiment(
                        strategy, cache_size, dataset, seed, device
                    )

                    # Update checkpoint
                    if success:
                        checkpoint["completed"].append(exp_id)
                    else:
                        checkpoint["failed"].append({
                            "exp_id": exp_id,
                            "time": datetime.now().isoformat(),
                            "elapsed": elapsed
                        })

                    save_checkpoint(checkpoint)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT RUN COMPLETE")
    print(f"  Completed: {len(checkpoint['completed'])}/{total_experiments}")
    print(f"  Failed: {len(checkpoint['failed'])}")
    print(f"{'='*60}\n")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 experiments")
    parser.add_argument("--stage", choices=["B", "C", "all"], default="B",
                        help="Stage to run: B (qmsum only), C (hotpotqa), all (both)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint reset.")

    if args.stage == "B":
        datasets = ["qmsum"]
    elif args.stage == "C":
        datasets = ["hotpotqa"]
    else:
        datasets = ["qmsum", "hotpotqa"]

    run_stage(datasets, args.device)


if __name__ == "__main__":
    main()
