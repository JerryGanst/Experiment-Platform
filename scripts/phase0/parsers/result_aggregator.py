"""
结果聚合器

将 54 次实验的结果聚合为 raw_results.csv
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import (
    generate_experiment_grid,
    RAW_RESULTS_DIR,
    RAW_RESULTS_CSV,
    DATASET_CONFIG,
    MODEL,
)
from .eval_parser import run_eval_and_parse, get_score_for_dataset
from runners.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def aggregate_results(
    checkpoint_path: Optional[Path] = None,
    raw_results_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    run_eval: bool = True,
) -> pd.DataFrame:
    """
    聚合所有实验结果为 CSV

    Args:
        checkpoint_path: 检查点文件路径
        raw_results_dir: 原始结果目录
        output_csv: 输出 CSV 路径
        run_eval: 是否运行评估 (如果 checkpoint 中没有分数)

    Returns:
        聚合后的 DataFrame
    """
    raw_results_dir = raw_results_dir or RAW_RESULTS_DIR
    output_csv = output_csv or RAW_RESULTS_CSV

    checkpoint = CheckpointManager(checkpoint_path)
    state = checkpoint.load()
    experiments = state.get("experiments", {})

    experiment_grid = generate_experiment_grid()

    rows = []
    for exp in experiment_grid:
        exp_id = exp["experiment_id"]
        pred_name = exp["pred_name"]
        dataset = exp["dataset"]

        row = {
            "experiment_id": exp_id,
            "strategy": exp["strategy"],
            "cache_size": exp["cache_size"],
            "dataset": dataset,
            "seed": exp["seed"],
            "pred_name": pred_name,
            "status": "pending",
            "score": None,
            "metric_type": DATASET_CONFIG.get(dataset, {}).get("metric", "unknown"),
            "duration_sec": None,
        }

        # 从 checkpoint 获取状态
        if exp_id in experiments:
            exp_state = experiments[exp_id]
            row["status"] = exp_state.get("status", "unknown")
            row["duration_sec"] = exp_state.get("duration_sec")

            # 尝试从 checkpoint 获取分数
            if "score" in exp_state and exp_state["score"] is not None:
                row["score"] = exp_state["score"]

        # 如果没有分数且需要运行评估
        if row["score"] is None and run_eval and row["status"] == "completed":
            output_path = raw_results_dir / pred_name
            if output_path.exists():
                score = get_score_for_dataset(output_path, dataset, MODEL)
                if score is not None:
                    row["score"] = score
                    # 更新 checkpoint 中的分数
                    checkpoint.mark_completed(
                        exp_id,
                        output_path=str(output_path / f"{dataset}.jsonl"),
                        score=score,
                        duration_sec=row["duration_sec"],
                    )

        rows.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(rows)

    # 保存 CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Aggregated results saved to {output_csv}")

    # 打印摘要
    _print_summary(df)

    return df


def _print_summary(df: pd.DataFrame):
    """打印结果摘要"""
    total = len(df)
    completed = len(df[df["status"] == "completed"])
    with_scores = len(df[df["score"].notna()])

    logger.info(f"\n{'='*60}")
    logger.info("Aggregation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {total}")
    logger.info(f"Completed: {completed}")
    logger.info(f"With scores: {with_scores}")

    if with_scores > 0:
        logger.info(f"\nScore statistics by strategy:")
        for strategy in df["strategy"].unique():
            strategy_df = df[(df["strategy"] == strategy) & (df["score"].notna())]
            if len(strategy_df) > 0:
                mean_score = strategy_df["score"].mean()
                std_score = strategy_df["score"].std()
                logger.info(f"  {strategy}: {mean_score:.2f} ± {std_score:.2f}")

        logger.info(f"\nScore statistics by cache_size:")
        for cache_size in sorted(df["cache_size"].unique()):
            cache_df = df[(df["cache_size"] == cache_size) & (df["score"].notna())]
            if len(cache_df) > 0:
                mean_score = cache_df["score"].mean()
                std_score = cache_df["score"].std()
                logger.info(f"  {cache_size}: {mean_score:.2f} ± {std_score:.2f}")


def load_raw_results(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    加载已聚合的结果

    Args:
        csv_path: CSV 文件路径

    Returns:
        DataFrame
    """
    csv_path = csv_path or RAW_RESULTS_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    return pd.read_csv(csv_path)


def get_completed_results(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    获取已完成且有分数的实验结果

    Args:
        df: 输入 DataFrame (可选，否则从文件加载)

    Returns:
        过滤后的 DataFrame
    """
    if df is None:
        df = load_raw_results()

    return df[(df["status"] == "completed") & (df["score"].notna())].copy()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 测试聚合
    df = aggregate_results(run_eval=False)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
