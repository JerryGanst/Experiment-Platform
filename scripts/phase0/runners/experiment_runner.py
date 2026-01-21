"""
单次实验执行器

封装 scripts/experiments/run_hace.py 的调用
"""
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import (
    HACE_RUNNER_SCRIPT,
    RAW_RESULTS_DIR,
    MODEL,
    WINDOW_SIZE,
    PREF_MODE,
    get_head_mode,
    get_pred_name,
)

logger = logging.getLogger(__name__)


def run_single_experiment(
    strategy: str,
    cache_size: int,
    dataset: str,
    seed: int,
    device: int = 0,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    执行单次 HACE 实验

    Args:
        strategy: 策略名 (high_entropy, low_entropy, uniform)
        cache_size: KV cache 大小 (tokens)
        dataset: 数据集名
        seed: 随机种子
        device: CUDA 设备号
        output_dir: 输出目录 (默认为 RAW_RESULTS_DIR)
        dry_run: 仅打印命令，不执行

    Returns:
        {
            "pred_name": str,
            "status": "success" | "failed" | "dry_run",
            "output_path": str,
            "duration_sec": float,
            "error": Optional[str],
            "command": str,
        }
    """
    pred_name = get_pred_name(strategy, cache_size, dataset, seed)
    head_mode = get_head_mode(strategy)
    output_dir = output_dir or RAW_RESULTS_DIR

    # 构建命令
    cmd = [
        "python",
        str(HACE_RUNNER_SCRIPT),
        "--model", MODEL,
        "--dataset", dataset,
        "--cache_size", str(cache_size),
        "--window_size", str(WINDOW_SIZE),
        "--pref_mode", PREF_MODE,
        "--pred_name", pred_name,
        "--device", str(device),
        "--output_dir", str(output_dir),
    ]

    # head_mode 参数 (uniform 时为空字符串，不传递)
    if head_mode:
        cmd.extend(["--head_mode", head_mode])

    # seed 参数 - 控制 sampling 随机性
    cmd.extend(["--seed", str(seed)])

    command_str = " ".join(cmd)
    logger.info(f"Command: {command_str}")

    result = {
        "pred_name": pred_name,
        "output_path": str(output_dir / pred_name / f"{dataset}.jsonl"),
        "command": command_str,
    }

    if dry_run:
        result["status"] = "dry_run"
        result["duration_sec"] = 0
        logger.info(f"[DRY RUN] Would execute: {command_str}")
        return result

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["HACE_PREF_MODE"] = PREF_MODE
    env["HACE_HEAD_MODE"] = head_mode

    # 执行
    start_time = time.time()

    try:
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
        )

        duration = time.time() - start_time
        result["duration_sec"] = duration

        if process.returncode == 0:
            result["status"] = "success"
            logger.info(f"Experiment completed in {duration:.1f}s: {pred_name}")
        else:
            result["status"] = "failed"
            result["error"] = process.stderr[:1000] if process.stderr else "Unknown error"
            logger.error(f"Experiment failed: {pred_name}\nError: {result['error']}")

    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error"] = "Timeout (1 hour)"
        result["duration_sec"] = time.time() - start_time
        logger.error(f"Experiment timeout: {pred_name}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["duration_sec"] = time.time() - start_time
        logger.error(f"Experiment error: {pred_name}\nException: {e}")

    return result


def check_experiment_output(pred_name: str, dataset: str, output_dir: Optional[Path] = None) -> bool:
    """
    检查实验输出是否存在且有效

    Args:
        pred_name: 预测目录名
        dataset: 数据集名
        output_dir: 输出目录

    Returns:
        True if output exists and is non-empty
    """
    output_dir = output_dir or RAW_RESULTS_DIR
    output_path = output_dir / pred_name / f"{dataset}.jsonl"

    if not output_path.exists():
        return False

    # 检查文件非空
    try:
        with open(output_path, "r") as f:
            first_line = f.readline()
            return bool(first_line.strip())
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试干运行
    result = run_single_experiment(
        strategy="high_entropy",
        cache_size=128,
        dataset="qmsum",
        seed=42,
        device=0,
        dry_run=True,
    )
    print(f"Dry run result: {result}")
