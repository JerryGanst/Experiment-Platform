"""
评估结果解析器

解析 eval.py 的输出，提取 ROUGE/F1 分数
"""
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import EVAL_SCRIPT, MODEL, DATASET_CONFIG

logger = logging.getLogger(__name__)


def run_eval_and_parse(
    dir_path: Path,
    model: str = MODEL,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    """
    运行 eval.py 并解析输出

    Args:
        dir_path: 预测结果目录 (包含 {dataset}.jsonl)
        model: 模型名
        dataset: 数据集名 (可选，用于验证)

    Returns:
        {dataset_name: score, ...}
    """
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--model", model,
        "--dir_path", str(dir_path),
    ]

    logger.info(f"Running eval: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Eval failed: {result.stderr}")
            return {}

        # 解析输出
        scores = parse_eval_output(result.stdout)

        # 也检查是否生成了 result.json
        result_json = dir_path / "result.json"
        if result_json.exists():
            with open(result_json, "r") as f:
                json_scores = json.load(f)
                # 合并结果
                scores.update(json_scores)

        return scores

    except subprocess.TimeoutExpired:
        logger.error("Eval timeout")
        return {}
    except Exception as e:
        logger.error(f"Eval error: {e}")
        return {}


def parse_eval_output(stdout: str) -> Dict[str, float]:
    """
    解析 eval.py 的标准输出

    eval.py 输出格式可能是:
    1. JSON 格式: {"qmsum": 24.52, "hotpotqa": 45.67}
    2. 打印格式: qmsum: 24.52

    Args:
        stdout: eval.py 的标准输出

    Returns:
        {dataset_name: score}
    """
    scores = {}

    # 尝试解析 JSON
    try:
        # 查找 JSON 对象
        json_match = re.search(r'\{[^{}]+\}', stdout)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
    except json.JSONDecodeError:
        pass

    # 尝试解析 key: value 格式
    # 例如: "qmsum: 24.52" 或 "hotpotqa score: 45.67"
    patterns = [
        r'(\w+):\s*([\d.]+)',  # dataset: score
        r'(\w+)\s+score:\s*([\d.]+)',  # dataset score: score
        r'ROUGE-L:\s*([\d.]+)',  # ROUGE-L: score (无数据集名)
        r'F1:\s*([\d.]+)',  # F1: score
    ]

    for pattern in patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                key, value = match
                try:
                    scores[key.lower()] = float(value)
                except ValueError:
                    continue

    return scores


def get_score_for_dataset(
    dir_path: Path,
    dataset: str,
    model: str = MODEL,
) -> Optional[float]:
    """
    获取特定数据集的分数

    Args:
        dir_path: 预测结果目录
        dataset: 数据集名
        model: 模型名

    Returns:
        分数 (ROUGE-L 或 F1)，失败返回 None
    """
    scores = run_eval_and_parse(dir_path, model, dataset)

    if not scores:
        return None

    # 根据数据集类型获取对应指标
    config = DATASET_CONFIG.get(dataset, {})
    metric_key = config.get("metric_key", dataset)

    # 尝试多种可能的 key
    possible_keys = [
        metric_key,
        dataset,
        "rouge-l",
        "rougel",
        "rouge_l",
        "f1",
        "score",
    ]

    for key in possible_keys:
        if key in scores:
            return scores[key]
        if key.lower() in scores:
            return scores[key.lower()]

    # 如果只有一个分数，返回它
    if len(scores) == 1:
        return list(scores.values())[0]

    logger.warning(f"Could not find score for {dataset} in {scores}")
    return None


def parse_result_json(result_json_path: Path) -> Dict[str, float]:
    """
    解析 result.json 文件

    Args:
        result_json_path: result.json 文件路径

    Returns:
        {dataset_name: score}
    """
    if not result_json_path.exists():
        return {}

    try:
        with open(result_json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse {result_json_path}: {e}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试解析
    test_outputs = [
        '{"qmsum": 24.52}',
        'qmsum: 24.52\nhotpotqa: 45.67',
        'ROUGE-L: 0.2452',
        'Processing... done.\n{"qmsum": 24.52, "hotpotqa": 45.67}\n',
    ]

    for output in test_outputs:
        print(f"Input: {output!r}")
        print(f"Parsed: {parse_eval_output(output)}")
        print()
