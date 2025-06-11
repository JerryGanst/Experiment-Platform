"""
评分工具模块 - 实现相对于Full KV基线的百分制评分系统
严格按照研究报告规范实现
"""
import json
import os
from pathlib import Path
import logging

# 导入LongBench评分函数
import sys
longbench_metrics_path = os.path.join(os.path.dirname(__file__), 'cakekv-main', 'cakekv-main', 'experiments', 'LongBench')
if longbench_metrics_path not in sys.path:
    sys.path.append(longbench_metrics_path)

try:
    from metrics import qa_f1_score, rouge_score
    print("[OK] LongBench评分模块加载成功")
except ImportError as e:
    print(f"[WARNING] LongBench评分模块加载失败: {e}")
    # 简单占位函数
    def qa_f1_score(pred, ref, **kwargs):
        return 0.5 if pred.strip() and ref.strip() else 0.0
    def rouge_score(pred, ref, **kwargs):
        return 0.5 if pred.strip() and ref.strip() else 0.0

logger = logging.getLogger(__name__)

# ❶ 数据集评分函数映射 (按研究报告要求)
DATASET_SCORERS = {
    "hotpotqa": qa_f1_score,     # QA任务，F1分数
    "multi_news": rouge_score,   # 摘要任务，ROUGE-L F1
}

# 基线文件路径
_BASELINE_FILE = Path(__file__).parent / "baseline_fullkv.json"

def score_dataset(dataset_name: str,
                  predictions: list,
                  references: list) -> float:
    """
    计算单个数据集的平均分 (0~1)
    按照研究报告规范实现
    
    Args:
        dataset_name: 数据集名称 (hotpotqa 或 multi_news)
        predictions: 模型预测结果列表
        references: 参考答案列表
        
    Returns:
        float: 平均分数 (0~1)
    """
    if dataset_name not in DATASET_SCORERS:
        raise ValueError(f"不支持的数据集: {dataset_name}. 仅支持: {list(DATASET_SCORERS.keys())}")
    
    scorer = DATASET_SCORERS[dataset_name]
    scores = []
    
    for pred, ref in zip(predictions, references):
        try:
            score = scorer(pred, ref)
            scores.append(score)
        except Exception as e:
            logger.warning(f"评分出错: {e}, 使用0分")
            scores.append(0.0)
    
    if not scores:
        return 0.0
    
    avg_score = sum(scores) / len(scores)
    logger.info(f"数据集 {dataset_name} 平均分: {avg_score:.4f}")
    return avg_score

def load_baseline():
    """加载基线分数"""
    if _BASELINE_FILE.exists():
        with open(_BASELINE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise FileNotFoundError("baseline_fullkv.json 不存在，请先跑 Full-KV 建立基线!")

def save_baseline(baseline_dict):
    """保存基线分数"""
    with open(_BASELINE_FILE, 'w', encoding='utf-8') as f:
        json.dump(baseline_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"基线分数已保存到 {_BASELINE_FILE}")

def calculate_relative_score(dataset_name: str, raw_score: float, is_full_kv: bool = False) -> dict:
    """
    计算相对于Full KV基线的得分
    
    Args:
        dataset_name: 数据集名称
        raw_score: 原始分数 (0~1)
        is_full_kv: 是否为Full KV模式
        
    Returns:
        dict: 包含原始分数和相对分数的字典
    """
    result = {
        "dataset": dataset_name,
        "raw_score": raw_score
    }
    
    if is_full_kv:
        # Full KV模式：更新基线并设为100分
        try:
            baseline = load_baseline()
        except FileNotFoundError:
            baseline = {}
        
        baseline[dataset_name] = raw_score
        save_baseline(baseline)
        result["relative_score"] = 100.0
        result["baseline_score"] = raw_score
        logger.info(f"Full KV基线已更新: {dataset_name} = {raw_score:.4f}")
    else:
        # 其他策略：计算相对分数
        baseline = load_baseline()
        if baseline.get(dataset_name) is None:
            raise ValueError(f"数据集 {dataset_name} 的基线分数不存在，请先运行 Full-KV!")
        
        baseline_score = baseline[dataset_name]
        if baseline_score == 0:
            # 基线为0的特殊处理
            relative_score = 100.0 if raw_score == 0 else 0.0
            logger.warning(f"数据集 {dataset_name} 基线分数为0，特殊处理")
        else:
            relative_score = 100.0 * raw_score / baseline_score
        
        result["relative_score"] = relative_score
        result["baseline_score"] = baseline_score
        logger.info(f"相对分数计算: {dataset_name} = {raw_score:.4f} / {baseline_score:.4f} * 100 = {relative_score:.2f}")
    
    return result

def aggregate_scores(results: list) -> dict:
    """
    聚合多个数据集的相对分数
    
    Args:
        results: 评分结果列表
        
    Returns:
        dict: 聚合后的分数统计
    """
    if not results:
        return {"average_relative_score": 0.0, "count": 0}
    
    relative_scores = [r["relative_score"] for r in results if "relative_score" in r]
    raw_scores = [r["raw_score"] for r in results if "raw_score" in r]
    
    aggregated = {
        "average_relative_score": sum(relative_scores) / len(relative_scores) if relative_scores else 0.0,
        "average_raw_score": sum(raw_scores) / len(raw_scores) if raw_scores else 0.0,
        "count": len(results),
        "datasets": [r["dataset"] for r in results],
        "individual_scores": results
    }
    
    logger.info(f"聚合分数: 平均相对分数 = {aggregated['average_relative_score']:.2f}")
    return aggregated

def format_score_report(aggregated_result: dict, strategy_name: str = "Unknown") -> str:
    """
    格式化分数报告
    
    Args:
        aggregated_result: 聚合的评分结果
        strategy_name: 策略名称
        
    Returns:
        str: 格式化的报告
    """
    report = f"\n{'='*50}\n"
    report += f"策略: {strategy_name}\n"
    report += f"{'='*50}\n"
    report += f"平均相对分数: {aggregated_result['average_relative_score']:.2f}/100\n"
    report += f"平均原始分数: {aggregated_result['average_raw_score']:.4f}\n"
    report += f"评测数据集数量: {aggregated_result['count']}\n"
    report += f"数据集列表: {', '.join(aggregated_result['datasets'])}\n"
    report += f"\n详细分数:\n"
    
    for result in aggregated_result['individual_scores']:
        report += f"  {result['dataset']}: {result['relative_score']:.2f}/100 "
        report += f"(原始: {result['raw_score']:.4f})\n"
    
    report += f"{'='*50}\n"
    return report 