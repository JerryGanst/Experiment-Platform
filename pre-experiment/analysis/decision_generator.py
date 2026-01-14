"""
决策生成器

根据 ANOVA 结果自动生成 Go/No-Go 决策文档
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import (
    DECISION_MD,
    SIGNIFICANCE_LEVEL,
    MIN_EFFECT_SIZE,
    TOTAL_EXPERIMENTS,
)

logger = logging.getLogger(__name__)


class DecisionGenerator:
    """
    决策生成器

    根据 ANOVA 结果判定假设验证结果
    """

    def __init__(
        self,
        anova_results: Dict[str, Any],
        significance_level: float = SIGNIFICANCE_LEVEL,
        min_effect_size: float = MIN_EFFECT_SIZE,
    ):
        """
        初始化

        Args:
            anova_results: ANOVA 分析结果 (按数据集分组)
            significance_level: 显著性水平 (默认 0.05)
            min_effect_size: 最小有意义效应量 (默认 0.02)
        """
        self.anova_results = anova_results
        self.alpha = significance_level
        self.min_effect = min_effect_size

    def evaluate(self) -> Dict[str, Any]:
        """
        评估假设验证结果

        Returns:
            {
                "decision": "PROCEED" | "RECONSIDER" | "ABANDON" | "REFINE",
                "justification": str,
                "dataset_results": {...},
                "recommendations": [str, ...],
            }
        """
        dataset_results = {}
        significant_datasets = []
        large_effect_datasets = []

        for dataset, results in self.anova_results.items():
            if "error" in results or dataset == "combined":
                continue

            interaction = results.get("interaction", {})
            p_value = interaction.get("p", 1.0)
            eta_sq = interaction.get("eta_squared", 0.0)

            is_significant = p_value < self.alpha
            has_large_effect = eta_sq >= self.min_effect

            dataset_results[dataset] = {
                "p_value": p_value,
                "eta_squared": eta_sq,
                "significant": is_significant,
                "practical_significance": has_large_effect,
            }

            if is_significant:
                significant_datasets.append(dataset)
            if has_large_effect:
                large_effect_datasets.append(dataset)

        # 决策逻辑
        total_datasets = len(dataset_results)
        sig_count = len(significant_datasets)
        effect_count = len(large_effect_datasets)

        if total_datasets == 0:
            decision = "ERROR"
            justification = "No valid ANOVA results to evaluate."
            recommendations = ["Re-run experiments and analysis."]

        elif sig_count == total_datasets and effect_count == total_datasets:
            # 所有数据集都显著且效应量大
            decision = "PROCEED"
            justification = (
                f"Interaction effect is statistically significant (p < {self.alpha}) "
                f"AND practically significant (η² > {self.min_effect}) for all {total_datasets} datasets. "
                "Core hypothesis validated."
            )
            recommendations = [
                "Proceed to Phase 1-4 full research plan.",
                "Focus on characterizing the interaction pattern.",
                "Develop adaptive strategy selection mechanism.",
            ]

        elif sig_count == total_datasets and effect_count < total_datasets:
            # 所有数据集都显著但效应量不够大
            decision = "RECONSIDER"
            justification = (
                f"Interaction effect is statistically significant (p < {self.alpha}) for all datasets, "
                f"BUT effect size is small (η² < {self.min_effect}) for some datasets. "
                f"Significant datasets: {significant_datasets}. "
                f"Large effect datasets: {large_effect_datasets}."
            )
            recommendations = [
                "Effect exists but may be too small for practical impact.",
                "Consider: Is this worth a paper?",
                "Option 1: Pivot to other research questions.",
                "Option 2: Focus on datasets with larger effects.",
                "Option 3: Investigate whether effect increases at more extreme compression rates.",
            ]

        elif sig_count > 0 and sig_count < total_datasets:
            # 部分数据集显著
            decision = "REFINE"
            justification = (
                f"Mixed results: Interaction is significant for {sig_count}/{total_datasets} datasets. "
                f"Significant datasets: {significant_datasets}. "
                f"Non-significant datasets: {[d for d in dataset_results if d not in significant_datasets]}."
            )
            recommendations = [
                "Narrow research scope to task types where effect exists.",
                f"Focus on datasets: {significant_datasets}.",
                "Investigate why effect differs across task types.",
                "Consider task-specific adaptive strategies.",
            ]

        else:
            # 没有数据集显著
            decision = "ABANDON"
            justification = (
                f"Interaction effect is NOT statistically significant (p >= {self.alpha}) "
                f"for any dataset. Core hypothesis NOT supported."
            )
            recommendations = [
                "Abandon current research direction.",
                "Document negative result.",
                "Pivot to alternative hypotheses:",
                "  - Layer-level adaptation instead of head-level",
                "  - Task-specific strategies",
                "  - Different entropy metrics",
            ]

        return {
            "decision": decision,
            "justification": justification,
            "dataset_results": dataset_results,
            "recommendations": recommendations,
            "criteria": {
                "significance_level": self.alpha,
                "min_effect_size": self.min_effect,
            },
        }


def generate_decision(
    anova_results: Dict[str, Any],
    output_path: Optional[Path] = None,
    checkpoint_summary: Optional[Dict] = None,
) -> str:
    """
    生成决策文档

    Args:
        anova_results: ANOVA 分析结果
        output_path: 输出 Markdown 路径
        checkpoint_summary: 检查点摘要 (可选)

    Returns:
        决策文档内容 (Markdown)
    """
    output_path = output_path or DECISION_MD

    generator = DecisionGenerator(anova_results)
    evaluation = generator.evaluate()

    # 生成 Markdown
    md_content = _generate_markdown(evaluation, anova_results, checkpoint_summary)

    # 保存文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Decision document saved to {output_path}")

    return md_content


def _generate_markdown(
    evaluation: Dict[str, Any],
    anova_results: Dict[str, Any],
    checkpoint_summary: Optional[Dict] = None,
) -> str:
    """生成 Markdown 格式的决策文档"""

    decision = evaluation["decision"]
    justification = evaluation["justification"]
    dataset_results = evaluation["dataset_results"]
    recommendations = evaluation["recommendations"]
    criteria = evaluation["criteria"]

    # 决策图标
    decision_icons = {
        "PROCEED": "PROCEED",
        "RECONSIDER": "RECONSIDER",
        "ABANDON": "ABANDON",
        "REFINE": "REFINE",
        "ERROR": "ERROR",
    }

    md = []

    # 标题
    md.append("# Phase 0 Hypothesis Validation Decision\n")

    # 元信息
    md.append("## Experiment Summary\n")
    md.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"- **Total Experiments Planned**: {TOTAL_EXPERIMENTS}")

    if checkpoint_summary:
        md.append(f"- **Completed**: {checkpoint_summary.get('completed', 'N/A')}")
        md.append(f"- **Failed**: {checkpoint_summary.get('failed', 'N/A')}")

    md.append("")

    # 假设回顾
    md.append("## Hypothesis Under Test\n")
    md.append("> **H1 (Alternative)**: The optimal KV cache eviction strategy varies with compression rate.")
    md.append("> Some strategies become relatively better/worse as compression changes.\n")
    md.append("> **H0 (Null)**: Optimal strategy is independent of compression rate.\n")

    # 判定标准
    md.append("## Judgment Criteria\n")
    md.append(f"- **Statistical Significance**: p < {criteria['significance_level']}")
    md.append(f"- **Practical Significance**: eta-squared > {criteria['min_effect_size']} ({criteria['min_effect_size']*100}%)\n")

    # ANOVA 结果
    md.append("## Statistical Results\n")

    for dataset, results in anova_results.items():
        if dataset == "combined" or "error" in results:
            continue

        md.append(f"### {dataset.upper()} Dataset\n")

        if "error" in results:
            md.append(f"**Error**: {results['error']}\n")
            continue

        # 主效应表格
        md.append("#### Main Effects\n")
        md.append("| Factor | F | p-value | Significant | eta-squared | Interpretation |")
        md.append("|--------|---|---------|-------------|-------------|----------------|")

        for factor, stats in results.get("main_effects", {}).items():
            sig = "Yes" if stats.get("significant") else "No"
            md.append(
                f"| {factor} | {stats.get('F', 0):.3f} | {stats.get('p', 1):.4f} | {sig} | "
                f"{stats.get('eta_squared', 0):.4f} | {stats.get('effect_interpretation', 'N/A')} |"
            )

        # 交互效应
        interaction = results.get("interaction", {})
        md.append("\n#### Interaction Effect\n")
        md.append("| Effect | F | p-value | Significant | eta-squared | Interpretation |")
        md.append("|--------|---|---------|-------------|-------------|----------------|")

        sig = "Yes" if interaction.get("significant") else "No"
        md.append(
            f"| {interaction.get('name', 'Interaction')} | {interaction.get('F', 0):.3f} | "
            f"{interaction.get('p', 1):.4f} | **{sig}** | "
            f"{interaction.get('eta_squared', 0):.4f} | {interaction.get('effect_interpretation', 'N/A')} |"
        )

        md.append("")

    # 决策
    md.append("---\n")
    md.append("## DECISION\n")
    md.append(f"### **{decision_icons.get(decision, decision)}**\n")
    md.append(f"### Justification\n")
    md.append(f"{justification}\n")

    # 各数据集结果摘要
    md.append("### Dataset-Level Results\n")
    md.append("| Dataset | p-value | eta-squared | Significant | Practical |")
    md.append("|---------|---------|-------------|-------------|-----------|")

    for dataset, dr in dataset_results.items():
        sig = "Yes" if dr.get("significant") else "No"
        prac = "Yes" if dr.get("practical_significance") else "No"
        md.append(
            f"| {dataset} | {dr.get('p_value', 1):.4f} | {dr.get('eta_squared', 0):.4f} | {sig} | {prac} |"
        )

    md.append("")

    # 建议
    md.append("## Recommendations\n")
    for rec in recommendations:
        md.append(f"- {rec}")

    md.append("")

    # 下一步
    md.append("---\n")
    md.append("## Next Steps\n")

    if decision == "PROCEED":
        md.append("1. Phase 1: Collect comprehensive interaction data across 10 datasets")
        md.append("2. Phase 2: Characterize interaction patterns")
        md.append("3. Phase 3: Develop adaptive strategy")
        md.append("4. Phase 4: Write paper")
    elif decision == "RECONSIDER":
        md.append("1. Discuss with advisor whether effect size is sufficient")
        md.append("2. Consider testing at more extreme compression rates")
        md.append("3. Evaluate alternative research directions")
    elif decision == "REFINE":
        md.append("1. Focus research on task types showing significant effects")
        md.append("2. Investigate why effect varies across tasks")
        md.append("3. Consider task-specific contributions")
    elif decision == "ABANDON":
        md.append("1. Document this negative result")
        md.append("2. Pivot to alternative hypotheses")
        md.append("3. Consider other aspects of KV cache optimization")

    md.append("\n---\n")
    md.append(f"*Generated automatically by Phase 0 Validation Framework*")
    md.append(f"*Timestamp: {datetime.now().isoformat()}*")

    return "\n".join(md)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试用模拟结果
    test_anova_results = {
        "qmsum": {
            "grand_mean": 25.0,
            "sample_size": 27,
            "design": "3 x 3",
            "main_effects": {
                "strategy": {
                    "F": 2.5,
                    "p": 0.12,
                    "significant": False,
                    "eta_squared": 0.08,
                    "effect_interpretation": "medium",
                },
                "cache_size": {
                    "F": 15.0,
                    "p": 0.001,
                    "significant": True,
                    "eta_squared": 0.35,
                    "effect_interpretation": "large",
                },
            },
            "interaction": {
                "name": "strategy x cache_size",
                "F": 4.5,
                "p": 0.015,
                "significant": True,
                "eta_squared": 0.12,
                "effect_interpretation": "medium",
            },
        },
        "hotpotqa": {
            "grand_mean": 45.0,
            "sample_size": 27,
            "design": "3 x 3",
            "main_effects": {
                "strategy": {
                    "F": 1.2,
                    "p": 0.32,
                    "significant": False,
                    "eta_squared": 0.04,
                    "effect_interpretation": "small",
                },
                "cache_size": {
                    "F": 8.0,
                    "p": 0.008,
                    "significant": True,
                    "eta_squared": 0.25,
                    "effect_interpretation": "large",
                },
            },
            "interaction": {
                "name": "strategy x cache_size",
                "F": 3.2,
                "p": 0.04,
                "significant": True,
                "eta_squared": 0.09,
                "effect_interpretation": "medium",
            },
        },
    }

    md = generate_decision(test_anova_results)
    print(md)
