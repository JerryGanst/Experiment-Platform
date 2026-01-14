"""
双因素方差分析 (Two-Way ANOVA)

不依赖 statsmodels，手动实现 ANOVA 计算
检验策略 × 压缩率的交互效应
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from .effect_size import (
    calculate_eta_squared,
    calculate_partial_eta_squared,
    calculate_omega_squared,
    interpret_effect_size,
)
from config.phase0_config import ANOVA_RESULTS_JSON, SIGNIFICANCE_LEVEL

logger = logging.getLogger(__name__)


class TwoWayANOVA:
    """
    双因素方差分析

    检验两个因子的主效应和交互效应
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dependent_var: str = "score",
        factor1: str = "strategy",
        factor2: str = "cache_size",
    ):
        """
        初始化

        Args:
            df: 数据 DataFrame
            dependent_var: 因变量列名
            factor1: 因子1列名 (策略)
            factor2: 因子2列名 (压缩率)
        """
        self.df = df.dropna(subset=[dependent_var]).copy()
        self.dv = dependent_var
        self.iv1 = factor1
        self.iv2 = factor2

        # 验证数据
        self._validate_data()

    def _validate_data(self):
        """验证数据完整性"""
        required_cols = [self.dv, self.iv1, self.iv2]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")

        # 检查各单元格是否有数据
        cell_counts = self.df.groupby([self.iv1, self.iv2]).size()
        if cell_counts.min() < 2:
            logger.warning(f"Some cells have < 2 observations: {cell_counts.min()}")

    def calculate(self) -> Dict[str, Any]:
        """
        执行双因素 ANOVA

        Returns:
            {
                "grand_mean": float,
                "main_effects": {
                    "factor1": {"SS": ..., "df": ..., "MS": ..., "F": ..., "p": ..., "eta_sq": ...},
                    "factor2": {...},
                },
                "interaction": {"SS": ..., "df": ..., "MS": ..., "F": ..., "p": ..., "eta_sq": ...},
                "error": {"SS": ..., "df": ..., "MS": ...},
                "total": {"SS": ..., "df": ...},
            }
        """
        # 获取因子水平
        levels1 = self.df[self.iv1].unique()
        levels2 = self.df[self.iv2].unique()

        a = len(levels1)  # 因子1水平数
        b = len(levels2)  # 因子2水平数
        N = len(self.df)  # 总样本数

        # 计算各种均值
        grand_mean = self.df[self.dv].mean()

        # 因子1边际均值
        means_iv1 = self.df.groupby(self.iv1)[self.dv].mean()

        # 因子2边际均值
        means_iv2 = self.df.groupby(self.iv2)[self.dv].mean()

        # 单元格均值
        cell_means = self.df.groupby([self.iv1, self.iv2])[self.dv].mean()
        cell_counts = self.df.groupby([self.iv1, self.iv2])[self.dv].count()

        # 每单元格的样本数 (假设均衡设计)
        n = N // (a * b)

        # ============ 计算平方和 ============

        # SS_total: 总平方和
        ss_total = np.sum((self.df[self.dv] - grand_mean) ** 2)

        # SS_A (因子1): n * b * Σ(mean_i - grand_mean)²
        ss_iv1 = n * b * np.sum((means_iv1 - grand_mean) ** 2)

        # SS_B (因子2): n * a * Σ(mean_j - grand_mean)²
        ss_iv2 = n * a * np.sum((means_iv2 - grand_mean) ** 2)

        # SS_AB (交互): n * Σ(cell_mean_ij - mean_i - mean_j + grand_mean)²
        ss_interaction = 0
        for level1 in levels1:
            for level2 in levels2:
                cell_mean = cell_means.get((level1, level2), grand_mean)
                mean_i = means_iv1.get(level1, grand_mean)
                mean_j = means_iv2.get(level2, grand_mean)
                cell_n = cell_counts.get((level1, level2), n)
                ss_interaction += cell_n * (cell_mean - mean_i - mean_j + grand_mean) ** 2

        # SS_error: SS_total - SS_A - SS_B - SS_AB
        ss_error = ss_total - ss_iv1 - ss_iv2 - ss_interaction
        ss_error = max(0, ss_error)  # 避免数值误差导致负值

        # ============ 计算自由度 ============
        df_iv1 = a - 1
        df_iv2 = b - 1
        df_interaction = (a - 1) * (b - 1)
        df_error = N - a * b
        df_total = N - 1

        # ============ 计算均方 ============
        ms_iv1 = ss_iv1 / df_iv1 if df_iv1 > 0 else 0
        ms_iv2 = ss_iv2 / df_iv2 if df_iv2 > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0

        # ============ 计算 F 值和 p 值 ============
        def calc_f_and_p(ms_effect, ms_error, df_effect, df_error):
            if ms_error == 0:
                return np.inf, 0.0
            f_value = ms_effect / ms_error
            p_value = 1 - stats.f.cdf(f_value, df_effect, df_error)
            return f_value, p_value

        f_iv1, p_iv1 = calc_f_and_p(ms_iv1, ms_error, df_iv1, df_error)
        f_iv2, p_iv2 = calc_f_and_p(ms_iv2, ms_error, df_iv2, df_error)
        f_interaction, p_interaction = calc_f_and_p(ms_interaction, ms_error, df_interaction, df_error)

        # ============ 计算效应量 ============
        eta_sq_iv1 = calculate_eta_squared(ss_iv1, ss_total)
        eta_sq_iv2 = calculate_eta_squared(ss_iv2, ss_total)
        eta_sq_interaction = calculate_eta_squared(ss_interaction, ss_total)

        partial_eta_sq_iv1 = calculate_partial_eta_squared(ss_iv1, ss_error)
        partial_eta_sq_iv2 = calculate_partial_eta_squared(ss_iv2, ss_error)
        partial_eta_sq_interaction = calculate_partial_eta_squared(ss_interaction, ss_error)

        # ============ 构建结果 ============
        results = {
            "grand_mean": grand_mean,
            "sample_size": N,
            "design": f"{a} x {b}",
            "main_effects": {
                self.iv1: {
                    "SS": ss_iv1,
                    "df": df_iv1,
                    "MS": ms_iv1,
                    "F": f_iv1,
                    "p": p_iv1,
                    "significant": p_iv1 < SIGNIFICANCE_LEVEL,
                    "eta_squared": eta_sq_iv1,
                    "partial_eta_squared": partial_eta_sq_iv1,
                    "effect_interpretation": interpret_effect_size(eta_sq_iv1),
                },
                self.iv2: {
                    "SS": ss_iv2,
                    "df": df_iv2,
                    "MS": ms_iv2,
                    "F": f_iv2,
                    "p": p_iv2,
                    "significant": p_iv2 < SIGNIFICANCE_LEVEL,
                    "eta_squared": eta_sq_iv2,
                    "partial_eta_squared": partial_eta_sq_iv2,
                    "effect_interpretation": interpret_effect_size(eta_sq_iv2),
                },
            },
            "interaction": {
                "name": f"{self.iv1} x {self.iv2}",
                "SS": ss_interaction,
                "df": df_interaction,
                "MS": ms_interaction,
                "F": f_interaction,
                "p": p_interaction,
                "significant": p_interaction < SIGNIFICANCE_LEVEL,
                "eta_squared": eta_sq_interaction,
                "partial_eta_squared": partial_eta_sq_interaction,
                "effect_interpretation": interpret_effect_size(eta_sq_interaction),
            },
            "error": {
                "SS": ss_error,
                "df": df_error,
                "MS": ms_error,
            },
            "total": {
                "SS": ss_total,
                "df": df_total,
            },
            "cell_means": cell_means.to_dict(),
            "marginal_means": {
                self.iv1: means_iv1.to_dict(),
                self.iv2: means_iv2.to_dict(),
            },
        }

        return results


def perform_two_way_anova(
    df: pd.DataFrame,
    dependent_var: str = "score",
    factor1: str = "strategy",
    factor2: str = "cache_size",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    便捷函数：执行双因素 ANOVA

    Args:
        df: 数据 DataFrame
        dependent_var: 因变量列名
        factor1: 因子1列名
        factor2: 因子2列名
        output_path: 输出 JSON 路径

    Returns:
        ANOVA 结果字典
    """
    anova = TwoWayANOVA(df, dependent_var, factor1, factor2)
    results = anova.calculate()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换 numpy 类型为 Python 原生类型
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_types(results), f, indent=2, ensure_ascii=False)

        logger.info(f"ANOVA results saved to {output_path}")

    return results


def run_anova_by_dataset(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    按数据集分别运行 ANOVA

    Args:
        df: 完整的结果 DataFrame
        output_path: 输出 JSON 路径

    Returns:
        {
            "qmsum": {ANOVA results},
            "hotpotqa": {ANOVA results},
            "combined": {ANOVA results for all data},
        }
    """
    output_path = output_path or ANOVA_RESULTS_JSON
    results = {}

    datasets = df["dataset"].unique()

    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ANOVA for dataset: {dataset}")
        logger.info(f"{'='*60}")

        dataset_df = df[df["dataset"] == dataset].copy()

        if len(dataset_df) < 10:
            logger.warning(f"Skipping {dataset}: insufficient data ({len(dataset_df)} samples)")
            continue

        try:
            results[dataset] = perform_two_way_anova(dataset_df)
            _print_anova_summary(results[dataset], dataset)
        except Exception as e:
            logger.error(f"ANOVA failed for {dataset}: {e}")
            results[dataset] = {"error": str(e)}

    # 合并数据的 ANOVA (可选)
    if len(df) >= 20:
        logger.info(f"\n{'='*60}")
        logger.info("Running ANOVA for combined data")
        logger.info(f"{'='*60}")
        try:
            results["combined"] = perform_two_way_anova(df)
            _print_anova_summary(results["combined"], "combined")
        except Exception as e:
            logger.error(f"Combined ANOVA failed: {e}")

    # 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_types(results), f, indent=2, ensure_ascii=False)

        logger.info(f"\nAll ANOVA results saved to {output_path}")

    return results


def _print_anova_summary(results: Dict, dataset_name: str):
    """打印 ANOVA 摘要"""
    if "error" in results:
        logger.error(f"ANOVA error: {results['error']}")
        return

    logger.info(f"\nANOVA Summary for {dataset_name}:")
    logger.info(f"  Grand Mean: {results['grand_mean']:.4f}")
    logger.info(f"  Sample Size: {results['sample_size']}")
    logger.info(f"  Design: {results['design']}")

    logger.info("\nMain Effects:")
    for factor, stats in results["main_effects"].items():
        sig_marker = "*" if stats["significant"] else ""
        logger.info(
            f"  {factor}: F({stats['df']}, {results['error']['df']}) = {stats['F']:.3f}, "
            f"p = {stats['p']:.4f}{sig_marker}, η² = {stats['eta_squared']:.4f} ({stats['effect_interpretation']})"
        )

    interaction = results["interaction"]
    sig_marker = "*" if interaction["significant"] else ""
    logger.info(f"\nInteraction ({interaction['name']}):")
    logger.info(
        f"  F({interaction['df']}, {results['error']['df']}) = {interaction['F']:.3f}, "
        f"p = {interaction['p']:.4f}{sig_marker}, η² = {interaction['eta_squared']:.4f} ({interaction['effect_interpretation']})"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 创建测试数据
    np.random.seed(42)

    strategies = ["high_entropy", "low_entropy", "uniform"]
    cache_sizes = [128, 512, 2048]

    data = []
    for strategy in strategies:
        for cache_size in cache_sizes:
            for seed in [42, 123, 456]:
                # 模拟：不同策略在不同压缩率下有不同表现 (交互效应)
                base_score = 25

                if strategy == "high_entropy":
                    # 高压缩时好，低压缩时差
                    score = base_score + (2048 - cache_size) / 100
                elif strategy == "low_entropy":
                    # 低压缩时好，高压缩时差
                    score = base_score + (cache_size - 128) / 100
                else:
                    score = base_score

                score += np.random.normal(0, 1)  # 添加噪声

                data.append({
                    "strategy": strategy,
                    "cache_size": cache_size,
                    "seed": seed,
                    "score": score,
                    "dataset": "qmsum",
                })

    df = pd.DataFrame(data)
    print(f"Test data shape: {df.shape}")

    # 运行 ANOVA
    results = perform_two_way_anova(df)
