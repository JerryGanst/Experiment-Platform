"""
交互效应可视化

绘制 Strategy × CacheSize 交互图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import INTERACTION_PLOT_PNG, STRATEGIES, CACHE_SIZES

logger = logging.getLogger(__name__)

# 颜色配置
STRATEGY_COLORS = {
    "high_entropy": "#1f77b4",  # 蓝色
    "low_entropy": "#ff7f0e",   # 橙色
    "uniform": "#2ca02c",       # 绿色
}

STRATEGY_MARKERS = {
    "high_entropy": "o",
    "low_entropy": "s",
    "uniform": "^",
}

STRATEGY_LABELS = {
    "high_entropy": "High Entropy",
    "low_entropy": "Low Entropy",
    "uniform": "Uniform",
}


class InteractionPlotter:
    """
    交互效应绘图器
    """

    def __init__(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        初始化

        Args:
            df: 结果数据 DataFrame
            figsize: 图像尺寸
        """
        self.df = df
        self.figsize = figsize

    def plot(
        self,
        output_path: Optional[Path] = None,
        show_error_bars: bool = True,
        show_significance: bool = True,
        anova_results: Optional[Dict] = None,
    ) -> plt.Figure:
        """
        绘制交互图

        Args:
            output_path: 输出路径
            show_error_bars: 是否显示误差条
            show_significance: 是否显示显著性标注
            anova_results: ANOVA 结果 (用于显著性标注)

        Returns:
            matplotlib Figure 对象
        """
        output_path = output_path or INTERACTION_PLOT_PNG

        datasets = self.df["dataset"].unique()
        n_datasets = len(datasets)

        fig, axes = plt.subplots(1, n_datasets, figsize=self.figsize)

        if n_datasets == 1:
            axes = [axes]

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            self._plot_single_dataset(
                ax,
                dataset,
                show_error_bars,
                show_significance,
                anova_results,
            )

        plt.tight_layout()

        # 保存图像
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Interaction plot saved to {output_path}")

        return fig

    def _plot_single_dataset(
        self,
        ax: plt.Axes,
        dataset: str,
        show_error_bars: bool,
        show_significance: bool,
        anova_results: Optional[Dict],
    ):
        """绘制单个数据集的交互图"""
        dataset_df = self.df[self.df["dataset"] == dataset]

        # 计算均值和标准误
        summary = dataset_df.groupby(["strategy", "cache_size"])["score"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        summary["se"] = summary["std"] / np.sqrt(summary["count"])

        # X轴位置
        cache_sizes = sorted(dataset_df["cache_size"].unique())
        x_positions = np.arange(len(cache_sizes))

        for strategy in STRATEGIES:
            strategy_data = summary[summary["strategy"] == strategy]

            if len(strategy_data) == 0:
                continue

            # 按 cache_size 排序
            strategy_data = strategy_data.set_index("cache_size").loc[cache_sizes].reset_index()

            y_values = strategy_data["mean"].values
            y_errors = strategy_data["se"].values if show_error_bars else None

            ax.errorbar(
                x_positions,
                y_values,
                yerr=y_errors,
                marker=STRATEGY_MARKERS.get(strategy, "o"),
                color=STRATEGY_COLORS.get(strategy, "gray"),
                label=STRATEGY_LABELS.get(strategy, strategy),
                linewidth=2,
                markersize=8,
                capsize=4,
            )

        # 设置轴标签
        ax.set_xlabel("Cache Size (tokens)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"{dataset.upper()}", fontsize=14, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cache_sizes)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 显示显著性标注
        if show_significance and anova_results:
            self._add_significance_annotation(ax, dataset, anova_results)

    def _add_significance_annotation(
        self,
        ax: plt.Axes,
        dataset: str,
        anova_results: Dict,
    ):
        """添加显著性标注"""
        if dataset not in anova_results:
            return

        results = anova_results[dataset]
        if "error" in results:
            return

        interaction = results.get("interaction", {})
        p_value = interaction.get("p", 1.0)
        eta_sq = interaction.get("eta_squared", 0.0)
        significant = interaction.get("significant", False)

        # 构建标注文本
        sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        annotation = f"Interaction: p={p_value:.4f} {sig_text}\nη²={eta_sq:.4f}"

        # 添加文本框
        props = dict(boxstyle="round", facecolor="wheat" if significant else "lightgray", alpha=0.8)
        ax.text(
            0.02, 0.98,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )


def create_interaction_plot(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    anova_results: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    便捷函数：创建交互图

    Args:
        df: 结果数据 DataFrame
        output_path: 输出路径
        anova_results: ANOVA 结果
        figsize: 图像尺寸

    Returns:
        matplotlib Figure
    """
    plotter = InteractionPlotter(df, figsize)
    return plotter.plot(
        output_path=output_path,
        show_error_bars=True,
        show_significance=True,
        anova_results=anova_results,
    )


def create_main_effect_plots(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    创建主效应图 (独立于交互图)

    Args:
        df: 结果数据
        output_dir: 输出目录

    Returns:
        {figure_name: Figure}
    """
    figures = {}

    # 策略主效应
    fig_strategy, ax = plt.subplots(figsize=(8, 6))

    strategy_means = df.groupby("strategy")["score"].agg(["mean", "std", "count"])
    strategy_means["se"] = strategy_means["std"] / np.sqrt(strategy_means["count"])

    strategies = list(STRATEGIES)
    x_pos = np.arange(len(strategies))

    bars = ax.bar(
        x_pos,
        [strategy_means.loc[s, "mean"] if s in strategy_means.index else 0 for s in strategies],
        yerr=[strategy_means.loc[s, "se"] if s in strategy_means.index else 0 for s in strategies],
        color=[STRATEGY_COLORS.get(s, "gray") for s in strategies],
        capsize=5,
    )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Main Effect: Strategy", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies])
    ax.grid(True, alpha=0.3, axis="y")

    figures["strategy_main_effect"] = fig_strategy

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_strategy.savefig(output_dir / "strategy_main_effect.png", dpi=150, bbox_inches="tight")

    # 压缩率主效应
    fig_cache, ax = plt.subplots(figsize=(8, 6))

    cache_means = df.groupby("cache_size")["score"].agg(["mean", "std", "count"])
    cache_means["se"] = cache_means["std"] / np.sqrt(cache_means["count"])

    cache_sizes = sorted(df["cache_size"].unique())
    x_pos = np.arange(len(cache_sizes))

    ax.bar(
        x_pos,
        [cache_means.loc[c, "mean"] if c in cache_means.index else 0 for c in cache_sizes],
        yerr=[cache_means.loc[c, "se"] if c in cache_means.index else 0 for c in cache_sizes],
        color="#5DA5DA",
        capsize=5,
    )

    ax.set_xlabel("Cache Size (tokens)", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Main Effect: Cache Size", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cache_sizes)
    ax.grid(True, alpha=0.3, axis="y")

    figures["cache_main_effect"] = fig_cache

    if output_dir:
        fig_cache.savefig(output_dir / "cache_main_effect.png", dpi=150, bbox_inches="tight")

    plt.close("all")

    return figures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)

    data = []
    for strategy in STRATEGIES:
        for cache_size in CACHE_SIZES:
            for seed in [42, 123, 456]:
                for dataset in ["qmsum", "hotpotqa"]:
                    # 模拟交互效应
                    base_score = 25 if dataset == "qmsum" else 45

                    if strategy == "high_entropy":
                        score = base_score + (2048 - cache_size) / 200
                    elif strategy == "low_entropy":
                        score = base_score + (cache_size - 128) / 200
                    else:
                        score = base_score

                    score += np.random.normal(0, 1.5)

                    data.append({
                        "strategy": strategy,
                        "cache_size": cache_size,
                        "seed": seed,
                        "dataset": dataset,
                        "score": score,
                    })

    df = pd.DataFrame(data)
    print(f"Test data shape: {df.shape}")

    # 创建图表
    fig = create_interaction_plot(df)
    print("Interaction plot created")

    plt.close("all")
