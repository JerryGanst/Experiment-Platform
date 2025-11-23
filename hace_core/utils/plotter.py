# Plotting utilities for experiment analysis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "create_memory_comparison_plots",
    "create_latency_plots",
    "create_tradeoff_plot",
]


def create_memory_comparison_plots(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None, output_dir: str):
    """Generate memory comparison plots.

    Args:
        baseline_df: DataFrame containing baseline metrics.
        h2o_df: DataFrame containing H2O metrics or ``None``.
        output_dir: Directory to save plots.

    Returns:
        List of file paths for the generated plots.
    """
    plot_paths: list[str] = []

    plt.figure(figsize=(12, 8))

    if h2o_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df["配置"] = "基线"

        h2o_df = h2o_df.copy()
        h2o_df["配置"] = h2o_df["heavy_ratio"].apply(lambda x: f"H2O-{int(x*100)}%")

        combined_df = pd.concat([baseline_df, h2o_df])
    else:
        baseline_df = baseline_df.copy()
        baseline_df["配置"] = "基线"
        combined_df = baseline_df

    grouped = combined_df.groupby(["配置", "kv_cache_length"]).agg({"peak_gpu_memory_mb": "mean"}).reset_index()

    for config in sorted(grouped["配置"].unique()):
        config_data = grouped[grouped["配置"] == config]
        plt.plot(
            config_data["kv_cache_length"],
            config_data["peak_gpu_memory_mb"],
            marker="o",
            label=config,
        )

    plt.title("KV缓存长度对内存使用的影响")
    plt.xlabel("KV缓存长度")
    plt.ylabel("峰值GPU内存 (MB)")
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    memory_kv_path = os.path.join(output_dir, "memory_vs_kv_length.png")
    plt.savefig(memory_kv_path, dpi=300)
    plt.close()
    plot_paths.append(memory_kv_path)

    plt.figure(figsize=(12, 8))
    grouped = combined_df.groupby(["配置", "batch_size"]).agg({"peak_gpu_memory_mb": "mean"}).reset_index()
    for config in sorted(grouped["配置"].unique()):
        config_data = grouped[grouped["配置"] == config]
        plt.plot(
            config_data["batch_size"],
            config_data["peak_gpu_memory_mb"],
            marker="o",
            label=config,
        )
    plt.title("批处理大小对内存使用的影响")
    plt.xlabel("批处理大小")
    plt.ylabel("峰值GPU内存 (MB)")
    plt.grid(True)
    plt.legend()
    memory_bs_path = os.path.join(output_dir, "memory_vs_batch_size.png")
    plt.savefig(memory_bs_path, dpi=300)
    plt.close()
    plot_paths.append(memory_bs_path)

    if h2o_df is not None:
        plt.figure(figsize=(14, 10))
        memory_savings = []
        for kv_length in sorted(combined_df["kv_cache_length"].unique()):
            for bs in sorted(combined_df["batch_size"].unique()):
                baseline_mem = baseline_df[
                    (baseline_df["kv_cache_length"] == kv_length)
                    & (baseline_df["batch_size"] == bs)
                ]["peak_gpu_memory_mb"].mean()

                for ratio in sorted(h2o_df["heavy_ratio"].unique()):
                    h2o_mem = h2o_df[
                        (h2o_df["kv_cache_length"] == kv_length)
                        & (h2o_df["batch_size"] == bs)
                        & (h2o_df["heavy_ratio"] == ratio)
                    ]["peak_gpu_memory_mb"].mean()

                    if not np.isnan(baseline_mem) and not np.isnan(h2o_mem):
                        saving = (baseline_mem - h2o_mem) / baseline_mem * 100
                        memory_savings.append(
                            {
                                "KV缓存长度": kv_length,
                                "批处理大小": bs,
                                "Heavy Ratio": f"{int(ratio*100)}%",
                                "内存节省(%)": saving,
                            }
                        )
        if memory_savings:
            savings_df = pd.DataFrame(memory_savings)
            pivot = pd.pivot_table(
                savings_df,
                values="内存节省(%)",
                index="批处理大小",
                columns=["KV缓存长度", "Heavy Ratio"],
                aggfunc="mean",
            )
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title("内存节省热力图 (%)，按KV缓存长度、批处理大小和Heavy Ratio")
            memory_heatmap_path = os.path.join(output_dir, "memory_saving_heatmap.png")
            plt.savefig(memory_heatmap_path, dpi=300)
            plt.close()
            plot_paths.append(memory_heatmap_path)
    return plot_paths


def create_latency_plots(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None, output_dir: str):
    """Generate latency comparison plots."""
    plot_paths: list[str] = []

    if h2o_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df["配置"] = "基线"

        h2o_df = h2o_df.copy()
        h2o_df["配置"] = h2o_df["heavy_ratio"].apply(lambda x: f"H2O-{int(x*100)}%")
        combined_df = pd.concat([baseline_df, h2o_df])
    else:
        baseline_df = baseline_df.copy()
        baseline_df["配置"] = "基线"
        combined_df = baseline_df

    plt.figure(figsize=(12, 8))
    grouped = combined_df.groupby(["配置", "batch_size"]).agg({"ttft_ms": "mean"}).reset_index()
    for config in sorted(grouped["配置"].unique()):
        config_data = grouped[grouped["配置"] == config]
        plt.plot(config_data["batch_size"], config_data["ttft_ms"], marker="o", label=config)
    plt.title("批处理大小对首个令牌生成时间的影响")
    plt.xlabel("批处理大小")
    plt.ylabel("TTFT (ms)")
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    ttft_path = os.path.join(output_dir, "ttft_vs_batch_size.png")
    plt.savefig(ttft_path, dpi=300)
    plt.close()
    plot_paths.append(ttft_path)

    plt.figure(figsize=(12, 8))
    grouped = combined_df.groupby(["配置", "batch_size"]).agg({"tpot_ms": "mean"}).reset_index()
    for config in sorted(grouped["配置"].unique()):
        config_data = grouped[grouped["配置"] == config]
        plt.plot(config_data["batch_size"], config_data["tpot_ms"], marker="o", label=config)
    plt.title("批处理大小对每令牌生成时间的影响")
    plt.xlabel("批处理大小")
    plt.ylabel("TPOT (ms)")
    plt.grid(True)
    plt.legend()
    tpot_path = os.path.join(output_dir, "tpot_vs_batch_size.png")
    plt.savefig(tpot_path, dpi=300)
    plt.close()
    plot_paths.append(tpot_path)

    plt.figure(figsize=(12, 8))
    grouped = combined_df.groupby(["配置", "batch_size"]).agg({"throughput_tokens_per_sec": "mean"}).reset_index()
    for config in sorted(grouped["配置"].unique()):
        config_data = grouped[grouped["配置"] == config]
        plt.plot(
            config_data["batch_size"],
            config_data["throughput_tokens_per_sec"],
            marker="o",
            label=config,
        )
    plt.title("批处理大小对吞吐量的影响")
    plt.xlabel("批处理大小")
    plt.ylabel("吞吐量 (tokens/sec)")
    plt.grid(True)
    plt.legend()
    throughput_path = os.path.join(output_dir, "throughput_vs_batch_size.png")
    plt.savefig(throughput_path, dpi=300)
    plt.close()
    plot_paths.append(throughput_path)

    return plot_paths


def create_tradeoff_plot(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None, output_dir: str):
    """Generate memory/throughput trade-off scatter plot."""
    if h2o_df is None:
        return None

    plt.figure(figsize=(12, 8))
    tradeoff_data = []
    for kv_length in sorted(baseline_df["kv_cache_length"].unique()):
        for bs in sorted(baseline_df["batch_size"].unique()):
            baseline_metrics = baseline_df[
                (baseline_df["kv_cache_length"] == kv_length)
                & (baseline_df["batch_size"] == bs)
            ].agg({"peak_gpu_memory_mb": "mean", "throughput_tokens_per_sec": "mean"})
            for ratio in sorted(h2o_df["heavy_ratio"].unique()):
                h2o_metrics = h2o_df[
                    (h2o_df["kv_cache_length"] == kv_length)
                    & (h2o_df["batch_size"] == bs)
                    & (h2o_df["heavy_ratio"] == ratio)
                ].agg({"peak_gpu_memory_mb": "mean", "throughput_tokens_per_sec": "mean"})
                if not baseline_metrics.empty and not h2o_metrics.empty:
                    memory_saving = (
                        (baseline_metrics["peak_gpu_memory_mb"] - h2o_metrics["peak_gpu_memory_mb"])
                        / baseline_metrics["peak_gpu_memory_mb"]
                        * 100
                    )
                    throughput_change = (
                        (h2o_metrics["throughput_tokens_per_sec"] - baseline_metrics["throughput_tokens_per_sec"])
                        / baseline_metrics["throughput_tokens_per_sec"]
                        * 100
                    )
                    tradeoff_data.append(
                        {
                            "KV缓存长度": kv_length,
                            "批处理大小": bs,
                            "Heavy Ratio": ratio,
                            "内存节省(%)": memory_saving,
                            "吞吐量变化(%)": throughput_change,
                        }
                    )
    if not tradeoff_data:
        return None
    tradeoff_df = pd.DataFrame(tradeoff_data)
    sns.scatterplot(
        data=tradeoff_df,
        x="内存节省(%)",
        y="吞吐量变化(%)",
        hue="Heavy Ratio",
        size="批处理大小",
        sizes=(50, 200),
        alpha=0.7,
    )
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="r", linestyle="-", alpha=0.3)
    plt.title("内存节省与吞吐量变化权衡图")
    plt.grid(True, alpha=0.3)
    best_memory = tradeoff_df.loc[tradeoff_df["内存节省(%)"].idxmax()]
    best_throughput = tradeoff_df.loc[tradeoff_df["吞吐量变化(%)"].idxmax()]
    best_balanced = tradeoff_df.loc[(tradeoff_df["内存节省(%)"] + tradeoff_df["吞吐量变化(%)"]).idxmax()]
    plt.annotate(
        f"最大内存节省\nKV={best_memory['KV缓存长度']}, BS={best_memory['批处理大小']}, HR={best_memory['Heavy Ratio']:.1f}",
        xy=(best_memory["内存节省(%)"], best_memory["吞吐量变化(%)"]),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )
    plt.annotate(
        f"最大吞吐量提升\nKV={best_throughput['KV缓存长度']}, BS={best_throughput['批处理大小']}, HR={best_throughput['Heavy Ratio']:.1f}",
        xy=(best_throughput["内存节省(%)"], best_throughput["吞吐量变化(%)"]),
        xytext=(10, -10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )
    plt.annotate(
        f"最佳平衡点\nKV={best_balanced['KV缓存长度']}, BS={best_balanced['批处理大小']}, HR={best_balanced['Heavy Ratio']:.1f}",
        xy=(best_balanced["内存节省(%)"], best_balanced["吞吐量变化(%)"]),
        xytext=(-10, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
    )
    tradeoff_path = os.path.join(output_dir, "memory_throughput_tradeoff.png")
    plt.savefig(tradeoff_path, dpi=300)
    plt.close()
    return tradeoff_path

