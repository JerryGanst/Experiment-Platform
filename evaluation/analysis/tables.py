"""Utilities for generating result tables."""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

__all__ = [
    "load_experiment_results",
    "create_performance_summary_table",
    "create_language_comparison_table",
    "create_best_config_table",
]


def load_experiment_results(baseline_dir: str, h2o_dir: str | None = None):
    """Load baseline and optional H2O experiment results from directories."""
    baseline_files: list[str] = []
    for root, _, files in os.walk(baseline_dir):
        for file in files:
            if file.endswith(".csv") and "baseline_results" in file and "summary" not in file:
                baseline_files.append(os.path.join(root, file))
    if not baseline_files:
        raise ValueError(f"No baseline result files found in {baseline_dir}")
    baseline_dfs: list[pd.DataFrame] = []
    for file in baseline_files:
        try:
            df = pd.read_csv(file)
            model_name = None
            if "opt-125m" in file:
                model_name = "opt-125m"
            elif "opt-350m" in file:
                model_name = "opt-350m"
            elif "opt-1.3b" in file:
                model_name = "opt-1.3b"
            if model_name and "model_name" not in df.columns:
                df["model_name"] = model_name
            dataset_name = None
            if "mmlu" in file:
                dataset_name = "mmlu"
            elif "hellaswag" in file:
                dataset_name = "hellaswag"
            elif "ceval" in file:
                dataset_name = "ceval"
            elif "race" in file:
                dataset_name = "race"
            if dataset_name and "dataset" not in df.columns:
                df["dataset"] = dataset_name
            if "language" not in df.columns:
                if dataset_name in ["mmlu", "hellaswag"]:
                    df["language"] = "english"
                elif dataset_name in ["ceval", "race"]:
                    df["language"] = "chinese"
            df["type"] = "baseline"
            baseline_dfs.append(df)
        except Exception as e:
            print(f"Error loading baseline file {file}: {e}")
    if not baseline_dfs:
        raise ValueError("Failed to load any baseline results")
    baseline_df = pd.concat(baseline_dfs, ignore_index=True)

    h2o_df: pd.DataFrame | None = None
    if h2o_dir:
        h2o_files: list[str] = []
        for root, _, files in os.walk(h2o_dir):
            for file in files:
                if file.endswith(".csv") and "h2o_results" in file and "summary" not in file:
                    h2o_files.append(os.path.join(root, file))
        if h2o_files:
            h2o_dfs: list[pd.DataFrame] = []
            for file in h2o_files:
                try:
                    df = pd.read_csv(file)
                    model_name = None
                    if "opt-125m" in file:
                        model_name = "opt-125m"
                    elif "opt-350m" in file:
                        model_name = "opt-350m"
                    elif "opt-1.3b" in file:
                        model_name = "opt-1.3b"
                    if model_name and "model_name" not in df.columns:
                        df["model_name"] = model_name
                    dataset_name = None
                    if "mmlu" in file:
                        dataset_name = "mmlu"
                    elif "hellaswag" in file:
                        dataset_name = "hellaswag"
                    elif "ceval" in file:
                        dataset_name = "ceval"
                    elif "race" in file:
                        dataset_name = "race"
                    if dataset_name and "dataset" not in df.columns:
                        df["dataset"] = dataset_name
                    if "language" not in df.columns:
                        if dataset_name in ["mmlu", "hellaswag"]:
                            df["language"] = "english"
                        elif dataset_name in ["ceval", "race"]:
                            df["language"] = "chinese"
                    df["type"] = "h2o"
                    h2o_dfs.append(df)
                except Exception as e:
                    print(f"Error loading H2O file {file}: {e}")
            if h2o_dfs:
                h2o_df = pd.concat(h2o_dfs, ignore_index=True)
        else:
            print(f"Warning: No H2O result files found in {h2o_dir}")
    return baseline_df, h2o_df


def create_performance_summary_table(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None = None):
    baseline_summary = baseline_df.groupby(["model_name", "kv_cache_length", "batch_size"]).agg({
        "peak_gpu_memory_mb": "mean",
        "ttft_ms": "mean",
        "tpot_ms": "mean",
        "throughput_tokens_per_sec": "mean",
    }).reset_index()
    baseline_summary["type"] = "baseline"
    baseline_summary["memory_saving_percent"] = 0
    baseline_summary["latency_improvement_percent"] = 0
    baseline_summary["throughput_improvement_percent"] = 0
    if h2o_df is None:
        baseline_summary = baseline_summary.rename(
            columns={
                "model_name": "模型",
                "type": "配置",
                "kv_cache_length": "KV长度",
                "batch_size": "批处理大小",
                "peak_gpu_memory_mb": "内存使用(MB)",
                "memory_saving_percent": "内存节省(%)",
                "ttft_ms": "TTFT(ms)",
                "tpot_ms": "TPOT(ms)",
                "latency_improvement_percent": "延迟改进(%)",
                "throughput_tokens_per_sec": "吞吐量(tokens/s)",
                "throughput_improvement_percent": "吞吐量改进(%)",
            }
        )
        baseline_summary = baseline_summary[
            [
                "配置",
                "模型",
                "KV长度",
                "批处理大小",
                "内存使用(MB)",
                "内存节省(%)",
                "TTFT(ms)",
                "TPOT(ms)",
                "吞吐量(tokens/s)",
            ]
        ]
        baseline_summary["配置"] = "基线"
        return baseline_summary

    h2o_summary = h2o_df.groupby(["model_name", "kv_cache_length", "batch_size", "heavy_ratio"]).agg({
        "peak_gpu_memory_mb": "mean",
        "ttft_ms": "mean",
        "tpot_ms": "mean",
        "throughput_tokens_per_sec": "mean",
    }).reset_index()
    h2o_summary["type"] = "h2o"
    summary_rows: list[dict] = []
    for _, baseline_row in baseline_summary.iterrows():
        summary_rows.append(
            {
                "配置": "基线",
                "模型": baseline_row["model_name"],
                "KV长度": baseline_row["kv_cache_length"],
                "批处理大小": baseline_row["batch_size"],
                "内存使用(MB)": round(baseline_row["peak_gpu_memory_mb"], 2),
                "内存节省(%)": "-",
                "TTFT(ms)": round(baseline_row["ttft_ms"], 2),
                "TPOT(ms)": round(baseline_row["tpot_ms"], 2),
                "吞吐量(tokens/s)": round(baseline_row["throughput_tokens_per_sec"], 2),
                "困惑度": "-",
                "准确率变化(%)": "-",
            }
        )
    for _, h2o_row in h2o_summary.iterrows():
        baseline_match = baseline_summary[
            (baseline_summary["model_name"] == h2o_row["model_name"])
            & (baseline_summary["kv_cache_length"] == h2o_row["kv_cache_length"])
            & (baseline_summary["batch_size"] == h2o_row["batch_size"])
        ]
        if not baseline_match.empty:
            baseline_match = baseline_match.iloc[0]
            memory_saving = (
                (baseline_match["peak_gpu_memory_mb"] - h2o_row["peak_gpu_memory_mb"])
                / baseline_match["peak_gpu_memory_mb"]
                * 100
            )
            latency_improvement = (
                (baseline_match["ttft_ms"] - h2o_row["ttft_ms"])
                / baseline_match["ttft_ms"]
                * 100
            )
            throughput_improvement = (
                (h2o_row["throughput_tokens_per_sec"] - baseline_match["throughput_tokens_per_sec"])
                / baseline_match["throughput_tokens_per_sec"]
                * 100
            )
            summary_rows.append(
                {
                    "配置": f"H2O-{int(h2o_row['heavy_ratio']*100)}%",
                    "模型": h2o_row["model_name"],
                    "KV长度": h2o_row["kv_cache_length"],
                    "批处理大小": h2o_row["batch_size"],
                    "内存使用(MB)": round(h2o_row["peak_gpu_memory_mb"], 2),
                    "内存节省(%)": round(memory_saving, 2),
                    "TTFT(ms)": round(h2o_row["ttft_ms"], 2),
                    "TPOT(ms)": round(h2o_row["tpot_ms"], 2),
                    "吞吐量(tokens/s)": round(h2o_row["throughput_tokens_per_sec"], 2),
                    "困惑度": "-",
                    "准确率变化(%)": "-",
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["模型", "KV长度", "批处理大小", "配置"])
    return summary_df


def create_language_comparison_table(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None = None):
    if h2o_df is None:
        language_df = pd.DataFrame(columns=["语言", "数据集", "内存节省(%)", "延迟改进(%)", "质量变化(%)"])
        language_groups = baseline_df.groupby(["language", "dataset"]).size().reset_index()
        for _, row in language_groups.iterrows():
            language_df = language_df.append(
                {
                    "语言": row["language"],
                    "数据集": row["dataset"],
                    "内存节省(%)": "-",
                    "延迟改进(%)": "-",
                    "质量变化(%)": "-",
                },
                ignore_index=True,
            )
        return language_df
    baseline_language = baseline_df.groupby(["language", "dataset"]).agg({
        "peak_gpu_memory_mb": "mean",
        "ttft_ms": "mean",
        "tpot_ms": "mean",
    }).reset_index()
    h2o_language = h2o_df.groupby(["language", "dataset"]).agg({
        "peak_gpu_memory_mb": "mean",
        "ttft_ms": "mean",
        "tpot_ms": "mean",
    }).reset_index()
    language_rows: list[dict] = []
    for _, baseline_row in baseline_language.iterrows():
        language = baseline_row["language"]
        dataset = baseline_row["dataset"]
        h2o_match = h2o_language[(h2o_language["language"] == language) & (h2o_language["dataset"] == dataset)]
        if not h2o_match.empty:
            h2o_match = h2o_match.iloc[0]
            memory_saving = (
                (baseline_row["peak_gpu_memory_mb"] - h2o_match["peak_gpu_memory_mb"])
                / baseline_row["peak_gpu_memory_mb"]
                * 100
            )
            latency_improvement = (
                (baseline_row["ttft_ms"] - h2o_match["ttft_ms"])
                / baseline_row["ttft_ms"]
                * 100
            )
            language_rows.append(
                {
                    "语言": language,
                    "数据集": dataset,
                    "内存节省(%)": round(memory_saving, 2),
                    "延迟改进(%)": round(latency_improvement, 2),
                    "质量变化(%)": "-",
                }
            )
        else:
            language_rows.append(
                {
                    "语言": language,
                    "数据集": dataset,
                    "内存节省(%)": "-",
                    "延迟改进(%)": "-",
                    "质量变化(%)": "-",
                }
            )
    language_df = pd.DataFrame(language_rows)
    language_df["语言"] = language_df["语言"].map({"english": "英文", "chinese": "中文"})
    language_df = language_df.sort_values(["语言", "数据集"])
    return language_df


def create_best_config_table(baseline_df: pd.DataFrame, h2o_df: pd.DataFrame | None = None):
    if h2o_df is None:
        config_df = pd.DataFrame([
            {
                "应用场景": "长文本生成",
                "推荐H2O配置": "待H2O实验完成后填充",
                "内存节省(%)": "-",
                "性能提升(%)": "-",
                "注意事项": "-",
            },
            {
                "应用场景": "批量处理",
                "推荐H2O配置": "待H2O实验完成后填充",
                "内存节省(%)": "-",
                "性能提升(%)": "-",
                "注意事项": "-",
            },
            {
                "应用场景": "实时对话",
                "推荐H2O配置": "待H2O实验完成后填充",
                "内存节省(%)": "-",
                "性能提升(%)": "-",
                "注意事项": "-",
            },
        ])
        return config_df
    config_rows: list[dict] = []
    long_text_configs = h2o_df[h2o_df["kv_cache_length"] >= 1024].copy()
    if not long_text_configs.empty:
        long_text_best = long_text_configs.groupby(["heavy_ratio"]).agg({
            "peak_gpu_memory_mb": "mean",
            "throughput_tokens_per_sec": "mean",
        }).reset_index()
        baseline_long = baseline_df[baseline_df["kv_cache_length"] >= 1024].agg({
            "peak_gpu_memory_mb": "mean",
            "throughput_tokens_per_sec": "mean",
        })
        long_text_best["memory_saving"] = (
            (baseline_long["peak_gpu_memory_mb"] - long_text_best["peak_gpu_memory_mb"])
            / baseline_long["peak_gpu_memory_mb"]
            * 100
        )
        long_text_best["throughput_improvement"] = (
            (long_text_best["throughput_tokens_per_sec"] - baseline_long["throughput_tokens_per_sec"])
            / baseline_long["throughput_tokens_per_sec"]
            * 100
        )
        best_long_config = long_text_best.loc[long_text_best["memory_saving"].idxmax()]
        config_rows.append(
            {
                "应用场景": "长文本生成",
                "推荐H2O配置": f"{int(best_long_config['heavy_ratio']*100)}%重要命中",
                "内存节省(%)": round(best_long_config["memory_saving"], 2),
                "性能提升(%)": round(best_long_config["throughput_improvement"], 2),
                "注意事项": "适用于需要长上下文处理的场景，如文档摘要和长对话",
            }
        )
    batch_configs = h2o_df[h2o_df["batch_size"] >= 8].copy()
    if not batch_configs.empty:
        batch_best = batch_configs.groupby(["heavy_ratio"]).agg({
            "peak_gpu_memory_mb": "mean",
            "throughput_tokens_per_sec": "mean",
        }).reset_index()
        baseline_batch = baseline_df[baseline_df["batch_size"] >= 8].agg({
            "peak_gpu_memory_mb": "mean",
            "throughput_tokens_per_sec": "mean",
        })
        batch_best["memory_saving"] = (
            (baseline_batch["peak_gpu_memory_mb"] - batch_best["peak_gpu_memory_mb"])
            / baseline_batch["peak_gpu_memory_mb"]
            * 100
        )
        batch_best["throughput_improvement"] = (
            (batch_best["throughput_tokens_per_sec"] - baseline_batch["throughput_tokens_per_sec"])
            / baseline_batch["throughput_tokens_per_sec"]
            * 100
        )
        best_batch_config = batch_best.loc[batch_best["throughput_improvement"].idxmax()]
        config_rows.append(
            {
                "应用场景": "批量处理",
                "推荐H2O配置": f"{int(best_batch_config['heavy_ratio']*100)}%重要命中",
                "内存节省(%)": round(best_batch_config["memory_saving"], 2),
                "性能提升(%)": round(best_batch_config["throughput_improvement"], 2),
                "注意事项": "适用于需要同时处理多个请求的场景，如API服务和批量生成",
            }
        )
    realtime_configs = h2o_df[h2o_df["batch_size"] <= 4].copy()
    if not realtime_configs.empty:
        realtime_best = realtime_configs.groupby(["heavy_ratio"]).agg({
            "peak_gpu_memory_mb": "mean",
            "ttft_ms": "mean",
        }).reset_index()
        baseline_realtime = baseline_df[baseline_df["batch_size"] <= 4].agg({
            "peak_gpu_memory_mb": "mean",
            "ttft_ms": "mean",
        })
        realtime_best["memory_saving"] = (
            (baseline_realtime["peak_gpu_memory_mb"] - realtime_best["peak_gpu_memory_mb"])
            / baseline_realtime["peak_gpu_memory_mb"]
            * 100
        )
        realtime_best["latency_improvement"] = (
            (baseline_realtime["ttft_ms"] - realtime_best["ttft_ms"])
            / baseline_realtime["ttft_ms"]
            * 100
        )
        best_realtime_config = realtime_best.loc[realtime_best["latency_improvement"].idxmax()]
        config_rows.append(
            {
                "应用场景": "实时对话",
                "推荐H2O配置": f"{int(best_realtime_config['heavy_ratio']*100)}%重要命中",
                "内存节省(%)": round(best_realtime_config["memory_saving"], 2),
                "性能提升(%)": round(best_realtime_config["latency_improvement"], 2),
                "注意事项": "适用于对响应延迟要求较高的场景，如在线聊天机器人",
            }
        )
    config_df = pd.DataFrame(config_rows)
    return config_df

