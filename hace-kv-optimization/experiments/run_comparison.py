"""
实验对比脚本

运行基线、H2O和/或CAKE对比实验，并生成比较报告和统计分析。
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import json

# 确保能导入项目根目录的config和模块
current_script_path = os.path.abspath(__file__)
pkg_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(pkg_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hace_core import config as global_config
from analysis.statistical_tests import StatisticsAnalyzer

# 日志设置
logger = logging.getLogger(__name__)
def setup_logging(log_file=None, level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING) # 减少matplotlib日志
    return logging.getLogger(__name__)

def load_and_validate_csv(csv_path: str, method_name: str) -> pd.DataFrame | None:
    """加载CSV文件并进行基本验证。"""
    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"{method_name} 结果CSV文件路径 '{csv_path}' 无效或文件不存在。")
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"{method_name} CSV文件 '{csv_path}' 为空。")
            return None
        if "success" in df.columns:
             df = df[df["success"] == True].copy() # 使用 .copy() 避免SettingWithCopyWarning
             if df.empty:
                 logger.warning(f"在 {method_name} CSV '{csv_path}' 中没有找到成功的实验记录。")
                 return None
        logger.info(f"成功加载并筛选 {method_name} CSV: {csv_path}, 记录数: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"加载 {method_name} CSV文件 '{csv_path}' 时出错: {e}", exc_info=True)
        return None

def generate_comparison_plots(combined_df: pd.DataFrame, metrics_config: list, vis_dir: str, plot_prefix: str):
    """生成对比图表。"""
    os.makedirs(vis_dir, exist_ok=True)
    plots_paths = {}
    
    comparison_dimensions = ["kv_cache_length", "batch_size"]
    if "heavy_ratio" in combined_df.columns: # H2O特定维度
        comparison_dimensions.append("heavy_ratio")
    if "allocation_strategy" in combined_df.columns: # CAKE特定维度
        comparison_dimensions.append("allocation_strategy")

    for metric_info in metrics_config:
        metric_name = metric_info["name"]
        metric_title = metric_info["title"]
        if metric_name not in combined_df.columns:
            logger.warning(f"指标 '{metric_name}' 不在合并的数据中，跳过绘图。")
            continue

        for dim in comparison_dimensions:
            if dim not in combined_df.columns:
                logger.warning(f"维度 '{dim}' 不在合并的数据中，跳过 {metric_name} vs {dim} 的绘图。")
                continue
            
            # 确保绘图时x轴有多个唯一值
            if combined_df[dim].nunique() < 2 and not (dim == "allocation_strategy" and combined_df[dim].nunique() ==1) :
                logger.info(f"维度 '{dim}' 的唯一值少于2个 ({combined_df[dim].unique()})，不适合为 '{metric_name}' 生成线图/箱线图。")
                # 如果只有一个点，可以考虑bar plot
                if combined_df[dim].nunique() == 1 and combined_df["method_type"].nunique() > 1:
                     try:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(data=combined_df[combined_df[dim] == combined_df[dim].unique()[0]], x="method_type", y=metric_name, hue="method_type", dodge=False)
                        plt.title(f"{metric_title} at {dim} = {combined_df[dim].unique()[0]}")
                        plt.ylabel(metric_title)
                        plt.xlabel("Method Type")
                        plt.tight_layout()
                        plot_filename = os.path.join(vis_dir, f"{plot_prefix}_bar_{dim}_{metric_name}.png")
                        plt.savefig(plot_filename)
                        plt.close()
                        plots_paths[f"{metric_name}_vs_{dim}_bar"] = plot_filename
                        logger.info(f"单一维度值条形图已保存: {plot_filename}")
                     except Exception as e:
                        logger.error(f"为 {metric_name} vs {dim} (单一值) 生成条形图失败: {e}", exc_info=True)
                continue
            
            plt.figure(figsize=(12, 7))
            try:
                if pd.api.types.is_numeric_dtype(combined_df[dim]):
                    # 对于数值型维度，使用线图
                    sns.lineplot(data=combined_df, x=dim, y=metric_name, hue="method_type", style="method_type", markers=True, dashes=False)
                else:
                    # 对于类别型维度 (如 allocation_strategy)，使用箱线图
                    # 按类别排序以获得一致的图表顺序
                    ordered_categories = sorted(combined_df[dim].astype(str).unique())
                    sns.boxplot(data=combined_df, x=dim, y=metric_name, hue="method_type", order=ordered_categories, notch=False)
                    plt.xticks(rotation=45, ha="right")
                
                plt.title(f"{metric_title} vs {dim}")
                plt.xlabel(str(dim).replace("_", " ").title())
                plt.ylabel(metric_title)
                if "memory" in metric_name.lower(): plt.ylim(bottom=0)
                plt.legend(title="Method Type")
                plt.tight_layout()
                plot_filename = os.path.join(vis_dir, f"{plot_prefix}_{dim}_{metric_name}.png")
                plt.savefig(plot_filename)
                plt.close()
                plots_paths[f"{metric_name}_vs_{dim}"] = plot_filename
                logger.info(f"对比图已保存: {plot_filename}")
            except Exception as e:
                logger.error(f"为 {metric_name} vs {dim} 生成图表失败: {e}", exc_info=True)
    return plots_paths

def generate_summary_tables(combined_df: pd.DataFrame, metrics_config: list) -> dict:
    """生成性能指标的汇总表格 (均值和标准差)。"""
    summary_tables = {}
    groupby_cols = ["method_type"]
    # 动态添加分组列 (如果存在)
    for col in ["kv_cache_length", "batch_size", "heavy_ratio", "allocation_strategy", "cache_budget"]:
        if col in combined_df.columns:
            groupby_cols.append(col)
    
    # 确保只使用实际存在的列进行分组
    valid_groupby_cols = [col for col in groupby_cols if col in combined_df.columns]
    if not valid_groupby_cols:
        logger.error("没有有效的分组列可用于生成汇总表。")
        return summary_tables

    for metric_info in metrics_config:
        metric_name = metric_info["name"]
        if metric_name in combined_df.columns:
            try:
                summary = combined_df.groupby(valid_groupby_cols)[metric_name].agg(["mean", "std", "count"]).reset_index()
                summary_tables[metric_name] = summary.to_dict(orient="records")
            except Exception as e:
                logger.error(f"为指标 '{metric_name}' 生成汇总表失败: {e}", exc_info=True)
    return summary_tables

def main():
    parser = argparse.ArgumentParser(description="运行基线、H2O 和/或 CAKE 的对比实验并生成报告与分析。")
    parser.add_argument("--experiment_type", type=str, required=True, 
                        choices=["compare_h2o_baseline", "compare_cake_baseline", "compare_all"],
                        help="要执行的对比类型。")
    parser.add_argument("--baseline_csv", type=str, required=False, help="基线实验结果的CSV文件路径。")
    parser.add_argument("--h2o_csv", type=str, required=False, help="H2O实验结果的CSV文件路径。")
    parser.add_argument("--cake_csv", type=str, required=False, help="CAKE实验结果的CSV文件路径。")
    # output_main_dir 是由 run_experiment.py 传递的，指向 <output_dir_base>/<run_name>/comparison_report
    parser.add_argument("--output_main_dir", type=str, required=True, help="保存对比报告和分析结果的主目录。")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别 DEBUG, INFO, WARNING, ERROR")
    
    # 允许传递来自主脚本的通用参数，尽管此脚本可能不直接使用全部
    # 这对于未来扩展或传递给统计分析器可能有用
    parser.add_argument("--datasets", type=str, help="数据集名称 (主要用于命名和记录)")
    parser.add_argument("--model_name", type=str, help="模型名称 (主要用于命名和记录)")
    # ... 可以添加其他 common_params 以供记录 ...

    args = parser.parse_args()

    global logger
    log_file_path = os.path.join(args.output_main_dir, f"log_comparison_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = setup_logging(log_file=log_file_path, level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info(f"开始对比实验分析: {args.experiment_type}")
    logger.info(f"参数: {args}")

    # 加载数据
    baseline_df = None
    h2o_df = None
    cake_df = None
    data_frames = []

    if args.experiment_type in ["compare_h2o_baseline", "compare_cake_baseline", "compare_all"]:
        if not args.baseline_csv:
            logger.error("所有对比实验都需要 baseline_csv。")
            sys.exit(1)
        baseline_df = load_and_validate_csv(args.baseline_csv, "Baseline")
        if baseline_df is None: sys.exit(1)
        baseline_df["method_type"] = global_config.STATISTICS_CONFIG.get("baseline_group_name", "Baseline")
        data_frames.append(baseline_df)

    if args.experiment_type in ["compare_h2o_baseline", "compare_all"]:
        if not args.h2o_csv:
            logger.error(f"实验类型 '{args.experiment_type}' 需要 h2o_csv。")
            sys.exit(1)
        h2o_df = load_and_validate_csv(args.h2o_csv, "H2O")
        if h2o_df is None: sys.exit(1)
        h2o_df["method_type"] = global_config.STATISTICS_CONFIG.get("h2o_group_name", "H2O")
        data_frames.append(h2o_df)

    if args.experiment_type in ["compare_cake_baseline", "compare_all"]:
        if not args.cake_csv:
            logger.error(f"实验类型 '{args.experiment_type}' 需要 cake_csv。")
            sys.exit(1)
        cake_df = load_and_validate_csv(args.cake_csv, "CAKE")
        if cake_df is None: sys.exit(1)
        cake_df["method_type"] = global_config.STATISTICS_CONFIG.get("cake_group_name", "CAKE")
        data_frames.append(cake_df)
    
    if not data_frames:
        logger.error("没有成功加载任何实验数据，无法进行对比。")
        sys.exit(1)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    logger.info(f"合并后的数据包含 {len(combined_df)} 条记录，方法类型: {combined_df['method_type'].unique()}")

    # 定义要分析的指标 (可以从config.py中获取或硬编码)
    metrics_to_analyze = global_config.STATISTICS_CONFIG.get("comparison_metrics", [])
    # 将其转换为绘图和统计分析器所需的格式
    metrics_config_for_plots = []
    for m_name in metrics_to_analyze:
        title = m_name.replace("_", " ").title()
        lower_is_better = any(term in m_name.lower() for term in ["ttft", "tpot", "latency", "memory", "time"])
        metrics_config_for_plots.append({"name": m_name, "title": title, "lower_better": lower_is_better})

    # 生成图表
    visualizations_dir = os.path.join(args.output_main_dir, "visualizations")
    plot_prefix = args.experiment_type
    logger.info(f"开始生成对比图表到 {visualizations_dir}，前缀: {plot_prefix}")
    generate_comparison_plots(combined_df, metrics_config_for_plots, visualizations_dir, plot_prefix)

    # 生成汇总表
    logger.info("生成性能汇总表...")
    summary_tables_data = generate_summary_tables(combined_df, metrics_config_for_plots)
    summary_tables_path = os.path.join(args.output_main_dir, f"summary_tables_{plot_prefix}.json")
    try:
        with open(summary_tables_path, 'w', encoding='utf-8') as f:
            json.dump(summary_tables_data, f, indent=4, ensure_ascii=False)
        logger.info(f"性能汇总表已保存到: {summary_tables_path}")
    except Exception as e:
        logger.error(f"保存汇总表失败: {e}", exc_info=True)

    # 执行统计分析
    analyzer = StatisticsAnalyzer(config=global_config.STATISTICS_CONFIG)
    statistical_analysis_results = {}

    analysis_output_dir = os.path.join(args.output_main_dir, "statistical_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)

    if args.experiment_type == "compare_h2o_baseline" and baseline_df is not None and h2o_df is not None:
        logger.info("执行 H2O vs Baseline 统计分析...")
        h2o_analysis = analyzer.analyze_h2o_performance(baseline_df, h2o_df, metrics=metrics_to_analyze, output_dir=analysis_output_dir)
        statistical_analysis_results["h2o_vs_baseline"] = h2o_analysis
    
    elif args.experiment_type == "compare_cake_baseline" and baseline_df is not None and cake_df is not None:
        logger.info("执行 CAKE vs Baseline 统计分析...")
        cake_analysis = analyzer.analyze_cake_performance(baseline_df, cake_df, metrics=metrics_to_analyze, output_dir=analysis_output_dir)
        statistical_analysis_results["cake_vs_baseline"] = cake_analysis

    elif args.experiment_type == "compare_all" and baseline_df is not None and h2o_df is not None and cake_df is not None:
        logger.info("执行 H2O vs Baseline 和 CAKE vs Baseline 统计分析...")
        h2o_analysis = analyzer.analyze_h2o_performance(baseline_df, h2o_df, metrics=metrics_to_analyze, output_dir=analysis_output_dir)
        statistical_analysis_results["h2o_vs_baseline"] = h2o_analysis
        
        cake_analysis = analyzer.analyze_cake_performance(baseline_df, cake_df, metrics=metrics_to_analyze, output_dir=analysis_output_dir)
        statistical_analysis_results["cake_vs_baseline"] = cake_analysis
        
        # 未来可以添加 H2O vs CAKE 的直接对比分析
        # logger.info("执行 H2O vs CAKE 统计分析...")
        # h2o_vs_cake_analyzer = StatisticsAnalyzer(config=global_config.STATISTICS_CONFIG) # 使用新的实例或配置
        # h2o_vs_cake_analysis = h2o_vs_cake_analyzer.analyze_method_performance(
        #     baseline_df=h2o_df, # H2O作为"基线"
        #     method_df=cake_df,  # CAKE作为"方法"
        #     method_name="CAKE_vs_H2O",
        #     method_params_cols=global_config.STATISTICS_CONFIG.get("default_method_params", {}).get("cake", []), # 使用CAKE参数进行内部影响分析
        #     metrics_to_compare=metrics_to_analyze,
        #     output_dir=os.path.join(analysis_output_dir, "h2o_vs_cake_analysis")
        # )
        # statistical_analysis_results["cake_vs_h2o"] = h2o_vs_cake_analysis

    else:
        logger.warning(f"未针对实验类型 '{args.experiment_type}' 执行特定的统计分析，或所需数据不完整。")

    # 保存完整的统计分析结果
    full_stat_report_path = os.path.join(analysis_output_dir, f"full_statistical_report_{plot_prefix}.json")
    try:
        def json_safe_converter(o):
            if isinstance(o, (np.integer, np.bool_)):
                return int(o) if isinstance(o, np.integer) else bool(o)
            if isinstance(o, np.floating): 
                return float(o) if not (np.isnan(o) or np.isinf(o)) else str(o) # nan/inf 转字符串
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (pd.Timestamp, datetime)):
                return o.isoformat()
            if pd.isna(o): return None
            logger.debug(f"Cannot serialize type {type(o)}: {o}")
            return str(o) # Default fallback

        with open(full_stat_report_path, 'w', encoding='utf-8') as f:
            json.dump(statistical_analysis_results, f, indent=4, default=json_safe_converter, ensure_ascii=False)
        logger.info(f"完整的统计分析报告已保存到: {full_stat_report_path}")
    except Exception as e:
        logger.error(f"保存完整统计报告失败: {e}", exc_info=True)

    logger.info(f"对比分析 {args.experiment_type} 完成。结果保存在: {args.output_main_dir}")

if __name__ == "__main__":
    main() 