"""
H2O与基线对比实验脚本

运行基线和H2O对比实验，并生成比较报告
"""
import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
from config import MODEL_CONFIG, EXPERIMENT_CONFIG, OUTPUT_CONFIG, MONITORING_CONFIG

# 设置日志
def setup_logging(log_file=None, level=logging.INFO):
    """设置日志记录"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建日志目录
    if log_file:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def run_baseline_experiment(args, output_dir):
    """
    运行基线实验
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        str: 结果CSV文件路径
    """
    cmd = ["python", "-m", "h2o_experiment.main"]
    
    # 添加参数
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--dataset", args.dataset])
    cmd.extend(["--language", args.language])
    
    if args.kv_cache_lengths:
        cmd.extend(["--kv_cache_lengths", args.kv_cache_lengths])
    
    if args.batch_sizes:
        cmd.extend(["--batch_sizes", args.batch_sizes])
    
    logger.info(f"Running baseline experiment with command: {' '.join(cmd)}")
    
    # 运行实验
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # 输出日志
    logger.info("Baseline experiment stdout:\n" + stdout)
    if stderr:
        logger.error("Baseline experiment stderr:\n" + stderr)
    
    # 找到生成的结果CSV文件
    csv_files = list(Path(output_dir).glob(f"baseline_results_{args.language}_{args.dataset}_*.csv"))
    if not csv_files:
        logger.error("No baseline results CSV found!")
        return None
    
    # 返回最新的CSV文件
    latest_csv = max(csv_files, key=os.path.getctime)
    logger.info(f"Baseline results saved to: {latest_csv}")
    return str(latest_csv)

def run_h2o_experiment(args, output_dir):
    """
    运行H2O实验
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        str: 结果CSV文件路径
    """
    cmd = ["python", "-m", "h2o_experiment.h2o_main"]
    
    # 添加参数
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--dataset", args.dataset])
    cmd.extend(["--language", args.language])
    
    if args.kv_cache_lengths:
        cmd.extend(["--kv_cache_lengths", args.kv_cache_lengths])
    
    if args.batch_sizes:
        cmd.extend(["--batch_sizes", args.batch_sizes])
    
    if args.h2o_ratios:
        cmd.extend(["--h2o_ratios", args.h2o_ratios])
    
    logger.info(f"Running H2O experiment with command: {' '.join(cmd)}")
    
    # 运行实验
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # 输出日志
    logger.info("H2O experiment stdout:\n" + stdout)
    if stderr:
        logger.error("H2O experiment stderr:\n" + stderr)
    
    # 找到生成的结果CSV文件
    csv_files = list(Path(output_dir).glob(f"h2o_results_{args.language}_{args.dataset}_*.csv"))
    if not csv_files:
        logger.error("No H2O results CSV found!")
        return None
    
    # 返回最新的CSV文件
    latest_csv = max(csv_files, key=os.path.getctime)
    logger.info(f"H2O results saved to: {latest_csv}")
    return str(latest_csv)

def generate_comparison_report(baseline_csv, h2o_csv, output_dir):
    """
    生成基线和H2O对比报告
    
    Args:
        baseline_csv: 基线结果CSV文件路径
        h2o_csv: H2O结果CSV文件路径
        output_dir: 输出目录
        
    Returns:
        str: 报告文件路径
    """
    if not baseline_csv or not h2o_csv:
        logger.error("Missing baseline or H2O results CSV!")
        return None
    
    # 加载数据
    try:
        baseline_df = pd.read_csv(baseline_csv)
        h2o_df = pd.read_csv(h2o_csv)
    except Exception as e:
        logger.error(f"Error loading CSV files: {e}")
        return None
    
    # 筛选成功的实验
    baseline_df = baseline_df[baseline_df["success"] == True]
    h2o_df = h2o_df[h2o_df["success"] == True]
    
    if baseline_df.empty or h2o_df.empty:
        logger.error("No successful experiments found in one or both CSV files!")
        return None
    
    # 创建报告目录
    vis_dir = os.path.join(output_dir, OUTPUT_CONFIG["visualizations_dir"])
    os.makedirs(vis_dir, exist_ok=True)
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 14})
    
    # 添加模型类型标签
    baseline_df["type"] = "Baseline"
    h2o_df["type"] = "H2O"
    
    # 合并数据
    combined_df = pd.concat([baseline_df, h2o_df], ignore_index=True)
    
    # 分析指标: GPU内存、TTFT、TPOT、吞吐量
    metrics = [
        {"name": "peak_gpu_memory_mb", "title": "Peak GPU Memory (MB)", "lower_better": True},
        {"name": "ttft_ms", "title": "Time to First Token (ms)", "lower_better": True},
        {"name": "tpot_ms", "title": "Time per Output Token (ms)", "lower_better": True},
        {"name": "throughput_tokens_per_sec", "title": "Throughput (tokens/sec)", "lower_better": False}
    ]
    
    # 图表列表
    plots = []
    
    # 1. 按KV缓存长度比较
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=combined_df, 
            x="kv_cache_length", 
            y=metric["name"], 
            hue="type",
            style="type",
            markers=True, 
            dashes=False,
            ax=ax
        )
        ax.set_title(f"{metric['title']} vs KV Cache Length")
        ax.set_xlabel("KV Cache Length")
        ax.set_ylabel(metric["title"])
        
        # 如果是内存使用，调整y轴从0开始
        if "memory" in metric["name"]:
            ax.set_ylim(bottom=0)
        
        plot_path = os.path.join(vis_dir, f"kv_length_{metric['name']}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plots.append(plot_path)
        plt.close(fig)
    
    # 2. 按批处理大小比较
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=combined_df, 
            x="batch_size", 
            y=metric["name"], 
            hue="type",
            style="type",
            markers=True, 
            dashes=False,
            ax=ax
        )
        ax.set_title(f"{metric['title']} vs Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(metric["title"])
        
        # 如果是内存使用，调整y轴从0开始
        if "memory" in metric["name"]:
            ax.set_ylim(bottom=0)
        
        plot_path = os.path.join(vis_dir, f"batch_size_{metric['name']}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plots.append(plot_path)
        plt.close(fig)
    
    # 3. 计算平均改进率
    improvements = {}
    for metric in metrics:
        name = metric["name"]
        baseline_mean = baseline_df[name].mean()
        h2o_mean = h2o_df[name].mean()
        
        if metric["lower_better"]:
            improvement = (baseline_mean - h2o_mean) / baseline_mean * 100
        else:
            improvement = (h2o_mean - baseline_mean) / baseline_mean * 100
        
        improvements[name] = {
            "baseline_mean": baseline_mean,
            "h2o_mean": h2o_mean,
            "improvement": improvement
        }
    
    # 生成HTML报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.html")
    
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>H2O vs Baseline Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric-card {{ 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0;
                    background-color: #f9f9f9;
                }}
                .improvement-positive {{ color: green; }}
                .improvement-negative {{ color: red; }}
                .plot-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>H2O vs Baseline Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Experiment Configuration</h2>
            <ul>
                <li>Model: {MODEL_CONFIG.get("model_name_or_path", "unknown")}</li>
                <li>Precision: {MODEL_CONFIG.get("precision", "fp16")}</li>
                <li>KV Cache Lengths: {', '.join(map(str, combined_df['kv_cache_length'].unique()))}</li>
                <li>Batch Sizes: {', '.join(map(str, combined_df['batch_size'].unique()))}</li>
                <li>Max New Tokens: {EXPERIMENT_CONFIG.get("max_new_tokens", 128)}</li>
            </ul>
            
            <h2>Performance Improvements Summary</h2>
        """)
        
        # 添加性能改进摘要
        for metric in metrics:
            name = metric["name"]
            imp = improvements[name]
            imp_class = "improvement-positive" if imp["improvement"] > 0 else "improvement-negative"
            
            f.write(f"""
            <div class="metric-card">
                <h3>{metric["title"]}</h3>
                <p>Baseline: {imp["baseline_mean"]:.2f}</p>
                <p>H2O: {imp["h2o_mean"]:.2f}</p>
                <p>Improvement: <span class="{imp_class}">{imp["improvement"]:.2f}%</span></p>
            </div>
            """)
        
        # 添加图表
        f.write("<h2>Visualization: KV Cache Length Impact</h2>")
        for i, metric in enumerate(metrics):
            plot_path = plots[i]
            rel_path = os.path.relpath(plot_path, output_dir)
            f.write(f"""
            <div class="plot-container">
                <h3>{metric["title"]}</h3>
                <img src="{rel_path}" alt="{metric['title']} vs KV Cache Length" width="800">
            </div>
            """)
        
        f.write("<h2>Visualization: Batch Size Impact</h2>")
        for i, metric in enumerate(metrics):
            plot_path = plots[i + len(metrics)]
            rel_path = os.path.relpath(plot_path, output_dir)
            f.write(f"""
            <div class="plot-container">
                <h3>{metric["title"]}</h3>
                <img src="{rel_path}" alt="{metric['title']} vs Batch Size" width="800">
            </div>
            """)
        
        # 添加详细数据表
        f.write("""
            <h2>Detailed Comparison Data</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>KV Length</th>
                    <th>Batch Size</th>
                    <th>Peak Memory (MB)</th>
                    <th>TTFT (ms)</th>
                    <th>TPOT (ms)</th>
                    <th>Throughput (tokens/sec)</th>
                </tr>
        """)
        
        # 计算每种配置的平均值
        agg_cols = ["type", "kv_cache_length", "batch_size"]
        if "heavy_ratio" in h2o_df.columns:
            agg_cols.append("heavy_ratio")
        
        agg_data = combined_df.groupby(agg_cols).agg({
            "peak_gpu_memory_mb": "mean",
            "ttft_ms": "mean",
            "tpot_ms": "mean",
            "throughput_tokens_per_sec": "mean"
        }).reset_index()
        
        for _, row in agg_data.iterrows():
            f.write(f"""
                <tr>
                    <td>{row['type']}</td>
                    <td>{row['kv_cache_length']}</td>
                    <td>{row['batch_size']}</td>
                    <td>{row['peak_gpu_memory_mb']:.2f}</td>
                    <td>{row['ttft_ms']:.2f}</td>
                    <td>{row['tpot_ms']:.2f}</td>
                    <td>{row['throughput_tokens_per_sec']:.2f}</td>
                </tr>
            """)
        
        f.write("""
            </table>
        </body>
        </html>
        """)
    
    logger.info(f"Comparison report generated: {report_path}")
    return report_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run H2O vs Baseline Comparison Experiments")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_CONFIG.get("results_dir", "./results"),
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        help="Dataset to use for evaluation")
    parser.add_argument("--language", type=str, default="english",
                        choices=["english", "chinese"],
                        help="Language of the dataset")
    parser.add_argument("--kv_cache_lengths", type=str, default=None,
                        help="Comma-separated list of KV cache lengths to test (overrides config)")
    parser.add_argument("--batch_sizes", type=str, default=None,
                        help="Comma-separated list of batch sizes to test (overrides config)")
    parser.add_argument("--h2o_ratios", type=str, default=None,
                        help="Comma-separated list of H2O ratios to test (overrides config)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline experiments")
    parser.add_argument("--skip_h2o", action="store_true",
                        help="Skip H2O experiments")
    parser.add_argument("--baseline_csv", type=str, default=None,
                        help="Use existing baseline CSV results")
    parser.add_argument("--h2o_csv", type=str, default=None,
                        help="Use existing H2O CSV results")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志目录
    logs_dir = OUTPUT_CONFIG.get("logs_dir", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"comparison_{timestamp}.log")
    global logger
    logger = setup_logging(log_file, level=getattr(logging, MONITORING_CONFIG.get("logging_level", "INFO")))
    
    logger.info("Starting H2O vs Baseline comparison experiment")
    
    baseline_csv = args.baseline_csv
    h2o_csv = args.h2o_csv
    
    # 运行基线实验
    if not args.skip_baseline and not baseline_csv:
        logger.info("Running baseline experiments...")
        baseline_csv = run_baseline_experiment(args, args.output_dir)
    
    # 运行H2O实验
    if not args.skip_h2o and not h2o_csv:
        logger.info("Running H2O experiments...")
        h2o_csv = run_h2o_experiment(args, args.output_dir)
    
    # 生成比较报告
    if baseline_csv or h2o_csv:
        logger.info("Generating comparison report...")
        report_path = generate_comparison_report(baseline_csv, h2o_csv, args.output_dir)
        
        if report_path:
            logger.info(f"Comparison complete. Report available at: {report_path}")
        else:
            logger.error("Failed to generate comparison report!")
    else:
        logger.error("No experiment results available for comparison!")

if __name__ == "__main__":
    main() 