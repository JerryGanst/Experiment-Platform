#!/usr/bin/env python
"""
实验命令行工具

提供简单的命令行界面启动不同类型的实验 (Baseline, H2O, CAKE)
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

# 导入项目根目录的config，以便脚本可以从任何位置运行
current_script_path = os.path.abspath(__file__)
pkg_dir = os.path.dirname(current_script_path)  # hace-kv-optimization dir
project_root = os.path.dirname(pkg_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hace_core import config as global_config  # 项目根目录的config

# 实验类型
EXPERIMENT_TYPES = {
    "baseline": "运行标准KV缓存基线测试",
    "cake": "运行CAKE层级感知KV缓存测试",
    "compare_cake_baseline": "运行基线和CAKE对比实验并生成报告",
    "compare_all": "运行基线和CAKE对比实验并生成报告"
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="KV缓存优化实验命令行工具 (支持Baseline, H2O, CAKE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例用法:\n"
               "  python hace-kv-optimization/experiments/run_experiment.py baseline --datasets mmlu --kv_lengths 1024 --batch_sizes 1\n"
               "  python hace-kv-optimization/experiments/run_experiment.py h2o --datasets mmlu --h2o_ratios 0.1,0.2\n"
               "  python hace-kv-optimization/experiments/run_experiment.py cake --datasets mmlu --allocation_strategies adaptive --cache_budgets 0.7\n"
               "  python hace-kv-optimization/experiments/run_experiment.py compare_all --datasets mmlu,gsm8k --run_name full_comparison_run"
    )
    
    parser.add_argument("experiment_type", choices=EXPERIMENT_TYPES.keys(),
                        help="要运行的实验类型。")
    
    # 通用参数
    default_output_dir = os.path.join(global_config.EXPERIMENT_CONFIG.get("output_base_dir", "results"), "cli_runs")
    parser.add_argument("--output_dir_base", type=str, default=default_output_dir,
                        help=f"保存所有实验运行结果的基础目录。默认: {default_output_dir}")
    parser.add_argument("--run_name", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="为本次运行指定一个名称，结果将保存在 <output_dir_base>/<run_name> 下。")
    parser.add_argument("--datasets", type=str, default=",".join(global_config.EXPERIMENT_CONFIG["datasets"]), 
                        help="逗号分隔的数据集列表。")
    # kv_lengths 和 batch_sizes 现在从各自的 main 脚本的默认值或config.py中获取
    # 保留这些参数用于在CLI覆盖，如果用户想为特定运行指定不同值
    parser.add_argument("--kv_lengths", type=str, default=None,
                        help="逗号分隔的KV缓存长度列表 (覆盖默认值)。")
    parser.add_argument("--batch_sizes", type=str, default=None,
                        help="逗号分隔的批处理大小列表 (覆盖默认值)。")
    
    # 模型参数 (大多从config.py获取默认值)
    parser.add_argument("--model_name", type=str, default=global_config.EXPERIMENT_CONFIG["model_name_or_path"],
                        help="要测试的模型名称或路径。")
    parser.add_argument("--precision", type=str, default=global_config.EXPERIMENT_CONFIG["precision"],
                        choices=["fp16", "fp32", "bf16"], help="模型精度。")
    parser.add_argument("--max_new_tokens", type=int, default=global_config.EXPERIMENT_CONFIG["max_new_tokens"],
                        help="生成的最大新token数量。")
    parser.add_argument("--repetitions", type=int, default=global_config.EXPERIMENT_CONFIG["repetitions"],
                        help="每个配置重复实验的次数。")
    parser.add_argument("--seed", type=int, default=global_config.EXPERIMENT_CONFIG.get("random_seed", 42),
                        help="用于可复现性的随机种子。")
    
    # H2O相关参数
    parser.add_argument("--h2o_ratios", type=str, default=",".join(map(str, global_config.EXPERIMENT_CONFIG["h2o_ratios"]) if "h2o_ratios" in global_config.EXPERIMENT_CONFIG else []),
                        help="H2O的heavy-hitter和recent比率列表，逗号分隔 (仅用于H2O和compare类实验)。")
    parser.add_argument("--eviction_strategies", type=str, default=",".join(global_config.EXPERIMENT_CONFIG["eviction_strategies"]) if "eviction_strategies" in global_config.EXPERIMENT_CONFIG else [],
                        help="H2O的驱逐策略列表 (仅用于H2O和compare类实验)。")

    # CAKE相关参数 (新增)
    parser.add_argument("--allocation_strategies", type=str, default=",".join(global_config.EXPERIMENT_CONFIG["layer_allocation_strategies"]) if "layer_allocation_strategies" in global_config.EXPERIMENT_CONFIG else [],
                        help="CAKE的层分配策略列表 (仅用于CAKE和compare类实验)。")
    parser.add_argument("--cache_budgets", type=str, default=",".join(map(str, global_config.EXPERIMENT_CONFIG["cache_budgets"]) if "cache_budgets" in global_config.EXPERIMENT_CONFIG else []),
                        help="CAKE的缓存预算比例列表 (仅用于CAKE和compare类实验)。")
    
    # 对比实验参数 (通用化)
    parser.add_argument("--baseline_results_file", type=str, default=None,
                        help="可选的预计算基线结果CSV文件路径。")
    parser.add_argument("--h2o_results_file", type=str, default=None,
                        help="可选的预计算H2O结果CSV文件路径 (用于compare_h2o_baseline, compare_all)。")
    parser.add_argument("--cake_results_file", type=str, default=None,
                        help="可选的预计算CAKE结果CSV文件路径 (用于compare_cake_baseline, compare_all)。") # 新增
    
    parser.add_argument("--skip_run_baseline", action="store_true", help="在对比实验中跳过运行新的基线实验 (如果提供了baseline_results_file则自动跳过)。")
    parser.add_argument("--skip_run_h2o", action="store_true", help="在对比实验中跳过运行新的H2O实验 (如果提供了h2o_results_file则自动跳过)。")
    parser.add_argument("--skip_run_cake", action="store_true", help="在对比实验中跳过运行新的CAKE实验 (如果提供了cake_results_file则自动跳过)。") # 新增

    parser.add_argument("--log_level", type=str, default=global_config.MONITORING_CONFIG.get("logging_level", "INFO"), 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="日志级别设置")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def run_single_experiment_script(script_name:str, common_params:list, specific_params:list, run_output_dir:str, log_level:str, run_env:dict):
    """辅助函数运行单个实验脚本 (如 h2o_main.py, cake_main.py)"""
    cmd = [sys.executable, os.path.join(pkg_dir, 'baselines', script_name)] + \
          ["--output_dir", run_output_dir, "--log_level", log_level] + \
          common_params + specific_params
    
    main_script_name = script_name.split('.')[0]
    log_file = os.path.join(run_output_dir, f"log_{main_script_name}.txt")
    
    print(f"\n--- 正在运行 {main_script_name} ---")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")
    
    try:
        with open(log_file, 'w') as lf:
            process = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=run_env)
            process.wait()
        if process.returncode == 0:
            print(f"{main_script_name} 成功完成!")
            # 查找生成的CSV文件，通常是 all_*_experiments_summary.csv
            summary_csv = [f for f in os.listdir(run_output_dir) if f.startswith("all_") and f.endswith("_summary.csv")]
            if summary_csv:
                return os.path.join(run_output_dir, summary_csv[0])
            else:
                print(f"警告: 未在 {run_output_dir} 中找到 {main_script_name} 的汇总CSV文件。")
                return None #表示运行了但没找到标准输出文件
        else:
            print(f"{main_script_name} 失败，退出代码：{process.returncode}。详情请查看日志: {log_file}")
            return False # 表示运行失败
    except KeyboardInterrupt:
        print(f"\n{main_script_name} 被用户中断")
        return False
    except Exception as e:
        print(f"运行 {main_script_name} 出错: {e}")
        return False

def run_comparison_script(common_params:list, comparison_specific_params:list, run_output_dir:str, log_level:str, run_env:dict):
    """辅助函数运行对比脚本 (run_comparison.py)"""
    cmd = [sys.executable, os.path.join(pkg_dir, 'experiments', "run_comparison.py")] + \
          ["--output_main_dir", run_output_dir, "--log_level", log_level] + \
          common_params + comparison_specific_params

    log_file = os.path.join(run_output_dir, "log_comparison.txt")
    print("\n--- 正在运行对比分析脚本 ---")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")
    try:
        with open(log_file, 'w') as lf:
            process = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=run_env)
            process.wait()
        if process.returncode == 0:
            print("对比分析成功完成!")
            return True
        else:
            print(f"对比分析失败，退出代码：{process.returncode}。详情请查看日志: {log_file}")
            return False
    except KeyboardInterrupt:
        print("\n对比分析被用户中断")
        return False
    except Exception as e:
        print(f"运行对比分析出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print(" KV Cache 优化实验命令行工具 (Baseline, H2O, CAKE)")
    print("=" * 70)
    
    args = parse_args()
    
    # 创建本次运行的特定输出目录
    # <output_dir_base>/<run_name>
    #   ﹂ baseline_run/
    #   ﹂ h2o_run/
    #   ﹂ cake_run/
    #   ﹂ comparison_report/
    current_run_main_output_dir = os.path.join(args.output_dir_base, args.run_name)
    os.makedirs(current_run_main_output_dir, exist_ok=True)
    print(f"所有此运行 ('{args.run_name}') 的结果将保存在: {current_run_main_output_dir}")

    # 设置PYTHONPATH环境变量，确保子进程能正确找到模块
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = pkg_dir + os.pathsep + run_env.get("PYTHONPATH", "")

    # 通用参数构造
    common_params = [
        "--model_name", args.model_name,
        "--precision", args.precision,
        "--max_new_tokens", str(args.max_new_tokens),
        "--repetitions", str(args.repetitions),
        "--seed", str(args.seed),
        "--datasets", args.datasets # Pass datasets to sub-scripts
    ]
    if args.kv_lengths: # 如果在CLI中指定，则覆盖默认值
        common_params.extend(["--kv_cache_lengths", args.kv_lengths])
    if args.batch_sizes:
        common_params.extend(["--batch_sizes", args.batch_sizes])

    # H2O特定参数
    h2o_params = []
    if args.h2o_ratios: h2o_params.extend(["--h2o_ratios", args.h2o_ratios])
    if args.eviction_strategies: h2o_params.extend(["--eviction_strategies", args.eviction_strategies])

    # CAKE特定参数
    cake_params = []
    if args.allocation_strategies: cake_params.extend(["--allocation_strategies", args.allocation_strategies])
    if args.cache_budgets: cake_params.extend(["--cache_budgets", args.cache_budgets])

    # --- 实验执行逻辑 ---
    baseline_csv_path = args.baseline_results_file
    h2o_csv_path = args.h2o_results_file
    cake_csv_path = args.cake_results_file

    # 运行基线实验
    if args.experiment_type == "baseline" or \
       (args.experiment_type in ["compare_h2o_baseline", "compare_cake_baseline", "compare_all"] and not args.skip_run_baseline and not baseline_csv_path):
        baseline_output_dir = os.path.join(current_run_main_output_dir, "baseline_run")
        os.makedirs(baseline_output_dir, exist_ok=True)
        # 运行基线实验
        run_result = run_single_experiment_script("baseline_main.py", common_params, [], baseline_output_dir, args.log_level, run_env)
        if isinstance(run_result, str): baseline_csv_path = run_result
        elif run_result is False and args.experiment_type != "baseline": # 如果在对比流程中运行失败，则中止
            print("基线实验失败，对比实验无法继续。")
            sys.exit(1)
    
    # 运行H2O实验
    if args.experiment_type == "h2o" or \
       (args.experiment_type in ["compare_h2o_baseline", "compare_all"] and not args.skip_run_h2o and not h2o_csv_path):
        h2o_output_dir = os.path.join(current_run_main_output_dir, "h2o_run")
        os.makedirs(h2o_output_dir, exist_ok=True)
        run_result = run_single_experiment_script("h2o_main.py", common_params, h2o_params, h2o_output_dir, args.log_level, run_env)
        if isinstance(run_result, str): h2o_csv_path = run_result
        elif run_result is False and args.experiment_type != "h2o":
            print("H2O实验失败，对比实验无法继续。")
            sys.exit(1)

    # 运行CAKE实验 (新增)
    if args.experiment_type == "cake" or \
       (args.experiment_type in ["compare_cake_baseline", "compare_all"] and not args.skip_run_cake and not cake_csv_path):
        cake_output_dir = os.path.join(current_run_main_output_dir, "cake_run")
        os.makedirs(cake_output_dir, exist_ok=True)
        run_result = run_single_experiment_script("cake_main.py", common_params, cake_params, cake_output_dir, args.log_level, run_env)
        if isinstance(run_result, str): cake_csv_path = run_result
        elif run_result is False and args.experiment_type != "cake":
            print("CAKE实验失败，对比实验无法继续。")
            sys.exit(1)
    
    # 运行对比分析
    if args.experiment_type.startswith("compare"):
        comparison_output_dir = os.path.join(current_run_main_output_dir, "comparison_report")
        os.makedirs(comparison_output_dir, exist_ok=True)
        
        comparison_params = ["--experiment_type", args.experiment_type]
        if baseline_csv_path: comparison_params.extend(["--baseline_csv", baseline_csv_path])
        if h2o_csv_path and args.experiment_type in ["compare_h2o_baseline", "compare_all"]:
             comparison_params.extend(["--h2o_csv", h2o_csv_path])
        if cake_csv_path and args.experiment_type in ["compare_cake_baseline", "compare_all"]:
             comparison_params.extend(["--cake_csv", cake_csv_path])
        
        # 将H2O和CAKE的参数也传递给对比脚本，以防它需要这些信息来解释结果或生成更详细的报告
        # (虽然当前run_comparison.py可能不直接使用它们，但为了未来扩展性)
        all_method_params = list(set(h2o_params + cake_params)) # 去重合并
        
        run_comparison_script(common_params + all_method_params, comparison_params, comparison_output_dir, args.log_level, run_env)

    print("\n" + "=" * 70)
    print(f"所有实验和/或分析已完成。结果位于: {current_run_main_output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main() 