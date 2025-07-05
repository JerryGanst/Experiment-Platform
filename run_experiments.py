#!/usr/bin/env python3
"""
统一实验运行入口脚本

支持的实验类型：
- baseline: 运行基线实验 (evaluation/baselines/)
- cake: 运行CAKE优化实验 (src/methods/cake/)
- h2o: 运行H2O优化实验 (src/methods/h2o/)
- comparison: 运行对比分析 (evaluation/experiments/)
- full_evaluation: 运行完整评估流程

按照重构后的目录结构设计
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent

def create_run_directory(tag):
    """创建运行目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_project_root() / "runs" / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def run_baseline_experiment(run_dir, args):
    """运行基线实验"""
    print("🔵 运行基线实验...")
    
    baseline_script = get_project_root() / "evaluation" / "baselines" / "fullkvcache_main.py"
    if not baseline_script.exists():
        print(f"❌ 基线脚本不存在: {baseline_script}")
        return False
    
    cmd = [
        sys.executable, str(baseline_script),
        "--output_dir", str(run_dir / "baseline_results"),
        "--datasets", args.datasets,
        "--kv_cache_lengths", args.kv_lengths,
        "--batch_sizes", args.batch_sizes
    ]
    
    if args.dry_run:
        print(f"将执行命令: {' '.join(cmd)}")
        return True
    
    # 设置环境变量以确保Python模块能够正确导入
    env = os.environ.copy()
    env['PYTHONPATH'] = str(get_project_root()) + (os.pathsep + env.get('PYTHONPATH', ''))
    
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode == 0

def run_cake_experiment(run_dir, args):
    """运行CAKE实验"""
    print("🟢 运行CAKE实验...")
    
    cake_script = get_project_root() / "src" / "methods" / "cake" / "cake_main.py"
    if not cake_script.exists():
        print(f"❌ CAKE脚本不存在: {cake_script}")
        return False
    
    cmd = [
        sys.executable, str(cake_script),
        "--output_dir", str(run_dir / "cake_results"),
        "--datasets", args.datasets,
        "--allocation_strategies", args.allocation_strategies,
        "--cache_budgets", args.cache_budgets
    ]
    
    if args.dry_run:
        print(f"将执行命令: {' '.join(cmd)}")
        return True
    
    # 设置环境变量以确保Python模块能够正确导入
    env = os.environ.copy()
    env['PYTHONPATH'] = str(get_project_root()) + (os.pathsep + env.get('PYTHONPATH', ''))
    
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode == 0

def run_h2o_experiment(run_dir, args):
    """运行H2O实验 (目前为简化版占位实现)"""
    print("🟡 运行H2O实验...")

    h2o_script = get_project_root() / "src" / "methods" / "h2o" / "h2o_main.py"
    if not h2o_script.exists():
        print(f"❌ H2O脚本不存在: {h2o_script}")
        return False

    cmd = [
        sys.executable, str(h2o_script),
        "--output_dir", str(run_dir / "h2o_results"),
        "--datasets", args.datasets,
        "--cache_budgets", args.cache_budgets,
    ]

    if args.dry_run:
        print(f"将执行命令: {' '.join(cmd)}")
        return True

    env = os.environ.copy()
    env['PYTHONPATH'] = str(get_project_root()) + (os.pathsep + env.get('PYTHONPATH', ''))

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode == 0

def run_comparison_analysis(run_dir, args):
    """运行对比分析"""
    print("📊 运行对比分析...")
    
    comparison_script = get_project_root() / "evaluation" / "experiments" / "run_comparison.py"
    if not comparison_script.exists():
        print(f"❌ 对比脚本不存在: {comparison_script}")
        return False
    
    # 查找结果文件
    baseline_csv = run_dir / "baseline_results" / "results.csv"
    cake_csv = run_dir / "cake_results" / "results.csv"
    
    cmd = [
        sys.executable, str(comparison_script),
        "--experiment_type", "compare_cake_baseline",
        "--output_main_dir", str(run_dir / "comparison_report")
    ]
    
    if baseline_csv.exists():
        cmd.extend(["--baseline_csv", str(baseline_csv)])
    if cake_csv.exists():
        cmd.extend(["--cake_csv", str(cake_csv)])
    
    if args.dry_run:
        print(f"将执行命令: {' '.join(cmd)}")
        return True
    
    # 设置环境变量以确保Python模块能够正确导入
    env = os.environ.copy()
    env['PYTHONPATH'] = str(get_project_root()) + (os.pathsep + env.get('PYTHONPATH', ''))
    
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode == 0

def run_full_evaluation(run_dir, args):
    """运行完整评估流程"""
    print("🎯 运行完整评估流程...")
    
    success = True
    
    # 1. 运行基线实验
    if not run_baseline_experiment(run_dir, args):
        print("❌ 基线实验失败")
        success = False
    
    # 2. 运行CAKE实验
    if not run_cake_experiment(run_dir, args):
        print("❌ CAKE实验失败")
        success = False
    
    # 3. 运行对比分析
    if success and not run_comparison_analysis(run_dir, args):
        print("❌ 对比分析失败")
        success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="统一实验运行入口")
    
    # 实验类型
    parser.add_argument('--experiment', 
                       choices=['baseline', 'cake', 'h2o', 'comparison', 'full_evaluation'], 
                       default='full_evaluation', 
                       help='实验类型')
    
    # 运行配置
    parser.add_argument('--tag', default='default', help='运行标签')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要运行的命令')
    
    # 实验参数
    parser.add_argument('--datasets', default='hotpotqa,multi_news', 
                       help='数据集列表，逗号分隔')
    parser.add_argument('--kv_lengths', default='128,1024', 
                       help='KV缓存长度列表，逗号分隔 (推荐: 128,1024)')
    parser.add_argument('--batch_sizes', default='1', 
                       help='批处理大小列表，逗号分隔')
    parser.add_argument('--allocation_strategies', default='adaptive', 
                       help='CAKE分配策略')
    parser.add_argument('--cache_budgets', default='0.7', 
                       help='CAKE缓存预算')
    
    args = parser.parse_args()
    
    print("🚀 统一实验运行器")
    print(f"实验类型: {args.experiment}")
    print(f"数据集: {args.datasets}")
    print(f"项目根目录: {get_project_root()}")
    
    # 创建运行目录
    run_dir = create_run_directory(args.tag)
    print(f"运行目录: {run_dir}")
    
    if args.dry_run:
        print("🔍 这是一个干运行，不会执行实际操作")
    
    # 执行相应的实验
    success = False
    
    if args.experiment == 'baseline':
        success = run_baseline_experiment(run_dir, args)
    elif args.experiment == 'cake':
        success = run_cake_experiment(run_dir, args)
    elif args.experiment == 'h2o':
        success = run_h2o_experiment(run_dir, args)
    elif args.experiment == 'comparison':
        success = run_comparison_analysis(run_dir, args)
    elif args.experiment == 'full_evaluation':
        success = run_full_evaluation(run_dir, args)
    
    if success:
        print("✅ 实验完成!")
        print(f"📁 结果保存在: {run_dir}")
    else:
        print("❌ 实验失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 