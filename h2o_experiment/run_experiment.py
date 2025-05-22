#!/usr/bin/env python
"""
H2O实验命令行工具

提供简单的命令行界面启动不同类型的实验
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

# 实验类型
EXPERIMENT_TYPES = {
    "baseline": "运行标准KV缓存基线测试",
    "h2o": "运行H2O优化KV缓存测试",
    "compare": "运行基线和H2O对比实验并生成报告"
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H2O KV缓存优化实验命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 实验类型
    parser.add_argument("experiment_type", choices=EXPERIMENT_TYPES.keys(),
                        help="实验类型: baseline(基线), h2o(H2O优化), compare(对比)")
    
    # 通用参数
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果保存目录")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        help="使用的数据集")
    parser.add_argument("--language", type=str, default="english", 
                        choices=["english", "chinese"],
                        help="数据集语言")
    parser.add_argument("--kv_lengths", type=str, default="128,512,1024",
                        help="KV缓存长度列表，逗号分隔")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8",
                        help="批处理大小列表，逗号分隔")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m",
                        help="要测试的模型名称")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "fp32", "bf16", "int8"],
                        help="模型精度")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="生成的最大新token数量")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="每个配置重复实验的次数")
    
    # H2O相关参数
    parser.add_argument("--h2o_ratios", type=str, default="0.1,0.2",
                        help="H2O比率列表，逗号分隔 (仅用于H2O和compare)")
    
    # 比较实验参数
    parser.add_argument("--baseline_csv", type=str, default=None,
                        help="已有的基线结果CSV (仅用于compare)")
    parser.add_argument("--h2o_csv", type=str, default=None,
                        help="已有的H2O结果CSV (仅用于compare)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="跳过基线实验 (仅用于compare)")
    parser.add_argument("--skip_h2o", action="store_true",
                        help="跳过H2O实验 (仅用于compare)")
    
    # 实验控制
    parser.add_argument("--repeat", type=int, default=None,
                        help="重复实验次数 (覆盖配置)")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def run_experiment(args):
    """
    运行指定类型的实验
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # 格式化命令行参数
    common_args = [
        "--output_dir", args.output_dir,
        "--dataset", args.dataset,
        "--language", args.language,
    ]
    
    if args.kv_lengths:
        common_args.extend(["--kv_cache_lengths", args.kv_lengths])
    
    if args.batch_sizes:
        common_args.extend(["--batch_sizes", args.batch_sizes])
    
    # 添加模型参数
    if hasattr(args, "model_name") and args.model_name:
        common_args.extend(["--model_name", args.model_name])
    
    if hasattr(args, "precision") and args.precision:
        common_args.extend(["--precision", args.precision])
    
    if hasattr(args, "device") and args.device:
        common_args.extend(["--device", args.device])
    
    if hasattr(args, "max_new_tokens") and args.max_new_tokens:
        common_args.extend(["--max_new_tokens", str(args.max_new_tokens)])
    
    if hasattr(args, "num_repeats") and args.num_repeats:
        common_args.extend(["--num_repeats", str(args.num_repeats)])
    
    # 根据实验类型运行相应命令
    if args.experiment_type == "baseline":
        print(f"正在运行基线实验 (dataset={args.dataset}, language={args.language})...")
        cmd = ["python", "h2o_experiment/main.py"] + common_args
        
    elif args.experiment_type == "h2o":
        print(f"正在运行H2O实验 (dataset={args.dataset}, language={args.language})...")
        cmd = ["python", "h2o_experiment/h2o_main.py"] + common_args
        
        if args.h2o_ratios:
            cmd.extend(["--h2o_ratios", args.h2o_ratios])
        
    elif args.experiment_type == "compare":
        print(f"正在运行对比实验 (dataset={args.dataset}, language={args.language})...")
        cmd = ["python", "h2o_experiment/run_comparison.py"] + common_args
        
        if args.h2o_ratios:
            cmd.extend(["--h2o_ratios", args.h2o_ratios])
        
        if args.baseline_csv:
            cmd.extend(["--baseline_csv", args.baseline_csv])
        
        if args.h2o_csv:
            cmd.extend(["--h2o_csv", args.h2o_csv])
        
        if args.skip_baseline:
            cmd.append("--skip_baseline")
        
        if args.skip_h2o:
            cmd.append("--skip_h2o")
    
    # 记录命令和启动时间
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始时间: {timestamp}")
    print(f"运行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        process = subprocess.Popen(cmd)
        process.wait()
        
        if process.returncode == 0:
            print("实验成功完成!")
        else:
            print(f"实验失败，退出代码：{process.returncode}")
    
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"运行实验出错: {e}")
    
    # 记录结束时间
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"结束时间: {end_timestamp}")

def main():
    """主函数"""
    # 打印欢迎信息
    print("=" * 60)
    print("H2O KV缓存优化实验命令行工具")
    print("=" * 60)
    
    # 显示可用实验类型
    print("可用实验类型:")
    for exp_type, desc in EXPERIMENT_TYPES.items():
        print(f"  - {exp_type}: {desc}")
    print("-" * 60)
    
    # 解析命令行参数
    args = parse_args()
    
    # 运行实验
    run_experiment(args)

if __name__ == "__main__":
    main() 