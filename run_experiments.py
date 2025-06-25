#!/usr/bin/env python3
"""
统一实验运行入口脚本
按照SYSTEM_READY.md的设计规范
"""
import os
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="运行CAKE/FullKV实验")
    parser.add_argument('--experiment', choices=['cake', 'fullkv', 'comparison'], 
                       default='cake', help='实验类型')
    parser.add_argument('--tag', default='default', help='运行标签')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要运行的命令')
    
    args = parser.parse_args()
    
    # 生成运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{timestamp}_{args.tag}"
    
    print(f"实验类型: {args.experiment}")
    print(f"运行目录: {run_dir}")
    print(f"配置文件: configs/")
    print(f"第三方代码: src/third_party/")
    
    if args.dry_run:
        print("这是一个干运行，不会执行实际操作")
        return
    
    # 确保目录存在
    os.makedirs(run_dir, exist_ok=True)
    print(f"已创建运行目录: {run_dir}")
    
    # 根据实验类型调用相应的脚本
    if args.experiment == 'cake':
        print("将调用CAKE实验脚本...")
        print("模块路径: src/third_party/cakekv-main/")
    elif args.experiment == 'fullkv':
        print("将调用FullKV实验脚本...")
        print("模块路径: src/third_party/fullkv/")
    else:
        print("将运行对比实验...")
    
    print("实验配置已准备完成!")

if __name__ == "__main__":
    main() 