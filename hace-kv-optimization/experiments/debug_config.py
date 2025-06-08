#!/usr/bin/env python3
"""
配置诊断脚本 - 检查DATASET_CONFIG是否正确加载
"""

import sys
import os
import json

# 添加路径
current_file_path = os.path.abspath(__file__)
project_dir = os.path.dirname(current_file_path)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

print("=== 配置诊断脚本 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"项目目录: {project_dir}")
print(f"Python路径: {sys.path[:3]}...")

try:
    # 导入配置
    from hace_core import config
    print("\n✅ config模块导入成功")
    
    # 检查DATASET_CONFIG
    DATASET_CONFIG = config.DATASET_CONFIG
    print(f"\n📋 DATASET_CONFIG类型: {type(DATASET_CONFIG)}")
    print(f"📋 DATASET_CONFIG顶级键: {list(DATASET_CONFIG.keys())}")
    
    # 检查available_datasets
    if "available_datasets" in DATASET_CONFIG:
        available_datasets = DATASET_CONFIG["available_datasets"]
        print(f"\n✅ available_datasets键存在")
        print(f"📋 available_datasets类型: {type(available_datasets)}")
        print(f"📋 available_datasets包含的数据集: {list(available_datasets.keys())}")
        
        # 检查mmlu配置
        if "mmlu" in available_datasets:
            mmlu_config = available_datasets["mmlu"]
            print(f"\n✅ mmlu配置存在: {mmlu_config}")
        else:
            print(f"\n❌ mmlu配置不存在")
            print(f"可用数据集: {list(available_datasets.keys())}")
    else:
        print(f"\n❌ available_datasets键不存在")
        print(f"当前顶级键: {list(DATASET_CONFIG.keys())}")
    
    # 模拟数据集配置获取过程
    print(f"\n🔍 模拟数据集配置获取:")
    dataset_name = "mmlu"
    dataset_config = DATASET_CONFIG.get("available_datasets", {}).get(dataset_name)
    print(f"DATASET_CONFIG.get('available_datasets', {{}}) = {DATASET_CONFIG.get('available_datasets', {})}")
    print(f"最终获取的配置: {dataset_config}")
    
    if dataset_config:
        print("✅ 数据集配置获取成功")
    else:
        print("❌ 数据集配置获取失败")
        
except ImportError as e:
    print(f"❌ config模块导入失败: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 诊断完成 ===") 