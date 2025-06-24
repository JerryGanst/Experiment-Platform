#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据路径修复验证脚本
验证重构后的脚本能否正确找到数据文件
"""

import os
import sys
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_path_resolution():
    """测试数据路径解析功能"""
    print("🧪 测试数据路径解析功能")
    print("=" * 50)
    
    # 模拟fullkvcache_main.py中的路径解析函数
    def load_local_jsonl_test(dataset_name, data_dir=None):
        # 动态确定数据目录路径
        if data_dir is None:
            # 获取脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 尝试多个可能的数据路径
            possible_data_dirs = [
                os.path.join(script_dir, "hace-kv-optimization", "run_result", "data"),  # run_result/data/
                os.path.join(script_dir, "data"),  # 项目根目录的data/
                os.path.join(script_dir, "hace-kv-optimization", "data"),  # hace-kv-optimization/data/
            ]
            
            data_dir = None
            for possible_dir in possible_data_dirs:
                test_file = os.path.join(possible_dir, f"{dataset_name}.jsonl")
                if os.path.exists(test_file):
                    data_dir = possible_dir
                    logger.info(f"✅ 找到数据文件: {test_file}")
                    return test_file
            
            if data_dir is None:
                logger.warning(f"❌ 在以下路径中未找到 {dataset_name}.jsonl:")
                for path in possible_data_dirs:
                    logger.warning(f"  - {os.path.join(path, f'{dataset_name}.jsonl')}")
                return None
        
        file_path = os.path.join(data_dir, f"{dataset_name}.jsonl")
        return file_path if os.path.exists(file_path) else None
    
    # 测试常见数据集
    test_datasets = ["hotpotqa", "mmlu", "narrativeqa"]
    
    for dataset in test_datasets:
        print(f"\n📊 测试数据集: {dataset}")
        found_path = load_local_jsonl_test(dataset)
        
        if found_path:
            print(f"  ✅ 成功找到: {found_path}")
            
            # 验证文件格式
            try:
                with open(found_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if first_line.strip():
                        sample = json.loads(first_line)
                        print(f"  📋 样本字段: {list(sample.keys())}")
                        print(f"  📏 文件大小: {os.path.getsize(found_path) / (1024*1024):.1f}MB")
                    else:
                        print("  ⚠️  文件为空")
            except Exception as e:
                print(f"  ❌ 文件读取失败: {e}")
        else:
            print(f"  ❌ 未找到数据文件")

def verify_path_manager_integration():
    """验证与PathManager的集成"""
    print("\n🔧 验证PathManager集成")
    print("=" * 50)
    
    try:
        # 检查是否存在utilities目录和path_config模块
        utilities_path = Path("utilities")
        if utilities_path.exists():
            sys.path.insert(0, str(utilities_path.parent))
            from utilities.path_config import PathManager
            
            pm = PathManager()
            print("✅ PathManager初始化成功")
            
            # 获取配置信息
            if "data_paths" in pm.paths:
                data_config = pm.paths["data_paths"]
                print(f"📁 配置的数据路径:")
                for key, path in data_config.items():
                    print(f"  - {key}: {path}")
                    if Path(path).exists():
                        print(f"    ✅ 路径存在")
                    else:
                        print(f"    ❌ 路径不存在")
            
            # 获取脚本路径
            try:
                baseline_script = pm.get_script_path("baseline")
                print(f"📜 基线脚本路径: {baseline_script}")
                if os.path.exists(baseline_script):
                    print("  ✅ 脚本存在")
                else:
                    print("  ❌ 脚本不存在")
            except Exception as e:
                print(f"  ❌ 获取脚本路径失败: {e}")
                
        else:
            print("❌ utilities目录不存在，跳过PathManager测试")
            
    except ImportError as e:
        print(f"❌ PathManager导入失败: {e}")
    except Exception as e:
        print(f"❌ PathManager测试失败: {e}")

def show_current_project_structure():
    """显示当前项目结构"""
    print("\n📂 当前项目结构")
    print("=" * 50)
    
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    
    # 显示关键目录和文件
    key_paths = [
        "data/",
        "hace-kv-optimization/",
        "hace-kv-optimization/run_result/",
        "hace-kv-optimization/run_result/data/",
        "hace-kv-optimization/run_result/fullkvcache_main.py",
        "hace-kv-optimization/run_result/cake_main.py",
        "utilities/",
        "path_config.json"
    ]
    
    for path_str in key_paths:
        path = current_dir / path_str
        if path.exists():
            if path.is_dir():
                items = list(path.iterdir())
                print(f"✅ {path_str} ({len(items)} 项)")
                # 显示数据文件
                if "data" in path_str:
                    jsonl_files = [f.name for f in items if f.suffix == '.jsonl']
                    if jsonl_files:
                        print(f"   📊 JSONL文件: {', '.join(jsonl_files[:3])}{'...' if len(jsonl_files) > 3 else ''}")
            else:
                size = path.stat().st_size
                print(f"✅ {path_str} ({size/1024:.1f}KB)")
        else:
            print(f"❌ {path_str} (不存在)")

if __name__ == "__main__":
    print("🔍 数据路径修复验证")
    print("=" * 60)
    
    show_current_project_structure()
    test_data_path_resolution()
    verify_path_manager_integration()
    
    print("\n✅ 验证完成！")
    print("\n💡 使用建议:")
    print("1. 如果找到了数据文件，说明路径修复成功")
    print("2. 如果未找到，请检查数据文件是否存在于正确位置")
    print("3. 可以运行 fullkvcache_main.py 测试实际效果") 