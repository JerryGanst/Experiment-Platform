#!/usr/bin/env python3
"""测试数据集加载大小"""

import sys
import os
sys.path.append(os.getcwd())

from evaluation.baselines.fullkvcache_main import load_local_jsonl_data

def test_dataset_sizes():
    datasets = ["hotpotqa", "qasper", "multifieldqa_en"]
    
    print("=" * 60)
    print("数据集大小测试")
    print("=" * 60)
    
    for dataset_name in datasets:
        print(f"\n[测试] {dataset_name}:")
        
        # 测试完整数据集
        full_data = load_local_jsonl_data(dataset_name, max_samples=None)
        full_size = len(full_data) if full_data else 0
        
        # 测试单样本
        single_data = load_local_jsonl_data(dataset_name, max_samples=1)
        single_size = len(single_data) if single_data else 0
        
        print(f"  完整数据集: {full_size} 样本")
        print(f"  单样本测试: {single_size} 样本")
        
        if full_size > 1:
            print(f"  ✅ 完整数据集加载正常")
        else:
            print(f"  ❌ 完整数据集加载异常")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_dataset_sizes() 