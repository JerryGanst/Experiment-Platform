#!/usr/bin/env python3
"""
测试配置修复和数据处理改进的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hace_core.config import validate_config, DATASET_CONFIG, EXPERIMENT_CONFIG
from hace_core.data.dataset_loader import prepare_batch, get_dataset_info, get_available_datasets

def test_config_validation():
    """测试配置验证功能"""
    print("=== 测试配置验证 ===")
    results = validate_config()
    
    if results["passed"]:
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("⚠️ 配置警告:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    return results["passed"]

def test_dataset_config_consistency():
    """测试数据集配置一致性"""
    print("\n=== 测试数据集配置一致性 ===")
    
    # 检查dataset_subset_size中的数据集是否都在available_datasets中定义
    available_datasets = set(DATASET_CONFIG["available_datasets"].keys())
    subset_size_datasets = set(EXPERIMENT_CONFIG["dataset_subset_size"].keys())
    
    undefined = subset_size_datasets - available_datasets
    if undefined:
        print(f"❌ 发现未定义的数据集: {undefined}")
        return False
    else:
        print("✅ 所有dataset_subset_size中的数据集都已正确定义")
        return True

def test_path_case_consistency():
    """测试路径大小写一致性"""
    print("\n=== 测试路径大小写一致性 ===")
    
    longbench_paths = set()
    for name, config in DATASET_CONFIG["available_datasets"].items():
        if "longbench" in config["path"].lower():
            longbench_paths.add(config["path"])
    
    if len(longbench_paths) > 1:
        print(f"❌ LongBench路径大小写不一致: {longbench_paths}")
        return False
    else:
        print("✅ LongBench路径大小写一致")
        return True

def test_prepare_batch_improvements():
    """测试prepare_batch方法的改进"""
    print("\n=== 测试prepare_batch方法改进 ===")
    
    # 模拟样本数据
    samples = [
        {"prompt": "Sample 1", "reference": "Ref 1"},
        {"prompt": "Sample 2", "reference": "Ref 2"},
        {"prompt": "Sample 3", "reference": "Ref 3"}
    ]
    
    # 模拟tokenizer
    class MockTokenizer:
        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            return {
                "input_ids": [[1, 2, 3]] * len(prompts),
                "attention_mask": [[1, 1, 1]] * len(prompts)
            }
    
    tokenizer = MockTokenizer()
    
    # 测试drop_last=True的情况
    batch_size = 4
    result = prepare_batch(samples, tokenizer, batch_size, drop_last=True)
    
    if result is None:
        print("✅ drop_last=True时，样本数不足正确返回None")
    else:
        print("❌ drop_last=True时，样本数不足应该返回None")
        return False
    
    # 测试drop_last=False的情况
    result = prepare_batch(samples, tokenizer, batch_size, drop_last=False)
    
    if result is not None and result.get("is_padded", False):
        print("✅ drop_last=False时，正确填充样本并标记is_padded")
    else:
        print("❌ drop_last=False时，应该填充样本并标记is_padded")
        return False
    
    return True

def test_dataset_info_functions():
    """测试数据集信息获取函数"""
    print("\n=== 测试数据集信息获取函数 ===")
    
    try:
        # 测试get_dataset_info
        info = get_dataset_info("mmlu")
        if info["language"] == "english" and info["name"] == "mmlu":
            print("✅ get_dataset_info正常工作")
        else:
            print("❌ get_dataset_info返回结果不正确")
            return False
        
        # 测试get_available_datasets
        datasets = get_available_datasets()
        if "english" in datasets and "chinese" in datasets:
            print("✅ get_available_datasets正常工作")
        else:
            print("❌ get_available_datasets返回结果不正确")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试配置修复和数据处理改进...\n")
    
    tests = [
        test_config_validation,
        test_dataset_config_consistency,
        test_path_case_consistency,
        test_prepare_batch_improvements,
        test_dataset_info_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！配置修复和数据处理改进成功。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)