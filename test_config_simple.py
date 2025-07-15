#!/usr/bin/env python3
"""
简化的配置测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_validation():
    """测试配置验证功能"""
    print("=== 测试配置验证 ===")
    
    # 直接导入配置
    from hace_core.config import validate_config, DATASET_CONFIG, EXPERIMENT_CONFIG
    
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
    
    from hace_core.config import DATASET_CONFIG, EXPERIMENT_CONFIG
    
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
    
    from hace_core.config import DATASET_CONFIG
    
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

def test_config_structure():
    """测试配置结构"""
    print("\n=== 测试配置结构 ===")
    
    from hace_core.config import DATASET_CONFIG
    
    # 检查所有数据集都有language字段
    missing_language = []
    for name, config in DATASET_CONFIG["available_datasets"].items():
        if "language" not in config:
            missing_language.append(name)
    
    if missing_language:
        print(f"❌ 以下数据集缺少language字段: {missing_language}")
        return False
    else:
        print("✅ 所有数据集都有language字段")
        return True

def main():
    """主测试函数"""
    print("开始测试配置修复...\n")
    
    tests = [
        test_config_validation,
        test_dataset_config_consistency,
        test_path_case_consistency,
        test_config_structure
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
        print("🎉 所有测试通过！配置修复成功。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)