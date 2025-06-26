#!/usr/bin/env python3
"""
测试评分系统修复
验证eval_utils导入和基线文件路径是否正确
"""

import sys
from pathlib import Path

def test_eval_utils_import():
    """测试eval_utils导入"""
    print("🧪 测试eval_utils导入...")
    
    # 测试从src/cake_runner导入
    try:
        project_root = Path(__file__).parent
        eval_utils_path = project_root / "src" / "cake_runner"
        if str(eval_utils_path) not in sys.path:
            sys.path.append(str(eval_utils_path))
        
        from eval_utils import score_dataset, load_baseline, save_baseline
        print("✅ 从src/cake_runner成功导入eval_utils")
        
        # 测试基线文件路径
        baseline = load_baseline()
        print(f"✅ 成功加载基线文件: {baseline}")
        
        return True, "src/cake_runner"
    except Exception as e:
        print(f"❌ 从src/cake_runner导入失败: {e}")
        
        # 测试从evaluation目录导入
        try:
            sys.path.insert(0, str(project_root / "evaluation"))
            from eval_utils import score_dataset, load_baseline, save_baseline
            print("✅ 从evaluation目录成功导入eval_utils")
            
            baseline = load_baseline()
            print(f"✅ 成功加载基线文件: {baseline}")
            
            return True, "evaluation"
        except Exception as e2:
            print(f"❌ 从evaluation目录导入也失败: {e2}")
            return False, None

def test_baseline_file():
    """测试基线文件内容"""
    print("\n🧪 测试基线文件内容...")
    
    baseline_file = Path(__file__).parent / "evaluation" / "baseline_fullkv.json"
    
    if not baseline_file.exists():
        print(f"❌ 基线文件不存在: {baseline_file}")
        return False
    
    import json
    try:
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        
        print(f"✅ 基线文件存在且可读: {baseline_file}")
        print(f"📊 基线内容:")
        for dataset, score in baseline.items():
            if not dataset.startswith('_'):
                print(f"   {dataset}: {score}")
        
        # 检查是否有非零值
        non_zero_scores = [v for k, v in baseline.items() 
                          if not k.startswith('_') and isinstance(v, (int, float)) and v > 0]
        
        if non_zero_scores:
            print(f"✅ 找到 {len(non_zero_scores)} 个非零基线分数")
            return True
        else:
            print("⚠️ 所有基线分数都为0，可能导致除零错误")
            return False
            
    except Exception as e:
        print(f"❌ 读取基线文件失败: {e}")
        return False

def test_relative_scoring():
    """测试相对评分计算"""
    print("\n🧪 测试相对评分计算...")
    
    try:
        # 先确保能导入
        project_root = Path(__file__).parent
        eval_utils_path = project_root / "src" / "cake_runner"
        if str(eval_utils_path) not in sys.path:
            sys.path.append(str(eval_utils_path))
        
        from eval_utils import calculate_relative_score
        
        # 测试不同情况的相对评分
        test_cases = [
            ("hotpotqa", 0.6527, False),  # 100%基线
            ("hotpotqa", 0.3264, False),  # 50%基线
            ("multi_news", 0.2845, False),  # 100%基线
            ("multi_news", 0.1423, False),  # 50%基线
        ]
        
        for dataset, raw_score, is_full_kv in test_cases:
            result = calculate_relative_score(dataset, raw_score, is_full_kv)
            print(f"   {dataset}: {raw_score:.4f} → {result['relative_score']:.2f}%")
        
        print("✅ 相对评分计算正常")
        return True
        
    except Exception as e:
        print(f"❌ 相对评分计算失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 评分系统修复验证测试")
    print("=" * 50)
    
    results = []
    
    # 测试导入
    import_success, import_source = test_eval_utils_import()
    results.append(("eval_utils导入", import_success))
    
    # 测试基线文件
    baseline_success = test_baseline_file()
    results.append(("基线文件", baseline_success))
    
    # 测试相对评分
    if import_success:
        scoring_success = test_relative_scoring()
        results.append(("相对评分", scoring_success))
    
    # 汇总结果
    print("\n📊 测试结果汇总:")
    print("=" * 50)
    all_passed = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！评分系统修复成功")
        if import_success:
            print(f"💡 建议使用: {import_source} 版本的eval_utils")
    else:
        print("\n⚠️ 部分测试失败，需要进一步修复")
    
    return all_passed

if __name__ == "__main__":
    main() 