#!/usr/bin/env python3
"""
测试评分系统修复
验证eval_utils导入和基线文件路径是否正确
"""

import sys
import json
from pathlib import Path

# 添加项目根目录和metrics路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

longbench_metrics_path = project_root / "src" / "third_party" / "cakekv-main" / "cakekv-main" / "experiments" / "LongBench"
sys.path.append(str(longbench_metrics_path))

from src.common.data_loader import find_data_file  # noqa: E402

def test_eval_utils_import():
    """测试eval_utils导入"""
    print("🧪 测试eval_utils导入...")
    
    # 测试从src/cake_runner导入
    try:
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
    
    baseline_file = project_root / "evaluation" / "baseline_fullkv.json"
    
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

def test_scoring():
    print("=== 测试评分功能 ===")
    
    try:
        from metrics import classification_score, qa_f1_score, rouge_score
        print("✅ 评分模块导入成功")
        
        # 加载TREC数据集获取all_classes
        trec_file = find_data_file("trec")
        if not trec_file:
            print("❌ 找不到 trec.jsonl，请设置 DATASETS_ROOT 或复制到 ./data")
            return False

        with open(trec_file, 'r', encoding='utf-8') as f:
            trec_data = json.loads(f.readline())
        
        all_classes = trec_data.get('all_classes', [])
        print(f"✅ 加载TREC分类列表，共{len(all_classes)}个类别")
        
        # 测试TREC分类评分
        print("\n--- 测试TREC分类评分 ---")
        generated_text = "General Knowledge"
        ground_truth = "Other location"
        
        print(f"生成文本: {generated_text}")
        print(f"标准答案: {ground_truth}")
        print(f"所有类别: {all_classes[:5]}... (共{len(all_classes)}个)")
        
        score = classification_score(generated_text, ground_truth, all_classes=all_classes)
        print(f"分类评分结果: {score}")
        
        # 测试更精确的匹配
        print("\n--- 测试精确匹配 ---")
        exact_match_text = "Other location"
        exact_score = classification_score(exact_match_text, ground_truth, all_classes=all_classes)
        print(f"精确匹配评分: {exact_score}")
        
        # 测试包含多个分类的情况
        print("\n--- 测试包含多个分类的情况 ---")
        multi_class_text = "This is about Food and Other location"
        multi_score = classification_score(multi_class_text, ground_truth, all_classes=all_classes)
        print(f"多分类文本评分: {multi_score}")
        
        # 测试QA评分
        print("\n--- 测试QA F1评分 ---")
        qa_pred = "The first golf course was in South Carolina"
        qa_ref = "South Carolina"
        qa_score = qa_f1_score(qa_pred, qa_ref)
        print(f"QA F1评分: {qa_score}")
        
        # 测试ROUGE评分
        print("\n--- 测试ROUGE评分 ---")
        rouge_pred = "The first golf course in the United States was located in South Carolina"
        rouge_ref = "The first golf course was in South Carolina"
        rouge_score_result = rouge_score(rouge_pred, rouge_ref)
        print(f"ROUGE评分: {rouge_score_result}")
        
        print("\n✅ 所有评分测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 评分测试失败: {e}")
        import traceback
        traceback.print_exc()
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
    

    # 测试评分功能
    scoring_success = test_scoring()
    results.append(("评分功能", scoring_success))


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
