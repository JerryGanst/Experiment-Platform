#!/usr/bin/env python3
"""
测试CoreCode改进效果的脚本
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from test_corecode_hotpotqa_fixed import CoreCodeEvaluator

def test_improvements():
    """测试改进效果"""
    print("="*60)
    print("🧪 测试CoreCode改进效果")
    print("="*60)
    
    # 创建评估器
    evaluator = CoreCodeEvaluator(cache_budget=0.7)
    
    # 测试F1分数计算改进
    print("\n📊 测试F1分数计算:")
    test_cases = [
        ("Paris", "Paris", "完全匹配"),
        ("The capital is Paris", "Paris", "包含答案"),
        ("Paris is the capital of France", "The capital of France is Paris", "语序不同"),
        ("It's Paris, the beautiful city", "Paris", "包含额外信息"),
        ("London", "Paris", "完全不匹配"),
    ]
    
    for pred, truth, desc in test_cases:
        f1 = evaluator.calculate_f1_score(pred, truth)
        print(f"  {desc}: F1={f1:.3f} (预测:'{pred}' vs 真实:'{truth}')")
    
    # 测试答案规范化
    print("\n🔧 测试答案规范化:")
    test_texts = [
        "The answer is Paris.",
        "Answer: Paris",
        "Based on the context, Paris",
        "  Paris  ",
        "PARIS"
    ]
    
    for text in test_texts:
        normalized = evaluator.normalize_answer(text)
        print(f"  '{text}' -> '{normalized}'")
    
    # 运行小规模评估
    print("\n🚀 运行小规模评估 (5个样本):")
    evaluator.evaluate(num_samples=5, kv_cache_length=1024)

if __name__ == "__main__":
    test_improvements()