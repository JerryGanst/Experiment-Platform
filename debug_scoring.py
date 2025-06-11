#!/usr/bin/env python3
"""
调试评分系统 - 测试hotpotqa评分
"""
import json
import sys
import os

# 添加路径以导入评分模块
sys.path.append('hace-kv-optimization')

try:
    # 尝试不同的导入路径
    try:
        from hace_core.evaluation.scoring import qa_f1_score
        print("✅ 评分模块导入成功 (hace_core.evaluation)")
    except ImportError:
        # 尝试从baselines目录导入
        sys.path.append('hace-kv-optimization/baselines')
        from fullkvcache_main import qa_f1_score
        print("✅ 评分模块导入成功 (fullkvcache_main)")
except ImportError as e:
    print(f"❌ 评分模块导入失败: {e}")
    print("尝试直接测试...")
    
    # 简单的F1评分实现用于测试
    def simple_f1_score(prediction, ground_truth):
        """简单的F1评分实现"""
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
            
        intersection = pred_tokens & gt_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    qa_f1_score = simple_f1_score
    print("✅ 使用简单F1评分实现")


def test_scoring():
    """测试评分功能"""
    
    # 查找Miller v. California样本
    miller_sample = None
    with open('data/hotpotqa.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            if 'Miller v. California' in sample.get('answers', []):
                miller_sample = sample
                print(f"✅ 找到Miller v. California样本 (第{i+1}行)")
                break
            if i >= 10:  # 只检查前10个样本
                break
    
    if not miller_sample:
        print("❌ 未找到Miller v. California样本，使用第一个样本")
        with open('data/hotpotqa.jsonl', 'r', encoding='utf-8') as f:
            miller_sample = json.loads(f.readline())
    
    print("\n📋 样本信息:")
    print(f"问题: {miller_sample['input'][:100]}...")
    print(f"标准答案: {miller_sample['answers']}")
    
    sample = miller_sample

    # 测试不同的预测答案
    test_cases = [
        "Miller v. California",  # 完全正确
        "Miller v California",  # 缺少点号
        "miller v. california",  # 小写
        "The case Miller v. California",  # 包含额外文字
        "Gates v. Collier",  # 错误答案
        "I don't know",  # 完全错误
        "",  # 空答案
    ]

    print("\n🧪 评分测试:")
    for i, prediction in enumerate(test_cases, 1):
        try:
            score = qa_f1_score(prediction, sample['answers'][0])
            print(f"测试 {i}: '{prediction}' -> 分数: {score:.3f}")
        except Exception as e:
            print(f"测试 {i}: '{prediction}' -> 错误: {e}")

    print("\n🔍 多答案测试:")
    # 测试多个标准答案的情况
    for prediction in ["Miller v. California", "Gates v. Collier"]:
        scores = []
        for gt in sample['answers']:
            score = qa_f1_score(prediction, gt)
            scores.append(score)
        max_score = max(scores)
        print(f"预测: '{prediction}' -> 最高分: {max_score:.3f}")


if __name__ == "__main__":
    test_scoring()