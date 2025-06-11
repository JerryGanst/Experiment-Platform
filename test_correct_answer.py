#!/usr/bin/env python3
"""
测试正确答案格式的评分
"""

import json
import sys

# 添加路径
sys.path.append('hace-kv-optimization/baselines')

try:
    from fullkvcache_main import qa_f1_score

    print("✅ 评分模块导入成功")
except ImportError as e:
    print(f"❌ 评分模块导入失败: {e}")
    exit(1)


def test_simple_scoring():
    """测试简单的评分情况"""

    # 读取第一个样本
    with open('data/hotpotqa.jsonl', 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())

    print("📋 第一个样本:")
    print(f"问题: {sample['input']}")
    print(f"标准答案: {sample['answers']}")

    # 测试正确答案
    correct_answers = ["yes", "Yes", "YES", "true", "True"]

    print("\n🧪 正确答案测试:")
    for answer in correct_answers:
        score = qa_f1_score(answer, sample['answers'][0])
        print(f"预测: '{answer}' vs 标准: '{sample['answers'][0]}' -> 分数: {score:.3f}")

    print("\n🧪 错误答案测试:")
    wrong_answers = ["no", "No", "false", "Miller v. California", "I don't know"]
    for answer in wrong_answers:
        score = qa_f1_score(answer, sample['answers'][0])
        print(f"预测: '{answer}' vs 标准: '{sample['answers'][0]}' -> 分数: {score:.3f}")


if __name__ == "__main__":
    test_simple_scoring()