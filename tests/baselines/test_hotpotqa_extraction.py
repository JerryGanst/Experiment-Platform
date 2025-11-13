#!/usr/bin/env python3
"""
测试Ground Truth提取功能
验证修复后的答案提取是否正常工作

迁移说明: 从 evaluation/baselines/test.py 迁移而来
路径已修正为基于项目根目录的绝对路径
"""

import json
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.common.data_loader import find_data_file  # noqa: E402


def extract_ground_truth_from_sample(sample, dataset_source="local"):
    """
    修复后的答案提取函数
    专门处理LongBench HotpotQA格式
    """

    print(f"[DEBUG] 提取答案 - 样本键: {list(sample.keys())}")

    # 1. 优先处理标准LongBench格式
    if 'answers' in sample and sample['answers']:
        answers = sample['answers']
        if isinstance(answers, list) and len(answers) > 0:
            # 过滤空答案
            valid_answers = [str(ans).strip() for ans in answers if ans and str(ans).strip()]
            if valid_answers:
                print(f"[DEBUG] ✅ 使用answers字段: {valid_answers}")
                return valid_answers
        elif isinstance(answers, str) and answers.strip():
            print(f"[DEBUG] ✅ 使用answers字符串: [{answers.strip()}]")
            return [answers.strip()]

    # 2. 处理processed sample的reference字段
    if 'reference' in sample and sample['reference']:
        reference = sample['reference']
        if isinstance(reference, list) and len(reference) > 0:
            valid_refs = [str(ref).strip() for ref in reference if ref and str(ref).strip()]
            if valid_refs:
                print(f"[DEBUG] ✅ 使用reference字段: {valid_refs}")
                return valid_refs
        elif isinstance(reference, str) and reference.strip():
            print(f"[DEBUG] ✅ 使用reference字符串: [{reference.strip()}]")
            return [reference.strip()]

    # 3. 处理嵌套的原始样本
    if 'original_sample' in sample:
        original = sample['original_sample']
        if isinstance(original, dict):
            for field in ['answers', 'answer', 'output', 'target']:
                if field in original and original[field]:
                    value = original[field]
                    if isinstance(value, list) and len(value) > 0:
                        valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                        if valid_vals:
                            print(f"[DEBUG] ✅ 使用原始样本{field}: {valid_vals}")
                            return valid_vals
                    elif isinstance(value, str) and value.strip():
                        print(f"[DEBUG] ✅ 使用原始样本{field}: [{value.strip()}]")
                        return [value.strip()]

    # 4. 其他常见答案字段
    answer_fields = ['answer', 'output', 'gold', 'target', 'label', 'ground_truth']
    for field in answer_fields:
        if field in sample and sample[field] is not None:
            value = sample[field]
            if isinstance(value, list) and len(value) > 0:
                valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                if valid_vals:
                    print(f"[DEBUG] ✅ 使用{field}字段: {valid_vals}")
                    return valid_vals
            elif isinstance(value, str) and value.strip():
                print(f"[DEBUG] ✅ 使用{field}字符串: [{value.strip()}]")
                return [value.strip()]
            elif isinstance(value, (int, float)):
                print(f"[DEBUG] ✅ 使用{field}数值: [{str(value)}]")
                return [str(value)]

    # 5. 最后的调试输出
    print(f"[DEBUG] ❌ 未找到有效答案字段")
    print(f"样本完整内容: {json.dumps(sample, indent=2, ensure_ascii=False)[:500]}...")

    # 返回空列表而不是["Unknown"]，让上层处理
    return []


def test_ground_truth_extraction():
    """测试Ground Truth提取功能"""

    print("🧪 测试Ground Truth提取功能")
    print("=" * 50)

    # 检查数据文件（支持 DATASETS_ROOT / 远端路径）
    data_file = find_data_file("hotpotqa")
    if not data_file:
        data_root_hint = os.environ.get("DATASETS_ROOT", project_root / "data")
        print(f"❌ 未找到 hotpotqa.jsonl (DATASETS_ROOT={data_root_hint})")
        print("💡 请在 SSH 会话中设置 DATASETS_ROOT 指向远端数据集目录")
        return False

    # 读取测试样本
    test_samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只测试前5个样本
                break
            line = line.strip()
            if line:
                try:
                    test_samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{i + 1}行JSON解析失败: {e}")
                    continue

    if not test_samples:
        print("❌ 没有找到有效的测试样本")
        return False

    print(f"📋 加载了 {len(test_samples)} 个测试样本")

    # 测试答案提取
    success_count = 0
    for i, sample in enumerate(test_samples):
        print(f"\n--- 样本 {i + 1} ---")
        print(f"输入: {sample.get('input', '')[:100]}...")

        # 使用修复后的提取函数
        ground_truth = extract_ground_truth_from_sample(sample, "local")

        if ground_truth and len(ground_truth) > 0:
            print(f"✅ 成功提取答案: {ground_truth}")
            success_count += 1
        else:
            print(f"❌ 答案提取失败")
            print(f"样本键: {list(sample.keys())}")
            # 显示样本内容用于调试
            print(f"样本内容: {json.dumps(sample, indent=2, ensure_ascii=False)[:300]}...")

    success_rate = (success_count / len(test_samples)) * 100
    print(f"\n📊 测试结果:")
    print(f"成功率: {success_rate:.1f}% ({success_count}/{len(test_samples)})")

    if success_rate >= 80:
        print("\n🎉 Ground Truth提取测试通过！")
        print("现在可以运行完整的基线实验")
        return True
    else:
        print("\n❌ Ground Truth提取测试失败")
        print("需要进一步调试数据格式问题")
        return False


def test_scoring_function():
    """测试评分功能"""

    print("\n🧪 测试评分功能")
    print("=" * 50)

    # 简单的F1评分实现用于测试
    def simple_qa_f1_score(prediction, ground_truth):
        """简单的F1评分实现"""
        pred_tokens = set(str(prediction).lower().split())
        gt_tokens = set(str(ground_truth).lower().split())

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        intersection = pred_tokens & gt_tokens
        if len(intersection) == 0:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # 测试用例
    test_cases = [
        ("yes", "yes", 1.0),
        ("Yes", "yes", 1.0),
        ("true", "yes", 0.0),
        ("Miller v. California", "Miller v. California", 1.0),
        ("Miller California", "Miller v. California", 0.67),
        ("no", "yes", 0.0),
        ("I don't know", "yes", 0.0)
    ]

    print("测试评分函数:")
    for prediction, ground_truth, expected in test_cases:
        score = simple_qa_f1_score(prediction, ground_truth)
        status = "✅" if abs(score - expected) < 0.1 else "❌"
        print(f"{status} 预测: '{prediction}' vs 标准: '{ground_truth}' -> 分数: {score:.3f} (期望: {expected:.3f})")

    return True


def create_test_fix_script():
    """创建用于修复fullkvcache_main.py的代码片段"""

    fix_code = '''
# 将此函数替换到 hace-kv-optimization/run_result/fullkvcache_main.py 中

def extract_ground_truth_from_sample(sample, dataset_source):
    """
    修复版：专门处理LongBench官方数据
    处理所有可能的答案字段格式
    """

    # 调试信息（可以在生产环境中移除）
    # logger.debug(f"提取答案 - 样本键: {list(sample.keys())}")

    # 1. 优先处理标准LongBench格式
    if 'answers' in sample and sample['answers']:
        answers = sample['answers']
        if isinstance(answers, list) and len(answers) > 0:
            # 过滤空答案
            valid_answers = [str(ans).strip() for ans in answers if ans and str(ans).strip()]
            if valid_answers:
                return valid_answers
        elif isinstance(answers, str) and answers.strip():
            return [answers.strip()]

    # 2. 处理processed sample的reference字段
    if 'reference' in sample and sample['reference']:
        reference = sample['reference']
        if isinstance(reference, list) and len(reference) > 0:
            valid_refs = [str(ref).strip() for ref in reference if ref and str(ref).strip()]
            if valid_refs:
                return valid_refs
        elif isinstance(reference, str) and reference.strip():
            return [reference.strip()]

    # 3. 处理嵌套的原始样本
    if 'original_sample' in sample:
        original = sample['original_sample']
        if isinstance(original, dict):
            for field in ['answers', 'answer', 'output', 'target']:
                if field in original and original[field]:
                    value = original[field]
                    if isinstance(value, list) and len(value) > 0:
                        valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                        if valid_vals:
                            return valid_vals
                    elif isinstance(value, str) and value.strip():
                        return [value.strip()]

    # 4. 其他常见答案字段
    answer_fields = ['answer', 'output', 'gold', 'target', 'label', 'ground_truth']
    for field in answer_fields:
        if field in sample and sample[field] is not None:
            value = sample[field]
            if isinstance(value, list) and len(value) > 0:
                valid_vals = [str(v).strip() for v in value if v and str(v).strip()]
                if valid_vals:
                    return valid_vals
            elif isinstance(value, str) and value.strip():
                return [value.strip()]
            elif isinstance(value, (int, float)):
                return [str(value)]

    # 5. 如果都没找到，记录警告但不返回["Unknown"]
    logger.warning(f"未找到有效答案字段，样本键: {list(sample.keys())}")

    # 返回空列表而不是["Unknown"]，让上层处理
    return []
'''

    with open("fixed_extract_function.py", "w", encoding="utf-8") as f:
        f.write(fix_code)

    print(f"\n📝 修复代码已保存到: fixed_extract_function.py")
    print("请将此函数复制到 hace-kv-optimization/run_result/fullkvcache_main.py 中")


def main():
    """主测试函数"""

    print("🔧 Ground Truth提取测试")
    print("=" * 70)

    # 测试Ground Truth提取
    if not test_ground_truth_extraction():
        print("\n💡 建议:")
        print("1. 检查数据文件格式是否正确")
        print("2. 运行: python fixed_download_script.py 重新下载数据")
        print("3. 如果仍有问题，可能需要手动调试数据格式")
        return False

    # 测试评分功能
    test_scoring_function()

    # 创建修复代码
    create_test_fix_script()

    print("\n🎉 所有测试完成！")
    print("\n📋 下一步操作:")
    print("1. 将 fixed_extract_function.py 中的代码替换到 fullkvcache_main.py")
    print("2. 运行完整实验测试修复效果")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 测试失败，请检查错误信息")
        sys.exit(1)
