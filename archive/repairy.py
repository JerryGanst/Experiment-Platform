#!/usr/bin/env python3
"""
修复基线文件数据结构
将嵌套结构转换为扁平结构
"""
import json
import sys
import os


def fix_baseline_structure():
    """修复基线文件的数据结构"""
    baseline_file = 'hace-kv-optimization/baseline_fullkv.json'

    print("=== 修复基线文件结构 ===")

    # 检查文件是否存在
    if not os.path.exists(baseline_file):
        print(f"❌ 文件不存在: {baseline_file}")
        return False

    # 读取当前文件
    try:
        with open(baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📄 原始数据结构: {list(data.keys())}")

        # 检查是否为嵌套结构
        if 'baseline_scores' in data:
            print("🔧 检测到嵌套结构，转换为扁平结构...")

            # 提取baseline_scores部分
            baseline_scores = data['baseline_scores']
            print(f"📊 提取的数据集: {list(baseline_scores.keys())}")

            # 创建新的扁平结构
            flat_data = baseline_scores.copy()

            # 添加元数据
            flat_data['_metadata'] = {
                'created_at': data.get('timestamp', '2025-06-17'),
                'source': 'structure_conversion',
                'original_structure': 'nested'
            }

            # 备份原文件
            backup_file = baseline_file + '.backup'
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"📦 原文件已备份到: {backup_file}")

            # 保存修复后的文件
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(flat_data, f, indent=2, ensure_ascii=False)

            print("✅ 基线文件结构已修复为扁平格式")

            # 验证修复结果
            with open(baseline_file, 'r', encoding='utf-8') as f:
                verified_data = json.load(f)

            datasets = [k for k in verified_data.keys() if not k.startswith('_')]
            print(f"📋 修复后的数据集: {datasets}")

            # 检查关键数据集
            if 'hotpotqa' in verified_data:
                print(f"✅ hotpotqa基线分数: {verified_data['hotpotqa']}")
            else:
                print("❌ hotpotqa仍然缺失")
                return False

            return True

        else:
            print("✅ 已经是扁平结构，无需转换")

            # 检查数据完整性
            datasets = [k for k in data.keys() if not k.startswith('_')]
            print(f"📋 当前数据集: {datasets}")

            if 'hotpotqa' in data:
                print(f"✅ hotpotqa基线分数: {data['hotpotqa']}")
                return True
            else:
                print("❌ hotpotqa缺失")
                return False

    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return False


def test_eval_utils():
    """测试eval_utils功能"""
    print("\n=== 测试eval_utils功能 ===")

    try:
        # 清除模块缓存
        if 'eval_utils' in sys.modules:
            del sys.modules['eval_utils']

        sys.path.insert(0, 'hace-kv-optimization')
        import eval_utils

        print("✅ eval_utils模块导入成功")

        # 测试load_baseline
        baseline = eval_utils.load_baseline()
        datasets = [k for k in baseline.keys() if not k.startswith('_')]
        print(f"📊 基线数据集: {datasets}")

        # 测试相对评分计算
        if 'hotpotqa' in baseline:
            result = eval_utils.calculate_relative_score('hotpotqa', 0.18, False)
            print(f"✅ 相对评分成功: {result['relative_score']:.2f}/100")

            # 显示详细结果
            print(f"📈 测试结果详情:")
            print(f"  数据集: {result['dataset']}")
            print(f"  原始分数: {result['raw_score']:.4f}")
            print(f"  基线分数: {result['baseline_score']:.4f}")
            print(f"  相对分数: {result['relative_score']:.2f}/100")

            return True
        else:
            print("❌ hotpotqa在基线数据中不存在")
            return False

    except Exception as e:
        print(f"❌ eval_utils测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 基线文件结构修复工具")
    print("=" * 50)

    # 1. 修复文件结构
    structure_fixed = fix_baseline_structure()

    if structure_fixed:
        # 2. 测试eval_utils功能
        eval_utils_works = test_eval_utils()

        if eval_utils_works:
            print("\n🎉 修复完成！基线评分系统现在应该正常工作了！")
            print("\n📋 下一步操作:")
            print("1. 重新运行基线实验:")
            print(
                "   python hace-kv-optimization/baselines/fullkvcache_main.py --enable_scoring --is_baseline_run --datasets hotpotqa")
            print("\n2. 或者运行你的诊断脚本:")
            print("   python test_baseline_system.py")
        else:
            print("\n⚠️ 结构已修复，但eval_utils仍有问题")
    else:
        print("\n❌ 结构修复失败")