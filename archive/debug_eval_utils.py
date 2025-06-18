#!/usr/bin/env python3
"""
调试和修复eval_utils.py的路径问题
"""
import os
import sys
from pathlib import Path


def debug_path_issue():
    """调试路径计算问题"""
    print("=== 路径问题调试 ===")

    # 1. 检查eval_utils.py的路径计算
    eval_utils_file = Path('hace-kv-optimization/eval_utils.py')
    print(f"eval_utils.py 位置: {eval_utils_file.absolute()}")

    # 2. 模拟eval_utils.py中的路径计算
    if eval_utils_file.exists():
        # 模拟 Path(__file__).parent.parent / "baseline_fullkv.json"
        simulated_path = eval_utils_file.parent.parent / "baseline_fullkv.json"
        print(f"eval_utils计算的基线文件路径: {simulated_path.absolute()}")
        print(f"该路径是否存在: {simulated_path.exists()}")

        # 3. 检查实际的基线文件位置
        actual_baseline = Path('baseline_fullkv.json')
        print(f"实际基线文件路径: {actual_baseline.absolute()}")
        print(f"实际基线文件是否存在: {actual_baseline.exists()}")

        # 4. 比较路径
        if simulated_path.absolute() == actual_baseline.absolute():
            print("✅ 路径计算正确")
        else:
            print("❌ 路径计算不匹配!")
            print(f"计算路径: {simulated_path.absolute()}")
            print(f"实际路径: {actual_baseline.absolute()}")

    # 5. 直接检查eval_utils.py中的_BASELINE_FILE变量
    try:
        sys.path.append('hace-kv-optimization')
        import eval_utils

        # 获取eval_utils模块中的_BASELINE_FILE路径
        baseline_file_path = eval_utils._BASELINE_FILE
        print(f"\neval_utils._BASELINE_FILE: {baseline_file_path}")
        print(f"_BASELINE_FILE.exists(): {baseline_file_path.exists()}")
        print(f"_BASELINE_FILE.absolute(): {baseline_file_path.absolute()}")

        # 列出该目录的内容
        parent_dir = baseline_file_path.parent
        print(f"\n基线文件所在目录: {parent_dir}")
        print("目录内容:")
        for item in parent_dir.iterdir():
            if 'baseline' in item.name.lower():
                print(f"  📄 {item.name} ({item.stat().st_size} bytes)")

    except Exception as e:
        print(f"❌ 检查eval_utils._BASELINE_FILE失败: {e}")


def fix_eval_utils_path():
    """修复eval_utils.py的路径问题"""
    print("\n=== 修复路径问题 ===")

    eval_utils_file = 'hace-kv-optimization/eval_utils.py'

    # 读取eval_utils.py内容
    try:
        with open(eval_utils_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找路径定义行
        if '_BASELINE_FILE = Path(__file__).parent.parent / "baseline_fullkv.json"' in content:
            print("✅ 找到路径定义")

            # 创建修复后的内容
            # 改为使用当前工作目录
            new_content = content.replace(
                '_BASELINE_FILE = Path(__file__).parent.parent / "baseline_fullkv.json"',
                '''# 修复路径问题：使用多种方法确保找到基线文件
import os
try:
    # 方法1：使用相对路径计算
    _BASELINE_FILE = Path(__file__).parent.parent / "baseline_fullkv.json"
    if not _BASELINE_FILE.exists():
        # 方法2：使用当前工作目录
        _BASELINE_FILE = Path(os.getcwd()) / "baseline_fullkv.json"
        if not _BASELINE_FILE.exists():
            # 方法3：使用绝对路径搜索
            for possible_path in [
                Path("baseline_fullkv.json"),
                Path("../baseline_fullkv.json"),
                Path("../../baseline_fullkv.json")
            ]:
                if possible_path.exists():
                    _BASELINE_FILE = possible_path.absolute()
                    break
except:
    _BASELINE_FILE = Path("baseline_fullkv.json")'''
            )

            # 备份原文件
            backup_file = eval_utils_file + '.backup'
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 原文件已备份到: {backup_file}")

            # 写入修复后的内容
            with open(eval_utils_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ eval_utils.py已修复")

            return True

        else:
            print("❌ 未找到路径定义行，可能文件结构已变化")
            return False

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False


def quick_test_after_fix():
    """修复后快速测试"""
    print("\n=== 修复后测试 ===")

    try:
        # 重新导入模块（清除缓存）
        if 'eval_utils' in sys.modules:
            del sys.modules['eval_utils']

        sys.path.insert(0, 'hace-kv-optimization')
        import eval_utils

        # 测试load_baseline
        baseline_data = eval_utils.load_baseline()
        print(f"✅ load_baseline成功!")
        print(f"📊 基线数据: {baseline_data}")

        # 测试相对评分计算
        result = eval_utils.calculate_relative_score('hotpotqa', 0.18, is_full_kv=False)
        print(f"✅ 相对评分计算成功: {result['relative_score']:.2f}/100")

        return True

    except Exception as e:
        print(f"❌ 修复后测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 1. 调试路径问题
    debug_path_issue()

    # 2. 修复路径问题
    fix_success = fix_eval_utils_path()

    if fix_success:
        # 3. 测试修复结果
        test_success = quick_test_after_fix()

        if test_success:
            print("\n🎉 问题已完全修复!")
            print("现在你可以重新运行基线实验了")
        else:
            print("\n⚠️ 路径已修复，但仍有其他问题")
    else:
        print("\n❌ 自动修复失败，需要手动处理")