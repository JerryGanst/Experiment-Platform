#!/usr/bin/env python3
"""诊断HACE实验结果是否真的不同。"""
import json
import hashlib
from pathlib import Path


def file_hash(path):
    """计算文件的MD5哈希。"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_files():
    """对比两个实验结果文件。"""
    results_dir = Path(__file__).parent.parent / "results_clean"

    file1 = results_dir / "hace_normal_128" / "qmsum.jsonl"
    file2 = results_dir / "hace_reverse_disp_128" / "qmsum.jsonl"

    print("=" * 80)
    print("HACE结果文件诊断")
    print("=" * 80)
    print()

    # 检查文件是否存在
    for name, path in [("normal", file1), ("reverse_disp", file2)]:
        if not path.exists():
            print(f"❌ {name}: 文件不存在")
            return
        print(f"✓ {name}: {path}")
        print(f"  大小: {path.stat().st_size} bytes")
        print(f"  修改时间: {path.stat().st_mtime}")

    print()

    # 计算哈希值
    hash1 = file_hash(file1)
    hash2 = file_hash(file2)

    print(f"文件哈希值:")
    print(f"  normal:       {hash1}")
    print(f"  reverse_disp: {hash2}")
    print()

    if hash1 == hash2:
        print("⚠️  警告：两个文件的哈希值完全相同！")
        print("这意味着两个文件内容100%相同，可能的原因：")
        print("  1. 环境变量HACE_PREF_MODE没有生效")
        print("  2. 两次运行实际上都使用了相同的模式")
        print("  3. 文件被意外复制了")
        print()
    else:
        print("✓ 两个文件内容不同")
        print()

    # 逐行对比
    with open(file1) as f1, open(file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    print(f"行数对比:")
    print(f"  normal: {len(lines1)} 行")
    print(f"  reverse_disp: {len(lines2)} 行")
    print()

    # 检查有多少行不同
    different_lines = 0
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            different_lines += 1
            if different_lines <= 3:  # 只显示前3个不同的样本
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                print(f"样本 {i+1} 不同:")
                print(f"  normal pred: {data1['pred'][:100]}...")
                print(f"  reverse pred: {data2['pred'][:100]}...")
                print()

    if different_lines == 0:
        print("❌ 所有预测结果完全相同！")
        print()
        print("建议:")
        print("1. 检查运行日志中是否有 '[HACE] Using pref_mode:' 消息")
        print("2. 确认两次运行使用了不同的 --pref_mode 参数")
        print("3. 可能需要重新运行其中一个实验")
    else:
        print(f"✓ 找到 {different_lines} 个不同的预测 ({100*different_lines/len(lines1):.1f}%)")


if __name__ == "__main__":
    compare_files()
