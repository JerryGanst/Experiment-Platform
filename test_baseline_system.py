#!/usr/bin/env python3
"""
修复fullkvcache_main.py中的DATASET_SCORING_MAP逻辑问题
确保在评分函数未定义时不会出现NameError
"""

import os
import re


def fix_dataset_scoring_map():
    """修复DATASET_SCORING_MAP的条件定义问题"""

    # 读取当前文件
    with open("fullkvcache_main.py", 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建备份
    backup_file = "fullkvcache_main.py.before_map_fix"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📁 已创建备份: {backup_file}")

    # 查找并替换评分模块导入部分
    scoring_import_pattern = r'# 导入评分模块.*?SCORING_AVAILABLE = False'

    new_scoring_import = '''# 导入评分模块
try:
    longbench_metrics_path = os.path.join(os.path.dirname(__file__), '..', 'cakekv-main', 'cakekv-main', 'experiments', 'LongBench')
    if longbench_metrics_path not in sys.path:
        sys.path.append(longbench_metrics_path)

    from metrics import (
        qa_f1_score, rouge_score, classification_score, 
        retrieval_score, count_score, code_sim_score,
        normalize_answer
    )
    SCORING_AVAILABLE = True
    print("[OK] 评分模块加载成功")
except ImportError as e:
    print(f"[WARNING] 评分模块加载失败: {e}")
    SCORING_AVAILABLE = False

    # 定义占位符函数，避免NameError
    def qa_f1_score(*args, **kwargs):
        return None
    def rouge_score(*args, **kwargs):
        return None
    def classification_score(*args, **kwargs):
        return None
    def retrieval_score(*args, **kwargs):
        return None
    def count_score(*args, **kwargs):
        return None
    def code_sim_score(*args, **kwargs):
        return None
    def normalize_answer(*args, **kwargs):
        return None

    print("[INFO] 已定义占位符评分函数")'''

    # 应用替换
    content = re.sub(scoring_import_pattern, new_scoring_import, content, flags=re.DOTALL)

    # 查找现有的DATASET_SCORING_MAP定义并删除
    map_pattern = r'# 数据集评分映射.*?^}'
    content = re.sub(map_pattern, '', content, flags=re.DOTALL | re.MULTILINE)

    # 在基线评分工具导入后添加条件化的DATASET_SCORING_MAP
    baseline_pattern = r'(except ImportError as e:\s+print\(f"\[WARNING\] 基线评分工具加载失败: \{e\}"\)\s+BASELINE_SCORING_AVAILABLE = False)'

    new_map_definition = r'''\1

# 数据集评分映射 - 条件化定义
DATASET_SCORING_MAP = {
    "mmlu": qa_f1_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}'''

    content = re.sub(baseline_pattern, new_map_definition, content)

    # 保存修复后的文件
    with open("fullkvcache_main.py", 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ DATASET_SCORING_MAP逻辑修复完成")
    return True


def verify_fix():
    """验证修复是否成功"""
    try:
        # 尝试编译Python文件
        import py_compile
        py_compile.compile("fullkvcache_main.py", doraise=True)
        print("✅ Python语法检查通过")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Python语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    print("🔧 修复DATASET_SCORING_MAP逻辑问题")
    print("=" * 50)

    print("🔍 问题诊断:")
    print("  - 评分模块导入失败 (SCORING_AVAILABLE = False)")
    print("  - 但DATASET_SCORING_MAP仍尝试引用未定义的函数")
    print("  - 导致NameError: name 'qa_f1_score' is not defined")

    print("\n🔧 修复方案:")
    print("  - 在导入失败时定义占位符函数")
    print("  - 确保DATASET_SCORING_MAP始终可以正确定义")

    if fix_dataset_scoring_map():
        print("\n🧪 验证修复...")
        if verify_fix():
            print("\n🎉 修复成功！")
            print("\n🚀 现在可以运行实验:")
            print(
                "python fullkvcache_main.py --datasets hotpotqa --kv_cache_lengths 512 --batch_sizes 1 --max_new_tokens 10 --repetitions 1 --enable_scoring --is_baseline_run")
        else:
            print("\n❌ 修复后仍有语法错误，请检查")
    else:
        print("\n❌ 修复失败")


if __name__ == "__main__":
    main()