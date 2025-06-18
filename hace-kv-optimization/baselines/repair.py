#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速重新集成基线评分系统到fullkvcache_main.py
安全地添加所有必要的功能，不破坏原有代码
"""

import os
import re


def reintegrate_baseline_system():
    """重新集成基线评分系统"""
    file_path = 'fullkvcache_main.py'

    print("=== 重新集成基线评分系统 ===")

    # 备份原始文件
    backup_path = file_path + '.before_reintegration'
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"📦 原始文件已备份到: {backup_path}")

    content = original_content

    # 1. 添加eval_utils导入（在其他导入后）
    eval_utils_import = '''
# 导入基线评分工具
try:
    eval_utils_path = os.path.join(os.path.dirname(__file__), '..')
    if eval_utils_path not in sys.path:
        sys.path.append(eval_utils_path)

    from eval_utils import (
        score_dataset,
        calculate_relative_score,
        aggregate_scores,
        format_score_report
    )
    BASELINE_SCORING_AVAILABLE = True
    print("[OK] 基线评分工具加载成功")
except ImportError as e:
    print(f"[WARNING] 基线评分工具加载失败: {e}")
    BASELINE_SCORING_AVAILABLE = False
'''

    # 在import语句后添加eval_utils导入
    import_pattern = r'(from hace_core.*?\n)'
    if re.search(import_pattern, content):
        content = re.sub(import_pattern, r'\1' + eval_utils_import, content, count=1)
        print("✅ 添加了eval_utils导入")

    # 2. 修改数据路径（安全的方式）
    content = content.replace('data_dir="../../data"', 'data_dir="../data"')
    content = content.replace("data_dir='../../data'", "data_dir='../data'")
    print("✅ 修复了数据路径")

    # 3. 添加评分参数到argparse
    argparse_addition = '''    parser.add_argument("--enable_scoring", action="store_true", help="Enable scoring evaluation")
    parser.add_argument("--is_baseline_run", action="store_true", help="Mark this as a baseline run for establishing Full KV baseline scores")
'''

    # 在现有参数后添加新参数
    if 'add_argument' in content and '--enable_scoring' not in content:
        # 找到最后一个add_argument的位置
        last_arg_match = None
        for match in re.finditer(r'parser\.add_argument\([^)]+\)', content):
            last_arg_match = match

        if last_arg_match:
            insert_pos = last_arg_match.end()
            content = content[:insert_pos] + '\n' + argparse_addition + content[insert_pos:]
            print("✅ 添加了评分参数")

    # 4. 添加基线处理逻辑（在main函数结束前）
    baseline_processing = '''
    # 处理基线评分（如果启用）
    if args.enable_scoring and args.is_baseline_run and BASELINE_SCORING_AVAILABLE:
        try:
            logger.info("开始处理基线评分...")

            # 收集所有实验的评分结果，建立基线
            baseline_scores = []

            for result in all_results_summary:
                if isinstance(result, dict) and 'experiment_id' in result:
                    # 查找对应的评分文件
                    experiment_id = result['experiment_id']

                    # 从实验ID中提取数据集名称
                    if 'ds_' in experiment_id:
                        dataset_part = experiment_id.split('ds_')[1].split('_')[0]

                        # 查找评分结果文件
                        for root, dirs, files in os.walk(main_output_dir):
                            for file in files:
                                if file.startswith(f"evaluation_results_") and experiment_id in file:
                                    eval_file_path = os.path.join(root, file)
                                    try:
                                        with open(eval_file_path, 'r', encoding='utf-8') as f:
                                            eval_data = json.load(f)
                                            if eval_data.get("average_score") is not None:
                                                score_result = calculate_relative_score(
                                                    dataset_name=dataset_part,
                                                    raw_score=eval_data["average_score"],
                                                    is_full_kv=True
                                                )
                                                baseline_scores.append(score_result)
                                                logger.info(f"基线分数已记录: {dataset_part} = {eval_data['average_score']:.4f}")
                                    except Exception as e:
                                        logger.warning(f"处理评分文件时出错 {eval_file_path}: {e}")

            if baseline_scores:
                # 生成基线报告
                aggregated = aggregate_scores(baseline_scores)
                report = format_score_report(aggregated, "Full KV (基线)")

                # 保存基线报告
                baseline_report_path = os.path.join(main_output_dir, "baseline_scoring_report.txt")
                with open(baseline_report_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                logger.info(f"基线评分报告已保存到: {baseline_report_path}")
                print(report)
            else:
                logger.warning("未找到有效的评分结果，无法建立基线")

        except Exception as baseline_error:
            logger.error(f"处理基线评分时出错: {baseline_error}")

    elif args.enable_scoring and not args.is_baseline_run:
        logger.info("评分已启用，但这不是基线运行，跳过基线建立")
'''

    # 在main函数结束前添加基线处理逻辑
    if 'logger.info("FullKVCache experiment suite finished.")' in content:
        content = content.replace(
            'logger.info("FullKVCache experiment suite finished.")',
            baseline_processing + '\n    logger.info("FullKVCache experiment suite finished.")'
        )
        print("✅ 添加了基线处理逻辑")

    # 5. 保存修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 基线评分系统重新集成完成")

    return True


def verify_integration():
    """验证集成是否成功"""
    print("\n=== 验证集成结果 ===")

    file_path = 'fullkvcache_main.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查关键功能
    checks = [
        ('eval_utils导入', 'from eval_utils import'),
        ('基线评分可用性', 'BASELINE_SCORING_AVAILABLE'),
        ('评分参数', '--enable_scoring'),
        ('基线运行参数', '--is_baseline_run'),
        ('相对评分计算', 'calculate_relative_score'),
        ('基线处理逻辑', '开始处理基线评分'),
        ('数据路径修复', '../data')
    ]

    for check_name, keyword in checks:
        if keyword in content:
            print(f"✅ {check_name}: 已集成")
        else:
            print(f"❌ {check_name}: 缺失")

    # 检查语法
    try:
        import ast
        ast.parse(content)
        print("✅ Python语法验证通过")
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 快速重新集成基线评分系统")
    print("=" * 50)

    success = reintegrate_baseline_system()

    if success:
        integration_ok = verify_integration()

        if integration_ok:
            print("\n🎉 基线评分系统重新集成成功！")
            print("\n📋 现在可以测试:")
            print(
                "python hace-kv-optimization/baselines/fullkvcache_main.py --enable_scoring --is_baseline_run --datasets hotpotqa --kv_cache_lengths 128 --repetitions 1")
        else:
            print("\n⚠️ 集成完成但可能有问题，请检查")
    else:
        print("\n❌ 重新集成失败")