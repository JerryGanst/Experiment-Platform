
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_experiment_results(baseline_dir, h2o_dir=None):
    """
    加载基线和H2O实验结果

    Args:
        baseline_dir: 基线实验结果目录
        h2o_dir: H2O实验结果目录 (可选)

    Returns:
        baseline_df: 基线数据框
        h2o_df: H2O数据框 (如果提供了h2o_dir)
    """
    # 查找基线结果文件
    baseline_files = []
    for root, _, files in os.walk(baseline_dir):
        for file in files:
            if file.endswith('.csv') and 'baseline_results' in file and 'summary' not in file:
                baseline_files.append(os.path.join(root, file))

    if not baseline_files:
        raise ValueError(f"No baseline result files found in {baseline_dir}")

    # 加载基线结果
    baseline_dfs = []
    for file in baseline_files:
        try:
            df = pd.read_csv(file)

            # 提取模型和数据集信息
            model_name = None
            if 'opt-125m' in file:
                model_name = 'opt-125m'
            elif 'opt-350m' in file:
                model_name = 'opt-350m'
            elif 'opt-1.3b' in file:
                model_name = 'opt-1.3b'

            if model_name and 'model_name' not in df.columns:
                df['model_name'] = model_name

            # 提取数据集信息
            dataset_name = None
            if 'mmlu' in file:
                dataset_name = 'mmlu'
            elif 'hellaswag' in file:
                dataset_name = 'hellaswag'
            elif 'ceval' in file:
                dataset_name = 'ceval'
            elif 'race' in file:
                dataset_name = 'race'

            if dataset_name and 'dataset' not in df.columns:
                df['dataset'] = dataset_name

            # 添加语言信息
            if 'language' not in df.columns:
                if dataset_name in ['mmlu', 'hellaswag']:
                    df['language'] = 'english'
                elif dataset_name in ['ceval', 'race']:
                    df['language'] = 'chinese'

            # 添加实验类型标记
            df['type'] = 'baseline'

            baseline_dfs.append(df)
        except Exception as e:
            print(f"Error loading baseline file {file}: {e}")

    if not baseline_dfs:
        raise ValueError("Failed to load any baseline results")

    baseline_df = pd.concat(baseline_dfs, ignore_index=True)

    # 如果提供了H2O结果目录，加载H2O结果
    h2o_df = None
    if h2o_dir:
        # 查找H2O结果文件
        h2o_files = []
        for root, _, files in os.walk(h2o_dir):
            for file in files:
                if file.endswith('.csv') and 'h2o_results' in file and 'summary' not in file:
                    h2o_files.append(os.path.join(root, file))

        if not h2o_files:
            print(f"Warning: No H2O result files found in {h2o_dir}")
        else:
            # 加载H2O结果
            h2o_dfs = []
            for file in h2o_files:
                try:
                    df = pd.read_csv(file)

                    # 提取模型信息
                    model_name = None
                    if 'opt-125m' in file:
                        model_name = 'opt-125m'
                    elif 'opt-350m' in file:
                        model_name = 'opt-350m'
                    elif 'opt-1.3b' in file:
                        model_name = 'opt-1.3b'

                    if model_name and 'model_name' not in df.columns:
                        df['model_name'] = model_name

                    # 提取数据集信息
                    dataset_name = None
                    if 'mmlu' in file:
                        dataset_name = 'mmlu'
                    elif 'hellaswag' in file:
                        dataset_name = 'hellaswag'
                    elif 'ceval' in file:
                        dataset_name = 'ceval'
                    elif 'race' in file:
                        dataset_name = 'race'

                    if dataset_name and 'dataset' not in df.columns:
                        df['dataset'] = dataset_name

                    # 添加语言信息
                    if 'language' not in df.columns:
                        if dataset_name in ['mmlu', 'hellaswag']:
                            df['language'] = 'english'
                        elif dataset_name in ['ceval', 'race']:
                            df['language'] = 'chinese'

                    # 添加实验类型标记
                    df['type'] = 'h2o'

                    h2o_dfs.append(df)
                except Exception as e:
                    print(f"Error loading H2O file {file}: {e}")

            if h2o_dfs:
                h2o_df = pd.concat(h2o_dfs, ignore_index=True)

    return baseline_df, h2o_df

def create_performance_summary_table(baseline_df, h2o_df=None):
    """
    创建主要性能指标汇总表

    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框 (可选)

    Returns:
        summary_df: 汇总表数据框
    """
    # 首先处理基线数据
    # 按配置分组计算平均值
    baseline_summary = baseline_df.groupby(['model_name', 'kv_cache_length', 'batch_size']).agg({
        'peak_gpu_memory_mb': 'mean',
        'ttft_ms': 'mean',
        'tpot_ms': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()

    # 添加实验类型和内存节省列
    baseline_summary['type'] = 'baseline'
    baseline_summary['memory_saving_percent'] = 0
    baseline_summary['latency_improvement_percent'] = 0
    baseline_summary['throughput_improvement_percent'] = 0

    # 如果没有H2O数据，直接返回基线摘要
    if h2o_df is None:
        # 重命名列以符合表格要求
        baseline_summary = baseline_summary.rename(columns={
            'model_name': '模型',
            'type': '配置',
            'kv_cache_length': 'KV长度',
            'batch_size': '批处理大小',
            'peak_gpu_memory_mb': '内存使用(MB)',
            'memory_saving_percent': '内存节省(%)',
            'ttft_ms': 'TTFT(ms)',
            'tpot_ms': 'TPOT(ms)',
            'latency_improvement_percent': '延迟改进(%)',
            'throughput_tokens_per_sec': '吞吐量(tokens/s)',
            'throughput_improvement_percent': '吞吐量改进(%)'
        })

        # 选择需要的列并按顺序排列
        baseline_summary = baseline_summary[[
            '配置', '模型', 'KV长度', '批处理大小', '内存使用(MB)', '内存节省(%)',
            'TTFT(ms)', 'TPOT(ms)', '吞吐量(tokens/s)'
        ]]

        # 重命名配置类型以更加友好
        baseline_summary['配置'] = '基线'

        return baseline_summary

    # 处理H2O数据
    # 按配置分组计算平均值
    h2o_summary = h2o_df.groupby(['model_name', 'kv_cache_length', 'batch_size', 'heavy_ratio']).agg({
        'peak_gpu_memory_mb': 'mean',
        'ttft_ms': 'mean',
        'tpot_ms': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()

    # 添加实验类型
    h2o_summary['type'] = 'h2o'

    # 合并基线和H2O数据，计算性能改进
    summary_rows = []

    # 先添加基线行
    for _, baseline_row in baseline_summary.iterrows():
        summary_rows.append({
            '配置': '基线',
            '模型': baseline_row['model_name'],
            'KV长度': baseline_row['kv_cache_length'],
            '批处理大小': baseline_row['batch_size'],
            '内存使用(MB)': round(baseline_row['peak_gpu_memory_mb'], 2),
            '内存节省(%)': '-',
            'TTFT(ms)': round(baseline_row['ttft_ms'], 2),
            'TPOT(ms)': round(baseline_row['tpot_ms'], 2),
            '吞吐量(tokens/s)': round(baseline_row['throughput_tokens_per_sec'], 2),
            '困惑度': '-',
            '准确率变化(%)': '-'
        })

    # 添加H2O行
    for _, h2o_row in h2o_summary.iterrows():
        # 查找匹配的基线配置
        baseline_match = baseline_summary[
            (baseline_summary['model_name'] == h2o_row['model_name']) &
            (baseline_summary['kv_cache_length'] == h2o_row['kv_cache_length']) &
            (baseline_summary['batch_size'] == h2o_row['batch_size'])
        ]

        if not baseline_match.empty:
            baseline_match = baseline_match.iloc[0]

            # 计算性能改进百分比
            memory_saving = ((baseline_match['peak_gpu_memory_mb'] - h2o_row['peak_gpu_memory_mb']) /
                            baseline_match['peak_gpu_memory_mb'] * 100)

            latency_improvement = ((baseline_match['ttft_ms'] - h2o_row['ttft_ms']) /
                                 baseline_match['ttft_ms'] * 100)

            throughput_improvement = ((h2o_row['throughput_tokens_per_sec'] - baseline_match['throughput_tokens_per_sec']) /
                                    baseline_match['throughput_tokens_per_sec'] * 100)

            summary_rows.append({
                '配置': f'H2O-{int(h2o_row["heavy_ratio"]*100)}%',
                '模型': h2o_row['model_name'],
                'KV长度': h2o_row['kv_cache_length'],
                '批处理大小': h2o_row['batch_size'],
                '内存使用(MB)': round(h2o_row['peak_gpu_memory_mb'], 2),
                '内存节省(%)': round(memory_saving, 2),
                'TTFT(ms)': round(h2o_row['ttft_ms'], 2),
                'TPOT(ms)': round(h2o_row['tpot_ms'], 2),
                '吞吐量(tokens/s)': round(h2o_row['throughput_tokens_per_sec'], 2),
                '困惑度': '-',  # 目前没有这个指标
                '准确率变化(%)': '-'  # 目前没有这个指标
            })

    # 创建汇总表
    summary_df = pd.DataFrame(summary_rows)

    # 按模型、KV长度、批处理大小排序
    summary_df = summary_df.sort_values(['模型', 'KV长度', '批处理大小', '配置'])

    return summary_df

def create_language_comparison_table(baseline_df, h2o_df=None):
    """
    创建不同语言性能对比表

    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框 (可选)

    Returns:
        language_df: 语言对比表数据框
    """
    # 如果没有H2O数据，则无法计算性能改进
    if h2o_df is None:
        # 创建一个空的语言比较表
        language_df = pd.DataFrame(columns=[
            '语言', '数据集', '内存节省(%)', '延迟改进(%)', '质量变化(%)'
        ])

        # 按语言和数据集添加行，但没有改进数据
        language_groups = baseline_df.groupby(['language', 'dataset']).size().reset_index()

        for _, row in language_groups.iterrows():
            language_df = language_df.append({
                '语言': row['language'],
                '数据集': row['dataset'],
                '内存节省(%)': '-',
                '延迟改进(%)': '-',
                '质量变化(%)': '-'
            }, ignore_index=True)

        return language_df

    # 按语言和数据集分组计算基线指标
    baseline_language = baseline_df.groupby(['language', 'dataset']).agg({
        'peak_gpu_memory_mb': 'mean',
        'ttft_ms': 'mean',
        'tpot_ms': 'mean'
    }).reset_index()

    # 按语言和数据集分组计算H2O指标
    h2o_language = h2o_df.groupby(['language', 'dataset']).agg({
        'peak_gpu_memory_mb': 'mean',
        'ttft_ms': 'mean',
        'tpot_ms': 'mean'
    }).reset_index()

    # 合并数据并计算改进百分比
    language_rows = []

    for _, baseline_row in baseline_language.iterrows():
        language = baseline_row['language']
        dataset = baseline_row['dataset']

        # 查找匹配的H2O数据
        h2o_match = h2o_language[
            (h2o_language['language'] == language) &
            (h2o_language['dataset'] == dataset)
        ]

        if not h2o_match.empty:
            h2o_match = h2o_match.iloc[0]

            # 计算性能改进
            memory_saving = ((baseline_row['peak_gpu_memory_mb'] - h2o_match['peak_gpu_memory_mb']) /
                            baseline_row['peak_gpu_memory_mb'] * 100)

            latency_improvement = ((baseline_row['ttft_ms'] - h2o_match['ttft_ms']) /
                                 baseline_row['ttft_ms'] * 100)

            language_rows.append({
                '语言': language,
                '数据集': dataset,
                '内存节省(%)': round(memory_saving, 2),
                '延迟改进(%)': round(latency_improvement, 2),
                '质量变化(%)': '-'  # 目前没有这个指标
            })
        else:
            # 没有H2O数据的行
            language_rows.append({
                '语言': language,
                '数据集': dataset,
                '内存节省(%)': '-',
                '延迟改进(%)': '-',
                '质量变化(%)': '-'
            })

    # 创建语言比较表
    language_df = pd.DataFrame(language_rows)

    # 标准化语言名称
    language_df['语言'] = language_df['语言'].map({
        'english': '英文',
        'chinese': '中文'
    })

    # 按语言和数据集排序
    language_df = language_df.sort_values(['语言', '数据集'])

    return language_df

def create_best_config_table(baseline_df, h2o_df=None):
    """
    创建最佳配置推荐表

    Args:
        baseline_df: 基线数据框
        h2o_df: H2O数据框 (可选)

    Returns:
        config_df: 最佳配置表数据框
    """
    # 如果没有H2O数据，则无法创建配置推荐
    if h2o_df is None:
        # 创建一个示例表格
        config_df = pd.DataFrame([
            {
                '应用场景': '长文本生成',
                '推荐H2O配置': '待H2O实验完成后填充',
                '内存节省(%)': '-',
                '性能提升(%)': '-',
                '注意事项': '-'
            },
            {
                '应用场景': '批量处理',
                '推荐H2O配置': '待H2O实验完成后填充',
                '内存节省(%)': '-',
                '性能提升(%)': '-',
                '注意事项': '-'
            },
            {
                '应用场景': '实时对话',
                '推荐H2O配置': '待H2O实验完成后填充',
                '内存节省(%)': '-',
                '性能提升(%)': '-',
                '注意事项': '-'
            }
        ])

        return config_df

    # 创建不同应用场景的最佳配置
    config_rows = []

    # 1. 长文本生成场景 - 优先选择高KV缓存长度和较大heavy_ratio的配置
    long_text_configs = h2o_df[h2o_df['kv_cache_length'] >= 1024].copy()

    if not long_text_configs.empty:
        # 按heavy_ratio分组找到平均内存节省最高的配置
        long_text_best = long_text_configs.groupby(['heavy_ratio']).agg({
            'peak_gpu_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        }).reset_index()

        # 比较基线数据
        baseline_long = baseline_df[baseline_df['kv_cache_length'] >= 1024].agg({
            'peak_gpu_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        })

        # 计算最佳heavy_ratio的性能改进
        long_text_best['memory_saving'] = ((baseline_long['peak_gpu_memory_mb'] - long_text_best['peak_gpu_memory_mb']) /
                                         baseline_long['peak_gpu_memory_mb'] * 100)

        long_text_best['throughput_improvement'] = ((long_text_best['throughput_tokens_per_sec'] - baseline_long['throughput_tokens_per_sec']) /
                                                 baseline_long['throughput_tokens_per_sec'] * 100)

        # 选择内存节省最高的heavy_ratio
        best_long_config = long_text_best.loc[long_text_best['memory_saving'].idxmax()]

        config_rows.append({
            '应用场景': '长文本生成',
            '推荐H2O配置': f"{int(best_long_config['heavy_ratio']*100)}%重要命中",
            '内存节省(%)': round(best_long_config['memory_saving'], 2),
            '性能提升(%)': round(best_long_config['throughput_improvement'], 2),
            '注意事项': '适用于需要长上下文处理的场景，如文档摘要和长对话'
        })

    # 2. 批量处理场景 - 优先选择高批处理大小和适中heavy_ratio的配置
    batch_configs = h2o_df[h2o_df['batch_size'] >= 8].copy()

    if not batch_configs.empty:
        # 按heavy_ratio分组找到平均吞吐量最高的配置
        batch_best = batch_configs.groupby(['heavy_ratio']).agg({
            'peak_gpu_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        }).reset_index()

        # 比较基线数据
        baseline_batch = baseline_df[baseline_df['batch_size'] >= 8].agg({
            'peak_gpu_memory_mb': 'mean',
            'throughput_tokens_per_sec': 'mean'
        })

        # 计算最佳heavy_ratio的性能改进
        batch_best['memory_saving'] = ((baseline_batch['peak_gpu_memory_mb'] - batch_best['peak_gpu_memory_mb']) /
                                     baseline_batch['peak_gpu_memory_mb'] * 100)

        batch_best['throughput_improvement'] = ((batch_best['throughput_tokens_per_sec'] - baseline_batch['throughput_tokens_per_sec']) /
                                             baseline_batch['throughput_tokens_per_sec'] * 100)

        # 选择吞吐量改进最高的heavy_ratio
        best_batch_config = batch_best.loc[batch_best['throughput_improvement'].idxmax()]

        config_rows.append({
            '应用场景': '批量处理',
            '推荐H2O配置': f"{int(best_batch_config['heavy_ratio']*100)}%重要命中",
            '内存节省(%)': round(best_batch_config['memory_saving'], 2),
            '性能提升(%)': round(best_batch_config['throughput_improvement'], 2),
            '注意事项': '适用于需要同时处理多个请求的场景，如API服务和批量生成'
        })

    # 3. 实时对话场景 - 优先选择低延迟和较高heavy_ratio的配置
    realtime_configs = h2o_df[h2o_df['batch_size'] <= 4].copy()

    if not realtime_configs.empty:
        # 按heavy_ratio分组找到平均TTFT最低的配置
        realtime_best = realtime_configs.groupby(['heavy_ratio']).agg({
            'peak_gpu_memory_mb': 'mean',
            'ttft_ms': 'mean'
        }).reset_index()

        # 比较基线数据
        baseline_realtime = baseline_df[baseline_df['batch_size'] <= 4].agg({
            'peak_gpu_memory_mb': 'mean',
            'ttft_ms': 'mean'
        })

        # 计算最佳heavy_ratio的性能改进
        realtime_best['memory_saving'] = ((baseline_realtime['peak_gpu_memory_mb'] - realtime_best['peak_gpu_memory_mb']) /
                                        baseline_realtime['peak_gpu_memory_mb'] * 100)

        realtime_best['latency_improvement'] = ((baseline_realtime['ttft_ms'] - realtime_best['ttft_ms']) /
                                             baseline_realtime['ttft_ms'] * 100)

        # 选择延迟改进最高的heavy_ratio
        best_realtime_config = realtime_best.loc[realtime_best['latency_improvement'].idxmax()]

        config_rows.append({
            '应用场景': '实时对话',
            '推荐H2O配置': f"{int(best_realtime_config['heavy_ratio']*100)}%重要命中",
            '内存节省(%)': round(best_realtime_config['memory_saving'], 2),
            '性能提升(%)': round(best_realtime_config['latency_improvement'], 2),
            '注意事项': '适用于对延迟敏感的场景，如交互式聊天机器人和实时助手'
        })

    # 创建配置推荐表
    config_df = pd.DataFrame(config_rows)

    return config_df

def save_tables_to_excel(tables, output_path):
    """
    将所有表格保存到Excel文件

    Args:
        tables: 表格字典，键为表格名称，值为数据框
        output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path) as writer:
        for table_name, df in tables.items():
            df.to_excel(writer, sheet_name=table_name, index=False)

    print(f"Tables saved to {output_path}")

def save_tables_to_csv(tables, output_dir):
    """
    将所有表格保存为CSV文件

    Args:
        tables: 表格字典，键为表格名称，值为数据框
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for table_name, df in tables.items():
        output_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(output_path, index=False)

    print(f"Tables saved to {output_dir}")

def save_tables_to_markdown(tables, output_dir):
    """
    将所有表格保存为Markdown文件

    Args:
        tables: 表格字典，键为表格名称，值为数据框
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for table_name, df in tables.items():
        output_path = os.path.join(output_dir, f"{table_name}.md")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {table_name}\n\n")
            f.write(df.to_markdown(index=False))

    print(f"Markdown tables saved to {output_dir}")

def save_tables_to_html(tables, output_dir):
    """
    将所有表格保存为HTML文件

    Args:
        tables: 表格字典，键为表格名称，值为数据框
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for table_name, df in tables.items():
        output_path = os.path.join(output_dir, f"{table_name}.html")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"<h1>{table_name}</h1>\n")
            f.write(df.to_html(index=False))

    print(f"HTML tables saved to {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate Experiment Result Tables")
    parser.add_argument("--baseline_dir", type=str, required=True,
                       help="Directory containing baseline experiment results")
    parser.add_argument("--h2o_dir", type=str, default=None,
                       help="Directory containing H2O experiment results (optional)")
    parser.add_argument("--output_dir", type=str, default="./tables",
                       help="Directory to save output tables")
    parser.add_argument("--format", type=str, default="excel",
                       choices=["excel", "csv", "markdown", "html", "all"],
                       help="Output format for tables")

    args = parser.parse_args()

    # 加载实验结果
    baseline_df, h2o_df = load_experiment_results(args.baseline_dir, args.h2o_dir)

    # 创建表格
    tables = {
        "主要性能指标汇总表": create_performance_summary_table(baseline_df, h2o_df),
        "不同语言性能对比表": create_language_comparison_table(baseline_df, h2o_df),
        "最佳配置推荐表": create_best_config_table(baseline_df, h2o_df)
    }

    # 保存表格
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format == "excel" or args.format == "all":
        save_tables_to_excel(tables, os.path.join(args.output_dir, "experiment_results_tables.xlsx"))

    if args.format == "csv" or args.format == "all":
        save_tables_to_csv(tables, os.path.join(args.output_dir, "csv"))

    if args.format == "markdown" or args.format == "all":
        save_tables_to_markdown(tables, os.path.join(args.output_dir, "markdown"))

    if args.format == "html" or args.format == "all":
        save_tables_to_html(tables, os.path.join(args.output_dir, "html"))

    print("Table generation completed!")

if __name__ == "__main__":
    main()