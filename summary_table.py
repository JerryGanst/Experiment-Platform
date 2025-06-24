# create_summary_table.py
import pandas as pd
import glob
import json
import os


def collect_all_results():
    """收集所有实验结果"""
    all_results = []

    # 数据集映射
    dataset_names = {
        'narrativeqa': 'NrtvQA',
        'qasper': 'Qasper',
        'multifieldqa_en': 'MF-en',
        'hotpotqa': 'HotpotQA',
        '2wikimqa': '2WikiMQA',
        'musique': 'Musique',
        'gov_report': 'GovReport',
        'qmsum': 'QMSum',
        'multi_news': 'MultiNews',
        'trec': 'TREC',
        'triviaqa': 'TriviaQA',
        'samsum': 'SAMSum',
        'passage_count': 'PCount',
        'passage_retrieval_en': 'PR-en',
        'lcc': 'Lcc',
        'repobench-p': 'RB-P'
    }

    # 搜索所有evaluation_results文件
    result_files = glob.glob('./**/evaluation_results_*.json', recursive=True)

    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'average_score' in data and 'dataset' in data:
                dataset = data['dataset']
                score = data['average_score']

                # 从文件路径提取KV cache长度
                kv_length = 128 if 'kv128' in file_path else 1024

                all_results.append({
                    'Dataset': dataset_names.get(dataset, dataset),
                    'KV_Length': kv_length,
                    'Score': score
                })

        except Exception as e:
            print(f"处理文件失败 {file_path}: {e}")

    return all_results


def create_summary_table():
    """创建汇总表格"""
    results = collect_all_results()
    df = pd.DataFrame(results)

    # 创建透视表
    pivot_table = df.pivot_table(
        index='Dataset',
        columns='KV_Length',
        values='Score',
        aggfunc='mean'
    )

    # 添加平均分列
    pivot_table['Avg.'] = pivot_table.mean(axis=1)

    # 格式化输出
    pivot_table = pivot_table.round(3)

    print("LongBench评估结果汇总:")
    print("=" * 50)
    print(pivot_table.to_string())

    # 保存到CSV
    pivot_table.to_csv('longbench_summary.csv')
    print(f"\n结果已保存到: longbench_summary.csv")

    return pivot_table


if __name__ == "__main__":
    create_summary_table()