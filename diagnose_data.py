# diagnose_data.py
import json
import os
from pathlib import Path

def diagnose_dataset(dataset_name):
    """诊断数据集格式"""
    # 查找数据文件
    possible_paths = [
        f"./data/{dataset_name}.jsonl",
        f"../data/{dataset_name}.jsonl",
        f"../../data/{dataset_name}.jsonl",
        f"./hace-kv-optimization/baselines/data/{dataset_name}.jsonl",
        f"./hace-kv-optimization/data/{dataset_name}.jsonl"
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print(f"❌ 找不到 {dataset_name}.jsonl")
        print("搜索路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"✅ 找到数据文件: {data_file}")
    
    # 读取前几个样本
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只看前3个样本
                    break
                
                line = line.strip()
                if not line:
                    continue
                    
                sample = json.loads(line)
                print(f"\n样本 {i+1}:")
                print(f"  字段: {list(sample.keys())}")
                
                # 打印每个字段的前100个字符
                for key, value in sample.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                    elif isinstance(value, list):
                        preview = f"列表，长度: {len(value)}"
                        if len(value) > 0:
                            first_item = value[0]
                            if isinstance(first_item, str):
                                preview += f", 第一个元素: {first_item[:50]}..."
                            else:
                                preview += f", 第一个元素类型: {type(first_item)}"
                    elif isinstance(value, dict):
                        preview = f"字典，字段: {list(value.keys())}"
                    else:
                        preview = str(value)
                    print(f"  {key}: {preview}")
                    
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")

def main():
    """主函数"""
    print("开始诊断数据集格式...")
    
    # 诊断您使用的数据集
    datasets = ["trec", "triviaqa", "samsum", "hotpotqa", "narrativeqa", "multi_news", 
                "2wikimqa", "musique", "gov_report", "qmsum", "qasper"]
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"诊断数据集: {ds}")
        print(f"{'='*60}")
        diagnose_dataset(ds)

if __name__ == "__main__":
    main() 