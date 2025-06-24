# validate_data_loading.py
import sys
import os
import json
import random
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hace-kv-optimization'))

logger = logging.getLogger(__name__)

def load_local_jsonl_data(dataset_name, max_samples=None):
    """加载本地JSONL数据文件"""
    possible_paths = [
        f"./data/{dataset_name}.jsonl",
        f"../data/{dataset_name}.jsonl", 
        f"./hace-kv-optimization/baselines/data/{dataset_name}.jsonl",
        f"./hace-kv-optimization/data/{dataset_name}.jsonl"
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print(f"❌ 找不到数据文件: {dataset_name}.jsonl")
        return []
    
    print(f"✅ 加载数据文件: {data_file}")
    
    data = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return []
    
    print(f"✅ 成功加载 {len(data)} 个样本")
    return data

def prepare_samples_for_evaluation(dataset, dataset_name, num_samples=100, random_seed=42):
    """准备用于评估的样本 - 修复版本"""
    random.seed(random_seed)
    print(f"准备 {num_samples} 个样本，数据集: {dataset_name}")
    
    samples = []
    
    # 处理不同的数据格式
    if hasattr(dataset, '__len__'):
        # 标准dataset对象
        dataset_items = dataset
    elif isinstance(dataset, list):
        # 直接的列表格式
        dataset_items = dataset
    else:
        print(f"❌ 未知的数据集格式: {type(dataset)}")
        return []
    
    # 采样
    num_available = len(dataset_items)
    num_to_sample = min(num_samples, num_available)
    indices = random.sample(range(num_available), num_to_sample)
    
    for idx in indices:
        item = dataset_items[idx]
        processed_sample = {}
        
        # 特殊数据集优先处理
        if dataset_name == "multi_news":
            # multi_news特殊处理 - input为空，使用context
            if "context" in item and "answers" in item:
                processed_sample["prompt"] = f"Summarize the following articles:\n{item['context']}\nSummary:"
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["prompt"] = ""
                processed_sample["reference"] = ""
                
        elif dataset_name == "gov_report":
            # gov_report特殊处理 - input为空，使用context
            if "context" in item and "answers" in item:
                processed_sample["prompt"] = f"Summarize the following government report:\n{item['context']}\nSummary:"
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["prompt"] = ""
                processed_sample["reference"] = ""
                
        elif dataset_name == "qmsum":
            # qmsum特殊处理 - 使用input作为问题，context作为会议内容
            if "input" in item and item["input"].strip() and "context" in item:
                query = item["input"].strip()
                processed_sample["prompt"] = f"Context: {item['context']}\nQuestion: {query}\nAnswer:"
            elif "context" in item:
                processed_sample["prompt"] = f"Summarize the following meeting:\n{item['context']}\nSummary:"
            else:
                processed_sample["prompt"] = ""
            
            if "answers" in item:
                if isinstance(item["answers"], list):
                    processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
                else:
                    processed_sample["reference"] = item["answers"]
            else:
                processed_sample["reference"] = ""
        
        # LongBench标准格式处理
        elif "input" in item and "answers" in item and item["input"].strip():
            # 标准LongBench格式 - input不为空
            processed_sample["prompt"] = item["input"]
            
            # 处理答案 - LongBench的答案可能是列表
            if isinstance(item["answers"], list):
                processed_sample["reference"] = item["answers"][0] if item["answers"] else ""
            else:
                processed_sample["reference"] = item["answers"]
                
        elif "context" in item and "question" in item:
            # 问答格式
            processed_sample["prompt"] = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
            if "answers" in item:
                answers = item["answers"]
                if isinstance(answers, list):
                    processed_sample["reference"] = answers[0] if answers else ""
                else:
                    processed_sample["reference"] = answers
            else:
                processed_sample["reference"] = item.get("answer", "")
                
        elif "dialogue" in item and "summary" in item:
            # 摘要格式（如samsum）
            processed_sample["prompt"] = f"Summarize the following dialogue:\n{item['dialogue']}\nSummary:"
            processed_sample["reference"] = item["summary"]
            
        elif "text" in item:
            # 通用文本格式
            processed_sample["prompt"] = item["text"]
            processed_sample["reference"] = item.get("summary", item.get("answer", ""))
            
        else:
            # 未知格式 - 记录并使用备用方案
            print(f"⚠️ 未知的数据格式，数据集: {dataset_name}, 可用字段: {list(item.keys())}")
            # 使用第一个看起来像输入的字段
            for key in ["input", "question", "text", "context", "dialogue"]:
                if key in item:
                    processed_sample["prompt"] = str(item[key])
                    break
            else:
                processed_sample["prompt"] = str(item)
                
            # 尝试找到答案
            for key in ["answers", "answer", "summary", "output"]:
                if key in item:
                    ref = item[key]
                    if isinstance(ref, list):
                        processed_sample["reference"] = ref[0] if ref else ""
                    else:
                        processed_sample["reference"] = str(ref)
                    break
            else:
                processed_sample["reference"] = ""
        
        # 验证处理结果
        if not processed_sample.get("reference"):
            print(f"⚠️ 样本 {idx} 没有找到参考答案，数据集: {dataset_name}")
        
        # 确保都有内容
        if not processed_sample.get("prompt"):
            print(f"⚠️ 样本 {idx} 没有找到输入提示，数据集: {dataset_name}")
            processed_sample["prompt"] = ""
            
        samples.append(processed_sample)
    
    print(f"✅ 成功准备 {len(samples)} 个样本")
    return samples

def validate_dataset_loading(dataset_name):
    """验证数据集加载和处理"""
    print(f"\n验证数据集: {dataset_name}")
    
    # 加载数据
    data = load_local_jsonl_data(dataset_name, max_samples=5)
    if not data:
        print(f"❌ 无法加载数据集 {dataset_name}")
        return False
    
    # 准备样本
    samples = prepare_samples_for_evaluation(data, dataset_name, num_samples=3)
    
    # 检查结果
    print(f"✅ 成功准备 {len(samples)} 个样本")
    for i, sample in enumerate(samples):
        print(f"\n样本 {i+1}:")
        print(f"  Prompt长度: {len(sample.get('prompt', ''))}")
        print(f"  Reference长度: {len(sample.get('reference', ''))}")
        
        # 显示内容预览
        prompt_preview = sample.get('prompt', '')[:200] + "..." if len(sample.get('prompt', '')) > 200 else sample.get('prompt', '')
        ref_preview = sample.get('reference', '')[:100] + "..." if len(sample.get('reference', '')) > 100 else sample.get('reference', '')
        
        print(f"  Prompt预览: {prompt_preview}")
        print(f"  Reference预览: {ref_preview}")
        
        # 检查是否有空的reference
        if not sample.get('reference', '').strip():
            print(f"  ⚠️ 警告: 参考答案为空!")
    
    return len(samples) > 0 and all(s.get('reference', '').strip() for s in samples)

def main():
    """主函数"""
    print("开始验证数据集加载...")
    
    # 测试关键数据集
    test_datasets = ["samsum", "hotpotqa", "narrativeqa", "trec", "triviaqa", "multi_news"]
    
    results = {}
    for ds in test_datasets:
        print(f"\n{'='*60}")
        print(f"验证数据集: {ds}")
        print(f"{'='*60}")
        results[ds] = validate_dataset_loading(ds)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("验证结果汇总:")
    print(f"{'='*60}")
    
    for ds, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{ds:15} - {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 个数据集验证通过")

if __name__ == "__main__":
    main() 