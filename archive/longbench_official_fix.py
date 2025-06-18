#!/usr/bin/env python3
"""
LongBench官方数据修复脚本
解决预处理问题，直接使用LongBench官方数据
基于探索结果：LongBench HotpotQA是独立版本，与原始HotpotQA不匹配
"""

import os
import re
import shutil
from datasets import load_dataset

def verify_longbench_access():
    """验证LongBench数据访问"""
    try:
        print("🔍 验证LongBench数据访问...")
        test_dataset = load_dataset("THUDM/LongBench", "hotpotqa", split="test")
        
        print(f"✅ LongBench访问成功: {len(test_dataset)} 个样本")
        
        # 显示前3个样本用于验证
        for i in range(min(3, len(test_dataset))):
            sample = test_dataset[i]
            print(f"样本 {i+1}: {sample['input'][:60]}... → {sample['answers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LongBench访问失败: {e}")
        return False

def create_backup():
    """创建备份文件"""
    target_file = "hace-kv-optimization/baselines/fullkvcache_main.py"
    if os.path.exists(target_file):
        backup_name = target_file + ".before_longbench_fix"
        shutil.copy2(target_file, backup_name)
        print(f"📁 已创建备份: {backup_name}")
        return True
    return False

def apply_longbench_fix():
    """应用LongBench官方数据修复"""
    
    target_file = "hace-kv-optimization/baselines/fullkvcache_main.py"
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 应用LongBench官方数据修复...")
    
    # 1. 添加LongBench直接加载函数
    longbench_loader = '''
def load_longbench_official_data(dataset_name: str, max_samples: int = None):
    """直接加载LongBench官方数据，绕过有问题的预处理"""
    from datasets import load_dataset
    
    print(f"🌐 加载LongBench官方数据: {dataset_name}")
    print(f"📋 原因：基于探索发现，LongBench是独立版本，更适合学术比较")
    
    try:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        print(f"✅ 成功加载LongBench: {len(dataset)} 个样本")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"📊 限制样本数量: {len(dataset)} 个样本")
        
        return dataset
        
    except Exception as e:
        print(f"❌ LongBench加载失败: {e}")
        raise
'''
    
    # 在imports后添加函数
    if 'load_longbench_official_data' not in content:
        # 找到最后一个import之后插入
        import_end = max(
            content.rfind('from hace_core.utils.unified_monitor import UnifiedMonitor'),
            content.rfind('from eval_utils import')
        )
        if import_end != -1:
            # 找到这行的结尾
            next_newline = content.find('\n', import_end)
            if next_newline != -1:
                content = content[:next_newline+1] + longbench_loader + content[next_newline+1:]
                print("  ✅ 添加了LongBench官方数据加载函数")
    
    # 2. 替换数据加载和样本准备逻辑
    # 找到样本准备的关键部分
    pattern_start = r'# 🔧 绕过有问题的预处理，直接使用原始数据'
    pattern_end = r'print\(f"\[DEBUG\] 预处理后样本: \{samples\[0\] if samples else \'No samples\'\}"\)'
    
    new_loading_logic = '''# LongBench官方数据加载：绕过预处理问题
        print(f"🔧 修复：直接使用LongBench官方数据，绕过预处理问题")

        # 加载LongBench官方数据
        longbench_raw = load_longbench_official_data(dataset_name, actual_num_samples_to_prepare)

        # 手动转换为实验格式，确保正确的字段映射
        samples = []
        for i, raw_sample in enumerate(longbench_raw):
            if 'input' not in raw_sample or 'answers' not in raw_sample:
                print(f"⚠️ 跳过LongBench样本 {i+1}，格式异常")
                continue
            
            # 标准化答案格式
            answers = raw_sample['answers']
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = [str(answers)]
            
            # 创建正确的实验样本格式
            experiment_sample = {
                'prompt': raw_sample['input'],
                'reference': answers,  # 关键修复：确保答案正确映射
                'context': raw_sample.get('context', ''),
                'sample_id': raw_sample.get('_id', f'longbench_{i}'),
                'original_sample': raw_sample,
                'data_source': 'longbench_official'
            }
            
            samples.append(experiment_sample)
            
            print(f"[DEBUG] LongBench样本 {i+1}: {experiment_sample['prompt'][:50]}... → {experiment_sample['reference']}")

        # 设置数据源标识
        dataset_source = "longbench_official"
        print(f"✅ LongBench数据准备完成: {len(samples)} 个样本")'''
    
    # 使用正则表达式查找并替换整个代码段
    full_pattern = r'# 🔧 绕过有问题的预处理，直接使用原始数据.*?print\(f"\[DEBUG\] 预处理后样本: \{samples\[0\] if samples else \'No samples\'\}"\)'
    
    import re
    match = re.search(full_pattern, content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), new_loading_logic)
        print("  ✅ 更新了数据加载逻辑")
    else:
        print("  ⚠️ 未找到预处理代码段，可能需要手动修改")
    
    # 3. 更新答案提取函数
    new_extract_function = '''def extract_ground_truth_from_sample(sample, dataset_source):
    """修复版：专门处理LongBench官方数据"""
    
    # LongBench官方数据处理
    if dataset_source == "longbench_official":
        # 优先使用标准化的reference字段
        if 'reference' in sample and sample['reference']:
            reference = sample['reference']
            if isinstance(reference, list) and reference:
                result = [str(item).strip() for item in reference if item and str(item).strip()]
                if result:
                    print(f"[DEBUG] ✅ LongBench答案: {result}")
                    return result
            elif isinstance(reference, str) and reference.strip():
                print(f"[DEBUG] ✅ LongBench答案: [{reference.strip()}]")
                return [reference.strip()]
        
        # 回退到原始样本
        if 'original_sample' in sample and 'answers' in sample['original_sample']:
            original_answers = sample['original_sample']['answers']
            if isinstance(original_answers, list) and original_answers:
                result = [str(item).strip() for item in original_answers if item and str(item).strip()]
                if result:
                    print(f"[DEBUG] ✅ LongBench原始答案: {result}")
                    return result
            elif isinstance(original_answers, str) and original_answers.strip():
                print(f"[DEBUG] ✅ LongBench原始答案: [{original_answers.strip()}]")
                return [original_answers.strip()]
    
    # 通用处理（其他数据源）
    answer_fields = ['reference', 'answers', 'answer', 'output']
    for field in answer_fields:
        if field in sample and sample[field]:
            value = sample[field]
            if isinstance(value, list) and value:
                result = [str(item).strip() for item in value if item and str(item).strip()]
                if result:
                    return result
            elif isinstance(value, str) and value.strip():
                return [value.strip()]
    
    print(f"[DEBUG] ❌ 未找到答案字段")
    return ["Unknown"]'''
    
    # 替换答案提取函数
    extract_pattern = r'def extract_ground_truth_from_sample\(.*?\n(.*?\n)*?.*?return \["Unknown"\]'
    content = re.sub(extract_pattern, new_extract_function, content, flags=re.DOTALL)
    print("  ✅ 更新了答案提取函数")
    
    # 保存修改
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("🎉 LongBench官方数据修复完成！")

def test_fix():
    """提供测试建议"""
    print("\n🧪 建议测试命令:")
    print("cd hace-kv-optimization/baselines")
    print("python fullkvcache_main.py --datasets hotpotqa --kv_cache_lengths 512 --batch_sizes 1 --max_new_tokens 10 --repetitions 1 --enable_scoring --is_baseline_run")

def main():
    print("🌐 LongBench官方数据修复脚本")
    print("基于探索结果：使用LongBench独立版本，确保学术标准")
    print("=" * 70)
    
    # 1. 验证LongBench访问
    if not verify_longbench_access():
        print("❌ 无法访问LongBench，请检查网络连接")
        return
    
    # 2. 创建备份
    if not create_backup():
        print("❌ 无法创建备份")
        return
    
    # 3. 应用修复
    try:
        apply_longbench_fix()
        print("\n🎉 修复成功！现在使用LongBench官方数据")
        print("📊 这确保了与其他LongBench研究的可比较性")
        test_fix()
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        print("可从备份恢复: cp hace-kv-optimization/baselines/fullkvcache_main.py.before_longbench_fix hace-kv-optimization/baselines/fullkvcache_main.py")

if __name__ == "__main__":
    main() 