#!/usr/bin/env python3
"""
下载正确的HotpotQA验证集（带答案）
使用特定commit hash避免test split问题
"""

import requests
import json
from pathlib import Path

def download_correct_hotpotqa():
    """下载带答案的HotpotQA验证集"""
    
    # 使用用户提供的特定commit hash URL
    url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/6c00d1e/hotpotqa/hotpotqa_val.jsonl"
    
    output_file = Path("data/hotpotqa.jsonl")
    
    print(f"🚀 下载HotpotQA验证集（带答案）...")
    print(f"URL: {url}")
    print(f"目标文件: {output_file}")
    
    try:
        # 确保目录存在
        output_file.parent.mkdir(exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 下载成功: {output_file}")
        print(f"文件大小: {output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # 验证文件内容
        print("\n🔍 验证文件内容...")
        sample_count = 0
        samples_with_answers = 0
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        sample_count += 1
                        
                        # 检查前几个样本
                        if line_num < 3:
                            print(f"样本 {line_num + 1}:")
                            print(f"  input长度: {len(data.get('input', ''))}")
                            print(f"  answers: {data.get('answers', 'N/A')}")
                            print(f"  length: {data.get('length', 'N/A')}")
                        
                        # 统计有答案的样本
                        if data.get('answers') and data['answers'] != '':
                            samples_with_answers += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n📊 验证结果:")
        print(f"总样本数: {sample_count}")
        print(f"有答案样本数: {samples_with_answers}")
        print(f"答案覆盖率: {samples_with_answers/sample_count*100:.1f}%")
        
        if samples_with_answers > 0:
            print("\n🎉 验证成功！文件包含答案，可以用于评分")
            return True
        else:
            print("\n❌ 警告：文件中没有找到答案")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_correct_hotpotqa()
    if success:
        print("\n下一步:")
        print("python hace-kv-optimization/fullkvcache_main_optimized.py --datasets 'hotpotqa' --kv_cache_lengths '512' --enable_scoring --is_baseline_run")
    else:
        print("\n请检查网络连接或手动下载文件") 