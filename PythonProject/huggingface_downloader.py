#!/usr/bin/env python3
"""
Hugging Face 模型下载器
支持多种模型的自动下载和缓存
"""

import os
import sys
from pathlib import Path
import argparse
from utilities.path_config import PathManager

def install_requirements():
    """安装必要的依赖"""
    try:
        import transformers
        import torch
        print("✅ 依赖已安装")
    except ImportError:
        print("📦 安装必要依赖...")
        os.system("pip install transformers torch tqdm accelerate")

def download_model(model_name, save_dir=None, cache_only=False):
    """
    下载Hugging Face模型
    
    Args:
        model_name: 模型名称 (如 'meta-llama/Llama-2-7b-chat-hf')
        save_dir: 保存目录 (可选)
        cache_only: 是否只缓存到默认位置
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"🚀 开始下载模型: {model_name}")
        
        # 配置缓存目录
        if save_dir and not cache_only:
            cache_dir = save_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = None
            
        print("📥 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print("📥 下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if save_dir and not cache_only:
            print(f"💾 保存到: {save_dir}")
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            
        print("✅ 模型下载完成！")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Hugging Face 模型下载器")
    parser.add_argument("--model", "-m", default="meta-llama/Llama-2-7b-chat-hf", 
                        help="模型名称")
    pm = PathManager()
    default_dir = pm.paths["model_paths"]["local_models"]
    parser.add_argument("--save-dir", "-s",
                        default=default_dir,
                        help="保存目录")
    parser.add_argument("--cache-only", action="store_true",
                        help="只缓存到默认位置")
    parser.add_argument("--list-models", action="store_true",
                        help="显示可用模型列表")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("🤗 推荐的开源模型:")
        models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf", 
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/DialoGPT-medium",
            "facebook/opt-6.7b",
            "EleutherAI/gpt-j-6b"
        ]
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        return
    
    # 安装依赖
    install_requirements()
    
    # 下载模型
    success = download_model(args.model, args.save_dir, args.cache_only)
    
    if success:
        print(f"🎉 成功下载模型: {args.model}")
        if not args.cache_only:
            print(f"📁 保存位置: {args.save_dir}")
    else:
        print("💥 下载失败，请检查网络连接和模型名称")

if __name__ == "__main__":
    main() 