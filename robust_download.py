#!/usr/bin/env python3
"""
健壮的模型下载脚本
专门针对网络慢、连接不稳定的情况
支持断点续传、智能重试、网络优化
"""

import os
import sys
import time
import signal
from pathlib import Path
from huggingface_hub import snapshot_download, HfFolder

def setup_environment():
    """设置环境变量"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = '/home/luxshare/.cache/huggingface'
    
    # 创建缓存目录
    cache_dir = Path(os.environ['HF_HOME'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🌐 镜像地址: {os.environ['HF_ENDPOINT']}")
    print(f"📁 缓存目录: {os.environ['HF_HOME']}")

def robust_download(model_name, local_path, max_retries=10):
    """健壮的下载函数，支持断点续传和重试"""
    print(f"\n🎯 开始下载: {model_name}")
    print(f"💾 保存路径: {local_path}")
    print(f"🔄 最大重试次数: {max_retries}")
    
    # 创建本地目录
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    # 获取 HuggingFace token
    token = None
    
    # 检查是否已登录
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ HuggingFace 已登录")
        else:
            print("⚠️  未登录 HuggingFace，某些模型可能无法下载")
            print("如需下载受限模型，请先运行: huggingface-cli login")
    except ImportError:
        print("⚠️  huggingface_hub 未安装，某些模型可能无法下载")
        print("安装方法: pip install huggingface_hub")
    
    # 检查已有文件
    existing_files = list(Path(local_path).glob("*"))
    if existing_files:
        print(f"📂 发现 {len(existing_files)} 个已有文件，将断点续传")
    
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 尝试 {attempt + 1}/{max_retries}")
            
            # 根据网络情况调整参数
            if attempt < 3:
                # 前3次尝试使用正常配置
                timeout = 600  # 10分钟
                max_workers = 2
            else:
                # 后续尝试使用保守配置
                timeout = 1800  # 30分钟
                max_workers = 1
                print("🐌 使用保守配置（单线程，长超时）")
            
            print(f"⚙️  超时时间: {timeout}秒, 并发数: {max_workers}")
            
            # 开始下载
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,  # 断点续传
                token=token,
                timeout=timeout,
                max_workers=max_workers,
            )
            
            print("\n✅ 下载完成！")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"\n❌ 尝试 {attempt + 1} 失败: {str(e)}")
            
            # 分析错误类型并给出建议
            if "timeout" in error_msg:
                print("🔸 超时错误 - 网络较慢，将增加超时时间")
            elif "connection" in error_msg:
                print("🔸 连接错误 - 网络不稳定，将重试")
            elif "403" in error_msg or "401" in error_msg:
                print("🔸 权限错误 - 需要登录HuggingFace账户")
                print("   请运行: huggingface-cli login")
                return False
            elif "404" in error_msg:
                print("🔸 模型不存在 - 请检查模型名称")
                return False
            
            if attempt < max_retries - 1:
                # 指数退避等待
                wait_time = min(2 ** attempt * 5, 60)  # 最多等待60秒
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                
                for i in range(wait_time):
                    print(f"\r⏳ 剩余: {wait_time - i} 秒", end="", flush=True)
                    time.sleep(1)
                print()
    
    print("❌ 所有重试都失败了")
    return False

def verify_model(local_path):
    """验证下载的模型"""
    try:
        print("\n🔍 验证模型...")
        
        # 检查基本文件
        config_file = Path(local_path) / "config.json"
        if not config_file.exists():
            print("❌ 缺少 config.json")
            return False
        
        # 检查模型文件
        model_files = (
            list(Path(local_path).glob("*.safetensors")) +
            list(Path(local_path).glob("*.bin")) +
            list(Path(local_path).glob("pytorch_model*.bin"))
        )
        
        if not model_files:
            print("❌ 没有找到模型权重文件")
            return False
        
        # 统计信息
        total_files = len(list(Path(local_path).glob("*")))
        total_size = sum(f.stat().st_size for f in Path(local_path).glob("*") if f.is_file())
        
        print(f"✅ 总文件数: {total_files}")
        print(f"✅ 权重文件: {len(model_files)}")
        print(f"✅ 总大小: {total_size / (1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 健壮模型下载器")
    print("   🔄 断点续传 | 🔁 智能重试 | �� 镜像加速")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("\n使用方法:")
        print("  python3 robust_download.py <模型名称> <本地路径>")
        print("\n示例:")
        print("  python3 robust_download.py mistralai/Mistral-7B-Instruct-v0.3 ./models/mistral-7b")
        print("  python3 robust_download.py microsoft/DialoGPT-medium ./models/dialogpt")
        print("  python3 robust_download.py facebook/opt-350m ./models/opt350m")
        print("\n特点:")
        print("  - 断点续传：网络断了重新运行继续下载")
        print("  - 智能重试：自动重试10次不放弃")
        print("  - 登录检测：自动检测HuggingFace登录状态")
        sys.exit(1)
    
    model_name = sys.argv[1]
    local_path = sys.argv[2]
    
    print(f"\n📦 模型: {model_name}")
    print(f"📁 路径: {local_path}")
    
    # 设置环境
    setup_environment()
    
    # 设置信号处理（优雅中断）
    def signal_handler(signum, frame):
        print("\n\n⚠️  收到中断信号...")
        print("💾 已下载的文件已保存，下次运行时自动续传")
        print("👋 再见！")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 开始下载
    success = robust_download(model_name, local_path)
    
    if not success:
        print("\n💔 下载失败")
        sys.exit(1)
    
    # 验证模型
    if not verify_model(local_path):
        print("\n⚠️  模型可能不完整，但文件已保存")
        print("🔄 可以重新运行脚本继续下载")
        sys.exit(1)
    
    print(f"\n🎉 模型下载成功！")
    print(f"�� 保存在: {local_path}")
    print("\n✨ 可以开始使用模型了！")

if __name__ == "__main__":
    main()
