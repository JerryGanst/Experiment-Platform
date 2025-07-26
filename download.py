#!/usr/bin/env python3
"""
专用VPN模型下载器 - 适配您的Clash配置
使用SOCKS5代理 localhost:7891 或 HTTP代理 localhost:7890
"""

import os
import sys
import requests
import time
from pathlib import Path
import argparse

def create_vpn_downloader():
    """创建配置了您的VPN代理的下载器"""
    
    # 根据您的Clash配置设置代理
    proxy_config = {
        'http': 'socks5://127.0.0.1:7891',
        'https': 'socks5://127.0.0.1:7891'
    }
    
    session = requests.Session()
    session.proxies = proxy_config
    
    # 设置User-Agent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    print(f"🌐 已配置代理: SOCKS5://127.0.0.1:7891")
    return session

def test_vpn_connection(session):
    """测试VPN连接"""
    print("🔍 测试VPN连接...")
    
    test_sites = [
        ("Google", "https://www.google.com"),
        ("HuggingFace", "https://huggingface.co"),
        ("GitHub", "https://github.com")
    ]
    
    for name, url in test_sites:
        try:
            start_time = time.time()
            response = session.get(url, timeout=10)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                print(f"  ✅ {name}: 连接成功 ({latency:.2f}s)")
            else:
                print(f"  ⚠️ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")

def download_model_with_vpn(model_name, save_dir="./models"):
    """使用VPN下载模型"""
    
    print(f"🚀 开始下载模型: {model_name}")
    
    # 创建VPN会话
    session = create_vpn_downloader()
    
    # 测试连接
    test_vpn_connection(session)
    
    # 镜像站点（优先使用官方源，因为有VPN）
    mirrors = [
        "https://huggingface.co",
        "https://hf-mirror.com"
    ]
    
    # 选择最快的镜像
    best_mirror = mirrors[0]
    best_time = float('inf')
    
    print("\n🔍 测试镜像速度...")
    for mirror in mirrors:
        try:
            start_time = time.time()
            response = session.get(f"{mirror}/api/models", timeout=5)
            if response.status_code == 200:
                latency = time.time() - start_time
                print(f"  {mirror}: {latency:.2f}s")
                if latency < best_time:
                    best_time = latency
                    best_mirror = mirror
        except:
            print(f"  {mirror}: 连接失败")
    
    print(f"✅ 选择镜像: {best_mirror}")
    
    # 核心文件列表
    files_to_download = [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors"  # 优先下载safetensors格式
    ]
    
    # 获取可用文件列表
    try:
        api_url = f"{best_mirror}/api/models/{model_name}"
        response = session.get(api_url, timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            available_files = [f['rfilename'] for f in model_info.get('siblings', [])]
            print(f"📋 模型包含 {len(available_files)} 个文件")
            
            # 扩展文件列表（包含所有必要文件）
            extended_files = [
                "config.json",
                "tokenizer.json", 
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.safetensors",
                "pytorch_model.bin"
            ]
            
            files_to_download = [f for f in extended_files if f in available_files]
            print(f"📥 将下载 {len(files_to_download)} 个文件")
    except Exception as e:
        print(f"⚠️ 无法获取文件列表: {e}")
    
    # 开始下载
    success_count = 0
    total_files = len(files_to_download)
    
    for i, filename in enumerate(files_to_download):
        print(f"\n📁 [{i+1}/{total_files}] 下载: {filename}")
        
        file_url = f"{best_mirror}/{model_name}/resolve/main/{filename}"
        local_path = os.path.join(save_dir, model_name, filename)
        
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        try:
            response = session.get(file_url, stream=True, timeout=60)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    start_time = time.time()
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                elapsed = time.time() - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                
                                print(f"\r  进度: {progress:.1f}% "
                                     f"({downloaded / 1024 / 1024:.1f}MB/"
                                     f"{total_size / 1024 / 1024:.1f}MB) "
                                     f"速度: {speed / 1024:.1f}KB/s", end='')
                
                print(f"\n  ✅ {filename} 下载完成!")
                success_count += 1
            else:
                print(f"  ❌ HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")

    print(f"\n🎉 下载完成! 成功: {success_count}/{total_files}")
    
    if success_count > 0:
        model_path = os.path.join(save_dir, model_name)
        print(f"📍 模型保存在: {model_path}")
        
        # 显示文件大小
        total_size = 0
        try:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            print(f"💾 总大小: {total_size / 1024 / 1024:.1f}MB")
        except:
            pass
    
    return success_count == total_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VPN模型下载器")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", help="模型名称")
    parser.add_argument("--save-dir", default="./models", help="保存目录")
    
    args = parser.parse_args()
    
    success = download_model_with_vpn(args.model, args.save_dir)
    
    if success:
        print("\n🎊 所有文件下载成功!")
    else:
        print("\n⚠️ 部分文件下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main()