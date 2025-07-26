#!/usr/bin/env python3
"""
快速模型下载器
支持断点续传、多线程下载和多种镜像站点
"""

import os
import sys
import requests
import threading
import time
from pathlib import Path
from typing import Optional, List
import argparse
from concurrent.futures import ThreadPoolExecutor

class FastModelDownloader:
    """快速模型下载器"""
    
    def __init__(self):
        self.mirrors = [
            "https://huggingface.co",
            "https://hf-mirror.com",
            "https://huggingface.co"  # 备用官方源
        ]
        self.chunk_size = 8192  # 8KB chunks
        self.max_retries = 3
        
    def test_mirror_speed(self, mirror: str) -> float:
        """测试镜像站点速度"""
        try:
            start_time = time.time()
            response = requests.get(f"{mirror}/api/models", timeout=5)
            if response.status_code == 200:
                return time.time() - start_time
            else:
                return float('inf')
        except:
            return float('inf')
    
    def select_fastest_mirror(self) -> str:
        """选择最快的镜像站点"""
        print("🔍 测试镜像站点速度...")
        
        speeds = {}
        for mirror in self.mirrors:
            speed = self.test_mirror_speed(mirror)
            speeds[mirror] = speed
            print(f"  {mirror}: {speed:.2f}s")
        
        fastest = min(speeds.items(), key=lambda x: x[1])
        print(f"✅ 选择最快镜像: {fastest[0]} ({fastest[1]:.2f}s)")
        return fastest[0]
    
    def download_file_with_resume(
        self, 
        url: str, 
        local_path: str, 
        desc: str = ""
    ) -> bool:
        """支持断点续传的文件下载"""
        
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 检查已下载的部分
        resume_pos = 0
        if os.path.exists(local_path):
            resume_pos = os.path.getsize(local_path)
            print(f"📂 发现已下载 {resume_pos / 1024 / 1024:.1f}MB，继续下载...")
        
        # 设置请求头进行断点续传
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            if response.status_code in [200, 206]:  # 200: 完整下载, 206: 部分下载
                total_size = resume_pos
                if 'content-length' in response.headers:
                    total_size += int(response.headers['content-length'])
                
                mode = 'ab' if resume_pos > 0 else 'wb'
                downloaded = resume_pos
                
                with open(local_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r⏬ {desc}: {progress:.1f}% "
                                     f"({downloaded / 1024 / 1024:.1f}MB / "
                                     f"{total_size / 1024 / 1024:.1f}MB)", end='')
                
                print(f"\n✅ {desc} 下载完成！")
                return True
            else:
                print(f"❌ 下载失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 下载错误: {e}")
            return False
    
    def download_model_files(
        self, 
        model_name: str, 
        save_dir: str,
        files_to_download: Optional[List[str]] = None
    ) -> bool:
        """下载模型文件"""
        
        # 选择最快的镜像
        mirror = self.select_fastest_mirror()
        
        # 默认下载的文件列表
        if files_to_download is None:
            files_to_download = [
                "config.json",
                "tokenizer.json", 
                "tokenizer_config.json",
                "special_tokens_map.json",
                "pytorch_model.bin",  # 或者 model.safetensors
                "model.safetensors"
            ]
        
        # 尝试获取文件列表
        api_url = f"{mirror}/api/models/{model_name}"
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                available_files = [f['rfilename'] for f in model_info.get('siblings', [])]
                print(f"📋 可用文件: {len(available_files)} 个")
                
                # 过滤实际存在的文件
                files_to_download = [f for f in files_to_download if f in available_files]
        except:
            print("⚠️  无法获取文件列表，使用默认列表")
        
        print(f"📥 准备下载 {len(files_to_download)} 个文件...")
        
        success_count = 0
        for file_name in files_to_download:
            file_url = f"{mirror}/{model_name}/resolve/main/{file_name}"
            local_path = os.path.join(save_dir, model_name, file_name)
            
            print(f"\n📁 下载: {file_name}")
            
            # 尝试下载，支持重试
            for attempt in range(self.max_retries):
                if self.download_file_with_resume(file_url, local_path, file_name):
                    success_count += 1
                    break
                else:
                    if attempt < self.max_retries - 1:
                        print(f"🔄 重试 ({attempt + 1}/{self.max_retries})...")
                        time.sleep(2)
                    else:
                        print(f"❌ {file_name} 下载失败")
        
        print(f"\n🎉 下载完成! 成功: {success_count}/{len(files_to_download)}")
        return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="快速模型下载器")
    parser.add_argument("model_name", help="模型名称 (如: mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--save-dir", default="./models", help="保存目录 (默认: ./models)")
    parser.add_argument("--files", nargs="+", help="指定要下载的文件")
    
    args = parser.parse_args()
    
    downloader = FastModelDownloader()
    
    print(f"🚀 开始下载模型: {args.model_name}")
    print(f"📂 保存到: {args.save_dir}")
    
    success = downloader.download_model_files(
        args.model_name, 
        args.save_dir,
        args.files
    )
    
    if success:
        print(f"\n✅ 模型下载完成!")
        print(f"📍 模型位置: {os.path.join(args.save_dir, args.model_name)}")
    else:
        print(f"\n❌ 模型下载失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 