#!/usr/bin/env python3
"""
VPN/代理模型下载器
支持各种代理设置、堡垒机环境和网络优化
"""

import os
import sys
import requests
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VPNModelDownloader:
    """支持VPN/代理的模型下载器"""
    
    def __init__(self, proxy_config: Optional[Dict] = None):
        self.proxy_config = proxy_config
        self.session = self._create_session()
        
        # 国内外镜像站点
        self.mirrors = [
            "https://huggingface.co",
            "https://hf-mirror.com",
            "https://huggingface-mirror.com",
            "https://hub.nuaa.cf",  # 南京航空航天大学镜像
            "https://hf.baai.ac.cn", # 智源镜像
        ]
        
        self.chunk_size = 32768  # 32KB chunks for better performance
        self.max_retries = 5
        self.timeout = 60
        
    def _create_session(self) -> requests.Session:
        """创建带有代理和重试机制的会话"""
        session = requests.Session()
        
        # 设置代理
        if self.proxy_config:
            session.proxies = self.proxy_config
            print(f"🌐 使用代理: {self.proxy_config}")
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置User-Agent
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        return session
    
    def test_network_connectivity(self) -> Dict[str, Any]:
        """测试网络连通性"""
        print("🔍 测试网络连通性...")
        
        results = {}
        
        # 测试基本连通性
        test_urls = [
            ("Google", "https://www.google.com"),
            ("GitHub", "https://github.com"),
            ("HuggingFace", "https://huggingface.co"),
            ("HF Mirror", "https://hf-mirror.com")
        ]
        
        for name, url in test_urls:
            try:
                start_time = time.time()
                response = self.session.get(url, timeout=10)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    results[name] = {"status": "✅", "latency": f"{latency:.2f}s"}
                else:
                    results[name] = {"status": "⚠️", "latency": f"{latency:.2f}s"}
            except Exception as e:
                results[name] = {"status": "❌", "error": str(e)[:50]}
        
        # 显示结果
        for name, result in results.items():
            if "error" in result:
                print(f"  {name}: {result['status']} ({result['error']})")
            else:
                print(f"  {name}: {result['status']} ({result['latency']})")
        
        return results
    
    def test_mirror_speed(self, mirror: str) -> float:
        """测试镜像站点速度"""
        try:
            start_time = time.time()
            # 测试小文件下载速度
            test_url = f"{mirror}/microsoft/DialoGPT-small/resolve/main/config.json"
            response = self.session.get(test_url, timeout=15)
            
            if response.status_code == 200:
                latency = time.time() - start_time
                # 计算下载速度 (bytes/second)
                speed = len(response.content) / latency if latency > 0 else 0
                return latency, speed
            else:
                return float('inf'), 0
        except Exception as e:
            print(f"  ❌ {mirror}: {e}")
            return float('inf'), 0
    
    def select_best_mirror(self) -> str:
        """选择最佳镜像站点"""
        print("🚀 测试镜像站点速度...")
        
        best_mirror = self.mirrors[0]
        best_score = float('inf')
        
        for mirror in self.mirrors:
            print(f"  测试 {mirror}...")
            latency, speed = self.test_mirror_speed(mirror)
            
            if latency < float('inf'):
                # 综合评分：延迟权重0.7 + 速度权重0.3
                score = latency * 0.7 + (1000000 / max(speed, 1)) * 0.3
                print(f"    延迟: {latency:.2f}s, 速度: {speed/1024:.1f}KB/s, 评分: {score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_mirror = mirror
            else:
                print(f"    ❌ 连接失败")
        
        print(f"✅ 选择镜像: {best_mirror}")
        return best_mirror
    
    def download_with_progress(
        self, 
        url: str, 
        local_path: str, 
        desc: str = ""
    ) -> bool:
        """带进度条的下载"""
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 检查断点续传
        resume_pos = 0
        if os.path.exists(local_path):
            resume_pos = os.path.getsize(local_path)
            if resume_pos > 0:
                print(f"📂 断点续传: 已下载 {resume_pos / 1024 / 1024:.1f}MB")
        
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        try:
            response = self.session.get(
                url, 
                headers=headers, 
                stream=True, 
                timeout=self.timeout
            )
            
            if response.status_code in [200, 206]:
                total_size = resume_pos
                if 'content-length' in response.headers:
                    total_size += int(response.headers['content-length'])
                
                downloaded = resume_pos
                start_time = time.time()
                
                mode = 'ab' if resume_pos > 0 else 'wb'
                
                with open(local_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 计算速度和进度
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = (downloaded - resume_pos) / elapsed
                                
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    eta = (total_size - downloaded) / speed if speed > 0 else 0
                                    
                                    print(f"\r⏬ {desc}: {progress:.1f}% "
                                         f"({downloaded / 1024 / 1024:.1f}MB/"
                                         f"{total_size / 1024 / 1024:.1f}MB) "
                                         f"速度: {speed / 1024:.1f}KB/s "
                                         f"剩余: {eta / 60:.1f}min", end='')
                
                print(f"\n✅ {desc} 下载完成！")
                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.reason}")
                return False
                
        except Exception as e:
            print(f"❌ 下载错误: {e}")
            return False
    
    def download_model(
        self, 
        model_name: str, 
        save_dir: str = "./models",
        essential_only: bool = False
    ) -> bool:
        """下载模型"""
        
        print(f"🤖 下载模型: {model_name}")
        
        # 网络连通性测试
        connectivity = self.test_network_connectivity()
        
        # 选择最佳镜像
        mirror = self.select_best_mirror()
        
        # 文件列表
        if essential_only:
            files = [
                "config.json",
                "tokenizer_config.json", 
                "model.safetensors"  # 优先下载safetensors格式
            ]
        else:
            files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.safetensors",
                "pytorch_model.bin"
            ]
        
        # 获取实际可用文件
        api_url = f"{mirror}/api/models/{model_name}"
        try:
            response = self.session.get(api_url, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                available_files = [f['rfilename'] for f in model_info.get('siblings', [])]
                files = [f for f in files if f in available_files]
                print(f"📋 找到 {len(files)} 个文件")
        except:
            print("⚠️ 无法获取文件列表，使用默认列表")
        
        # 下载文件
        success_count = 0
        total_files = len(files)
        
        for i, filename in enumerate(files):
            print(f"\n📁 [{i+1}/{total_files}] {filename}")
            
            file_url = f"{mirror}/{model_name}/resolve/main/{filename}"
            local_path = os.path.join(save_dir, model_name, filename)
            
            # 重试下载
            for attempt in range(self.max_retries):
                if self.download_with_progress(file_url, local_path, filename):
                    success_count += 1
                    break
                else:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"🔄 重试 {attempt + 1}/{self.max_retries}，等待 {wait_time}s...")
                        time.sleep(wait_time)
        
        print(f"\n🎉 下载完成! 成功: {success_count}/{total_files}")
        
        if success_count > 0:
            model_path = os.path.join(save_dir, model_name)
            print(f"📍 模型保存在: {model_path}")
            
            # 显示文件大小
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            print(f"💾 总大小: {total_size / 1024 / 1024:.1f}MB")
        
        return success_count == total_files

def main():
    parser = argparse.ArgumentParser(description="VPN/代理模型下载器")
    parser.add_argument("model_name", help="模型名称")
    parser.add_argument("--save-dir", default="./models", help="保存目录")
    parser.add_argument("--proxy", help="代理设置 (如: http://127.0.0.1:1080)")
    parser.add_argument("--socks-proxy", help="SOCKS代理 (如: socks5://127.0.0.1:1080)")
    parser.add_argument("--essential-only", action="store_true", help="只下载核心文件")
    parser.add_argument("--test-network", action="store_true", help="仅测试网络连通性")
    
    args = parser.parse_args()
    
    # 配置代理
    proxy_config = None
    if args.proxy:
        proxy_config = {"http": args.proxy, "https": args.proxy}
    elif args.socks_proxy:
        proxy_config = {"http": args.socks_proxy, "https": args.socks_proxy}
    
    downloader = VPNModelDownloader(proxy_config)
    
    if args.test_network:
        print("🔧 网络连通性测试模式")
        downloader.test_network_connectivity()
        sys.exit(0)
    
    success = downloader.download_model(
        args.model_name,
        args.save_dir,
        args.essential_only
    )
    
    if success:
        print("\n🎊 所有文件下载成功!")
    else:
        print("\n⚠️ 部分文件下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 