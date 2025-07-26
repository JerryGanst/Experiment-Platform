#!/usr/bin/env python3
"""
简化VPN下载器 - 使用HTTP代理，无额外依赖
"""

import os
import sys
import requests
import time
import argparse

def download_with_http_proxy(model_name, save_dir="./models", proxy_port=7890):
    """使用HTTP代理下载模型"""
    
    # 配置HTTP代理
    proxy_config = {
        'http': f'http://127.0.0.1:{proxy_port}',
        'https': f'http://127.0.0.1:{proxy_port}'
    }
    
    session = requests.Session()
    session.proxies = proxy_config
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    print(f"🌐 HTTP代理已配置: 127.0.0.1:{proxy_port}")
    
    # 测试连接
    try:
        response = session.get('https://huggingface.co/api/models', timeout=10)
        print(f"✅ 连接测试成功: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        print("💡 请确保Clash正在运行且HTTP代理端口可用")
        return False
    
    # 下载配置
    mirror = 'https://huggingface.co'
    files = [
        'config.json',
        'tokenizer_config.json', 
        'model.safetensors',
        'tokenizer.json',
        'special_tokens_map.json'
    ]
    
    print(f"🤖 开始下载模型: {model_name}")
    
    # 获取可用文件列表
    try:
        api_response = session.get(f'{mirror}/api/models/{model_name}', timeout=15)
        if api_response.status_code == 200:
            available_files = [f['rfilename'] for f in api_response.json().get('siblings', [])]
            files = [f for f in files if f in available_files]
            print(f"📋 找到 {len(files)} 个文件")
        else:
            print("⚠️ 无法获取文件列表，使用默认列表")
    except Exception as e:
        print(f"⚠️ API调用失败: {e}")
    
    # 下载文件
    success_count = 0
    total_files = len(files)
    
    for i, filename in enumerate(files):
        print(f"\n📁 [{i+1}/{total_files}] 下载: {filename}")
        
        file_url = f'{mirror}/{model_name}/resolve/main/{filename}'
        local_path = os.path.join(save_dir, model_name, filename)
        
        # 创建目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = session.get(file_url, stream=True, timeout=300)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(local_path, 'wb') as f:
                        start_time = time.time()
                        last_update = start_time
                        
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # 每5秒更新一次进度
                                current_time = time.time()
                                if current_time - last_update >= 5:
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        speed = downloaded / (current_time - start_time)
                                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                                        
                                        print(f"  进度: {progress:.1f}% "
                                             f"({downloaded / 1024 / 1024:.1f}MB/"
                                             f"{total_size / 1024 / 1024:.1f}MB) "
                                             f"速度: {speed / 1024:.1f}KB/s "
                                             f"剩余: {eta / 60:.1f}min")
                                    last_update = current_time
                    
                    print(f"  ✅ {filename} 下载完成!")
                    success_count += 1
                    break
                    
                elif response.status_code == 404:
                    print(f"  ⚠️ 文件不存在: {filename}")
                    break
                else:
                    print(f"  ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ 下载错误: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  🔄 重试 {attempt + 1}/{max_retries}，等待 {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ {filename} 下载失败，已达到最大重试次数")
    
    print(f"\n🎉 下载完成! 成功: {success_count}/{total_files}")
    
    if success_count > 0:
        model_path = os.path.join(save_dir, model_name)
        print(f"📍 模型保存在: {model_path}")
        
        # 计算总大小
        try:
            total_size = 0
            for root, dirs, files_found in os.walk(model_path):
                for file in files_found:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            print(f"💾 总大小: {total_size / 1024 / 1024 / 1024:.2f}GB")
        except:
            pass
    
    return success_count == total_files

def main():
    parser = argparse.ArgumentParser(description="简化VPN模型下载器")
    parser.add_argument("model_name", help="模型名称 (如: mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--save-dir", default="./models", help="保存目录")
    parser.add_argument("--proxy-port", type=int, default=7890, help="HTTP代理端口")
    
    args = parser.parse_args()
    
    success = download_with_http_proxy(args.model_name, args.save_dir, args.proxy_port)
    
    if success:
        print("\n🎊 所有文件下载成功!")
    else:
        print("\n⚠️ 部分文件下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 