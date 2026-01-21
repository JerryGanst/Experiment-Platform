#!/usr/bin/env python3
"""
模型配置检查脚本
用于验证模型路径是否正确配置，避免硬编码路径问题
"""

import os
import sys
from pathlib import Path

# 导入配置模块
try:
    from src.hace.config import get_model_path, MODEL_CONFIG, EXPERIMENT_CONFIG
except ImportError:
    print("❌ 无法导入配置模块，请确保项目结构正确")
    sys.exit(1)

def check_model_config():
    """检查模型配置"""
    print("🔍 检查模型配置...")
    print("=" * 50)
    
    # 检查环境变量
    model_env = os.environ.get("HACE_MODEL_PATH")
    if model_env:
        print(f"✅ 找到环境变量 HACE_MODEL_PATH: {model_env}")
        if Path(model_env).exists():
            print("   ✅ 路径存在")
        else:
            print("   ⚠️  路径不存在，但可能是 Hugging Face Hub 路径")
    else:
        print("ℹ️  未设置环境变量 HACE_MODEL_PATH")
    extra_env = os.environ.get("CORECODE_MODEL_PATH")
    if extra_env:
        print(f"✅ 找到 CORECODE_MODEL_PATH: {extra_env}")
    
    # 检查当前配置
    try:
        current_path = get_model_path(MODEL_CONFIG["model_name_or_path"])
        print(f"\n📍 当前模型路径: {current_path}")
        
        # 检查路径类型
        if current_path.startswith("C:/") or current_path.startswith("C:\\"):
            print("❌ 警告：检测到 Windows 风格路径，SSH 环境无法使用")
            print("   请设置 CORECODE_MODEL_PATH/HACE_MODEL_PATH 指向 Linux 路径，或将模型复制到服务器上。")
            return False
        elif current_path.startswith("./"):
            print("✅ 使用相对路径配置")
            if Path(current_path).exists():
                print("   ✅ 本地模型文件存在")
            else:
                print("   ❌ 本地模型文件不存在")
                return False
        elif current_path.startswith("/") or current_path[1:3] == ":/":
            print("✅ 使用绝对路径配置")
            if Path(current_path).exists():
                print("   ✅ 模型文件存在")
            else:
                print("   ❌ 模型文件不存在")
                return False
        else:
            print("✅ 模型路径配置正常")
            
    except FileNotFoundError as e:
        print(f"\n❌ 模型路径配置错误:")
        print(f"   {str(e)}")
        return False
    
    # 检查所有可能的本地路径
    print("\n📂 检查可能的模型存储位置:")
    
    local_paths = [
        Path("./models/mistral-7b-instruct-v0.3"),
        Path.home() / "models" / "mistral-7b-instruct-v0.3",
        Path("./models"),
        Path.home() / "models"
    ]
    
    for path in local_paths:
        if path.exists():
            print(f"   ✅ {path} 存在")
        else:
            print(f"   ❌ {path} 不存在")
    
    # 显示配置信息
    try:
        print(f"\n⚙️  MODEL_CONFIG 模型路径: {MODEL_CONFIG['model_name_or_path']}")
        print(f"⚙️  EXPERIMENT_CONFIG 模型路径: {EXPERIMENT_CONFIG['model_name_or_path']}")
    except Exception as e:
        print(f"\n⚠️  无法读取配置: {str(e)}")
    
    return True

def show_setup_guide():
    """显示设置指南"""
    print("\n" + "=" * 50)
    print("📖 模型路径设置指南:")
    print("=" * 50)
    
    print("\n1️⃣ 环境变量设置 (推荐):")
    print("   export CORECODE_MODEL_PATH='/mnt/models/mistral-7b-instruct-v0.3'")
    print("   export HACE_MODEL_PATH='/mnt/models/mistral-7b-instruct-v0.3'")
    print("   # 可写入 ~/.bashrc 或 .env.corecode 后 source")
    
    print("\n2️⃣ 本地/远端模型目录:")
    print("   mkdir models")
    print("   # 将模型文件放入 models/mistral-7b-instruct-v0.3/")
    
    print("\n3️⃣ 用户目录:")
    print("   mkdir ~/models")
    print("   # 将模型文件放入 ~/models/mistral-7b-instruct-v0.3/")

if __name__ == "__main__":
    print("🎯 模型配置检查工具")
    print("修复硬编码路径问题，确保跨平台兼容性")
    
    success = check_model_config()
    show_setup_guide()
    
    if success:
        print("\n✅ 配置检查完成")
    else:
        print("\n❌ 发现配置问题，请参考上述指南进行修复")
        sys.exit(1) 
