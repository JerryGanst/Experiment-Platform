#!/usr/bin/env python3
"""
模型文件清理脚本
用于删除下载的模型文件，恢复堡垒机环境
"""

import os
import shutil
import argparse
from pathlib import Path

def cleanup_models(models_dir="./models", confirm=True):
    """清理模型文件"""
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"📂 目录不存在: {models_path}")
        return
    
    # 计算总大小
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(models_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
                file_count += 1
            except:
                pass
    
    print(f"📊 发现模型文件:")
    print(f"   目录: {models_path.absolute()}")
    print(f"   文件数: {file_count}")
    print(f"   总大小: {total_size / 1024 / 1024 / 1024:.2f}GB")
    
    # 列出模型
    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
    if model_dirs:
        print(f"\n📁 模型列表:")
        for model_dir in model_dirs:
            model_size = 0
            model_files = 0
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    try:
                        model_size += os.path.getsize(os.path.join(root, file))
                        model_files += 1
                    except:
                        pass
            print(f"   {model_dir.name}: {model_files}个文件, {model_size / 1024 / 1024:.1f}MB")
    
    if confirm:
        print(f"\n⚠️  确认删除模型目录? (y/N): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 取消清理")
            return False
    
    # 删除目录
    try:
        shutil.rmtree(models_path)
        print(f"✅ 已删除模型目录: {models_path}")
        print(f"💾 释放空间: {total_size / 1024 / 1024 / 1024:.2f}GB")
        return True
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False

def cleanup_specific_model(model_name, models_dir="./models"):
    """删除特定模型"""
    
    model_path = Path(models_dir) / model_name
    
    if not model_path.exists():
        print(f"📂 模型不存在: {model_path}")
        return
    
    # 计算大小
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(root, file))
            except:
                pass
    
    print(f"🗑️  删除模型: {model_name}")
    print(f"   大小: {total_size / 1024 / 1024 / 1024:.2f}GB")
    
    try:
        shutil.rmtree(model_path)
        print(f"✅ 已删除: {model_path}")
        return True
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False

def show_disk_usage(models_dir="./models"):
    """显示磁盘使用情况"""
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"📂 目录不存在: {models_path}")
        return
    
    print(f"💾 磁盘使用情况:")
    
    # 计算models目录大小
    total_size = 0
    for root, dirs, files in os.walk(models_path):
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(root, file))
            except:
                pass
    
    print(f"   模型目录: {total_size / 1024 / 1024 / 1024:.2f}GB")
    
    # 系统磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage(models_path.parent)
        print(f"   总空间: {total / 1024 / 1024 / 1024:.1f}GB")
        print(f"   已使用: {used / 1024 / 1024 / 1024:.1f}GB")
        print(f"   可用: {free / 1024 / 1024 / 1024:.1f}GB")
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="模型文件清理工具")
    parser.add_argument("--models-dir", default="./models", help="模型目录")
    parser.add_argument("--model", help="删除特定模型")
    parser.add_argument("--all", action="store_true", help="删除所有模型")
    parser.add_argument("--show", action="store_true", help="仅显示信息，不删除")
    parser.add_argument("--force", action="store_true", help="强制删除，无需确认")
    
    args = parser.parse_args()
    
    if args.show:
        show_disk_usage(args.models_dir)
    elif args.model:
        cleanup_specific_model(args.model, args.models_dir)
    elif args.all:
        cleanup_models(args.models_dir, not args.force)
    else:
        print("🧹 模型清理工具")
        print("\n使用方法:")
        print("  python3 cleanup_models.py --show              # 显示磁盘使用")
        print("  python3 cleanup_models.py --all               # 删除所有模型")
        print("  python3 cleanup_models.py --model MODEL_NAME  # 删除特定模型")
        print("  python3 cleanup_models.py --all --force       # 强制删除，无需确认")
        
        show_disk_usage(args.models_dir)

if __name__ == "__main__":
    main() 