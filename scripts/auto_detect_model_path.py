#!/usr/bin/env python3
"""
自动检测模型路径并更新测试脚本

这个脚本会自动搜索系统中的Mistral模型，并更新test_corecode_hotpotqa_fixed.py中的默认路径
"""

import os
import sys
import platform
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


def find_mistral_models() -> List[Tuple[str, str]]:
    """
    搜索系统中的Mistral模型
    
    Returns:
        List of (path, model_name) tuples
    """
    models_found = []
    
    # 定义可能的模型位置
    search_paths = []
    
    # Windows路径
    if platform.system() == "Windows":
        search_paths.extend([
            r"C:\Users\Administrator\mistral_models",
            r"C:\models",
            r"D:\models",
            os.path.expanduser("~\\.cache\\huggingface\\hub"),
            os.path.expanduser("~\\models"),
        ])
    
    # Linux/Mac路径
    else:
        search_paths.extend([
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/models"),
            "/opt/models",
            "/data/models",
            "/home/models",
        ])
    
    # 添加环境变量中的路径
    if "HF_HOME" in os.environ:
        search_paths.append(os.environ["HF_HOME"])
    if "MODEL_PATH" in os.environ:
        search_paths.append(os.environ["MODEL_PATH"])
    
    # 搜索模型
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue
            
        # 递归搜索
        for root, dirs, files in os.walk(base_path):
            # 检查是否是Mistral模型目录
            if "config.json" in files:
                config_path = os.path.join(root, "config.json")
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        
                    # 检查是否是Mistral模型
                    model_type = config.get("model_type", "")
                    if "mistral" in model_type.lower() or "mistral" in root.lower():
                        # 提取模型名称
                        model_name = os.path.basename(root)
                        # 如果是Hugging Face缓存目录，尝试提取更友好的名称
                        if "models--" in root:
                            parts = root.split("models--")[1].split(os.sep)
                            if len(parts) >= 2:
                                model_name = f"{parts[0]}/{parts[1]}"
                        
                        models_found.append((root, model_name))
                        print(f"✅ 找到Mistral模型: {model_name}")
                        print(f"   路径: {root}")
                        
                except Exception as e:
                    continue
                    
            # 限制搜索深度
            if len(root.split(os.sep)) - len(base_path.split(os.sep)) > 5:
                dirs.clear()  # 不再深入搜索
    
    return models_found


def find_huggingface_models() -> List[Tuple[str, str]]:
    """搜索Hugging Face在线可用的Mistral模型"""
    online_models = [
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3 (在线)"),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1 (在线)"),
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B-v0.1 (在线)"),
    ]
    
    print("\n📡 可用的在线模型:")
    for model_id, name in online_models:
        print(f"   - {name}: {model_id}")
    
    return online_models


def select_model(models: List[Tuple[str, str]]) -> Optional[str]:
    """让用户选择模型"""
    if not models:
        return None
        
    print("\n请选择要使用的模型:")
    for i, (path, name) in enumerate(models):
        print(f"{i+1}. {name}")
        print(f"   路径: {path}")
    
    while True:
        try:
            choice = input("\n请输入选择的编号 (或输入 'q' 退出): ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][0]
            else:
                print("❌ 无效的选择，请重新输入")
        except ValueError:
            print("❌ 请输入有效的数字")


def update_test_script(model_path: str) -> bool:
    """更新测试脚本中的模型路径"""
    # 获取脚本所在目录，然后找到项目根目录
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    script_path = project_root / "test_corecode_hotpotqa_fixed.py"
    
    if not script_path.exists():
        print(f"❌ 找不到测试脚本: {script_path}")
        return False
    
    try:
        # 读取文件内容
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式替换两处模型路径
        # 1. 替换类初始化中的默认值
        pattern1 = r'def __init__\(self, model_name=[^,]+, cache_budget'
        replacement1 = f'def __init__(self, model_name="{model_path}", cache_budget'
        content = re.sub(pattern1, replacement1, content)
        
        # 2. 替换main函数中的默认值
        pattern2 = r'parser\.add_argument\("--model", type=str, default=[^,]+,'
        replacement2 = f'parser.add_argument("--model", type=str, default="{model_path}",'
        content = re.sub(pattern2, replacement2, content)
        
        # 写回文件
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 成功更新测试脚本的模型路径为: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ 更新测试脚本失败: {e}")
        return False


def main():
    """主函数"""
    print("🔍 开始搜索Mistral模型...")
    print(f"🖥️  当前系统: {platform.system()}")
    
    # 搜索本地模型
    local_models = find_mistral_models()
    
    # 获取在线模型
    online_models = find_huggingface_models()
    
    # 合并所有模型
    all_models = local_models + online_models
    
    if not all_models:
        print("\n❌ 未找到任何Mistral模型")
        print("建议：")
        print("1. 确保模型已下载到正确的位置")
        print("2. 设置MODEL_PATH环境变量指向模型目录")
        print("3. 使用在线模型（需要网络连接）")
        return
    
    # 选择模型
    selected_model = select_model(all_models)
    
    if selected_model:
        print(f"\n✅ 选择的模型: {selected_model}")
        
        # 更新测试脚本
        if update_test_script(selected_model):
            print("\n🎉 模型路径已成功更新！")
            print("现在可以运行: python test_corecode_hotpotqa_fixed.py")
        else:
            print("\n❌ 更新失败，请手动修改模型路径")
    else:
        print("\n👋 已取消")


if __name__ == "__main__":
    main() 