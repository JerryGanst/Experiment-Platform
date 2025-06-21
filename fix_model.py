# 第五步：模型路径修复脚本
# fix_model_paths.py

import sys
import os
import json
import shutil
from pathlib import Path

# 添加utilities到路径
sys.path.append('utilities')
from utilities.path_config import PathManager


def find_model_directories():
    """查找所有可能的模型目录"""
    pm = PathManager()
    current_user = pm.home_dir.name

    print(f"🔍 查找用户 {current_user} 的模型目录...")

    # 可能的模型路径
    possible_paths = [
        Path(f"C:/Users/{current_user}/mistral_models"),
        Path(f"C:/Users/{current_user}/models"),
        Path(f"C:/Users/{current_user}/huggingface"),
        Path(f"C:/Users/{current_user}/.cache/huggingface"),
        Path("C:/models"),
        Path("D:/models"),
        Path("E:/models"),
        # 检查是否模型在旧路径但可以移动
        Path("C:/Users/JerryGanst/mistral_models")
    ]

    found_models = []

    for path in possible_paths:
        if path.exists():
            print(f"📁 找到目录: {path}")

            # 查找Mistral模型
            for item in path.iterdir():
                if item.is_dir() and ("7B" in item.name or "mistral" in item.name.lower() or "Mistral" in item.name):
                    model_path = item
                    # 检查是否包含模型文件
                    if any(f.name in ["config.json", "pytorch_model.bin", "model.safetensors"] for f in
                           model_path.glob("*")):
                        found_models.append(model_path)
                        print(f"  ✅ 找到模型: {model_path}")

    return found_models


def setup_model_path():
    """设置正确的模型路径"""
    pm = PathManager()
    current_user = pm.home_dir.name

    print("🔧 开始模型路径设置...")

    # 1. 查找现有模型
    found_models = find_model_directories()

    if found_models:
        print(f"\\n🎯 找到 {len(found_models)} 个模型:")
        for i, model_path in enumerate(found_models):
            print(f"  {i + 1}. {model_path}")

        # 使用第一个找到的模型
        selected_model = found_models[0]
        print(f"\\n✅ 选择模型: {selected_model}")

        # 更新配置
        pm.update_config("model_paths", "local_models", str(selected_model.parent))
        pm.update_config("model_paths", "default_model", selected_model.name)

        return str(selected_model)

    else:
        print("\\n⚠️  未找到现有模型，设置自动下载...")

        # 设置使用HuggingFace Hub
        hub_model = "mistralai/Mistral-7B-Instruct-v0.3"
        pm.update_config("model_paths", "hub_model", hub_model)
        pm.update_config("model_paths", "use_hub", True)

        print(f"✅ 已配置使用HuggingFace模型: {hub_model}")
        return hub_model


def update_experiment_configs():
    """更新实验配置文件中的模型路径"""
    print("\\n🔄 更新实验配置文件...")

    pm = PathManager()
    current_model_path = pm.get_model_path()

    # 要更新的配置文件
    config_files = [
        "hace-kv-optimization/hace_core/config.py",
        "hace-kv-optimization/experiments/run_experiment.py"
    ]

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 替换旧的用户路径
                old_patterns = [
                    'C:/Users/JerryGanst/mistral_models/7B-Instruct-v0.3',
                    'C:\\Users\\JerryGanst\\mistral_models\\7B-Instruct-v0.3',
                    '/Users/JerryGanst/mistral_models/7B-Instruct-v0.3'
                ]

                updated = False
                for old_pattern in old_patterns:
                    if old_pattern in content:
                        content = content.replace(old_pattern, current_model_path)
                        updated = True
                        print(f"  🔄 更新 {config_file}: {old_pattern} → {current_model_path}")

                if updated:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ 已更新: {config_file}")
                else:
                    print(f"  ℹ️  无需更新: {config_file}")

            except Exception as e:
                print(f"  ❌ 更新失败 {config_file}: {e}")
        else:
            print(f"  ⚠️  文件不存在: {config_file}")


def create_model_test_script():
    """创建模型路径测试脚本"""
    test_script = '''# test_model_path.py - 测试模型路径
import sys
sys.path.append('utilities')
from utilities.path_config import PathManager
from pathlib import Path

def test_model_path():
    """测试模型路径是否正确"""
    print("🔍 测试模型路径...")

    pm = PathManager()
    model_path = pm.get_model_path()

    print(f"📍 配置的模型路径: {model_path}")

    # 检查路径是否存在
    path_obj = Path(model_path)

    if path_obj.exists():
        print("✅ 模型路径存在")

        # 检查模型文件
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        found_files = []

        for model_file in model_files:
            file_path = path_obj / model_file
            if file_path.exists():
                found_files.append(model_file)

        if found_files:
            print(f"✅ 找到模型文件: {found_files}")
            return True
        else:
            print("⚠️  模型路径存在但缺少模型文件")
            return False
    else:
        print("❌ 模型路径不存在")

        # 检查是否应该使用HuggingFace Hub
        if "mistralai/" in model_path:
            print("💡 建议使用HuggingFace Hub自动下载")
            return "hub"

        return False

def suggest_fix():
    """建议修复方案"""
    result = test_model_path()

    print("\\n🔧 修复建议:")

    if result is True:
        print("✅ 模型路径正确，可以开始实验")
        print("🚀 运行命令:")
        print("   cd hace-kv-optimization/baselines")
        print("   python fullkvcache_main.py --enable_scoring --is_baseline_run --datasets hotpotqa --max_new_tokens 50")

    elif result == "hub":
        print("💡 使用HuggingFace Hub模型")
        print("🚀 运行命令:")
        print("   cd hace-kv-optimization/baselines")
        print("   python fullkvcache_main.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --enable_scoring --is_baseline_run --datasets hotpotqa --max_new_tokens 50")

    else:
        print("❌ 需要先修复模型路径")
        print("🔧 运行命令:")
        print("   python fix_model_paths.py")

if __name__ == "__main__":
    suggest_fix()
'''

    with open('test_model_path.py', 'w', encoding='utf-8') as f:
        f.write(test_script)

    print("✅ 已创建模型路径测试脚本: test_model_path.py")


def create_direct_command():
    """创建直接运行命令"""
    pm = PathManager()

    # 检查模型路径
    model_path = pm.get_model_path()

    if Path(model_path).exists():
        model_param = f"--model_name {model_path}"
    else:
        # 使用HuggingFace Hub
        model_param = "--model_name mistralai/Mistral-7B-Instruct-v0.3"

    command = f'''# 直接运行基线实验的命令
cd hace-kv-optimization/baselines
python fullkvcache_main.py {model_param} --enable_scoring --is_baseline_run --datasets hotpotqa --max_new_tokens 50 --kv_cache_lengths 512 --batch_sizes 1 --repetitions 1
'''

    with open('run_baseline_fixed.bat', 'w', encoding='utf-8') as f:
        f.write(command)

    print("✅ 已创建修复后的运行脚本: run_baseline_fixed.bat")
    print(f"🚀 建议运行命令: {model_param}")


def main():
    """主修复流程"""
    print("🔧 第五步：模型路径修复")
    print("=" * 50)

    # 1. 查找并设置模型路径
    model_path = setup_model_path()

    # 2. 更新配置文件
    update_experiment_configs()

    # 3. 创建测试脚本
    create_model_test_script()

    # 4. 创建直接运行命令
    create_direct_command()

    print("\\n✅ 第五步完成！")
    print("🎯 下一步操作:")
    print("  1. 运行测试: python test_model_path.py")
    print("  2. 或直接实验: run_baseline_fixed.bat")


if __name__ == "__main__":
    main()