#!/usr/bin/env python3
"""
自动检测模型路径并输出环境变量设置建议。

该脚本面向 SSH/远端 GPU 场景：会扫描常见的 Linux 目录以及环境变量，
列出可用的 Mistral/Mixtral 模型，并生成 CORECODE_MODEL_PATH 的设置指令。
"""

import json
import os
import platform
from pathlib import Path
from typing import List, Optional, Tuple


def candidate_paths() -> List[str]:
    """返回默认扫描路径列表"""
    paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/models"),
        "/opt/models",
        "/data/models",
        "/home/models",
        "/mnt/models",
        "/srv/models",
    ]

    for env_key in ("CORECODE_MODEL_PATH", "HACE_MODEL_PATH", "MODEL_PATH", "HF_HOME"):
        value = os.environ.get(env_key)
        if value:
            paths.append(value)
    return paths


def find_mistral_models() -> List[Tuple[str, str]]:
    """搜索本地/远端磁盘上的 Mistral 模型"""
    models: List[Tuple[str, str]] = []

    for base_path in candidate_paths():
        if not os.path.exists(base_path):
            continue

        for root, dirs, files in os.walk(base_path):
            if "config.json" not in files:
                # 限制递归深度，避免遍历整块磁盘
                if len(root.split(os.sep)) - len(base_path.split(os.sep)) > 5:
                    dirs.clear()
                continue

            config_path = os.path.join(root, "config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    config = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue

            model_type = config.get("model_type", "").lower()
            if "mistral" not in model_type and "mixtral" not in model_type:
                continue

            model_name = os.path.basename(root)
            if "models--" in root:
                parts = root.split("models--")[1].split(os.sep)
                if len(parts) >= 2:
                    model_name = f"{parts[0]}/{parts[1]}"

            models.append((root, model_name))
            print(f"✅ 本地模型: {model_name}\n   路径: {root}")

    return models


def find_huggingface_models() -> List[Tuple[str, str]]:
    """列出可直接从 Hugging Face Hub 加载的模型"""
    online_models = [
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3 (HF Hub)"),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1 (HF Hub)"),
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B-v0.1 (HF Hub)"),
    ]

    print("\n📡 可用的在线模型:")
    for model_id, label in online_models:
        print(f"   - {label}: {model_id}")

    return online_models


def select_model(models: List[Tuple[str, str]]) -> Optional[str]:
    """交互式选择模型路径"""
    if not models:
        return None

    print("\n请选择要使用的模型:")
    for idx, (path, name) in enumerate(models, start=1):
        print(f"{idx}. {name}")
        print(f"   路径: {path}")

    while True:
        choice = input("\n请输入编号 (或输入 'q' 退出): ").strip()
        if choice.lower() == "q":
            return None
        try:
            selected = int(choice) - 1
            if 0 <= selected < len(models):
                return models[selected][0]
        except ValueError:
            pass
        print("❌ 无效选择，请重新输入。")


def write_env_file(model_path: str) -> Optional[Path]:
    """在项目根目录创建 .env.corecode 方便 SSH 会话加载"""
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env.corecode"

    try:
        with open(env_file, "w", encoding="utf-8") as fh:
            fh.write("# 自动生成：source 该文件即可设置模型路径\n")
            fh.write(f'export CORECODE_MODEL_PATH="{model_path}"\n')
        return env_file
    except OSError as exc:
        print(f"⚠️ 无法写入 {env_file}: {exc}")
        return None


def main() -> None:
    print("🔍 开始搜索 Mistral/Mixtral 模型...")
    print(f"🖥️ 当前系统: {platform.system()}")

    local_models = find_mistral_models()
    online_models = find_huggingface_models()
    all_models = local_models + online_models

    if not all_models:
        print("\n❌ 未找到任何模型")
        print("建议: 先下载模型或设置 CORECODE_MODEL_PATH 指向远端挂载目录")
        return

    selected = select_model(all_models)
    if not selected:
        print("\n👋 已取消操作")
        return

    print(f"\n✅ 选中的模型路径: {selected}")
    env_file = write_env_file(selected)

    print("\n🎯 接下来可以执行：")
    print(f'1. export CORECODE_MODEL_PATH="{selected}"')
    if env_file:
        print(f"2. 或者 source {env_file} 自动加载")
    print("3. python tests/integration/test_corecode_hotpotqa_fixed.py --model \"$CORECODE_MODEL_PATH\"")


if __name__ == "__main__":
    main()
