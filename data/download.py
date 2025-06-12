from huggingface_hub import snapshot_download
from pathlib import Path
import os


def download_mistral_model():
    # 设置模型存储路径
    mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    mistral_models_path.mkdir(parents=True, exist_ok=True)

    print(f"下载路径: {mistral_models_path}")

    try:
        # 下载完整的模型文件 - 包含transformers需要的所有文件
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            # 不指定allow_patterns，下载所有必要文件
            local_dir=mistral_models_path,
            resume_download=True,  # 支持断点续传
            local_dir_use_symlinks=False  # 直接复制文件而不使用符号链接
        )
        print("模型下载完成！")

        # 检查下载的文件
        for file in mistral_models_path.iterdir():
            print(f"已下载: {file.name} ({file.stat().st_size / (1024 * 1024 * 1024):.2f} GB)")

    except Exception as e:
        print(f"下载过程中出现错误: {e}")


if __name__ == "__main__":
    download_mistral_model()