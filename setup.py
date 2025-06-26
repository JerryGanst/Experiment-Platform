#!/usr/bin/env python3
"""
Experiment Platform 安装脚本
"""
from setuptools import setup, find_packages

setup(
    name="experiment-platform",
    version="1.0.0",
    description="KV缓存优化实验平台",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.43.3",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "datasets>=2.10.0",
        "pynvml>=11.0.0",
        "psutil>=5.8.0",
        "rouge>=1.0.1",
        "rouge_score>=0.1.2",
        "fuzzywuzzy>=0.18.0",
        "jieba==0.42.1",
        "tqdm==4.66.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "jinja2==3.1.0",
        "pyyaml>=6.0",
        "toml>=0.10.2",
        "icetk==0.0.7",
        "packaging>=21.0",
    ],
    package_data={
        "": ["*.yaml", "*.csv", "*.json", "*.md"],
    },
    include_package_data=True,
) 