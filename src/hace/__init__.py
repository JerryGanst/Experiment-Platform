"""
HACE Core - KV Cache 优化实验平台核心库

子模块：
- core: CAKE-AdaKV 统一分配器和集成框架
- data: 统一数据加载接口
- models: 模型加载、推理后端、转换器
- utils: 工具函数

使用示例:
    # 核心分配器
    from src.hace.core import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig

    # 数据加载
    from src.hace.data import load_data

    # 模型加载
    from src.hace.models import load_model_and_tokenizer
"""

from .algorithms import BaseHACEAlgorithm, SimpleHACEAlgorithm
# 直接导入核心子模块，避免 __getattr__ 导致递归
from . import config, core, data, models, utils


__all__ = [
    "BaseHACEAlgorithm",
    "SimpleHACEAlgorithm",
    # 子模块（延迟加载）
    "core",
    "data",
    "models",
    "utils",
    "config",
]
