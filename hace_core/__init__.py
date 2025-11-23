"""
HACE Core - KV Cache 优化实验平台核心库

子模块：
- core: CAKE-AdaKV 统一分配器和集成框架
- data: 统一数据加载接口
- models: 模型加载、推理后端、转换器
- utils: 工具函数

使用示例:
    # 核心分配器
    from hace_core.core import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig

    # 数据加载
    from hace_core.data import load_data

    # 模型加载
    from hace_core.models import load_model_and_tokenizer
"""

from .algorithms import BaseHACEAlgorithm, SimpleHACEAlgorithm

# 延迟导入子模块以避免循环依赖
def __getattr__(name):
    """延迟导入子模块"""
    if name == "core":
        from . import core
        return core
    elif name == "data":
        from . import data
        return data
    elif name == "models":
        from . import models
        return models
    elif name == "utils":
        from . import utils
        return utils
    elif name == "config":
        from . import config
        return config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
