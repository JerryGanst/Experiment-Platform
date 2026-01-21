"""
基线方法执行模块

包含纯粹的基线实现，不使用任何KV缓存优化：
- fullkvcache_main: 完整KV缓存基线实验
- baseline_main: 标准基线实验
- test: 基线测试工具
- data/: 基线测试数据

这些基线为优化方法提供性能对比的参考标准。

Note: BaseKVCacheMethod 已移至 src.hace.methods.base
"""

# 导入 BaseKVCacheMethod 以保持向后兼容
from src.hace.methods.base import BaseKVCacheMethod

__all__ = [
    'fullkvcache_main',
    'baseline_main',
    'test',
    'BaseKVCacheMethod',
] 