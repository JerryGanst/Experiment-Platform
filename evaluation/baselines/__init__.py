"""
基线方法执行模块

包含纯粹的基线实现，不使用任何KV缓存优化：
- fullkvcache_main: 完整KV缓存基线实验
- baseline_main: 标准基线实验
- base_method: 基础方法定义
- test: 基线测试工具
- data/: 基线测试数据

这些基线为优化方法提供性能对比的参考标准。
"""

__all__ = [
    'fullkvcache_main',
    'baseline_main', 
    'base_method',
    'test'
] 