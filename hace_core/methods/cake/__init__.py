"""
CAKE (Cache-Aware KV Eviction) 方法模块

包含：
- batch_runner: 批量运行器
- eval_utils: 评估工具
- fullkvcache_main_optimized: 优化的完整 KV Cache 主程序
- monitor_wrapper: 监控包装器
"""

# 延迟导入避免依赖问题
def __getattr__(name):
    if name == "batch_runner":
        from . import batch_runner
        return batch_runner
    elif name == "eval_utils":
        from . import eval_utils
        return eval_utils
    elif name == "fullkvcache_main_optimized":
        from . import fullkvcache_main_optimized
        return fullkvcache_main_optimized
    elif name == "monitor_wrapper":
        from . import monitor_wrapper
        return monitor_wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "batch_runner",
    "eval_utils",
    "fullkvcache_main_optimized",
    "monitor_wrapper",
]
