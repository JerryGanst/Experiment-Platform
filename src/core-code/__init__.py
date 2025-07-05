"""
CAKE-AdaKV 统一集成核心代码

这是我们的创新方法，独立于原始的CAKE和AdaKV实现。
提供层级-头级协同优化的完整解决方案。

主要模块：
- unified_allocator: 统一的分配器
- indicator_normalizer: 指标归一化器  
- budget_manager: 预算管理器
- strategy_selector: 策略选择器
- integration_framework: 集成框架
"""

# 核心组件导入
try:
    from .unified_allocator import UnifiedCakeAdaKVAllocator
    from .indicator_normalizer import IndicatorNormalizer, BudgetNormalizer
    from .strategy_selector import StrategySelector, RobustKeyHeadDetector
    from .memory_manager import UnifiedMemoryManager
    from .integration_framework import CakeAdaKVIntegration
except ImportError as e:
    # 开发环境下的回退处理
    import warnings
    warnings.warn(f"部分模块导入失败，可能缺少依赖: {e}")
    
    # 提供空的占位符类
    class UnifiedCakeAdaKVAllocator:
        pass
    class IndicatorNormalizer:
        pass
    class BudgetNormalizer:
        pass
    class StrategySelector:
        pass
    class RobustKeyHeadDetector:
        pass
    class UnifiedMemoryManager:
        pass
    class CakeAdaKVIntegration:
        pass

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "UnifiedCakeAdaKVAllocator",
    "IndicatorNormalizer", 
    "BudgetNormalizer",
    "StrategySelector",
    "RobustKeyHeadDetector", 
    "UnifiedMemoryManager",
    "CakeAdaKVIntegration"
]