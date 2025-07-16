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

# 将核心组件提升到包级别，方便外部调用
# 这也解决了相对导入的问题
from .unified_allocator import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig
from .indicator_normalizer import IndicatorNormalizer, BudgetNormalizer, NormalizationConfig
from .strategy_selector import StrategySelector, RobustKeyHeadDetector, AllocationStrategy, StrategyConfig
from .memory_manager import UnifiedMemoryManager
from .integration_framework import CakeAdaKVIntegration, IntegrationConfig
from .launcher import main as run_launcher

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "UnifiedCakeAdaKVAllocator",
    "UnifiedCacheConfig",
    "IndicatorNormalizer",
    "BudgetNormalizer",
    "NormalizationConfig",
    "StrategySelector",
    "RobustKeyHeadDetector",
    "AllocationStrategy",
    "StrategyConfig",
    "UnifiedMemoryManager",
    "CakeAdaKVIntegration",
    "IntegrationConfig",
    "run_launcher"
]