"""
CAKE-AdaKV 统一集成核心代码

这是独立于原始 CAKE 和 AdaKV 实现的创新方法，
提供层级-头级协同优化的完整解决方案。

主要模块：
- unified_allocator: 统一的 KV Cache 分配器
- indicator_normalizer: 指标归一化器
- strategy_selector: 策略选择器
- memory_manager: 内存管理器（KV Cache 锚点协调）
- integration_framework: 集成框架

使用示例:
    from src.hace.core import (
        UnifiedCakeAdaKVAllocator,
        UnifiedCacheConfig,
        CakeAdaKVIntegration,
        IntegrationConfig,
    )

    # 创建配置
    config = UnifiedCacheConfig(
        total_cache_budget=2000,
        num_layers=32,
        num_heads=32,
    )

    # 创建分配器
    allocator = UnifiedCakeAdaKVAllocator(config)
"""

from .unified_allocator import (
    UnifiedCakeAdaKVAllocator,
    UnifiedCacheConfig,
)
from .indicator_normalizer import (
    IndicatorNormalizer,
    BudgetNormalizer,
    NormalizationConfig,
)
from .strategy_selector import (
    StrategySelector,
    RobustKeyHeadDetector,
    AllocationStrategy,
    StrategyConfig,
)
from .memory_manager import (
    UnifiedMemoryManager,
    MemoryConfig,
)
from .integration_framework import (
    CakeAdaKVIntegration,
    IntegrationConfig,
    create_integration,
)
from .launcher import main as run_launcher

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    # Allocator
    "UnifiedCakeAdaKVAllocator",
    "UnifiedCacheConfig",
    # Normalizer
    "IndicatorNormalizer",
    "BudgetNormalizer",
    "NormalizationConfig",
    # Strategy
    "StrategySelector",
    "RobustKeyHeadDetector",
    "AllocationStrategy",
    "StrategyConfig",
    # Memory
    "UnifiedMemoryManager",
    "MemoryConfig",
    # Integration
    "CakeAdaKVIntegration",
    "IntegrationConfig",
    "create_integration",
    # Launcher
    "run_launcher",
]
