"""
CAKE-AdaKV 统一集成核心代码 - 兼容层

!!! 废弃警告 !!!
此模块已迁移至 hace_core.core，请更新导入：

    # 旧方式（已废弃）
    from src.core_code import UnifiedCakeAdaKVAllocator

    # 新方式（推荐）
    from hace_core.core import UnifiedCakeAdaKVAllocator

迁移指南：
    src.core_code.* -> hace_core.core.*

详见 docs/REFACTORING_PLAN.md
"""

import warnings

# 发出废弃警告
warnings.warn(
    "src.core_code 已迁移至 hace_core.core，请更新导入。"
    "详见 docs/REFACTORING_PLAN.md",
    DeprecationWarning,
    stacklevel=2
)

# 从新位置重新导出所有内容
try:
    from hace_core.core import (
        # Allocator
        UnifiedCakeAdaKVAllocator,
        UnifiedCacheConfig,
        # Normalizer
        IndicatorNormalizer,
        BudgetNormalizer,
        NormalizationConfig,
        # Strategy
        StrategySelector,
        RobustKeyHeadDetector,
        AllocationStrategy,
        StrategyConfig,
        # Memory
        UnifiedMemoryManager,
        MemoryConfig,
        # Integration
        CakeAdaKVIntegration,
        IntegrationConfig,
        create_integration,
        # Launcher
        run_launcher,
    )

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
        "MemoryConfig",
        "CakeAdaKVIntegration",
        "IntegrationConfig",
        "create_integration",
        "run_launcher",
    ]

except ImportError as e:
    # 如果新模块不可用，回退到原有实现
    warnings.warn(
        f"无法从 hace_core.core 导入，使用本地实现: {e}",
        ImportWarning,
        stacklevel=2
    )

    from .unified_allocator import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig
    from .indicator_normalizer import IndicatorNormalizer, BudgetNormalizer, NormalizationConfig
    from .strategy_selector import StrategySelector, RobustKeyHeadDetector, AllocationStrategy, StrategyConfig
    from .memory_manager import UnifiedMemoryManager
    from .integration_framework import CakeAdaKVIntegration, IntegrationConfig
    from .launcher import main as run_launcher

    # 补充缺失的导出
    try:
        from .memory_manager import MemoryConfig
    except ImportError:
        MemoryConfig = None

    try:
        from .integration_framework import create_integration
    except ImportError:
        create_integration = None

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
        "run_launcher",
    ]
