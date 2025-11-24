"""Compatibility layer for legacy :mod:`src.core_code` imports.

Core implementations were migrated to :mod:`hace_core.core`. This shim
re-exports the public API to keep existing call sites working during the
transition.
"""

from hace_core.core import (
    AllocationStrategy,
    BudgetNormalizer,
    CakeAdaKVIntegration,
    IntegrationConfig,
    IndicatorNormalizer,
    MemoryConfig,
    NormalizationConfig,
    RobustKeyHeadDetector,
    StrategyConfig,
    StrategySelector,
    UnifiedCakeAdaKVAllocator,
    UnifiedCacheConfig,
    UnifiedMemoryManager,
    create_integration,
    run_launcher,
)

__all__ = [
    "AllocationStrategy",
    "BudgetNormalizer",
    "CakeAdaKVIntegration",
    "IntegrationConfig",
    "IndicatorNormalizer",
    "MemoryConfig",
    "NormalizationConfig",
    "RobustKeyHeadDetector",
    "StrategyConfig",
    "StrategySelector",
    "UnifiedCakeAdaKVAllocator",
    "UnifiedCacheConfig",
    "UnifiedMemoryManager",
    "create_integration",
    "run_launcher",
]
