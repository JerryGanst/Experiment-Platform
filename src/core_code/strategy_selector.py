"""Compatibility wrapper for :mod:`src.core_code.strategy_selector`."""

from hace_core.core.strategy_selector import *  # noqa: F401,F403

__all__ = [
    "StrategySelector",
    "RobustKeyHeadDetector",
    "AllocationStrategy",
    "StrategyConfig",
]
