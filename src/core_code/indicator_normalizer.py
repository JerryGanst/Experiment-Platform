"""Compatibility wrapper for :mod:`src.core_code.indicator_normalizer`."""

from hace_core.core.indicator_normalizer import *  # noqa: F401,F403

__all__ = [
    "IndicatorNormalizer",
    "BudgetNormalizer",
    "NormalizationConfig",
]
