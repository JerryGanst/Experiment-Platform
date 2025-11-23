"""Adaptive KV cache baseline.

This simplified version maintains a FIFO cache with a fixed budget.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Tuple

from ..base_method import BaseKVCacheMethod


class AdaKVBaseline(BaseKVCacheMethod):
    """A toy AdaKV baseline keeping at most ``cache_budget`` entries."""

    def __init__(self, cache_budget: int = 512) -> None:
        self.cache_budget = cache_budget
        self.entries: Deque[Tuple[Any, Any]] = deque(maxlen=cache_budget)

    def initialize_cache(self) -> None:
        self.entries = deque(maxlen=self.cache_budget)

    def update_cache(self, key_states: Any, value_states: Any) -> None:
        self.entries.append((key_states, value_states))

    def get_method_specific_metrics(self) -> dict:
        return {
            "cache_budget": self.cache_budget,
            "cached_blocks": len(self.entries),
        }
