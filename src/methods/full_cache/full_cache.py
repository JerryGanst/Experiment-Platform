"""Full cache baseline implementation.

This baseline keeps all key/value pairs without any eviction.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from ..base_method import BaseKVCacheMethod


class FullCacheBaseline(BaseKVCacheMethod):
    """Store the entire KV history for evaluation purposes."""

    def __init__(self) -> None:
        self.entries: List[Tuple[Any, Any]] = []

    def initialize_cache(self) -> None:
        self.entries = []

    def update_cache(self, key_states: Any, value_states: Any) -> None:
        """Append new key/value states to the cache."""
        self.entries.append((key_states, value_states))

    def get_method_specific_metrics(self) -> dict:
        return {"cached_blocks": len(self.entries)}
