"""Streaming LLM baseline using a sliding window cache."""

from __future__ import annotations

from typing import Any, List, Tuple

from ..base_method import BaseKVCacheMethod


class StreamingLLMBaseline(BaseKVCacheMethod):
    """Maintain only the most recent ``window_size`` key/value pairs."""

    def __init__(self, window_size: int = 2048) -> None:
        self.window_size = window_size
        self.entries: List[Tuple[Any, Any]] = []

    def initialize_cache(self) -> None:
        self.entries = []

    def update_cache(self, key_states: Any, value_states: Any) -> None:
        self.entries.append((key_states, value_states))
        if len(self.entries) > self.window_size:
            # Drop tokens outside the window
            self.entries = self.entries[-self.window_size:]

    def get_method_specific_metrics(self) -> dict:
        return {
            "window_size": self.window_size,
            "cached_blocks": len(self.entries),
        }
