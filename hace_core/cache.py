"""Core cache implementation for the HACE method."""

from typing import Any, Dict

from .base import BaseKVCacheMethod


class HACECache(BaseKVCacheMethod):
    """Implementation skeleton for HACE KV cache management."""

    def __init__(
        self,
        head_importance_threshold: float = 0.1,
        adaptive_window_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.head_importance_threshold = head_importance_threshold
        self.adaptive_window_size = adaptive_window_size
        self.cache_state: Dict[str, Any] | None = None

    def initialize_cache(self, model_config: Dict[str, Any] | None = None) -> None:
        """Initialize cache data structures for HACE.

        Parameters
        ----------
        model_config:
            Optional dictionary describing the model.  The current
            implementation only uses this to determine how many layers and
            heads may be present, but the field is kept generic for future
            extension.
        """
        self.cache_state = {}

    def update_cache(
        self,
        layer_idx: int,
        head_idx: int,
        key: Any,
        value: Any,
        importance: float | None = None,
    ) -> None:
        """Update cache contents for a single attention head.

        Tokens whose importance score is below ``head_importance_threshold`` are
        ignored.  For the remaining tokens, only the most recent
        ``adaptive_window_size`` entries are retained.
        """
        if self.cache_state is None:
            self.initialize_cache()

        if importance is not None and importance < self.head_importance_threshold:
            return

        key_value_list = self.cache_state.setdefault((layer_idx, head_idx), [])
        key_value_list.append((key, value, importance))
        if len(key_value_list) > self.adaptive_window_size:
            # Keep only the newest entries
            self.cache_state[(layer_idx, head_idx)] = key_value_list[-self.adaptive_window_size:]

    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """Return HACE-specific metrics.

        Currently returns the number of cached entries.
        TODO: expose more detailed metrics such as eviction counts.
        """
        if self.cache_state is None:
            return {"cache_entries": 0, "cached_tokens": 0}

        total_tokens = sum(len(v) for v in self.cache_state.values())
        return {"cache_entries": len(self.cache_state), "cached_tokens": total_tokens}
