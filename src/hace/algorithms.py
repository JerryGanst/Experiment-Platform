"""Skeleton implementations of core HACE algorithms."""

from __future__ import annotations

from typing import Any, Iterable, Tuple


class BaseHACEAlgorithm:
    """Base class for HACE algorithms."""

    def __init__(self, head_threshold: float = 0.1, window_size: int = 64) -> None:
        self.head_threshold = head_threshold
        self.window_size = window_size

    def compute_head_importance(self, attn_scores: Any) -> Iterable[float]:
        """Compute importance for each attention head.

        This method should return an iterable of importance scores for every
        head in ``attn_scores``.
        """
        raise NotImplementedError

    def select_tokens_to_keep(
        self, importance_scores: Iterable[float], cache_state: Any
    ) -> Tuple[Any, Any]:
        """Select which tokens to keep given importance scores and cache state."""
        raise NotImplementedError

    def update_state(
        self, layer_idx: int, head_idx: int, key: Any, value: Any, score: float
    ) -> None:
        """Update any internal state for this algorithm."""
        raise NotImplementedError


class SimpleHACEAlgorithm(BaseHACEAlgorithm):
    """A minimal reference algorithm used for testing."""

    def compute_head_importance(self, attn_scores: Any) -> Iterable[float]:
        # Placeholder: mean attention score as importance
        return attn_scores.mean(dim=-1).tolist()

    def select_tokens_to_keep(
        self, importance_scores: Iterable[float], cache_state: Any
    ) -> Tuple[Any, Any]:
        # Placeholder: keep tokens for heads whose score is above threshold
        keep_mask = [s >= self.head_threshold for s in importance_scores]
        return keep_mask, cache_state

    def update_state(
        self, layer_idx: int, head_idx: int, key: Any, value: Any, score: float
    ) -> None:
        # No stateful behaviour in the skeleton
        pass
