from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class BaseKVCacheMethod(ABC):
    """Abstract base class for KV cache manipulation methods."""

    @abstractmethod
    def initialize_cache(self) -> None:
        """Initialize any internal cache structures."""
        raise NotImplementedError

    @abstractmethod
    def update_cache(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Update the cache given new key/value states."""
        raise NotImplementedError

    @abstractmethod
    def get_method_specific_metrics(self) -> dict:
        """Return metrics collected during cache usage specific to the method."""
        raise NotImplementedError

    def get_memory_usage(self) -> float:
        """Return the current GPU memory usage in megabytes."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

