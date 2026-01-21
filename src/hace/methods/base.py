from __future__ import annotations

from abc import ABC, abstractmethod

# Torch is optional for this lightweight stub; if missing we create a dummy
# module with the minimal attributes used in this file. This avoids a hard
# dependency when running on environments without GPU/torch installed.
try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import types, sys

    torch_stub = types.ModuleType("torch")

    def _max_mem():
        return 0

    torch_stub.cuda = types.ModuleType("torch.cuda")  # type: ignore
    torch_stub.cuda.is_available = lambda: False  # type: ignore
    torch_stub.cuda.max_memory_allocated = _max_mem  # type: ignore
    sys.modules["torch"] = torch_stub
    import torch  # type: ignore


class BaseKVCacheMethod(ABC):
    """Abstract base class for KV cache manipulation methods.

    This is the single source of truth for the base class.
    Import from: from src.hace.methods.base import BaseKVCacheMethod
    """

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