from abc import ABC, abstractmethod

class BaseKVCacheMethod(ABC):
    """Abstract base class for KV cache optimization methods."""

    @abstractmethod
    def initialize_cache(self, *args, **kwargs):
        """Initialize and return cache object"""
        raise NotImplementedError

    @abstractmethod
    def update_cache(self, *args, **kwargs):
        """Update the cache with new key/value states"""
        raise NotImplementedError
