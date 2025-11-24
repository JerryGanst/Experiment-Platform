"""Compatibility wrapper for :mod:`src.core_code.unified_allocator`.

Delegates to :mod:`hace_core.core.unified_allocator`.
"""

from hace_core.core.unified_allocator import *  # noqa: F401,F403

__all__ = [
    "UnifiedCakeAdaKVAllocator",
    "UnifiedCacheConfig",
]
