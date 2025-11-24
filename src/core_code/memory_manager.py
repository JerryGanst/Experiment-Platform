"""Compatibility wrapper for :mod:`src.core_code.memory_manager`."""

from hace_core.core.memory_manager import *  # noqa: F401,F403

__all__ = [
    "UnifiedMemoryManager",
    "MemoryConfig",
]
