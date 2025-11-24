"""Compatibility wrapper for :mod:`src.core_code.integration_framework`.

Exports are forwarded to :mod:`hace_core.core.integration_framework`.
"""

from hace_core.core.integration_framework import *  # noqa: F401,F403

__all__ = [
    "CakeAdaKVIntegration",
    "IntegrationConfig",
    "create_integration",
]
