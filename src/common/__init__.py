"""Compatibility package for legacy :mod:`src.common` imports.

The original implementations now live under :mod:`hace_core.data`. This
shim preserves existing entry points (e.g. ``src.common.data_loader``)
so older scripts and tests keep working while the refactor completes.
"""

from .data_loader import (
    find_data_file,
    find_local_file,
    get_configs_directory,
    get_data_root,
    get_project_root,
    get_runs_directory,
    load_local_jsonl_data,
)

__all__ = [
    "find_data_file",
    "find_local_file",
    "get_configs_directory",
    "get_data_root",
    "get_project_root",
    "get_runs_directory",
    "load_local_jsonl_data",
]
