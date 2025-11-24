"""Compatibility shim for :mod:`src.common.data_loader`.

This module forwards legacy imports to the refactored implementations in
``hace_core.data`` so existing callers continue to work.
"""

from hace_core.data import (
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
