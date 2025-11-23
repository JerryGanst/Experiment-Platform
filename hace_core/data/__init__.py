"""
数据加载子包

提供统一的数据加载接口，支持 HuggingFace 和本地文件。

推荐使用:
    from hace_core.data import load_data

    # 自动检测数据源
    data = load_data("hotpotqa")

    # 显式指定 HuggingFace
    data = load_data("hf://THUDM/LongBench", subset="hotpotqa")

    # 显式指定本地文件
    data = load_data("local://data/custom.jsonl")
"""

# 统一加载接口（无外部依赖）
from .unified_loader import (
    # 主接口
    load_data,
    DataSource,
    DataSourceType,
    # 底层加载器
    load_local_jsonl,
    load_huggingface_dataset,
    # 路径工具
    get_project_root,
    get_data_root,
    find_local_file,
    get_runs_directory,
    get_configs_directory,
    # 便捷函数
    list_local_datasets,
    validate_sample,
    # 兼容性导出
    find_data_file,
    load_local_jsonl_data,
)


# 延迟导入：dataset_loader 依赖 datasets 库
def __getattr__(name):
    """延迟导入需要外部依赖的模块"""
    _dataset_loader_exports = {
        "load_dataset_split",
        "prepare_samples_for_evaluation",
        "prepare_batch",
        "get_dataset_info",
        "get_available_datasets",
    }

    if name in _dataset_loader_exports:
        from . import dataset_loader
        return getattr(dataset_loader, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 统一接口
    "load_data",
    "DataSource",
    "DataSourceType",
    # 底层加载器
    "load_local_jsonl",
    "load_huggingface_dataset",
    "load_dataset_split",
    # 样本处理
    "prepare_samples_for_evaluation",
    "prepare_batch",
    # 数据集信息
    "get_dataset_info",
    "get_available_datasets",
    "list_local_datasets",
    "validate_sample",
    # 路径工具
    "get_project_root",
    "get_data_root",
    "find_local_file",
    "find_data_file",
    "get_runs_directory",
    "get_configs_directory",
    # 兼容性
    "load_local_jsonl_data",
]
