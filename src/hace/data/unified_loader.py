"""
统一数据加载模块

提供统一的数据加载接口，支持：
- HuggingFace datasets (hf://dataset_name)
- 本地 JSONL 文件 (local://path 或 file://path)
- 自动检测数据源类型

使用方式：
    from src.hace.data.unified_loader import load_data, DataSource

    # 自动检测
    data = load_data("hotpotqa")

    # 显式指定 HuggingFace
    data = load_data("hf://THUDM/LongBench", subset="hotpotqa")

    # 显式指定本地文件
    data = load_data("local://data/hotpotqa.jsonl")
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型"""
    HUGGINGFACE = "huggingface"
    LOCAL_FILE = "local"
    AUTO = "auto"


@dataclass
class DataSource:
    """数据源描述"""
    source_type: DataSourceType
    path: str
    subset: Optional[str] = None
    split: str = "validation"

    @classmethod
    def from_string(cls, source: str) -> "DataSource":
        """从字符串解析数据源

        支持格式：
        - "hf://dataset_name" -> HuggingFace
        - "local://path/to/file.jsonl" -> 本地文件
        - "file://path/to/file.jsonl" -> 本地文件
        - "dataset_name" -> 自动检测
        """
        if source.startswith("hf://"):
            return cls(
                source_type=DataSourceType.HUGGINGFACE,
                path=source[5:],
            )
        elif source.startswith("local://") or source.startswith("file://"):
            prefix_len = 8 if source.startswith("local://") else 7
            return cls(
                source_type=DataSourceType.LOCAL_FILE,
                path=source[prefix_len:],
            )
        else:
            return cls(
                source_type=DataSourceType.AUTO,
                path=source,
            )


# ============== 路径工具 ==============

def get_project_root() -> Path:
    """获取项目根目录路径"""
    current_file = Path(__file__).resolve()
    # 从 hace_core/data/ 回到项目根目录
    return current_file.parents[2]


def get_data_root() -> Path:
    """根据环境变量或默认位置返回数据集根目录"""
    env_path = os.environ.get("DATASETS_ROOT")
    if env_path:
        resolved = Path(env_path).expanduser()
        logger.debug("使用 DATASETS_ROOT: %s", resolved)
        return resolved
    return get_project_root() / "data"


def find_local_file(filename: str) -> Optional[Path]:
    """查找本地数据文件

    搜索顺序：
    1. DATASETS_ROOT 环境变量指定的目录
    2. 项目 data/ 目录
    3. 当前目录
    """
    # 如果是绝对路径，直接检查
    if os.path.isabs(filename):
        path = Path(filename)
        if path.exists():
            return path
        return None

    project_root = get_project_root()
    data_root = get_data_root()

    # 确保有 .jsonl 扩展名
    if not filename.endswith(".jsonl") and not filename.endswith(".json"):
        filename_with_ext = f"{filename}.jsonl"
    else:
        filename_with_ext = filename

    search_paths = [
        data_root / filename_with_ext,
        project_root / "data" / filename_with_ext,
        project_root / "data" / "raw" / filename_with_ext,
        Path(filename_with_ext),
        Path(f"./data/{filename_with_ext}"),
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"找到本地数据文件: {path}")
            return path

    logger.warning(f"未找到本地数据文件: {filename}")
    logger.debug(f"搜索路径: {[str(p) for p in search_paths]}")
    return None


# ============== 本地文件加载 ==============

def load_local_jsonl(
    filepath: Union[str, Path],
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从本地 JSONL 文件加载数据

    Args:
        filepath: 文件路径（可以是文件名或完整路径）
        max_samples: 最大样本数，None 表示加载全部

    Returns:
        数据列表

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析失败
    """
    # 查找文件
    if isinstance(filepath, str):
        found_path = find_local_file(filepath)
        if found_path is None:
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        filepath = found_path

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    data = []
    errors = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
                if max_samples and len(data) >= max_samples:
                    break
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"跳过无效 JSON 行 {line_num}: {str(e)[:50]}")

    if errors > 3:
        logger.warning(f"共跳过 {errors} 行无效 JSON")

    logger.info(f"从本地加载 {len(data)} 条样本: {filepath}")
    return data


# ============== HuggingFace 加载 ==============

def load_huggingface_dataset(
    path: str,
    subset: Optional[str] = None,
    split: str = "validation",
    trust_remote_code: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从 HuggingFace 加载数据集

    Args:
        path: 数据集路径（如 "THUDM/LongBench"）
        subset: 子集名称（如 "hotpotqa"）
        split: 数据分割（如 "validation", "test"）
        trust_remote_code: 是否信任远程代码
        max_samples: 最大样本数

    Returns:
        数据列表
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "需要安装 datasets 库: pip install datasets"
        )

    logger.info(f"从 HuggingFace 加载: {path} (subset={subset}, split={split})")

    try:
        if subset:
            dataset = hf_load_dataset(
                path,
                name=subset,
                split=split,
                trust_remote_code=trust_remote_code
            )
        else:
            dataset = hf_load_dataset(
                path,
                split=split,
                trust_remote_code=trust_remote_code
            )

        # 转换为列表
        if max_samples and len(dataset) > max_samples:
            data = [dataset[i] for i in range(max_samples)]
        else:
            data = list(dataset)

        logger.info(f"从 HuggingFace 加载 {len(data)} 条样本")
        return data

    except Exception as e:
        logger.error(f"HuggingFace 加载失败: {e}")
        raise


# ============== 统一加载接口 ==============

def load_data(
    source: Union[str, DataSource, Dict[str, Any]],
    subset: Optional[str] = None,
    split: str = "validation",
    max_samples: Optional[int] = None,
    prefer_local: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """统一数据加载接口

    Args:
        source: 数据源，支持：
            - 字符串: "hf://path", "local://path", 或数据集名称
            - DataSource 对象
            - 配置字典 {"path": "...", "subset": "..."}
        subset: HuggingFace 数据集子集
        split: 数据分割
        max_samples: 最大样本数
        prefer_local: 自动检测时是否优先使用本地文件
        **kwargs: 传递给底层加载器的其他参数

    Returns:
        数据列表

    Examples:
        # 自动检测
        data = load_data("hotpotqa")

        # HuggingFace
        data = load_data("hf://THUDM/LongBench", subset="hotpotqa")

        # 本地文件
        data = load_data("local://data/custom.jsonl")

        # 配置字典
        data = load_data({"path": "THUDM/LongBench", "subset": "hotpotqa"})
    """
    # 解析数据源
    if isinstance(source, str):
        ds = DataSource.from_string(source)
        if subset:
            ds.subset = subset
        ds.split = split
    elif isinstance(source, DataSource):
        ds = source
    elif isinstance(source, dict):
        # 配置字典格式
        ds = DataSource(
            source_type=DataSourceType.AUTO,
            path=source.get("path", ""),
            subset=source.get("subset") or subset,
            split=source.get("split", split),
        )
    else:
        raise TypeError(f"不支持的数据源类型: {type(source)}")

    # 根据类型加载
    if ds.source_type == DataSourceType.LOCAL_FILE:
        return load_local_jsonl(ds.path, max_samples=max_samples)

    elif ds.source_type == DataSourceType.HUGGINGFACE:
        return load_huggingface_dataset(
            ds.path,
            subset=ds.subset,
            split=ds.split,
            max_samples=max_samples,
            **kwargs
        )

    else:  # AUTO
        # 自动检测：优先本地，其次 HuggingFace
        if prefer_local:
            local_path = find_local_file(ds.path)
            if local_path:
                return load_local_jsonl(local_path, max_samples=max_samples)

        # 尝试 HuggingFace
        try:
            return load_huggingface_dataset(
                ds.path,
                subset=ds.subset,
                split=ds.split,
                max_samples=max_samples,
                **kwargs
            )
        except Exception as hf_error:
            # 如果 HuggingFace 失败且还没尝试本地
            if not prefer_local:
                local_path = find_local_file(ds.path)
                if local_path:
                    logger.info(f"HuggingFace 失败，回退到本地文件")
                    return load_local_jsonl(local_path, max_samples=max_samples)
            raise hf_error


# ============== 兼容性导出 ==============

# 从原 src/common/data_loader.py 导出的函数
def find_data_file(dataset_name: str) -> Optional[Path]:
    """兼容旧接口：查找数据文件路径"""
    return find_local_file(dataset_name)


def load_local_jsonl_data(
    dataset_name: str,
    max_samples: Optional[int] = None
) -> Optional[List[Dict[str, Any]]]:
    """兼容旧接口：从本地 JSONL 文件加载数据"""
    try:
        return load_local_jsonl(dataset_name, max_samples=max_samples)
    except FileNotFoundError:
        return None


def get_runs_directory() -> Path:
    """获取 runs 目录路径"""
    runs_dir = get_project_root() / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir


def get_configs_directory() -> Path:
    """获取配置文件目录路径"""
    return get_project_root() / "configs"


# ============== 便捷函数 ==============

def list_local_datasets() -> List[str]:
    """列出本地可用的数据集"""
    data_root = get_data_root()
    datasets = []

    if data_root.exists():
        for f in data_root.glob("*.jsonl"):
            datasets.append(f.stem)

    return sorted(datasets)


def validate_sample(sample: Dict[str, Any]) -> bool:
    """验证样本内容是否有效"""
    if not isinstance(sample, dict):
        return False

    # 检查关键字段
    content_fields = [
        'input', 'context', 'text', 'prompt',
        'question', 'document', 'articles'
    ]

    for field in content_fields:
        if field in sample and sample[field]:
            value = sample[field]
            if isinstance(value, str) and value.strip():
                return True

    return False
