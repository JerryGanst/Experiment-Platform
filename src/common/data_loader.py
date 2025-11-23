"""
统一数据加载模块 - 兼容层

!!! 废弃警告 !!!
此模块已废弃，请使用 hace_core.data 替代：

    # 旧方式（已废弃）
    from src.common.data_loader import find_data_file, load_local_jsonl_data

    # 新方式（推荐）
    from hace_core.data import load_data, find_local_file

迁移指南：
    - find_data_file -> hace_core.data.find_local_file
    - load_local_jsonl_data -> hace_core.data.load_local_jsonl
    - get_project_root -> hace_core.data.get_project_root
    - get_data_root -> hace_core.data.get_data_root
"""

import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path

# 发出废弃警告
warnings.warn(
    "src.common.data_loader 已废弃，请使用 hace_core.data 替代。"
    "详见 docs/REFACTORING_PLAN.md",
    DeprecationWarning,
    stacklevel=2
)

# 从新模块导入并重新导出
try:
    from hace_core.data.unified_loader import (
        get_project_root,
        get_data_root,
        find_local_file as find_data_file,
        load_local_jsonl as _load_local_jsonl,
        get_runs_directory,
        get_configs_directory,
        validate_sample as validate_sample_content,
    )

    def load_local_jsonl_data(
        dataset_name: str,
        max_samples: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """从本地 JSONL 文件加载数据（兼容旧接口）

        已废弃：请使用 hace_core.data.load_local_jsonl
        """
        try:
            return _load_local_jsonl(dataset_name, max_samples=max_samples)
        except FileNotFoundError:
            return None

    def load_config_file(config_name: str, file_type: str = "csv") -> Optional[Path]:
        """加载配置文件路径"""
        configs_dir = get_configs_directory()
        config_file = configs_dir / f"{config_name}.{file_type}"

        if config_file.exists():
            return config_file
        else:
            import logging
            logging.getLogger(__name__).warning(f"配置文件不存在: {config_file}")
            return None

except ImportError:
    # 如果新模块不可用，保留原有实现作为后备
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    def get_project_root() -> Path:
        """获取项目根目录路径"""
        current_file = Path(__file__).resolve()
        return current_file.parents[2]

    def get_data_root() -> Path:
        """根据环境变量或默认位置返回数据集根目录"""
        env_path = os.environ.get("DATASETS_ROOT")
        if env_path:
            resolved = Path(env_path).expanduser()
            logger.debug("使用DATASETS_ROOT: %s", resolved)
            return resolved
        project_root = get_project_root()
        return project_root / "data"

    def find_data_file(dataset_name: str) -> Optional[Path]:
        """查找数据文件路径"""
        project_root = get_project_root()
        data_root = get_data_root()

        possible_paths = [
            data_root / f"{dataset_name}.jsonl",
            project_root / "data" / f"{dataset_name}.jsonl",
            Path(f"./{dataset_name}.jsonl"),
            Path(f"./data/{dataset_name}.jsonl"),
        ]

        for path in possible_paths:
            if path.exists():
                logger.info("找到数据文件: %s", path)
                return path

        logger.warning("未找到数据文件: %s.jsonl", dataset_name)
        return None

    def load_local_jsonl_data(
        dataset_name: str,
        max_samples: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """从本地 JSONL 文件加载数据"""
        data_path = find_data_file(dataset_name)
        if not data_path:
            return None

        try:
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                            if max_samples and len(data) >= max_samples:
                                break
                        except json.JSONDecodeError:
                            pass
            logger.info(f"从本地加载 {dataset_name}，共 {len(data)} 条样本")
            return data
        except Exception as e:
            logger.error(f"加载本地JSONL文件失败: {e}")
            return None

    def validate_sample_content(sample: Dict[str, Any], dataset_name: str) -> bool:
        """验证样本内容是否有效"""
        if not isinstance(sample, dict):
            return False
        content_fields = ['input', 'context', 'text', 'prompt', 'question']
        for field in content_fields:
            if field in sample and sample[field] and str(sample[field]).strip():
                return True
        return False

    def get_runs_directory() -> Path:
        """获取 runs 目录路径"""
        project_root = get_project_root()
        runs_dir = project_root / "runs"
        runs_dir.mkdir(exist_ok=True)
        return runs_dir

    def get_configs_directory() -> Path:
        """获取配置文件目录路径"""
        project_root = get_project_root()
        return project_root / "configs"

    def load_config_file(config_name: str, file_type: str = "csv") -> Optional[Path]:
        """加载配置文件路径"""
        configs_dir = get_configs_directory()
        config_file = configs_dir / f"{config_name}.{file_type}"
        if config_file.exists():
            return config_file
        return None


__all__ = [
    "get_project_root",
    "get_data_root",
    "find_data_file",
    "load_local_jsonl_data",
    "validate_sample_content",
    "get_runs_directory",
    "get_configs_directory",
    "load_config_file",
]
