"""
统一数据加载模块 - 适配重构后的项目结构
去重所有重复的数据加载函数
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """获取项目根目录路径"""
    current_file = Path(__file__).resolve()
    return current_file.parents[2]  # 从src/common/回到项目根目录


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
    """查找数据文件路径 - 优先使用 DATASETS_ROOT"""
    project_root = get_project_root()
    data_root = get_data_root()

    possible_paths = [
        data_root / f"{dataset_name}.jsonl",
        project_root / "data" / f"{dataset_name}.jsonl",
        project_root / "hace-kv-optimization" / "data" / f"{dataset_name}.jsonl",
        project_root / "hace-kv-optimization" / "baselines" / "data" / f"{dataset_name}.jsonl",
        Path(f"./{dataset_name}.jsonl"),
        Path(f"./data/{dataset_name}.jsonl"),
    ]

    for path in possible_paths:
        if path.exists():
            logger.info("📂 找到数据文件: %s", path)
            return path

    logger.warning("❌ 未找到数据文件: %s.jsonl", dataset_name)
    logger.warning("搜索路径: %s", [str(p) for p in possible_paths])
    logger.warning(
        "提示: 可以设置 DATASETS_ROOT 环境变量指向远端或本地数据集目录"
    )
    return None


def load_local_jsonl_data(dataset_name: str, max_samples: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """
    从本地JSONL文件加载数据 - 统一版本
    
    Args:
        dataset_name: 数据集名称
        max_samples: 最大样本数，None表示加载全部
        
    Returns:
        数据列表，失败时返回None
    """
    data_path = find_data_file(dataset_name)
    if not data_path:
        return None

    try:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                        if max_samples and len(data) >= max_samples:
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_num}: {line[:50]}... 错误: {e}")

        logger.info(f"✅ 从本地加载 {dataset_name}，共 {len(data)} 条样本")
        return data

    except Exception as e:
        logger.error(f"❌ 加载本地JSONL文件失败: {e}")
        return None


def validate_sample_content(sample: Dict[str, Any], dataset_name: str) -> bool:
    """验证样本内容是否有效"""
    if not isinstance(sample, dict):
        return False
    
    # 检查关键字段
    content_fields = ['input', 'context', 'text', 'prompt', 'question', 'document', 'articles']
    for field in content_fields:
        if field in sample and sample[field] and str(sample[field]).strip():
            return True
    
    logger.warning(f"⚠️ 数据集 {dataset_name} 的样本缺少有效内容字段")
    return False


def get_runs_directory() -> Path:
    """获取runs目录路径"""
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
    else:
        logger.warning(f"配置文件不存在: {config_file}")
        return None 
