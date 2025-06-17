"""script_resolver.py
动态发现项目中可执行脚本路径，避免硬编码。

用法示例：
    from utilities.script_resolver import ScriptResolver
    resolver = ScriptResolver()
    baseline_main = resolver.get_script_path('baseline')
"""
from pathlib import Path
from typing import Dict, List


class ScriptResolver:
    """智能脚本发现与解析器。"""

    # 默认脚本名称模式，可根据需要扩展
    _DEFAULT_PATTERNS: Dict[str, List[str]] = {
        "baseline": [
            "baseline_main.py",
            "baseline.py",
            "fullkvcache_main.py",  # 项目中常见的基线脚本
        ],
        "fullkvcache": [
            "fullkvcache_main.py",
            "full_kv_cache_main.py",
        ],
        "experiment": [
            "run_experiment.py",
            "experiment.py",
        ],
    }

    def __init__(self, base_dir: Path | str | None = None, patterns: Dict[str, List[str]] | None = None):
        """初始化解析器。

        Args:
            base_dir: 开始搜索脚本的根目录（默认为当前工作目录）。
            patterns: 自定义脚本名称模式，若为空则使用默认模式。
        """
        self.base_dir: Path = Path(base_dir) if base_dir else Path.cwd()
        self.patterns: Dict[str, List[str]] = patterns or self._DEFAULT_PATTERNS
        self._cache: Dict[str, Path] = {}
        self._discover_scripts()

    # ---------------------------------------------------------------------
    # 公共 API
    # ---------------------------------------------------------------------
    def get_script_path(self, script_type: str) -> Path:
        """获取指定类型脚本的绝对路径。

        Raises:
            FileNotFoundError: 如果未找到对应脚本。
        """
        script_type = script_type.lower()
        if script_type not in self._cache:
            raise FileNotFoundError(f"未找到 '{script_type}' 类型的脚本")
        return self._cache[script_type]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _discover_scripts(self) -> None:
        """遍历 base_dir，找到满足模式的脚本并缓存路径。"""
        for script_type, names in self.patterns.items():
            for name in names:
                candidate = self.base_dir / name
                if candidate.is_file():
                    # 优先第一个匹配到的脚本
                    self._cache.setdefault(script_type, candidate.resolve())
                    break 