"""
断点续跑管理器

支持:
- 实验状态持久化
- 崩溃恢复
- 进度查询
"""
import json
import shutil
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import CHECKPOINT_FILE, CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


@contextmanager
def simple_file_lock(lock_path: Path):
    """简单的文件锁实现 (使用 fcntl)"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


class CheckpointManager:
    """
    断点管理器

    使用原子写入确保崩溃安全
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_FILE
        self.lock_path = self.checkpoint_path.with_suffix(".lock")
        self._ensure_dir()

    def _ensure_dir(self):
        """确保检查点目录存在"""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_lock(self):
        """获取文件锁"""
        return simple_file_lock(self.lock_path)

    def load(self) -> Dict[str, Any]:
        """
        加载检查点状态

        Returns:
            检查点字典，如果不存在则返回空状态
        """
        with self._get_lock():
            if not self.checkpoint_path.exists():
                return self._create_empty_state()

            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint: {e}, creating new state")
                return self._create_empty_state()

    def save(self, state: Dict[str, Any]) -> None:
        """
        原子性保存检查点

        使用临时文件+重命名确保原子性
        """
        state["last_updated"] = datetime.now().isoformat()

        with self._get_lock():
            # 写入临时文件
            temp_path = self.checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            # 原子重命名
            shutil.move(str(temp_path), str(self.checkpoint_path))

    def _create_empty_state(self) -> Dict[str, Any]:
        """创建空的检查点状态"""
        return {
            "version": "1.0",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_experiments": 0,
            "experiments": {},
        }

    def is_completed(self, experiment_id: str) -> bool:
        """检查实验是否已完成"""
        state = self.load()
        exp_state = state.get("experiments", {}).get(experiment_id, {})
        return exp_state.get("status") == "completed"

    def is_failed(self, experiment_id: str) -> bool:
        """检查实验是否失败"""
        state = self.load()
        exp_state = state.get("experiments", {}).get(experiment_id, {})
        return exp_state.get("status") == "failed"

    def get_retry_count(self, experiment_id: str) -> int:
        """获取实验的重试次数"""
        state = self.load()
        exp_state = state.get("experiments", {}).get(experiment_id, {})
        return exp_state.get("retry_count", 0)

    def mark_started(self, experiment_id: str) -> None:
        """标记实验开始"""
        state = self.load()

        if "experiments" not in state:
            state["experiments"] = {}

        state["experiments"][experiment_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "retry_count": state["experiments"].get(experiment_id, {}).get("retry_count", 0),
        }

        self.save(state)

    def mark_completed(
        self,
        experiment_id: str,
        output_path: str,
        score: Optional[float] = None,
        duration_sec: Optional[float] = None,
    ) -> None:
        """标记实验完成"""
        state = self.load()

        if "experiments" not in state:
            state["experiments"] = {}

        state["experiments"][experiment_id] = {
            "status": "completed",
            "started_at": state["experiments"].get(experiment_id, {}).get("started_at"),
            "completed_at": datetime.now().isoformat(),
            "output_path": output_path,
            "score": score,
            "duration_sec": duration_sec,
        }

        self.save(state)

    def mark_failed(self, experiment_id: str, error: str) -> None:
        """标记实验失败"""
        state = self.load()

        if "experiments" not in state:
            state["experiments"] = {}

        retry_count = state["experiments"].get(experiment_id, {}).get("retry_count", 0) + 1

        state["experiments"][experiment_id] = {
            "status": "failed",
            "started_at": state["experiments"].get(experiment_id, {}).get("started_at"),
            "failed_at": datetime.now().isoformat(),
            "error": error,
            "retry_count": retry_count,
        }

        self.save(state)

    def get_pending(self, experiment_ids: List[str], max_retries: int = 2) -> List[str]:
        """
        获取待执行的实验列表

        Args:
            experiment_ids: 所有实验ID
            max_retries: 最大重试次数

        Returns:
            待执行的实验ID列表
        """
        state = self.load()
        experiments = state.get("experiments", {})

        pending = []
        for exp_id in experiment_ids:
            if exp_id not in experiments:
                pending.append(exp_id)
            elif experiments[exp_id].get("status") == "failed":
                if experiments[exp_id].get("retry_count", 0) < max_retries:
                    pending.append(exp_id)
            elif experiments[exp_id].get("status") == "running":
                # 可能是上次崩溃，重新执行
                pending.append(exp_id)

        return pending

    def get_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        state = self.load()
        experiments = state.get("experiments", {})

        completed = sum(1 for e in experiments.values() if e.get("status") == "completed")
        failed = sum(1 for e in experiments.values() if e.get("status") == "failed")
        running = sum(1 for e in experiments.values() if e.get("status") == "running")

        return {
            "total": state.get("total_experiments", 0),
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": state.get("total_experiments", 0) - completed - failed - running,
            "started_at": state.get("started_at"),
            "last_updated": state.get("last_updated"),
        }

    def reset(self) -> None:
        """重置检查点 (删除所有进度)"""
        with self._get_lock():
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            logger.info("Checkpoint reset")


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    manager = CheckpointManager()
    print("Initial state:", manager.load())

    manager.mark_started("test_exp_1")
    print("After start:", manager.get_summary())

    manager.mark_completed("test_exp_1", "/path/to/output", score=0.85, duration_sec=120.5)
    print("After complete:", manager.get_summary())

    manager.mark_failed("test_exp_2", "CUDA OOM")
    print("After fail:", manager.get_summary())

    # 清理测试
    manager.reset()
