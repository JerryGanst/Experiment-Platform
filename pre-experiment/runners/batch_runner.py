"""
批量实验执行器

支持:
- 54次实验的自动化运行
- 断点续跑
- 进度报告
"""
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.phase0_config import (
    generate_experiment_grid,
    get_experiment_id,
    TOTAL_EXPERIMENTS,
    ensure_directories,
)
from .checkpoint_manager import CheckpointManager
from .experiment_runner import run_single_experiment, check_experiment_output

logger = logging.getLogger(__name__)


class BatchRunner:
    """
    批量实验执行器

    使用检查点实现断点续跑
    """

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_retries: int = 2,
    ):
        self.checkpoint = checkpoint_manager or CheckpointManager()
        self.max_retries = max_retries
        self.experiment_grid = generate_experiment_grid()

    def run(
        self,
        device: int = 0,
        skip_completed: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        运行所有实验

        Args:
            device: CUDA 设备号
            skip_completed: 是否跳过已完成的实验
            dry_run: 仅打印命令，不执行

        Returns:
            执行摘要
        """
        ensure_directories()

        # 更新总实验数
        state = self.checkpoint.load()
        state["total_experiments"] = TOTAL_EXPERIMENTS
        self.checkpoint.save(state)

        # 获取待执行实验
        all_exp_ids = [exp["experiment_id"] for exp in self.experiment_grid]

        if skip_completed:
            pending_ids = self.checkpoint.get_pending(all_exp_ids, self.max_retries)
        else:
            pending_ids = all_exp_ids

        pending_exps = [
            exp for exp in self.experiment_grid
            if exp["experiment_id"] in pending_ids
        ]

        logger.info(f"Total experiments: {TOTAL_EXPERIMENTS}")
        logger.info(f"Pending experiments: {len(pending_exps)}")
        logger.info(f"Skipped (completed/max retries): {TOTAL_EXPERIMENTS - len(pending_exps)}")

        if dry_run:
            logger.info("[DRY RUN MODE] No experiments will be executed")

        results = []
        start_time = time.time()

        for i, exp in enumerate(pending_exps, 1):
            exp_id = exp["experiment_id"]
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment {i}/{len(pending_exps)}: {exp_id}")
            logger.info(f"{'='*60}")

            # 标记开始
            if not dry_run:
                self.checkpoint.mark_started(exp_id)

            # 执行实验
            result = run_single_experiment(
                strategy=exp["strategy"],
                cache_size=exp["cache_size"],
                dataset=exp["dataset"],
                seed=exp["seed"],
                device=device,
                dry_run=dry_run,
            )

            # 更新检查点
            if not dry_run:
                if result["status"] == "success":
                    self.checkpoint.mark_completed(
                        exp_id,
                        output_path=result["output_path"],
                        duration_sec=result.get("duration_sec"),
                    )
                else:
                    self.checkpoint.mark_failed(
                        exp_id,
                        error=result.get("error", "Unknown error"),
                    )

            results.append(result)

            # 打印进度
            summary = self.checkpoint.get_summary()
            self._print_progress(summary, i, len(pending_exps))

        total_duration = time.time() - start_time

        return {
            "total": TOTAL_EXPERIMENTS,
            "executed": len(pending_exps),
            "results": results,
            "duration_sec": total_duration,
            "summary": self.checkpoint.get_summary(),
        }

    def _print_progress(self, summary: Dict, current: int, total: int):
        """打印进度信息"""
        completed = summary["completed"]
        failed = summary["failed"]
        total_exp = summary["total"]

        progress_pct = (completed / total_exp * 100) if total_exp > 0 else 0

        logger.info(f"\nProgress: {completed}/{total_exp} ({progress_pct:.1f}%)")
        logger.info(f"  Completed: {completed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Current batch: {current}/{total}")

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.checkpoint.get_summary()

    def reset(self) -> None:
        """重置所有进度"""
        self.checkpoint.reset()
        logger.info("All progress reset")


def run_batch(
    device: int = 0,
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    便捷函数：运行批量实验

    Args:
        device: CUDA 设备号
        resume: 是否从断点恢复
        dry_run: 仅打印命令，不执行

    Returns:
        执行摘要
    """
    runner = BatchRunner()
    return runner.run(device=device, skip_completed=resume, dry_run=dry_run)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 测试干运行
    result = run_batch(device=0, resume=True, dry_run=True)
    print(f"\nExecution summary:")
    print(f"  Total experiments: {result['total']}")
    print(f"  Would execute: {result['executed']}")
