#!/usr/bin/env python
"""
Phase 0 假设验证实验 - 主入口

用法:
    # 运行实验 (支持断点续跑)
    python pre-experiment/main.py run --device 0

    # 从断点恢复
    python pre-experiment/main.py run --resume --device 0

    # 仅分析 (不运行实验)
    python pre-experiment/main.py analyze

    # 仅绘图
    python pre-experiment/main.py plot

    # 完整流程
    python pre-experiment/main.py all --device 0

    # 干运行 (预览)
    python pre-experiment/main.py run --dry-run

    # 查看状态
    python pre-experiment/main.py status

    # 重置进度
    python pre-experiment/main.py reset
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.phase0_config import (
    TOTAL_EXPERIMENTS,
    RESULTS_DIR,
    RAW_RESULTS_CSV,
    ANOVA_RESULTS_JSON,
    INTERACTION_PLOT_PNG,
    DECISION_MD,
    ensure_directories,
    validate_config,
)
from runners.batch_runner import BatchRunner
from runners.checkpoint_manager import CheckpointManager
from parsers.result_aggregator import aggregate_results, get_completed_results
from analysis.two_way_anova import run_anova_by_dataset
from analysis.decision_generator import generate_decision
from visualization.interaction_plot import create_interaction_plot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "phase0.log", mode="a"),
    ],
)
logger = logging.getLogger("phase0")


def cmd_run(args):
    """运行实验"""
    logger.info("=" * 60)
    logger.info("Phase 0: Running Experiments")
    logger.info("=" * 60)

    # 验证配置
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    ensure_directories()

    runner = BatchRunner()
    result = runner.run(
        device=args.device,
        skip_completed=args.resume,
        dry_run=args.dry_run,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Execution Summary")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {result['total']}")
    logger.info(f"Executed this run: {result['executed']}")
    logger.info(f"Duration: {result['duration_sec']:.1f}s")

    summary = result["summary"]
    logger.info(f"Completed: {summary['completed']}")
    logger.info(f"Failed: {summary['failed']}")

    return 0


def cmd_analyze(args):
    """分析结果"""
    logger.info("=" * 60)
    logger.info("Phase 0: Analyzing Results")
    logger.info("=" * 60)

    ensure_directories()

    # 聚合结果
    logger.info("\n1. Aggregating results...")
    df = aggregate_results(run_eval=True)

    completed_df = get_completed_results(df)
    if len(completed_df) < 10:
        logger.warning(f"Only {len(completed_df)} completed experiments. Need more data for analysis.")
        return 1

    # 运行 ANOVA
    logger.info("\n2. Running Two-Way ANOVA...")
    anova_results = run_anova_by_dataset(completed_df, output_path=ANOVA_RESULTS_JSON)

    # 生成决策
    logger.info("\n3. Generating decision document...")
    checkpoint = CheckpointManager()
    generate_decision(
        anova_results,
        output_path=DECISION_MD,
        checkpoint_summary=checkpoint.get_summary(),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Results CSV: {RAW_RESULTS_CSV}")
    logger.info(f"ANOVA results: {ANOVA_RESULTS_JSON}")
    logger.info(f"Decision document: {DECISION_MD}")

    return 0


def cmd_plot(args):
    """生成可视化"""
    logger.info("=" * 60)
    logger.info("Phase 0: Generating Plots")
    logger.info("=" * 60)

    ensure_directories()

    # 加载数据
    if not RAW_RESULTS_CSV.exists():
        logger.error(f"Results file not found: {RAW_RESULTS_CSV}")
        logger.info("Run 'python main.py analyze' first.")
        return 1

    df = get_completed_results()

    if len(df) < 10:
        logger.warning(f"Only {len(df)} completed experiments. Plots may not be meaningful.")

    # 加载 ANOVA 结果
    anova_results = None
    if ANOVA_RESULTS_JSON.exists():
        import json
        with open(ANOVA_RESULTS_JSON) as f:
            anova_results = json.load(f)

    # 生成交互图
    logger.info("Creating interaction plot...")
    create_interaction_plot(df, output_path=INTERACTION_PLOT_PNG, anova_results=anova_results)

    logger.info(f"\nPlot saved to: {INTERACTION_PLOT_PNG}")

    return 0


def cmd_all(args):
    """运行完整流程"""
    logger.info("=" * 60)
    logger.info("Phase 0: Full Pipeline")
    logger.info("=" * 60)

    # 1. 运行实验
    logger.info("\n[Step 1/3] Running experiments...")
    ret = cmd_run(args)
    if ret != 0:
        return ret

    # 2. 分析结果
    logger.info("\n[Step 2/3] Analyzing results...")
    ret = cmd_analyze(args)
    if ret != 0:
        return ret

    # 3. 生成可视化
    logger.info("\n[Step 3/3] Generating plots...")
    ret = cmd_plot(args)
    if ret != 0:
        return ret

    logger.info("\n" + "=" * 60)
    logger.info("Phase 0 Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults directory: {RESULTS_DIR}")
    logger.info(f"Decision document: {DECISION_MD}")
    logger.info("\nPlease review the decision document for next steps.")

    return 0


def cmd_status(args):
    """查看状态"""
    checkpoint = CheckpointManager()
    summary = checkpoint.get_summary()

    print("\n" + "=" * 60)
    print("Phase 0 Experiment Status")
    print("=" * 60)
    print(f"Total experiments: {TOTAL_EXPERIMENTS}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Running: {summary['running']}")
    print(f"Pending: {summary['pending']}")
    print(f"\nStarted at: {summary['started_at'] or 'Not started'}")
    print(f"Last updated: {summary['last_updated'] or 'N/A'}")

    if summary['completed'] > 0:
        progress = summary['completed'] / TOTAL_EXPERIMENTS * 100
        print(f"\nProgress: {progress:.1f}%")

    # 检查结果文件
    print("\nOutput files:")
    for name, path in [
        ("Raw results CSV", RAW_RESULTS_CSV),
        ("ANOVA results", ANOVA_RESULTS_JSON),
        ("Interaction plot", INTERACTION_PLOT_PNG),
        ("Decision document", DECISION_MD),
    ]:
        status = "EXISTS" if path.exists() else "NOT FOUND"
        print(f"  {name}: {status}")

    return 0


def cmd_reset(args):
    """重置进度"""
    if not args.force:
        confirm = input("Are you sure you want to reset all progress? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    checkpoint = CheckpointManager()
    checkpoint.reset()
    print("Progress reset successfully.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 Hypothesis Validation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run 命令
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    run_parser.add_argument("--resume", action="store_true", default=True,
                           help="Resume from checkpoint (default: True)")
    run_parser.add_argument("--no-resume", dest="resume", action="store_false",
                           help="Start fresh, ignoring checkpoint")
    run_parser.add_argument("--dry-run", action="store_true",
                           help="Print commands without executing")
    run_parser.set_defaults(func=cmd_run)

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.set_defaults(func=cmd_analyze)

    # plot 命令
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.set_defaults(func=cmd_plot)

    # all 命令
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    all_parser.add_argument("--resume", action="store_true", default=True,
                           help="Resume from checkpoint")
    all_parser.add_argument("--dry-run", action="store_true",
                           help="Print commands without executing")
    all_parser.set_defaults(func=cmd_all)

    # status 命令
    status_parser = subparsers.add_parser("status", help="Show experiment status")
    status_parser.set_defaults(func=cmd_status)

    # reset 命令
    reset_parser = subparsers.add_parser("reset", help="Reset all progress")
    reset_parser.add_argument("--force", "-f", action="store_true",
                             help="Skip confirmation")
    reset_parser.set_defaults(func=cmd_reset)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # 确保目录存在
    ensure_directories()

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
