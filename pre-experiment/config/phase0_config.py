"""
Phase 0 假设验证实验配置

核心假设: "最优KV cache驱逐策略随压缩率变化"
实验设计: 3策略 × 3压缩率 × 2数据集 × 3重复 = 54次实验
"""
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

# =============================================================================
# 实验因子设计
# =============================================================================

# 策略 (head_mode): 头级别预算分配方式
STRATEGIES = ["high_entropy", "low_entropy", "uniform"]

# 压缩率 (cache_size): KV cache 总预算 (tokens)
CACHE_SIZES = [128, 512, 2048]

# 数据集: 摘要任务 + QA任务
# Stage B: 仅 qmsum; Stage C: 添加 hotpotqa
DATASETS = ["qmsum"]

# 随机种子: 3次重复
SEEDS = [42, 123, 456]

# 总实验数
TOTAL_EXPERIMENTS = len(STRATEGIES) * len(CACHE_SIZES) * len(DATASETS) * len(SEEDS)  # 54

# =============================================================================
# 控制变量 (固定)
# =============================================================================

MODEL = "qwen2.5-7b-instruct"
WINDOW_SIZE = 32  # 滑动窗口大小
PREF_MODE = "normal"  # 层级偏好模式 (固定为 normal)

# 数据集特定配置
DATASET_CONFIG = {
    "qmsum": {
        "metric": "rouge",
        "metric_key": "rouge-l",  # 使用 ROUGE-L
        "max_new_tokens": 512,
        "sample_count": 200,
    },
    "hotpotqa": {
        "metric": "f1",
        "metric_key": "f1",
        "max_new_tokens": 128,
        "sample_count": 200,
    },
}

# =============================================================================
# 统计判定标准
# =============================================================================

SIGNIFICANCE_LEVEL = 0.05  # p 值阈值
MIN_EFFECT_SIZE = 0.02  # 最小有意义效应量 (2%)

# 效应量解释 (Cohen's guidelines for eta-squared)
EFFECT_SIZE_THRESHOLDS = {
    "small": 0.01,
    "medium": 0.06,
    "large": 0.14,
}

# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# pre-experiment 目录
PRE_EXPERIMENT_DIR = PROJECT_ROOT / "pre-experiment"

# 结果目录
RESULTS_DIR = PRE_EXPERIMENT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"

# 输出文件
CHECKPOINT_FILE = CHECKPOINTS_DIR / "progress.json"
RAW_RESULTS_CSV = RESULTS_DIR / "raw_results.csv"
ANOVA_RESULTS_JSON = RESULTS_DIR / "anova_results.json"
INTERACTION_PLOT_PNG = RESULTS_DIR / "interaction_plot.png"
DECISION_MD = RESULTS_DIR / "decision.md"

# 依赖的外部脚本
HACE_RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_hace_qmsum.py"
EVAL_SCRIPT = PROJECT_ROOT / "src" / "third_party" / "cakekv-main" / "cakekv-main" / "experiments" / "LongBench" / "eval.py"

# =============================================================================
# 辅助函数
# =============================================================================

def get_experiment_id(strategy: str, cache_size: int, dataset: str, seed: int) -> str:
    """生成唯一实验标识符"""
    return f"{strategy}_{cache_size}_{dataset}_{seed}"


def get_pred_name(strategy: str, cache_size: int, dataset: str, seed: int) -> str:
    """生成预测输出目录名"""
    return f"phase0_{strategy}_{cache_size}_{dataset}_s{seed}"


def generate_experiment_grid() -> List[Dict[str, Any]]:
    """
    生成完整的实验配置网格

    Returns:
        List of 54 experiment configurations
    """
    grid = []
    for strategy, cache_size, dataset, seed in product(STRATEGIES, CACHE_SIZES, DATASETS, SEEDS):
        grid.append({
            "strategy": strategy,
            "cache_size": cache_size,
            "dataset": dataset,
            "seed": seed,
            "experiment_id": get_experiment_id(strategy, cache_size, dataset, seed),
            "pred_name": get_pred_name(strategy, cache_size, dataset, seed),
        })
    return grid


def get_head_mode(strategy: str) -> str:
    """
    将策略名映射到 head_mode 参数

    uniform 策略对应空字符串 (禁用头级别差异化)
    """
    if strategy == "uniform":
        return ""
    return strategy


def ensure_directories():
    """确保所有必要的目录存在"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 验证配置
# =============================================================================

def validate_config():
    """验证配置的正确性"""
    errors = []

    # 检查脚本是否存在
    if not HACE_RUNNER_SCRIPT.exists():
        errors.append(f"HACE runner script not found: {HACE_RUNNER_SCRIPT}")

    if not EVAL_SCRIPT.exists():
        errors.append(f"Eval script not found: {EVAL_SCRIPT}")

    # 检查数据集配置
    for dataset in DATASETS:
        if dataset not in DATASET_CONFIG:
            errors.append(f"Missing config for dataset: {dataset}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True


if __name__ == "__main__":
    # 验证配置
    validate_config()

    # 打印实验网格摘要
    grid = generate_experiment_grid()
    print(f"Total experiments: {len(grid)}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Cache sizes: {CACHE_SIZES}")
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    print(f"\nFirst 5 experiments:")
    for exp in grid[:5]:
        print(f"  {exp['experiment_id']}")
