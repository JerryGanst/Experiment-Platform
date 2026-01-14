"""
效应量计算模块

计算 eta-squared, partial eta-squared, Cohen's d 等效应量指标
"""
import numpy as np
from typing import Union, List, Tuple


def calculate_eta_squared(ss_effect: float, ss_total: float) -> float:
    """
    计算 eta-squared (η²)

    η² = SS_effect / SS_total

    表示效应解释的总方差比例

    Args:
        ss_effect: 效应的平方和
        ss_total: 总平方和

    Returns:
        eta-squared 值 (0-1)
    """
    if ss_total <= 0:
        return 0.0
    return ss_effect / ss_total


def calculate_partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    """
    计算 partial eta-squared (η²_p)

    η²_p = SS_effect / (SS_effect + SS_error)

    更常用于 ANOVA，表示效应在移除其他效应后的方差解释比例

    Args:
        ss_effect: 效应的平方和
        ss_error: 误差平方和

    Returns:
        partial eta-squared 值 (0-1)
    """
    denominator = ss_effect + ss_error
    if denominator <= 0:
        return 0.0
    return ss_effect / denominator


def calculate_omega_squared(ss_effect: float, df_effect: int, ms_error: float, ss_total: float) -> float:
    """
    计算 omega-squared (ω²)

    ω² 是 η² 的无偏估计，更适合小样本

    ω² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)

    Args:
        ss_effect: 效应的平方和
        df_effect: 效应的自由度
        ms_error: 误差均方
        ss_total: 总平方和

    Returns:
        omega-squared 值 (可能为负，表示效应可忽略)
    """
    numerator = ss_effect - df_effect * ms_error
    denominator = ss_total + ms_error

    if denominator <= 0:
        return 0.0

    omega_sq = numerator / denominator
    return max(0.0, omega_sq)  # omega² 可以为负，但通常截断为 0


def calculate_cohens_d(group1: Union[np.ndarray, List], group2: Union[np.ndarray, List]) -> float:
    """
    计算 Cohen's d

    d = (M1 - M2) / SD_pooled

    用于两组比较的效应量

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        Cohen's d 值
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_effect_size(value: float, measure: str = "eta_squared") -> str:
    """
    解释效应量大小

    Args:
        value: 效应量值
        measure: 效应量类型 ("eta_squared", "cohens_d")

    Returns:
        解释文本 ("negligible", "small", "medium", "large")
    """
    if measure in ["eta_squared", "partial_eta_squared", "omega_squared"]:
        # Cohen's guidelines for eta-squared
        if value < 0.01:
            return "negligible"
        elif value < 0.06:
            return "small"
        elif value < 0.14:
            return "medium"
        else:
            return "large"

    elif measure == "cohens_d":
        # Cohen's guidelines for d
        abs_value = abs(value)
        if abs_value < 0.2:
            return "negligible"
        elif abs_value < 0.5:
            return "small"
        elif abs_value < 0.8:
            return "medium"
        else:
            return "large"

    return "unknown"


def calculate_effect_size_confidence_interval(
    effect_size: float,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    计算效应量的置信区间 (近似)

    使用 Fisher 变换的近似方法

    Args:
        effect_size: eta-squared 值
        n: 样本量
        confidence: 置信水平

    Returns:
        (lower_bound, upper_bound)
    """
    from scipy import stats

    # 近似标准误
    se = np.sqrt((4 * effect_size * (1 - effect_size) ** 2) / (n - 1))

    z = stats.norm.ppf((1 + confidence) / 2)

    lower = max(0, effect_size - z * se)
    upper = min(1, effect_size + z * se)

    return (lower, upper)


if __name__ == "__main__":
    # 测试
    print("Effect size interpretation:")
    for value in [0.005, 0.03, 0.08, 0.20]:
        print(f"  η² = {value}: {interpret_effect_size(value)}")

    print("\nCohen's d interpretation:")
    for value in [0.1, 0.3, 0.6, 1.0]:
        print(f"  d = {value}: {interpret_effect_size(value, 'cohens_d')}")

    # 测试 eta-squared 计算
    ss_effect = 100
    ss_total = 500
    print(f"\nη² = {calculate_eta_squared(ss_effect, ss_total):.3f}")

    # 测试 Cohen's d
    group1 = [1, 2, 3, 4, 5]
    group2 = [2, 3, 4, 5, 6]
    print(f"Cohen's d = {calculate_cohens_d(group1, group2):.3f}")
