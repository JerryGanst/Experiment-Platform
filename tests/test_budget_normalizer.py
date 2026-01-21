import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 从正确路径导入 BudgetNormalizer
from src.hace.core.indicator_normalizer import BudgetNormalizer

# 测试用例: (raw_budgets, total_budget, min_budget)
TEST_CASES = [
    ([10, 10, 1], 15, 1),          # 基本缩减 + 回收
    ([2, 2], 2, 1),                # 无须调整
    ([3, 3, 3], 5, 1),            # 超额回收, 需多轮
    ([10, 1, 1], 5, 1),           # 原缺陷场景: excess >= len(candidates)
]

for raw, total, min_b in TEST_CASES:
    try:
        res = BudgetNormalizer.normalize_to_budget(raw, total, min_b)
        assert sum(res) == total, f"预算不守恒: {res} (sum={sum(res)}) ≠ {total}"
        assert all(x >= min_b for x in res), f"违反最小预算: {res}"
        print(f"✅ 输入 {raw}, total={total}, min={min_b} -> 输出 {res}")
    except Exception as e:
        print(f"❌ 输入 {raw}, total={total}, min={min_b} 触发异常: {e}") 