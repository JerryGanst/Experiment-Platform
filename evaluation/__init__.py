"""
统一评估模块

整合了基线执行、数据分析和实验管理功能：

- baselines/: 基线方法的执行和测试
- analysis/: 数据分析、统计测试和报告生成  
- experiments/: 实验管理和对比分析
- monitoring/: 监控和可视化（位于src/monitoring/）

这个模块提供了完整的实验评估工作流：
1. 执行基线实验 (baselines)
2. 运行优化方法实验 (通过experiments调用src/methods)  
3. 分析和对比结果 (analysis)
4. 生成报告和可视化 (analysis + monitoring)
"""

# 避免在导入 evaluation 时递归导入 baselines，导致缺失模块报错。
# 需要时请在调用处显式导入对应子模块。
__version__ = "1.0.0"
__all__ = []
