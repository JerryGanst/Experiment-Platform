"""
实验数据分析模块

专门用于分析基线和优化方法的实验结果：

- statistical_tests: 统计分析器（t检验、ANOVA、回归分析等）
- tables: 性能对比表格生成工具
- generate_report: HTML/Excel报告生成器

与基线模块紧密配合，提供完整的实验分析流程。
注意：图表生成功能位于hace_core.utils.plotter模块。
"""

from .statistical_tests import StatisticsAnalyzer
from .tables import (
    load_experiment_results,
    create_performance_summary_table,
    create_language_comparison_table,
    create_best_config_table
)
from .generate_report import generate_html_report

__all__ = [
    'StatisticsAnalyzer',
    'load_experiment_results',
    'create_performance_summary_table', 
    'create_language_comparison_table',
    'create_best_config_table',
    'generate_html_report'
]
