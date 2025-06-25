"""
实验管理模块

负责协调基线和优化方法的实验执行：

- run_experiment: 主实验执行器，支持多种方法对比
- run_comparison: 实验结果对比分析器
- enhanced_batch_runner: 批量实验执行器
- debug_config: 配置诊断工具

这个模块连接了基线执行、优化方法和结果分析。
"""

__all__ = [
    'run_experiment',
    'run_comparison',
    'enhanced_batch_runner', 
    'debug_config'
]
