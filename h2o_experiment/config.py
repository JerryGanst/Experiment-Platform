"""
配置文件，存储实验的所有参数设置
"""

# 硬件配置信息
HARDWARE_CONFIG = {
    "gpu": "NVIDIA RTX 4070 SUPER",
    "cpu": "Intel Core i7-13700KF",
    "memory": "64GB DDR5-6000MHz"
}

# 模型配置
MODEL_CONFIG = {
    "model_name_or_path": "facebook/opt-1.3b",  # 可以根据实际使用的模型更改
    "precision": "fp16",  # 或 "bf16", "int8" 等
    "device": "cuda"
}

# 实验配置
EXPERIMENT_CONFIG = {
    "kv_cache_lengths": [128, 256, 512, 1024, 2048],  # KV缓存长度
    "batch_sizes": [1, 4, 8, 16],  # 批处理大小
    "repeat_times": 5,  # 每个配置重复实验的次数
    "random_seed": 42,  # 随机种子
    "max_new_tokens": 128,  # 每次生成的最大令牌数
    
    # H2O配置参数
    "h2o_ratios": [0.1, 0.2, 0.3],  # heavy_ratio和recent_ratio的可能值
    "eviction_strategies": ["attention", "time_decay", "hybrid"],  # 缓存驱逐策略
    "attention_thresholds": [0.005, 0.01, 0.02, 0.05]  # 注意力分数阈值
}

# 数据集配置
DATASET_CONFIG = {
    "english": {
        "mmlu": {"path": "cais/mmlu", "subset": "all", "description": "多领域推理任务"},
        "hellaswag": {"path": "hellaswag", "subset": None, "description": "常识推理任务"},
        "longbench": {"path": "THUDM/longbench", "subset": "default", "description": "长文本理解基准"},
        "squad2": {"path": "squad_v2", "subset": None, "description": "阅读理解问答任务"}
    },
    "chinese": {
        "cluewsc2020": {"path": "cluewsc2020", "subset": None, "description": "指代消解任务"},
        "ceval": {"path": "ceval", "subset": "all", "description": "中文多领域评测基准"},
        "race": {"path": "race", "subset": "all", "description": "中文阅读理解任务"},
        "openeval": {"path": "openeval", "subset": None, "description": "中文全面评估基准"}
    }
}

# 监控配置
MONITORING_CONFIG = {
    # GPU监控
    "gpu_monitoring": True,  # 是否启用GPU监控
    "gpu_monitoring_interval": 0.1,  # GPU监控间隔(秒)
    
    # 系统监控
    "system_monitoring": True,  # 是否启用系统状态监控
    "system_monitoring_interval": 1.0,  # 系统监控间隔(秒)
    "memory_monitoring": True,  # 是否监控内存使用
    "cpu_monitoring": True,  # 是否监控CPU使用率
    
    # 温度监控
    "temperature_monitoring": False,  # 是否启用温度监控
    "temperature_monitoring_interval": 2.0,  # 温度监控间隔(秒)
    
    # 一般设置
    "logging_level": "INFO",  # 日志级别
    "auto_save": True,  # 是否自动保存监控数据
    "monitoring_dir": "./monitoring"  # 监控数据保存目录
}

# 输出配置
OUTPUT_CONFIG = {
    "results_dir": "./results",  # 结果目录
    "logs_dir": "./logs",  # 日志目录
    "visualizations_dir": "./visualizations",  # 可视化输出目录
    "save_model_outputs": True,  # 是否保存模型输出
    "cache_stats_dir": "./cache_stats",  # 缓存统计数据目录
    "quality_metrics_dir": "./quality_metrics"  # 质量指标目录
}

# 统计分析配置
STATISTICS_CONFIG = {
    "significance_level": 0.05,  # 统计显著性水平
    "outlier_detection": {
        "method": "zscore",  # 异常值检测方法
        "threshold": 3.0  # 异常值检测阈值
    },
    "cross_validation": {
        "enabled": True,  # 是否启用交叉验证
        "n_splits": 5  # 交叉验证分割数
    }
}

# 实验验证配置
VALIDATION_CONFIG = {
    "control_experiments": True,  # 是否进行对照实验
    "ablation_studies": {  # 消融实验配置
        "only_heavy_hitter": True,  # 仅使用Heavy Hitter
        "only_recent": True,  # 仅使用Recent
        "different_ratios": True  # 不同比例组合
    }
}

# 质量评估配置
QUALITY_CONFIG = {
    "perplexity": True,  # 是否计算困惑度
    "bleu_score": True,  # 是否计算BLEU分数
    "rouge_score": True,  # 是否计算ROUGE分数
    "accuracy": True,  # 是否计算准确率
    "consistency": {  # 一致性评估配置
        "enabled": True,
        "num_runs": 3  # 评估一致性的运行次数
    }
} 