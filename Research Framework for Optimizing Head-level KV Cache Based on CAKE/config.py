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
    "model_name_or_path": "NousResearch/Llama-2-7b-hf",
    "precision": "fp16",  # or "bf16", "fp32"
    "datasets": ["mmlu", "gsm8k", "winogrande", "arc_challenge", "hellaswag", "truthful_qa_mc"],
    "dataset_subset_size": { # Number of samples to use for each dataset, null for all
        "mmlu": 100, # Using 100 for MMLU for quicker tests
        "gsm8k": 100,
        "winogrande": None, # Using all samples for others
        "arc_challenge": None,
        "hellaswag": None,
        "truthful_qa_mc": None,
        "pubmed_qa": 100, # Example for a potentially large dataset
        "cais/mmlu-zh": 50 # Example for a Chinese dataset
    },
    "kv_cache_lengths": [512, 1024, 2048], # Example KV cache lengths for baseline
    "batch_sizes": [1, 4, 8],           # Example batch sizes
    "max_new_tokens": 256,
    "repetitions": 3, # Number of times to repeat each experiment for statistical significance

    # H2O specific configurations
    "h2o_enabled": True, # Global switch for H2O experiments
    "h2o_ratios": [0.1, 0.2, 0.3], # Heavy-hitter ratios for H2O
    "eviction_strategies": ["attention", "time_decay", "hybrid"], # H2O eviction strategies
    "h2o_kv_cache_lengths": [512, 1024], # KV cache lengths specifically for H2O tests

    # CAKE specific configurations
    "cake_enabled": True,
    "layer_allocation_strategies": ["uniform", "adaptive", "attention_based"],
    "layer_analysis_configs": {
        "attention_pattern_analysis": True,
        "layer_importance_scoring": True,
        "dynamic_allocation": True
    },
    "cache_budgets": [0.5, 0.7, 0.9],  # Total cache budget as a proportion of original
    "cake_kv_cache_lengths": [512, 1024], # KV cache lengths specifically for CAKE tests


    # Head-level optimization configurations (for future extension)
    "head_level_optimization": False,
    "head_analysis_enabled": False,
    "head_selection_strategy": "top_k", # e.g., top_k, threshold-based
    "head_k_value": 10, # Number of heads to keep if strategy is top_k
    
    "output_base_dir": "results",
    "enable_monitoring": True,
    "monitor_interval": 0.5 # seconds
}

# 数据集配置
DATASET_CONFIG = {
    "english": {
        "mmlu": {"path": "cais/mmlu", "subset": "all", "description": "多领域推理任务"},
        "hellaswag": {"path": "hellaswag", "subset": None, "description": "常识推理任务"},
        "longbench": {"path": "THUDM/longbench", "subset": "default", "description": "长文本理解基准"},
        "squad2": {"path": "squad_v2", "subset": None, "description": "阅读理解问答任务"},
        "max_new_tokens": EXPERIMENT_CONFIG["max_new_tokens"], # Default from main config
        "generate_config": {
            "max_new_tokens": EXPERIMENT_CONFIG["max_new_tokens"], # Ensure this is passed for generation
            "do_sample": False, # For reproducibility and focused performance testing
            "temperature": 0.7, # Only relevant if do_sample=True
            "top_p": 0.9,       # Only relevant if do_sample=True
        }
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
    "significance_level": 0.05,
    "output_dir": f"{EXPERIMENT_CONFIG['output_base_dir']}/analysis_reports",
    "baseline_group_name": "Baseline",
    "h2o_group_name": "H2O",
    "cake_group_name": "CAKE", # NEW
    "comparison_metrics": ["TTFT_ms", "TPOT_ms", "Throughput_tok_sec", "GPU_Memory_Peak_MB"],
    "detailed_analysis_params_h2o": ["heavy_ratio", "recent_ratio", "eviction_strategy"],
    "detailed_analysis_params_cake": ["allocation_strategy", "cache_budget"], # NEW
    "default_method_params": { # NEW section for easier management
        "h2o": ["heavy_ratio", "recent_ratio", "eviction_strategy"],
        "cake": ["allocation_strategy", "cache_budget"]
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

# Configuration for H2O specific model preparation
H2O_MODEL_CONFIG = {
    "heavy_ratio_default": 0.1,
    "recent_ratio_default": 0.1,
    "supported_models_h2o": ["llama", "mistral", "falcon", "gptneox", "phi"], # Models confirmed to work with H2O conversion
    "kv_cache_parts": 2 # Default for H2O, e.g., heavy and recent parts
}

# Configuration for CAKE specific model preparation (NEW)
CAKE_MODEL_CONFIG = {
    "default_allocation_strategy": "adaptive",
    "default_cache_budget": 0.8,
    "supported_models_cake": ["llama", "mistral", "falcon", "gptneox", "phi"], # Models to be tested with CAKE
    "dynamic_allocation_default": True,
    "layer_analysis_default": {
         "attention_pattern_analysis": True,
         "layer_importance_scoring": True
    }
} 