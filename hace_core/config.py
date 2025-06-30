"""
配置文件，存储实验的所有参数设置
"""
import os
from pathlib import Path

# 模型路径配置 - 支持环境变量和多种配置方式
def get_model_path(model_name="mistral-7b-instruct-v0.3"):
    """
    获取模型路径，优先级：
    1. 环境变量 HACE_MODEL_PATH
    2. 项目本地 models 目录
    3. 用户主目录下的 models 目录
    4. 对于HuggingFace格式的模型名（包含/），直接返回原始路径
    如果以上都不存在，将抛出错误提示用户配置
    """
    
    def _is_huggingface_hub_path(path):
        """
        判断是否为HuggingFace Hub路径
        HuggingFace Hub路径特征：
        - 包含 '/' 
        - 不以 './' 或 '../' 开头（本地相对路径）
        - 不以 '/' 开头（绝对路径）
        - 不包含常见的本地路径指示符如 ':' (Windows驱动器)
        """
        if not isinstance(path, str) or '/' not in path:
            return False
        
        # 排除本地路径格式
        if (path.startswith('./') or 
            path.startswith('../') or 
            path.startswith('/') or
            ':' in path):  # Windows驱动器路径如C:/path
            return False
            
        # HuggingFace Hub路径通常是 organization/model 格式
        # 进一步验证：应该只有一个'/'，且不包含文件扩展名
        parts = path.split('/')
        if len(parts) == 2 and all(part.strip() for part in parts):
            # 检查是否不包含常见的文件扩展名
            if not any(part.endswith(ext) for part in parts 
                      for ext in ['.bin', '.safetensors', '.json', '.txt', '.py']):
                return True
        
        return False
    
    # 优先级1：检查环境变量 HACE_MODEL_PATH
    if "HACE_MODEL_PATH" in os.environ:
        model_path = os.environ["HACE_MODEL_PATH"]
        # 如果环境变量指定的是HuggingFace格式路径，直接返回，不进行本地存在性检查
        if _is_huggingface_hub_path(model_path):
            return model_path
        # 对于本地路径，检查是否存在
        if Path(model_path).exists():
            return model_path
        else:
            raise FileNotFoundError(f"环境变量指定的模型路径不存在: {model_path}")
    
    # 优先级2：如果输入的model_name本身是HuggingFace Hub路径，直接返回
    if _is_huggingface_hub_path(model_name):
        return model_name
    
    # 优先级3：检查项目本地models目录
    local_model_path = Path("./models") / model_name
    if local_model_path.exists():
        return str(local_model_path)
    
    # 优先级4：检查用户主目录的models目录
    try:
        home_model_path = Path.home() / "models" / model_name
        if home_model_path.exists():
            return str(home_model_path)
    except (OSError, RuntimeError):
        # 无法确定用户主目录时跳过这个检查
        pass
    
    # 如果都不存在，抛出错误提示用户配置
    raise FileNotFoundError(
        f"找不到模型 '{model_name}'，请使用以下方式之一配置模型路径：\n"
        f"1. 设置环境变量: HACE_MODEL_PATH=你的模型路径\n"
        f"2. 将模型放入: ./models/{model_name}/\n"
        f"3. 将模型放入: ~/models/{model_name}/\n"
        f"运行 'python check_model_config.py' 获取详细配置指南"
    )

def get_resolved_model_path(config_dict):
    """
    从配置字典中获取解析后的模型路径
    
    Args:
        config_dict: 包含model_name_or_path键的配置字典
        
    Returns:
        str: 解析后的模型路径
    """
    model_name = config_dict.get("model_name_or_path")
    if not model_name:
        raise ValueError("配置中未找到model_name_or_path字段")
    
    return get_model_path(model_name)

# 硬件配置信息
HARDWARE_CONFIG = {
    "gpu": "NVIDIA RTX 4090",
    "cpu": "Intel Core i7-13700KF",
    "memory": "64GB DDR5-6000MHz"
}

# 模型配置
MODEL_CONFIG = {
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",  # 动态解析的模型名称，支持HuggingFace Hub或本地路径
    "precision": "fp16",  # 或 "bf16", "int8" 等
    "device": "cuda"
}

# 实验配置
EXPERIMENT_CONFIG = {
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",  # 动态解析的模型名称，支持HuggingFace Hub或本地路径
    "precision": "fp16",  # or "bf16", "fp32"
    "use_relative_paths": True,        # 新增：强制使用相对路径
    "auto_create_dirs": True,          # 新增：自动创建目录
    "baseline_search_patterns": [      # 新增：基线文件搜索模式
        "./**/evaluation_results_*.json",
        "./fullkvcache_run_*/ds_*/evaluation_results_*.json",
        "./results/**/evaluation_results_*.json"
    ],
    # 多模型实验配置
    "multi_model_experiments": True,
    "experiment_models": [
        "meta-llama/Llama-2-7b-chat",           # 默认主模型
        "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Llama 3.1 8B
        "mistralai/Mistral-7B-Instruct-v0.3"      # Mistral 7B v0.3
    ],
    
    # 显存管理配置
    "memory_management": {
        "auto_max_memory": True,  # 是否自动设置max_memory参数
        "manual_max_memory": {    # 手动设置显存上限 (仅在auto_max_memory=False时使用)
            0: "50000MB",  # GPU 0的显存上限 - 临时设置为50GB用于Linux分支
            # 1: "50000MB",  # GPU 1的显存上限 (多GPU时)
        },
        "memory_buffer_ratio": 0.05,  # 显存缓冲比例 (5%缓冲)
        "force_no_cpu_offload": True   # 强制避免CPU offload
    },
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
    "kv_cache_lengths": [128, 256, 512, 1024, 2048], # Complete KV cache lengths for baseline
    "batch_sizes": [1, 4, 8],           # Example batch sizes
    "max_new_tokens": 256,
    "repetitions": 3, # Number of times to repeat each experiment for statistical significance

    # H2O specific configurations
    "h2o_enabled": True, # Global switch for H2O experiments
    "h2o_ratios": [0.1, 0.2, 0.3], # Heavy-hitter ratios for H2O
    "eviction_strategies": ["attention", "time_decay", "hybrid"], # H2O eviction strategies
    "h2o_kv_cache_lengths": [128, 256, 512, 1024, 2048], # KV cache lengths specifically for H2O tests

    # CAKE specific configurations
    "cake_enabled": True,
    "layer_allocation_strategies": ["uniform", "adaptive", "attention_based"],
    "layer_analysis_configs": {
        "attention_pattern_analysis": True,
        "layer_importance_scoring": True,
        "dynamic_allocation": True
    },
    "cache_budgets": [0.5, 0.7, 0.9],  # Total cache budget as a proportion of original
    "cake_kv_cache_lengths": [128, 256, 512, 1024, 2048], # KV cache lengths specifically for CAKE tests


    # Head-level optimization configurations (for future extension)
    "head_level_optimization": False,
    "head_analysis_enabled": False,
    "head_selection_strategy": "top_k", # e.g., top_k, threshold-based
    "head_k_value": 10, # Number of heads to keep if strategy is top_k
    
    "output_base_dir": "./results",
    "enable_monitoring": True,
    "monitor_interval": 0.5 # seconds
}

# 数据集配置
DATASET_CONFIG = {
    # 添加available_datasets键
    "available_datasets": {
        # 英文数据集
        "mmlu": {"path": "cais/mmlu", "subset": "all", "description": "多领域推理任务"},
        "hellaswag": {"path": "hellaswag", "subset": None, "description": "常识推理任务"},
        "longbench": {"path": "THUDM/longbench", "subset": "default", "description": "长文本理解基准"},
        "squad2": {"path": "squad_v2", "subset": None, "description": "阅读理解问答任务"},
        "gsm8k": {"path": "gsm8k", "subset": "main", "description": "数学推理任务"},
        "winogrande": {"path": "winogrande", "subset": "winogrande_xl", "description": "常识推理任务"},
        "arc_challenge": {"path": "ai2_arc", "subset": "ARC-Challenge", "description": "科学推理任务"},
        "truthful_qa_mc": {"path": "truthful_qa", "subset": "multiple_choice", "description": "真实性问答任务"},
        
        # LongBench 单文档问答
        "narrativeqa": {"path": "THUDM/longbench", "subset": "narrativeqa", "description": "基于小说和电影剧本的问答"},
        "qasper": {"path": "THUDM/longbench", "subset": "qasper", "description": "基于科学论文的问答"},
        "multifieldqa_en": {"path": "THUDM/longbench", "subset": "multifieldqa_en", "description": "多领域英文问答"},
        
        # LongBench 多文档问答
        "hotpotqa": {"path": "THUDM/LongBench", "subset": "hotpotqa", "description": "多跳推理问答"},
        "2wikimqa": {"path": "THUDM/longbench", "subset": "2wikimqa", "description": "基于维基百科的多文档问答"},
        "musique": {"path": "THUDM/longbench", "subset": "musique", "description": "多步骤推理问答"},
        
        # LongBench 摘要生成
        "gov_report": {"path": "THUDM/longbench", "subset": "gov_report", "description": "政府报告摘要"},
        "qmsum": {"path": "THUDM/longbench", "subset": "qmsum", "description": "会议摘要"},
        "multi_news": {"path": "THUDM/LongBench", "subset": "multi_news", "description": "多新闻摘要"},
        
        # LongBench 少样本学习
        "trec": {"path": "THUDM/longbench", "subset": "trec", "description": "问题分类"},
        "triviaqa": {"path": "THUDM/longbench", "subset": "triviaqa", "description": "知识问答"},
        "samsum": {"path": "THUDM/longbench", "subset": "samsum", "description": "对话摘要"},
        
        # LongBench 合成任务
        "passage_count": {"path": "THUDM/longbench", "subset": "passage_count", "description": "段落计数"},
        "passage_retrieval_en": {"path": "THUDM/longbench", "subset": "passage_retrieval_en", "description": "段落检索"},
        
        # LongBench 代码任务
        "lcc": {"path": "THUDM/longbench", "subset": "lcc", "description": "代码补全"},
        "repobench-p": {"path": "THUDM/longbench", "subset": "repobench-p", "description": "代码库级别任务"},
        
        # NeedleBench 任务
        "single_needle": {"path": "custom/needlebench", "subset": "single_needle", "description": "单针检索任务"},
        "multi_needle_retrieval": {"path": "custom/needlebench", "subset": "multi_needle_retrieval", "description": "多针检索任务"},
        "multi_needle_reasoning": {"path": "custom/needlebench", "subset": "multi_needle_reasoning", "description": "多针推理任务"},
        
        # 中文数据集
        "cluewsc2020": {"path": "cluewsc2020", "subset": None, "description": "指代消解任务"},
        "ceval": {"path": "ceval", "subset": "all", "description": "中文多领域评测基准"},
        "race": {"path": "race", "subset": "all", "description": "中文阅读理解任务"},
        "openeval": {"path": "openeval", "subset": None, "description": "中文全面评估基准"}
    },
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
    "supported_models_cake": ["llama", "mistral", "qwen2"], # 仅包含CAKE核心代码实际支持的模型
    "dynamic_allocation_default": True,
    "layer_analysis_default": {
        "attention_pattern_analysis": True,
        "layer_importance_scoring": True
    }
}

"""
使用示例：
--------

# 在下游代码中使用时，需要解析实际路径：
from hace_core.config import MODEL_CONFIG, get_resolved_model_path

# 错误的用法（会收到模型名称而不是解析后的路径）：
model_path = MODEL_CONFIG["model_name_or_path"]  # 'mistralai/Mistral-7B-Instruct-v0.3'

# 正确的用法（会收到解析后的完整路径）：
model_path = get_resolved_model_path(MODEL_CONFIG)  # 实际路径，如本地路径或HuggingFace Hub路径

# 对于多模型实验：
from hace_core.config import EXPERIMENT_CONFIG, get_model_path

for model_name in EXPERIMENT_CONFIG["experiment_models"]:
    resolved_path = get_model_path(model_name)
    # 使用 resolved_path 进行模型加载

# 模型路径解析优先级：
# 1. 环境变量 HACE_MODEL_PATH（如果设置）
# 2. HuggingFace Hub路径（如果是 organization/model 格式）
# 3. 项目本地 ./models/模型名/ 目录
# 4. 用户主目录 ~/models/模型名/ 目录
# 
# 示例配置方式：
# - 环境变量：export HACE_MODEL_PATH="C:/Users/Administrator/mistral_models/7B-Instruct-v0.3"
# - 本地目录：./models/mistralai--Mistral-7B-Instruct-v0.3/
# - HuggingFace：直接使用 "mistralai/Mistral-7B-Instruct-v0.3"
""" 