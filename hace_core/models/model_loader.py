"""
模型加载与配置模块
"""
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(config):
    """
    加载模型和分词器
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    logger.info(f"Loading model: {config['model_name_or_path']}")
    
    # 确定模型精度
    torch_dtype = torch.float16 if config["precision"] == "fp16" else torch.float32
    if config["precision"] == "bf16" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    
    # 显存管理配置
    memory_mgmt = config.get("memory_management", {})
    auto_max_memory = memory_mgmt.get("auto_max_memory", True)
    buffer_ratio = memory_mgmt.get("memory_buffer_ratio", 0.05)
    manual_max_memory = memory_mgmt.get("manual_max_memory", {})
    
    # 智能显存分配：检测GPU显存并设置max_memory
    max_memory_config = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        max_memory_config = {}
        
        if auto_max_memory:
            # 自动检测并配置显存
            for i in range(gpu_count):
                # 获取每个GPU的总显存（以MB为单位）
                total_memory = torch.cuda.get_device_properties(i).total_memory
                total_memory_mb = int(total_memory / (1024 * 1024))
                
                # 计算可用显存（总显存 - 缓冲区）
                available_ratio = 1.0 - buffer_ratio
                max_memory_mb = int(total_memory_mb * available_ratio)
                
                # 为不同大小的GPU设置更保守的上限
                if total_memory_mb >= 24000:  # 24GB+ GPU (如RTX 4090, A100等)
                    max_memory_mb = min(23000, max_memory_mb)
                elif total_memory_mb >= 16000:  # 16GB+ GPU (如RTX 4080等)
                    max_memory_mb = min(15000, max_memory_mb)
                elif total_memory_mb >= 12000:  # 12GB+ GPU (如RTX 4070Ti等)
                    max_memory_mb = min(11000, max_memory_mb)
                elif total_memory_mb >= 8000:   # 8GB+ GPU (如RTX 4070等)
                    max_memory_mb = min(7000, max_memory_mb)
                
                max_memory_config[i] = f"{max_memory_mb}MB"
                logger.info(f"GPU {i}: 总显存 {total_memory_mb}MB, 自动设置上限 {max_memory_mb}MB (缓冲比例: {buffer_ratio:.1%})")
        else:
            # 使用手动配置的显存限制
            max_memory_config = manual_max_memory.copy()
            logger.info(f"使用手动配置的显存上限: {max_memory_config}")
    
    # 加载模型
    try:
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        # 如果有可用GPU且配置了max_memory，添加该参数
        if max_memory_config:
            model_kwargs["max_memory"] = max_memory_config
            logger.info(f"使用精细显存控制: {max_memory_config}")
        
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            **model_kwargs
        )
        logger.info(f"Model loaded successfully with dtype: {torch_dtype}")
        
        # 显示实际的显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
                logger.info(f"GPU {i} 显存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
                
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name_or_path"],
            use_fast=True
        )
        # 确保分词器有正确的填充令牌设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise
    
    return model, tokenizer

def configure_model_for_kv_cache_length(model, kv_cache_length):
    """
    配置模型以使用特定的KV缓存长度
    
    Args:
        model: 要配置的模型
        kv_cache_length: KV缓存的最大长度
        
    Returns:
        model: 配置后的模型
    """
    logger.info(f"Configuring model for KV cache length: {kv_cache_length}")
    
    try:
        # 设置最大长度属性
        if hasattr(model.config, "max_position_embeddings"):
            original_max_length = model.config.max_position_embeddings
            model.config.max_position_embeddings = max(original_max_length, kv_cache_length)
            logger.info(f"Updated max_position_embeddings from {original_max_length} to {model.config.max_position_embeddings}")
        
        # 处理特定模型架构
        model_type = model.config.model_type
        
        # OPT模型
        if model_type == "opt":
            # 确保注意力掩码可以处理更长的序列长度
            if hasattr(model, "model") and hasattr(model.model, "decoder"):
                for layer in model.model.decoder.layers:
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "max_position_embeddings"):
                        layer.self_attn.max_position_embeddings = max(layer.self_attn.max_position_embeddings, kv_cache_length)
        
        # LLaMA模型
        elif model_type == "llama":
            # 确保RoPE嵌入可以处理更长的序列长度
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                for layer in model.model.layers:
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                        layer.self_attn.rotary_emb.max_position_embeddings = max(
                            layer.self_attn.rotary_emb.max_position_embeddings, 
                            kv_cache_length
                        )
        
        logger.info(f"Model {model_type} configured successfully for KV cache length: {kv_cache_length}")
    except Exception as e:
        logger.warning(f"Could not fully configure KV cache length: {e}")
    
    return model

def prepare_model_for_baseline(model):
    """
    准备用于基线测试的模型，确保使用默认KV缓存机制
    
    Args:
        model: 要准备的模型
        
    Returns:
        model: 准备好的模型
    """
    logger.info("Preparing model for baseline testing with default KV cache")
    
    # 确保模型使用缓存
    model.config.use_cache = True
    
    # 返回准备好的模型
    return model

def prepare_model_for_h2o(model, heavy_ratio=0.1, recent_ratio=0.1, 
                           eviction_strategy="attention", attention_threshold=0.01):
    """
    准备用于H2O测试的模型，应用H2O KV缓存优化
    
    Args:
        model: 要准备的模型
        heavy_ratio: Heavy Hitter令牌的比例
        recent_ratio: Recent令牌的比例
        eviction_strategy: 缓存驱逐策略 ('attention', 'time_decay', 'hybrid')
        attention_threshold: 注意力分数阈值，用于确定重要命中令牌的标准
        
    Returns:
        model: 应用了H2O优化的模型
    """
    from .h2o_converter import apply_h2o_to_model, init_h2o_cache
    
    logger.info(f"Preparing model for H2O testing with heavy_ratio={heavy_ratio}, recent_ratio={recent_ratio}, "
                f"eviction_strategy={eviction_strategy}, attention_threshold={attention_threshold}")
    
    # 获取模型类型
    model_type = model.config.model_type.lower()
    
    # 应用H2O优化
    model = apply_h2o_to_model(
        model, 
        model_type, 
        heavy_ratio=heavy_ratio, 
        recent_ratio=recent_ratio,
        eviction_strategy=eviction_strategy,
        attention_threshold=attention_threshold,
        enable_cache_stats=True
    )
    
    # 初始化H2O缓存
    model = init_h2o_cache(model)
    
    return model

def is_model_type_supported_by_h2o(model):
    """
    检查模型类型是否支持H2O优化
    
    Args:
        model: 要检查的模型
        
    Returns:
        bool: 是否支持H2O优化
    """
    from .h2o_converter import get_supported_model_types
    
    model_type = model.config.model_type.lower()
    supported_types = get_supported_model_types()
    
    is_supported = model_type in supported_types
    if not is_supported:
        logger.warning(f"模型类型 '{model_type}' 不受H2O优化支持。支持的类型: {', '.join(supported_types)}")
    
    return is_supported

def prepare_model_for_cake(model, 
                           cake_experiment_config: dict, 
                           cake_model_specific_config: dict):
    """
    准备用于CAKE测试的模型，应用CAKE层级感知KV缓存优化。

    Args:
        model: 要准备的PyTorch模型。
        cake_experiment_config: 实验相关的CAKE配置 (来自EXPERIMENT_CONFIG)。
        cake_model_specific_config: 模型准备相关的CAKE配置 (来自CAKE_MODEL_CONFIG)。
        
    Returns:
        model: 应用了CAKE优化的模型。
    """
    from .cake_converter import convert_model_to_cake # 使用高级封装函数
    
    logger.info(f"Preparing model for CAKE testing with experiment config: {cake_experiment_config} and model specific config: {cake_model_specific_config}")
    
    # 获取模型的HuggingFace配置字典
    model_hf_config = model.config.to_dict()

    # 应用CAKE优化
    # convert_model_to_cake 会处理所有内部逻辑，包括LayerAnalyzer和CAKECacheManager的初始化与应用
    model = convert_model_to_cake(
        model,
        model_config_hf=model_hf_config,
        cake_exp_config=cake_experiment_config,
        cake_model_config=cake_model_specific_config
    )
    
    # 确保模型使用缓存 (CAKE转换器内部也应该会设置，这里作为双重保险)
    model.config.use_cache = True
    
    logger.info("Model successfully prepared for CAKE testing.")
    return model

def is_model_type_supported_by_cake(model, cake_model_config: dict) -> bool:
    """
    检查模型类型是否被CAKE_MODEL_CONFIG声明为支持。

    Args:
        model: 要检查的模型。
        cake_model_config: CAKE模型配置字典 (来自config.py的CAKE_MODEL_CONFIG)。

    Returns:
        bool: 模型类型是否在支持列表中。
    """
    model_type = model.config.model_type.lower()
    supported_types = cake_model_config.get("supported_models_cake", [])
    
    is_supported = model_type in supported_types
    if not is_supported:
        logger.warning(f"模型类型 '{model_type}' 未在CAKE_MODEL_CONFIG中被列为支持。支持的类型: {', '.join(supported_types)}。建议使用: NousResearch/Llama-2-7b-hf、mistralai/Mistral-7B-v0.1、Qwen/Qwen2-7B 等模型")
    else:
        logger.info(f"模型类型 '{model_type}' 被CAKE支持。")
    
    return is_supported 