"""
H2O模型转换模块

基于utils_hh中的实现，提供将普通模型转换为使用H2O缓存机制的模型的功能
"""
import torch
import logging
import math
import sys
import os
from torch import nn
from typing import Optional, Tuple, Union

# 配置导入路径
current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
hace_optimization_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(hace_optimization_dir)
utils_hh_dir = os.path.join(project_root, "utils_hh")

# 确保utils_hh目录在sys.path中
if utils_hh_dir not in sys.path:
    sys.path.insert(0, utils_hh_dir)

# 设置logger
logger = logging.getLogger(__name__)

# 导入H2O实现模块
try:
    # 从utils_hh目录导入
    from modify_opt import convert_kvcache_opt_heavy_recent, HeavyRecentCacheLayer, OPTAttention_heavy_hitter
    h2o_opt_available = True
except ImportError as e:
    logger.warning(f"H2O OPT实现模块无法导入: {e}")
    h2o_opt_available = False

try:
    # 从utils_hh目录导入llama实现(如果存在)
    from modify_llama import convert_kvcache_llama_heavy_recent
    h2o_llama_available = True
except ImportError as e:
    logger.warning(f"H2O LLaMA实现模块无法导入: {e}")
    h2o_llama_available = False

try:
    # 从utils_hh目录导入gptneox实现(如果存在)
    from modify_gptneox import convert_kvcache_gpt_neox_heavy_recent
    h2o_gptneox_available = True
except ImportError as e:
    logger.warning(f"H2O GPT-NeoX实现模块无法导入: {e}")
    h2o_gptneox_available = False

# H2O配置常量
EVICTION_STRATEGIES = ["attention", "time_decay", "hybrid"]
DEFAULT_ATTENTION_THRESHOLD = 0.01

def check_h2o_availability():
    """
    检查H2O实现是否可用
    
    Returns:
        bool: 是否可用H2O实现
    """
    if not os.path.exists(utils_hh_dir):
        logger.warning(f"utils_hh目录不存在: {utils_hh_dir}")
        return False
    
    available = h2o_opt_available or h2o_llama_available or h2o_gptneox_available
    
    if not available:
        logger.warning("没有可用的H2O实现模块。请确保utils_hh目录包含必要的实现文件。")
    
    return available

def apply_h2o_to_model(model, model_type, heavy_ratio=0.1, recent_ratio=0.1, 
                      eviction_strategy="attention", attention_threshold=DEFAULT_ATTENTION_THRESHOLD,
                      enable_cache_stats=True):
    """
    将H2O KV缓存优化应用于模型
    
    Args:
        model: 要优化的模型
        model_type: 模型类型，支持"opt", "llama", "gpt_neox"
        heavy_ratio: Heavy Hitter令牌的比例 (0.1, 0.2, 0.3)
        recent_ratio: Recent令牌的比例
        eviction_strategy: 缓存驱逐策略 ('attention', 'time_decay', 'hybrid')
        attention_threshold: 注意力分数阈值，用于确定重要命中令牌的标准
        enable_cache_stats: 是否启用缓存统计收集功能
        
    Returns:
        model: 应用了H2O优化的模型
    """
    if not check_h2o_availability():
        raise ImportError("H2O实现模块不可用，无法应用H2O优化")
    
    # 针对特定模型类型检查可用性
    if model_type == "opt" and not h2o_opt_available:
        raise ImportError("H2O OPT实现模块不可用，无法应用H2O优化到OPT模型")
    
    if model_type == "llama" and not h2o_llama_available:
        raise ImportError("H2O LLaMA实现模块不可用，无法应用H2O优化到LLaMA模型")
    
    if model_type == "gpt_neox" and not h2o_gptneox_available:
        raise ImportError("H2O GPT-NeoX实现模块不可用，无法应用H2O优化到GPT-NeoX模型")
    
    # 验证参数
    if eviction_strategy not in EVICTION_STRATEGIES:
        logger.warning(f"不支持的驱逐策略: {eviction_strategy}，使用默认的'attention'策略")
        eviction_strategy = "attention"
    
    # 设置H2O配置
    logger.info(f"正在应用H2O缓存优化，heavy_ratio={heavy_ratio}, recent_ratio={recent_ratio}, "
               f"eviction_strategy={eviction_strategy}, attention_threshold={attention_threshold}")
    
    try:
        # 设置共享配置参数
        model.config.heavy_ratio = heavy_ratio
        model.config.recent_ratio = recent_ratio
        model.config.use_cache = True
        model.config.eviction_strategy = eviction_strategy
        model.config.attention_threshold = attention_threshold
        model.config.enable_cache_stats = enable_cache_stats
        
        # 对不同模型类型应用不同的H2O实现
        if model_type == "opt":
            # 应用OPT的H2O转换
            model = convert_kvcache_opt_heavy_recent(model, model.config)
            logger.info("成功应用OPT模型的H2O缓存优化")
            
            # 修改OPT模型的驱逐策略
            if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                for layer in model.model.decoder.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'heavy_recent'):
                        layer.self_attn.heavy_recent.eviction_strategy = eviction_strategy
                        layer.self_attn.heavy_recent.attention_threshold = attention_threshold
            
        elif model_type == "llama":
            # 应用LLaMA的H2O转换
            model = convert_kvcache_llama_heavy_recent(model, model.config)
            logger.info("成功应用LLaMA模型的H2O缓存优化")
            
            # 修改LLaMA模型的驱逐策略
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for layer in model.model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'heavy_recent'):
                        layer.self_attn.heavy_recent.eviction_strategy = eviction_strategy
                        layer.self_attn.heavy_recent.attention_threshold = attention_threshold
            
        elif model_type == "gpt_neox":
            # 应用GPT-NeoX的H2O转换
            model = convert_kvcache_gpt_neox_heavy_recent(model, model.config)
            logger.info("成功应用GPT-NeoX模型的H2O缓存优化")
            
            # 修改GPT-NeoX模型的驱逐策略
            if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                for layer in model.gpt_neox.layers:
                    if hasattr(layer, 'attention') and hasattr(layer.attention, 'heavy_recent'):
                        layer.attention.heavy_recent.eviction_strategy = eviction_strategy
                        layer.attention.heavy_recent.attention_threshold = attention_threshold
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}。目前支持: opt, llama, gpt_neox")
        
        # 添加获取缓存统计数据的方法
        if enable_cache_stats:
            model = _add_cache_statistics_methods(model, model_type)
        
        return model
    
    except Exception as e:
        logger.error(f"应用H2O优化时出错: {str(e)}")
        raise

def _add_cache_statistics_methods(model, model_type):
    """
    为模型添加获取缓存统计数据的方法
    
    Args:
        model: 模型
        model_type: 模型类型
        
    Returns:
        model: 添加了统计方法的模型
    """
    def get_cache_statistics(self):
        """获取H2O缓存的统计数据"""
        stats = {
            'cache_hit_rate': 0.0,
            'heavy_hitter_distribution': [],
            'eviction_frequency': 0,
            'attention_distribution': []
        }
        
        # 收集各层的统计数据
        layers_stats = []
        
        if model_type == "opt":
            if hasattr(self, 'model') and hasattr(self.model, 'decoder'):
                for i, layer in enumerate(self.model.decoder.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'heavy_recent'):
                        layer_stats = layer.self_attn.heavy_recent.get_statistics()
                        layer_stats['layer_idx'] = i
                        layers_stats.append(layer_stats)
        
        elif model_type == "llama":
            if hasattr(self, 'model') and hasattr(self.model, 'layers'):
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'heavy_recent'):
                        layer_stats = layer.self_attn.heavy_recent.get_statistics()
                        layer_stats['layer_idx'] = i
                        layers_stats.append(layer_stats)
        
        elif model_type == "gpt_neox":
            if hasattr(self, 'gpt_neox') and hasattr(self.gpt_neox, 'layers'):
                for i, layer in enumerate(self.gpt_neox.layers):
                    if hasattr(layer, 'attention') and hasattr(layer.attention, 'heavy_recent'):
                        layer_stats = layer.attention.heavy_recent.get_statistics()
                        layer_stats['layer_idx'] = i
                        layers_stats.append(layer_stats)
        
        # 如果找到了统计数据
        if layers_stats:
            # 计算平均缓存命中率
            hit_rates = [s.get('cache_hit_rate', 0.0) for s in layers_stats]
            stats['cache_hit_rate'] = sum(hit_rates) / len(hit_rates)
            
            # 收集重要命中分布
            stats['heavy_hitter_distribution'] = [
                {
                    'layer_idx': s['layer_idx'],
                    'distribution': s.get('heavy_hitter_distribution', [])
                }
                for s in layers_stats
            ]
            
            # 计算总驱逐频率
            stats['eviction_frequency'] = sum([s.get('eviction_frequency', 0) for s in layers_stats])
            
            # 收集注意力分布
            stats['attention_distribution'] = [
                {
                    'layer_idx': s['layer_idx'],
                    'distribution': s.get('attention_distribution', [])
                }
                for s in layers_stats
            ]
        
        return stats
    
    # 添加方法到模型
    model.get_cache_statistics = get_cache_statistics.__get__(model)
    
    return model

def init_h2o_cache(model):
    """
    初始化模型的H2O缓存
    
    Args:
        model: 应用了H2O优化的模型
        
    Returns:
        model: 初始化缓存后的模型
    """
    if hasattr(model, 'init_cache'):
        logger.info("正在初始化H2O缓存...")
        model.init_cache()
    else:
        logger.warning("模型没有init_cache方法，H2O缓存可能未正确应用")
    
    return model

def get_supported_model_types():
    """
    获取支持H2O优化的模型类型列表
    
    Returns:
        list: 支持的模型类型列表
    """
    return ["opt", "llama", "gpt_neox"]

def get_eviction_strategies():
    """
    获取支持的驱逐策略列表
    
    Returns:
        list: 支持的驱逐策略列表
    """
    return EVICTION_STRATEGIES.copy() 