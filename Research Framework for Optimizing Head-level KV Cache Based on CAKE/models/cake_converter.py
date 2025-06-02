"""
CAKE模型转换模块

此模块作为您现有CAKE实验平台与cakekv-main核心功能的桥接层。
它提供统一的接口来应用CAKE优化到模型上。
"""
import sys
import os
import torch
import logging
from typing import Optional, Dict, Any

# 添加cakekv-main到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
cakekv_path = os.path.join(project_root, "cakekv-main", "cakekv-main")
if cakekv_path not in sys.path:
    sys.path.insert(0, cakekv_path)

# 导入CAKE核心功能
try:
    from cake.cake_cache import CakeCache, CakeprefillKVCache, CakeDecodingKVCache_LayerWise
    from cake.utils import CompressConfig, adjust_budgets
    from cake.monkeypatch import (
        replace_flashllama_attn_with_cakeattn,
        replace_flashmistral_attn_with_cakeattn,
        replace_flashqwen2_attn_with_cakeattn
    )
    CAKE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入CAKE核心功能: {e}")
    CAKE_AVAILABLE = False

logger = logging.getLogger(__name__)

class CAKEExperimentConfig:
    """CAKE实验配置管理器，用于管理实验相关的CAKE参数"""
    
    def __init__(self, 
                 allocation_strategy: str = "adaptive",
                 cache_budget: float = 0.8,
                 window_size: int = 32,
                 gamma: float = 0.8,
                 tau1: float = 1.0,
                 tau2: float = 1.0,
                 **kwargs):
        """
        初始化CAKE实验配置
        
        Args:
            allocation_strategy: 层级分配策略 ("uniform", "adaptive", "attention_based")
            cache_budget: 总缓存预算比例 (0.0-1.0)
            window_size: 窗口大小
            gamma: 注意力分数计算的gamma参数
            tau1: 熵的权重参数
            tau2: 方差的权重参数
        """
        self.allocation_strategy = allocation_strategy
        self.cache_budget = cache_budget
        self.window_size = window_size
        self.gamma = gamma
        self.tau1 = tau1
        self.tau2 = tau2
        self.extra_config = kwargs
        
        logger.info(f"CAKE实验配置初始化: 策略={allocation_strategy}, 预算={cache_budget}, 窗口={window_size}")

    def to_compress_config(self) -> 'CompressConfig':
        """转换为CAKE核心代码期望的CompressConfig格式"""
        if not CAKE_AVAILABLE:
            raise ImportError("CAKE核心功能不可用")
            
        # 计算cache_size基于预算
        # 这里使用一个默认值，实际应根据模型配置计算
        cache_size = int(1024 * self.cache_budget)  # 示例计算
        
        return CompressConfig(
            compress=True,
            cache_size=cache_size,
            window_size=self.window_size,
            hyper={
                'gamma': self.gamma,
                'tau1': self.tau1,
                'tau2': self.tau2,
                'allocation_strategy': self.allocation_strategy
            }
        )

class CAKEModelAdapter:
    """CAKE模型适配器，负责将CAKE应用到不同类型的模型上"""
    
    def __init__(self, experiment_config: CAKEExperimentConfig):
        self.config = experiment_config
        self.supported_models = {
            "llama": replace_flashllama_attn_with_cakeattn,
            "mistral": replace_flashmistral_attn_with_cakeattn,
            "qwen2": replace_flashqwen2_attn_with_cakeattn
        }
        
    def apply_cake_to_model(self, model, model_config_hf: Dict[str, Any]) -> torch.nn.Module:
        """
        将CAKE应用到模型上
        
        Args:
            model: 要应用CAKE的PyTorch模型
            model_config_hf: HuggingFace模型配置字典
            
        Returns:
            应用了CAKE的模型
        """
        if not CAKE_AVAILABLE:
            raise ImportError("CAKE核心功能不可用，请检查cakekv-main目录")
            
        model_type = model_config_hf.get("model_type", "").lower()
        logger.info(f"开始为 {model_type} 模型应用CAKE")
        
        # 检查模型类型是否支持
        if model_type not in self.supported_models:
            raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: {list(self.supported_models.keys())}")
        
        # 配置CAKE参数到模型配置中
        self._configure_model_for_cake(model, model_config_hf)
        
        # 应用CAKE的monkey patch
        cake_patch_fn = self.supported_models[model_type]
        logger.info(f"应用 {model_type} 的CAKE补丁")
        cake_patch_fn()
        
        # 标记模型已应用CAKE
        model.is_cake_enabled = True
        model.cake_config = self.config
        
        logger.info("CAKE应用完成！")
        return model
    
    def _configure_model_for_cake(self, model, model_config_hf: Dict[str, Any]):
        """配置模型以支持CAKE"""
        num_layers = (model_config_hf.get("num_hidden_layers") or 
                     model_config_hf.get("n_layer") or 
                     model_config_hf.get("num_layers") or 
                     len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 32)
        
        # 计算每层的预算分配
        layer_budgets = self._calculate_layer_budgets(num_layers, model_config_hf)
        
        # 创建CAKE配置对象并添加到模型配置中
        if not hasattr(model.config, 'key_size'):
            model.config.key_size = layer_budgets
        if not hasattr(model.config, 'window_size'):
            model.config.window_size = [self.config.window_size] * num_layers
        if not hasattr(model.config, 'gamma'):
            model.config.gamma = self.config.gamma
        if not hasattr(model.config, 'tau1'):
            model.config.tau1 = self.config.tau1
        if not hasattr(model.config, 'tau2'):
            model.config.tau2 = self.config.tau2
        
        # 初始化CAKE特定的状态
        if not hasattr(model.config, 'prefill'):
            model.config.prefill = [True] * num_layers
        if not hasattr(model.config, 'prefill_cake_evict'):
            model.config.prefill_cake_evict = [None] * num_layers
            for i in range(num_layers):
                model.config.prefill_cake_evict[i] = CakeprefillKVCache(
                    cache_size=layer_budgets[i],
                    window_size=self.config.window_size,
                    num_layers=num_layers
                )
        if not hasattr(model.config, 'decoding_evict'):
            model.config.decoding_evict = [None] * num_layers
            
        logger.info(f"模型配置完成: {num_layers}层, 预算分配={layer_budgets[:5]}...")
    
    def _calculate_layer_budgets(self, num_layers: int, model_config_hf: Dict[str, Any]) -> list:
        """
        根据配置的分配策略计算每层的缓存预算
        
        Args:
            num_layers: 模型层数
            model_config_hf: 模型配置
            
        Returns:
            每层的预算列表
        """
        # 获取基础参数
        max_seq_len = model_config_hf.get("max_position_embeddings", 2048)
        total_budget = int(max_seq_len * self.config.cache_budget)
        
        if self.config.allocation_strategy == "uniform":
            # 均匀分配
            budget_per_layer = total_budget // num_layers
            layer_budgets = [budget_per_layer] * num_layers
            
        elif self.config.allocation_strategy == "adaptive":
            # 自适应分配：前几层获得更多预算
            weights = [(num_layers - i) / sum(range(1, num_layers + 1)) for i in range(num_layers)]
            layer_budgets = [int(total_budget * w) for w in weights]
            
        elif self.config.allocation_strategy == "attention_based":
            # 基于注意力的分配（简化版本，实际可以更复杂）
            # 中间层获得更多预算
            weights = []
            for i in range(num_layers):
                if i < num_layers // 3:
                    weights.append(0.8)  # 前1/3层
                elif i < 2 * num_layers // 3:
                    weights.append(1.2)  # 中间1/3层
                else:
                    weights.append(1.0)  # 后1/3层
            
            total_weight = sum(weights)
            layer_budgets = [int(total_budget * w / total_weight) for w in weights]
            
        else:
            logger.warning(f"未知的分配策略 {self.config.allocation_strategy}，使用均匀分配")
            budget_per_layer = total_budget // num_layers
            layer_budgets = [budget_per_layer] * num_layers
        
        # 使用CAKE的adjust_budgets函数进行最终调整
        layer_budgets = adjust_budgets(layer_budgets, total_budget, max_seq_len, num_layers)
        
        logger.debug(f"层级预算计算完成: 总预算={total_budget}, 策略={self.config.allocation_strategy}")
        return layer_budgets

def apply_cake_to_model(
    model: torch.nn.Module,
    model_config_hf: Dict[str, Any],
    cake_experiment_config: Dict[str, Any],
    cake_model_specific_config: Dict[str, Any]
) -> torch.nn.Module:
    """
    高级接口：将CAKE应用到模型上
    
    Args:
        model: 要应用CAKE的PyTorch模型
        model_config_hf: HuggingFace模型配置字典
        cake_experiment_config: 实验相关的CAKE配置
        cake_model_specific_config: 模型特定的CAKE配置
        
    Returns:
        应用了CAKE的模型
    """
    if not CAKE_AVAILABLE:
        logger.error("CAKE核心功能不可用，请检查cakekv-main目录是否存在且正确")
        raise ImportError("CAKE核心功能不可用")
    
    # 检查模型是否已经应用了CAKE
    if hasattr(model, 'is_cake_enabled') and model.is_cake_enabled:
        logger.info("模型已经应用了CAKE，跳过重复应用")
        return model
    
    # 提取实验配置参数
    allocation_strategy = "uniform"  # 默认值
    cache_budget = 0.8  # 默认值
    
    if "layer_allocation_strategies" in cake_experiment_config:
        strategies = cake_experiment_config["layer_allocation_strategies"]
        allocation_strategy = strategies[0] if strategies else "uniform"
    
    if "cache_budgets" in cake_experiment_config:
        budgets = cake_experiment_config["cache_budgets"]
        cache_budget = budgets[0] if budgets else 0.8
    
    # 从模型特定配置中获取其他参数
    window_size = cake_model_specific_config.get("window_size", 32)
    gamma = cake_model_specific_config.get("gamma", 0.8)
    tau1 = cake_model_specific_config.get("tau1", 1.0)
    tau2 = cake_model_specific_config.get("tau2", 1.0)
    
    # 创建CAKE实验配置
    experiment_config = CAKEExperimentConfig(
        allocation_strategy=allocation_strategy,
        cache_budget=cache_budget,
        window_size=window_size,
        gamma=gamma,
        tau1=tau1,
        tau2=tau2
    )
    
    # 创建适配器并应用CAKE
    adapter = CAKEModelAdapter(experiment_config)
    return adapter.apply_cake_to_model(model, model_config_hf)

def convert_model_to_cake(model, model_config_hf, cake_exp_config, cake_model_config):
    """
    兼容性函数：与之前的接口保持一致
    """
    return apply_cake_to_model(model, model_config_hf, cake_exp_config, cake_model_config)

def is_cake_available() -> bool:
    """检查CAKE核心功能是否可用"""
    return CAKE_AVAILABLE

def get_supported_model_types() -> list:
    """获取支持的模型类型列表"""
    if not CAKE_AVAILABLE:
        return []
    return ["llama", "mistral", "qwen2"]

# 为了向后兼容，保留一些类定义（但实际功能由cakekv-main提供）
class LayerAnalyzer:
    """向后兼容的层级分析器类（实际功能由CAKE核心提供）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.warning("使用兼容性LayerAnalyzer，实际分析由CAKE核心提供")
    
    def compute_layer_importance_scores(self, strategy: str = "uniform") -> Dict[int, float]:
        """兼容性方法：返回简单的重要性分数"""
        if strategy == "uniform":
            # 假设12层模型，返回均匀分数
            num_layers = 12
            return {i: 1.0/num_layers for i in range(num_layers)}
        elif strategy == "adaptive":
            # 前几层更重要
            num_layers = 12
            weights = [(num_layers - i) for i in range(num_layers)]
            total = sum(weights)
            return {i: w/total for i, w in enumerate(weights)}
        else:
            logger.warning(f"未支持的策略: {strategy}, 使用uniform")
            num_layers = 12
            return {i: 1.0/num_layers for i in range(num_layers)}

class CAKECacheManager:
    """向后兼容的缓存管理器类（实际功能由CAKE核心提供）"""
    
    def __init__(self, model_config, cake_specific_config, layer_importance_scores, num_layers):
        self.model_config = model_config
        self.config = cake_specific_config
        self.layer_importance_scores = layer_importance_scores
        self.num_layers = num_layers
        logger.warning("使用兼容性CAKECacheManager，实际缓存管理由CAKE核心提供")
    
    def get_cache_budget_for_layer(self, layer_idx: int) -> int:
        """兼容性方法"""
        return 512  # 默认预算
    
    def apply_cake_to_attention_layer(self, attention_layer, layer_idx: int):
        """兼容性方法"""
        logger.info(f"兼容性方法：为层 {layer_idx} 应用CAKE（实际由核心代码处理）")

if __name__ == '__main__':
    # 测试CAKE功能是否可用
    logging.basicConfig(level=logging.INFO)
    
    if is_cake_available():
        logger.info("✅ CAKE核心功能可用")
        logger.info(f"支持的模型类型: {get_supported_model_types()}")
        
        # 测试配置创建
        config = CAKEExperimentConfig(
            allocation_strategy="adaptive",
            cache_budget=0.8,
            window_size=32
        )
        logger.info(f"CAKE实验配置创建成功: {config.allocation_strategy}")
        
    else:
        logger.error("❌ CAKE核心功能不可用，请检查cakekv-main目录") 