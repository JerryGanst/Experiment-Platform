"""
CAKE-AdaKV 统一预热和记忆管理机制

解决的核心问题：
1. 统一输入格式，避免双重预热
2. 记忆锚点的协同管理
3. 溢出记忆的智能处理
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class UnifiedCacheConfig:
    """统一的缓存配置"""
    # 共享配置
    window_size: int = 32  # 滑动窗口大小（记忆锚点）
    cache_size: int = 512  # 总缓存大小
    
    # CAKE特定配置
    tau1: float = 1.0  # H指标温度
    tau2: float = 1.0  # V指标温度
    gamma: float = 0.5  # 驱逐评分权重
    
    # AdaKV特定配置
    floor_alpha: float = 0.5  # 最小分配比例
    beta: float = 20.0  # 集中度敏感参数
    kernel_size: int = 7  # 平滑核大小
    
    # 统一预热配置
    warmup_steps: int = 10  # 预热步数
    warmup_mode: str = "unified"  # unified, cake_only, adakv_only


class UnifiedMemoryAnchor:
    """统一的记忆锚点管理器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.anchor_tokens = {}  # 层级 -> 锚点token索引
        self.anchor_importance = {}  # 层级 -> 锚点重要性评分
        
    def identify_anchors(
        self, 
        attention_weights: torch.Tensor,
        layer_idx: int,
        method: str = "hybrid"
    ) -> torch.Tensor:
        """
        识别记忆锚点
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            layer_idx: 层索引
            method: 锚点识别方法 - "cake", "adakv", "hybrid"
            
        Returns:
            anchor_indices: 锚点token的索引
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        if method == "cake" or method == "hybrid":
            # CAKE方法：基于注意力均值+方差
            attn_mean = attention_weights.mean(dim=-2)
            attn_var = attention_weights.var(dim=-2)
            cake_importance = attn_mean + self.config.gamma * attn_var
            
        if method == "adakv" or method == "hybrid":
            # AdaKV方法：基于注意力集中度
            # 计算每个token被关注的总和
            adakv_importance = attention_weights.sum(dim=-2)
            
        if method == "hybrid":
            # 混合方法：结合两种重要性评分
            importance = 0.6 * cake_importance + 0.4 * adakv_importance
        elif method == "cake":
            importance = cake_importance
        else:  # adakv
            importance = adakv_importance
            
        # 聚合到token级别
        token_importance = importance.mean(dim=1)  # [batch, seq_len]
        
        # 识别锚点：选择重要性最高的window_size个token
        _, anchor_indices = torch.topk(
            token_importance, 
            min(self.config.window_size, seq_len), 
            dim=-1
        )
        
        # 保存锚点信息
        self.anchor_tokens[layer_idx] = anchor_indices
        self.anchor_importance[layer_idx] = token_importance.gather(
            dim=-1, 
            index=anchor_indices
        )
        
        return anchor_indices
    
    def protect_anchors(
        self,
        eviction_scores: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        保护记忆锚点不被驱逐
        
        Args:
            eviction_scores: 驱逐评分
            layer_idx: 层索引
            
        Returns:
            protected_scores: 保护后的评分
        """
        if layer_idx not in self.anchor_tokens:
            return eviction_scores
            
        # 将锚点的驱逐评分设为最大值，确保不被驱逐
        anchor_indices = self.anchor_tokens[layer_idx]
        protected_scores = eviction_scores.clone()
        
        # 使用scatter操作高效地更新锚点评分
        max_score = eviction_scores.max() + 1.0
        protected_scores.scatter_(
            dim=-1,
            index=anchor_indices,
            value=max_score
        )
        
        return protected_scores


class UnifiedWarmupManager:
    """统一的预热管理器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.warmup_cache = {}  # 预热期间的缓存数据
        self.warmup_stats = {}  # 预热统计信息
        self.current_step = 0
        
    def unified_warmup(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        统一预热过程，避免重复计算
        
        Returns:
            warmup_info: 包含预热结果的字典
        """
        warmup_info = {
            'layer_stats': [],
            'head_stats': [],
            'anchor_info': {},
            'overflow_strategy': {}
        }
        
        with torch.no_grad():
            # 1. 单次前向传播收集所有信息
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=True
            )
            
            attentions = outputs.attentions
            past_key_values = outputs.past_key_values
            
            # 2. 统一分析每层的注意力模式
            for layer_idx, layer_attn in enumerate(attentions):
                # 计算CAKE指标
                h_indicator = self._compute_h_indicator(layer_attn)
                v_indicator = self._compute_v_indicator(layer_attn)
                
                # 计算AdaKV指标
                concentration_scores = self._compute_concentration(layer_attn)
                
                # 保存层级统计
                warmup_info['layer_stats'].append({
                    'h_indicator': h_indicator,
                    'v_indicator': v_indicator,
                    'mean_concentration': concentration_scores.mean().item()
                })
                
                # 保存头级统计
                warmup_info['head_stats'].append(concentration_scores)
        
        # 3. 基于统一的预热结果决定策略
        warmup_info['overflow_strategy'] = self._determine_overflow_strategy(
            warmup_info['layer_stats']
        )
        
        self.warmup_cache = warmup_info
        return warmup_info
    
    def _compute_h_indicator(self, attention_weights: torch.Tensor) -> float:
        """计算H指标（空间分散度）"""
        # 使用熵来量化
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10), 
            dim=-1
        )
        return entropy.mean().item()
    
    def _compute_v_indicator(self, attention_weights: torch.Tensor) -> float:
        """计算V指标（时间变化度）"""
        # 使用方差来量化
        variance = torch.var(attention_weights, dim=-2)
        return variance.mean().item()
    
    def _compute_concentration(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算每个头的集中度"""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 计算每个头的熵
        entropy_per_head = []
        for h in range(num_heads):
            attn_h = attention_weights[:, h, :, :]
            entropy = -torch.sum(attn_h * torch.log(attn_h + 1e-10), dim=-1)
            entropy_per_head.append(entropy.mean())
        
        entropy_per_head = torch.stack(entropy_per_head)
        
        # 转换为集中度
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))
        concentration_scores = 1.0 - (entropy_per_head / max_entropy)
        
        return concentration_scores
    
    def _determine_overflow_strategy(self, layer_stats: List[Dict]) -> Dict:
        """
        基于预热统计决定溢出处理策略
        """
        strategy = {}
        
        for layer_idx, stats in enumerate(layer_stats):
            h_val = stats['h_indicator']
            v_val = stats['v_indicator']
            
            # 基于指标决定策略
            if h_val > 0.7 and v_val > 0.5:
                # 高分散+高动态：混合策略
                strategy[layer_idx] = {
                    'method': 'hybrid',
                    'cake_weight': 0.6,
                    'adakv_weight': 0.4,
                    'anchor_protection': True
                }
            elif h_val > 0.7:
                # 高分散：偏向CAKE
                strategy[layer_idx] = {
                    'method': 'cake_dominant',
                    'cake_weight': 0.8,
                    'adakv_weight': 0.2,
                    'anchor_protection': True
                }
            elif v_val > 0.5:
                # 高动态：偏向AdaKV
                strategy[layer_idx] = {
                    'method': 'adakv_dominant',
                    'cake_weight': 0.3,
                    'adakv_weight': 0.7,
                    'anchor_protection': True
                }
            else:
                # 低分散+低动态：基础策略
                strategy[layer_idx] = {
                    'method': 'balanced',
                    'cake_weight': 0.5,
                    'adakv_weight': 0.5,
                    'anchor_protection': False
                }
        
        return strategy


class UnifiedOverflowHandler:
    """统一的溢出记忆处理器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.overflow_buffer = {}  # 溢出缓冲区
        self.compression_stats = {}  # 压缩统计
        
    def handle_overflow(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        strategy: Dict,
        current_budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理超出预算的记忆
        
        Args:
            key_states: 键状态
            value_states: 值状态
            layer_idx: 层索引
            strategy: 溢出策略
            current_budget: 当前预算
            
        Returns:
            compressed_keys, compressed_values: 压缩后的KV
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if seq_len <= current_budget:
            return key_states, value_states
            
        # 计算需要压缩的token数量
        overflow_size = seq_len - current_budget
        
        if strategy['method'] == 'hybrid':
            # 混合压缩策略
            compressed_k, compressed_v = self._hybrid_compression(
                key_states, value_states, 
                current_budget, 
                strategy['cake_weight'],
                strategy['adakv_weight']
            )
        elif strategy['method'] == 'cake_dominant':
            # CAKE主导的压缩
            compressed_k, compressed_v = self._cake_compression(
                key_states, value_states, current_budget
            )
        elif strategy['method'] == 'adakv_dominant':
            # AdaKV主导的压缩
            compressed_k, compressed_v = self._adakv_compression(
                key_states, value_states, current_budget
            )
        else:
            # 平衡压缩
            compressed_k, compressed_v = self._balanced_compression(
                key_states, value_states, current_budget
            )
        
        # 保存溢出信息用于后续分析
        self.overflow_buffer[layer_idx] = {
            'overflow_size': overflow_size,
            'compression_ratio': current_budget / seq_len,
            'method_used': strategy['method']
        }
        
        return compressed_k, compressed_v
    
    def _hybrid_compression(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        cake_weight: float,
        adakv_weight: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """混合压缩策略"""
        # 实现混合压缩逻辑
        # 这里简化示例，实际应该结合两种方法的优势
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # 为简化，这里使用加权平均的重要性评分
        # 实际实现应该更复杂
        keep_indices = torch.randperm(seq_len)[:budget]
        keep_indices = keep_indices.sort()[0]
        
        compressed_keys = keys.index_select(dim=2, index=keep_indices)
        compressed_values = values.index_select(dim=2, index=keep_indices)
        
        return compressed_keys, compressed_values
    
    def _cake_compression(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CAKE压缩策略"""
        # 基于CAKE的驱逐策略
        # 保留最近的window_size个token + 重要的历史token
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # 保留最近的窗口
        recent_size = min(self.config.window_size, budget)
        historical_budget = budget - recent_size
        
        if historical_budget > 0:
            # 选择历史token（这里简化，实际应该基于重要性）
            historical_indices = torch.randperm(seq_len - recent_size)[:historical_budget]
            recent_indices = torch.arange(seq_len - recent_size, seq_len)
            keep_indices = torch.cat([historical_indices, recent_indices]).sort()[0]
        else:
            keep_indices = torch.arange(seq_len - recent_size, seq_len)
        
        compressed_keys = keys.index_select(dim=2, index=keep_indices)
        compressed_values = values.index_select(dim=2, index=keep_indices)
        
        return compressed_keys, compressed_values
    
    def _adakv_compression(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AdaKV压缩策略"""
        # 基于头级自适应的压缩
        # 每个头可能保留不同数量的token
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # 简化实现：均匀分配
        # 实际应该基于每个头的集中度
        keep_indices = torch.randperm(seq_len)[:budget].sort()[0]
        
        compressed_keys = keys.index_select(dim=2, index=keep_indices)
        compressed_values = values.index_select(dim=2, index=keep_indices)
        
        return compressed_keys, compressed_values
    
    def _balanced_compression(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """平衡压缩策略"""
        # 简单的均匀采样
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        keep_indices = torch.linspace(0, seq_len-1, budget).long()
        
        compressed_keys = keys.index_select(dim=2, index=keep_indices)
        compressed_values = values.index_select(dim=2, index=keep_indices)
        
        return compressed_keys, compressed_values


class UnifiedCakeAdaKV:
    """统一的CAKE-AdaKV实现"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.warmup_manager = UnifiedWarmupManager(config)
        self.memory_anchor = UnifiedMemoryAnchor(config)
        self.overflow_handler = UnifiedOverflowHandler(config)
        
        # 缓存预热结果，避免重复计算
        self.warmup_completed = False
        self.warmup_results = None
        
    def initialize(self, model, sample_input):
        """
        统一初始化过程
        """
        if not self.warmup_completed:
            # 执行统一预热
            self.warmup_results = self.warmup_manager.unified_warmup(
                model, sample_input
            )
            self.warmup_completed = True
            
        return self.warmup_results
    
    def process_kv_cache(
        self,
        past_key_values,
        attention_weights,
        layer_idx: int
    ):
        """
        统一处理KV缓存
        """
        # 1. 识别记忆锚点
        anchor_indices = self.memory_anchor.identify_anchors(
            attention_weights, 
            layer_idx,
            method="hybrid"
        )
        
        # 2. 获取该层的策略
        strategy = self.warmup_results['overflow_strategy'][layer_idx]
        
        # 3. 计算预算（基于CAKE的层级分配）
        layer_budget = self._compute_layer_budget(layer_idx)
        
        # 4. 处理溢出
        compressed_keys, compressed_values = self.overflow_handler.handle_overflow(
            past_key_values.key_cache[layer_idx],
            past_key_values.value_cache[layer_idx],
            layer_idx,
            strategy,
            layer_budget
        )
        
        # 5. 更新缓存
        past_key_values.key_cache[layer_idx] = compressed_keys
        past_key_values.value_cache[layer_idx] = compressed_values
        
        return past_key_values
    
    def _compute_layer_budget(self, layer_idx: int) -> int:
        """计算层级预算"""
        # 基于预热结果计算
        layer_stats = self.warmup_results['layer_stats'][layer_idx]
        
        # 简化的预算计算
        base_budget = self.config.cache_size - self.config.window_size
        
        # 基于H和V指标调整
        h_factor = 1.0 + layer_stats['h_indicator'] * 0.2
        v_factor = 1.0 + layer_stats['v_indicator'] * 0.1
        
        adjusted_budget = int(base_budget * h_factor * v_factor)
        
        return min(adjusted_budget, self.config.cache_size)


# 使用示例
def create_unified_cache_system(model_config):
    """
    创建统一的缓存系统
    """
    config = UnifiedCacheConfig(
        window_size=32,
        cache_size=512,
        tau1=1.0,
        tau2=1.0,
        gamma=0.5,
        floor_alpha=0.5,
        beta=20.0,
        warmup_mode="unified"
    )
    
    return UnifiedCakeAdaKV(config)