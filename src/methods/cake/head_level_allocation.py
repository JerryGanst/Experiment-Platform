"""
CAKE + AdaKV 集成：基于层级分配的头级自适应预分配
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class HeadLevelAllocator:
    """在CAKE层级分配基础上进行头级预分配"""
    
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        window_size: int = 32,
        floor_alpha: float = 0.5,  # 最小分配比例
        beta: float = 20.0,  # 注意力集中度敏感参数
        kernel_size: int = 7,  # 池化核大小
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.window_size = window_size
        self.floor_alpha = floor_alpha
        self.beta = beta
        self.kernel_size = kernel_size
        
    def compute_head_concentration(
        self, 
        attention_weights: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        计算每个头的注意力集中度
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            layer_idx: 当前层索引
            
        Returns:
            concentration_scores: [num_kv_heads]
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 1. 计算每个头的注意力熵（集中度的反向指标）
        entropy_per_head = []
        for h in range(num_heads):
            attn_h = attention_weights[:, h, :, :]
            # 计算熵
            entropy = -torch.sum(attn_h * torch.log(attn_h + 1e-10), dim=-1)
            entropy_per_head.append(entropy.mean())
        
        entropy_per_head = torch.stack(entropy_per_head)
        
        # 2. 将多头映射到KV头
        entropy_per_head = entropy_per_head.reshape(self.num_kv_heads, self.num_groups)
        entropy_per_kv_head = entropy_per_head.mean(dim=1)
        
        # 3. 转换为集中度（低熵=高集中度）
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))
        concentration_scores = 1.0 - (entropy_per_kv_head / max_entropy)
        
        return concentration_scores
    
    def adaptive_head_allocation(
        self,
        layer_budget: int,
        concentration_scores: torch.Tensor,
        cake_h_indicator: float,  # CAKE的H指标
        cake_v_indicator: float,  # CAKE的V指标
    ) -> List[int]:
        """
        基于CAKE层级信息和头级集中度进行自适应分配
        
        Args:
            layer_budget: 该层的总预算（来自CAKE）
            concentration_scores: 每个KV头的集中度
            cake_h_indicator: CAKE计算的空间分散度
            cake_v_indicator: CAKE计算的时间变化度
            
        Returns:
            head_budgets: 每个KV头的预算分配
        """
        # 1. 基础分配
        base_budget_per_head = layer_budget / self.num_kv_heads
        min_budget_per_head = int(self.floor_alpha * base_budget_per_head)
        
        # 2. 结合CAKE指标调整集中度权重
        # 高H指标（分散）-> 更均匀的头级分配
        # 高V指标（动态）-> 更自适应的头级分配
        spatial_weight = 1.0 / (1.0 + cake_h_indicator)  # H越大，权重越小
        temporal_weight = 1.0 + cake_v_indicator  # V越大，权重越大
        
        # 3. 计算调整后的分配权重
        adjusted_scores = concentration_scores ** (self.beta * spatial_weight * temporal_weight)
        allocation_weights = adjusted_scores / adjusted_scores.sum()
        
        # 4. 分配预算
        head_budgets = []
        remaining_budget = layer_budget
        
        for i in range(self.num_kv_heads):
            # 计算该头的预算
            ideal_budget = int(allocation_weights[i] * layer_budget)
            # 确保满足最小预算约束
            actual_budget = max(ideal_budget, min_budget_per_head)
            # 确保不超过剩余预算
            actual_budget = min(actual_budget, remaining_budget)
            
            head_budgets.append(actual_budget)
            remaining_budget -= actual_budget
        
        # 5. 处理剩余预算（如果有）
        if remaining_budget > 0:
            # 分配给集中度最高的头
            sorted_indices = torch.argsort(concentration_scores, descending=True)
            for idx in sorted_indices:
                if remaining_budget <= 0:
                    break
                head_budgets[idx] += 1
                remaining_budget -= 1
        
        return head_budgets
    
    def compute_eviction_priority(
        self,
        attention_scores: torch.Tensor,
        value_norms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算每个token的驱逐优先级（结合注意力和值向量信息）
        
        Args:
            attention_scores: [batch, num_heads, seq_len, seq_len]
            value_norms: [batch, num_heads, seq_len, head_dim] 的范数
            
        Returns:
            priority_scores: [batch, num_kv_heads, seq_len]
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # 1. 计算注意力重要性（类似CAKE的方法）
        attn_mean = attention_scores.mean(dim=-2)  # 被关注程度
        attn_var = attention_scores.var(dim=-2)   # 关注模式的变化
        
        # 2. 结合均值和方差
        importance = attn_mean + 0.5 * attn_var
        
        # 3. 应用平滑
        if self.kernel_size > 1:
            importance = F.avg_pool1d(
                importance.reshape(batch_size * num_heads, -1).unsqueeze(1),
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1
            ).squeeze(1).reshape(batch_size, num_heads, -1)
        
        # 4. 聚合到KV头级别
        importance = importance.reshape(batch_size, self.num_kv_heads, self.num_groups, -1)
        priority_scores = importance.mean(dim=2)
        
        # 5. 如果有值向量信息，结合进来
        if value_norms is not None:
            value_importance = value_norms.mean(dim=-1)  # [batch, num_heads, seq_len]
            value_importance = value_importance.reshape(
                batch_size, self.num_kv_heads, self.num_groups, -1
            ).mean(dim=2)
            # 结合注意力和值向量的重要性
            priority_scores = 0.7 * priority_scores + 0.3 * value_importance
        
        return priority_scores


class CakeAdaKVIntegration:
    """CAKE和AdaKV的集成实现"""
    
    def __init__(
        self,
        model_config: Dict,
        cake_config: Dict,
        adakv_config: Dict
    ):
        self.num_layers = model_config['num_layers']
        self.num_heads = model_config['num_heads']
        self.num_kv_heads = model_config.get('num_kv_heads', self.num_heads)
        
        # 初始化每层的头级分配器
        self.head_allocators = [
            HeadLevelAllocator(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                **adakv_config
            )
            for _ in range(self.num_layers)
        ]
        
        self.cake_config = cake_config
        self.adakv_config = adakv_config
        
    def allocate_budgets(
        self,
        cake_outputs: Dict,
        attention_patterns: List[torch.Tensor]
    ) -> Dict:
        """
        基于CAKE的层级输出进行头级预算分配
        
        Args:
            cake_outputs: CAKE的输出，包含层级预算和指标
            attention_patterns: 每层的注意力模式
            
        Returns:
            allocation_results: 包含层级和头级分配结果
        """
        allocation_results = {
            'layer_budgets': cake_outputs['layer_budgets'],
            'head_budgets': [],
            'eviction_priorities': []
        }
        
        for layer_idx in range(self.num_layers):
            # 获取该层的CAKE输出
            layer_budget = cake_outputs['layer_budgets'][layer_idx]
            h_indicator = cake_outputs['h_indicators'][layer_idx]
            v_indicator = cake_outputs['v_indicators'][layer_idx]
            
            # 获取该层的注意力模式
            attn_weights = attention_patterns[layer_idx]
            
            # 计算头级集中度
            allocator = self.head_allocators[layer_idx]
            concentration_scores = allocator.compute_head_concentration(
                attn_weights, layer_idx
            )
            
            # 进行头级分配
            head_budgets = allocator.adaptive_head_allocation(
                layer_budget,
                concentration_scores,
                h_indicator,
                v_indicator
            )
            
            # 计算驱逐优先级
            eviction_priority = allocator.compute_eviction_priority(attn_weights)
            
            allocation_results['head_budgets'].append(head_budgets)
            allocation_results['eviction_priorities'].append(eviction_priority)
        
        return allocation_results
    
    def apply_allocation(
        self,
        key_states: List[torch.Tensor],
        value_states: List[torch.Tensor],
        allocation_results: Dict
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        应用分配结果，执行实际的缓存驱逐
        
        Args:
            key_states: 每层的键状态
            value_states: 每层的值状态
            allocation_results: 分配结果
            
        Returns:
            compressed_keys, compressed_values: 压缩后的KV缓存
        """
        compressed_keys = []
        compressed_values = []
        
        for layer_idx in range(self.num_layers):
            layer_keys = key_states[layer_idx]
            layer_values = value_states[layer_idx]
            head_budgets = allocation_results['head_budgets'][layer_idx]
            priorities = allocation_results['eviction_priorities'][layer_idx]
            
            # 对每个KV头进行压缩
            batch_size, num_kv_heads, seq_len, head_dim = layer_keys.shape
            
            compressed_k_heads = []
            compressed_v_heads = []
            
            for head_idx in range(num_kv_heads):
                budget = head_budgets[head_idx]
                head_priorities = priorities[:, head_idx, :]  # [batch, seq_len]
                
                # 选择top-k个token
                _, indices = torch.topk(head_priorities, budget, dim=-1)
                indices = indices.unsqueeze(-1).expand(-1, -1, head_dim)
                
                # 提取保留的KV
                head_k = layer_keys[:, head_idx:head_idx+1, :, :]
                head_v = layer_values[:, head_idx:head_idx+1, :, :]
                
                compressed_k = head_k.gather(dim=2, index=indices.unsqueeze(1))
                compressed_v = head_v.gather(dim=2, index=indices.unsqueeze(1))
                
                compressed_k_heads.append(compressed_k)
                compressed_v_heads.append(compressed_v)
            
            # 合并所有头
            compressed_keys.append(torch.cat(compressed_k_heads, dim=1))
            compressed_values.append(torch.cat(compressed_v_heads, dim=1))
        
        return compressed_keys, compressed_values


# 使用示例
def integrate_cake_adakv(model, cake_cache, attention_patterns):
    """
    将CAKE和AdaKV集成的示例函数
    """
    # 配置
    model_config = {
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads,
        'num_kv_heads': model.config.num_key_value_heads
    }
    
    cake_config = {
        'tau1': 1.0,
        'tau2': 1.0,
        'gamma': 0.5
    }
    
    adakv_config = {
        'window_size': 32,
        'floor_alpha': 0.5,
        'beta': 20.0,
        'kernel_size': 7
    }
    
    # 初始化集成器
    integrator = CakeAdaKVIntegration(model_config, cake_config, adakv_config)
    
    # 从CAKE获取层级分配结果
    cake_outputs = {
        'layer_budgets': cake_cache.layer_budget,
        'h_indicators': [score[0] for score in cake_cache.pref_scores],  # 简化示例
        'v_indicators': [score[1] for score in cake_cache.pref_scores]   # 简化示例
    }
    
    # 执行头级分配
    allocation_results = integrator.allocate_budgets(cake_outputs, attention_patterns)
    
    # 应用分配（实际压缩KV缓存）
    compressed_keys, compressed_values = integrator.apply_allocation(
        cake_cache.key_cache,
        cake_cache.value_cache,
        allocation_results
    )
    
    return compressed_keys, compressed_values, allocation_results