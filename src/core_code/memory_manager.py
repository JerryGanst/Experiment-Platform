"""
内存管理器模块

核心创新：
1. 统一的内存锚点管理
2. 智能溢出处理
3. 跨算法的内存协调

这是我们独立开发的创新组件，不依赖原始CAKE/AdaKV代码。
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryConfig:
    """内存管理配置"""
    enable_anchor_coordination: bool = True  # 启用锚点协调
    overflow_strategy: str = "adaptive"  # 溢出策略：adaptive, conservative, aggressive
    anchor_merge_threshold: float = 0.8  # 锚点合并阈值
    memory_efficiency_mode: bool = True  # 内存效率模式


class UnifiedMemoryManager:
    """
    统一内存管理器
    
    核心创新：
    - 协调CAKE和AdaKV的内存使用模式
    - 智能锚点管理：避免冲突，优化重用
    - 自适应溢出处理：根据内存压力动态调整策略
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.memory_anchors = {}  # 内存锚点
        self.overflow_history = {}  # 溢出历史
        
    def coordinate_memory_anchors(
        self,
        cake_anchors: Dict[int, List[int]],
        adakv_anchors: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        """
        协调内存锚点
        
        Args:
            cake_anchors: CAKE算法的锚点
            adakv_anchors: AdaKV算法的锚点
            
        Returns:
            统一的锚点配置
        """
        if not self.config.enable_anchor_coordination:
            # 简单合并
            unified_anchors = {}
            for layer_idx in set(cake_anchors.keys()) | set(adakv_anchors.keys()):
                cake_layer = cake_anchors.get(layer_idx, [])
                adakv_layer = adakv_anchors.get(layer_idx, [])
                unified_anchors[layer_idx] = list(set(cake_layer + adakv_layer))
            return unified_anchors
        
        # 智能协调
        unified_anchors = {}
        
        for layer_idx in set(cake_anchors.keys()) | set(adakv_anchors.keys()):
            cake_layer = set(cake_anchors.get(layer_idx, []))
            adakv_layer = set(adakv_anchors.get(layer_idx, []))
            
            # 计算重叠度
            intersection = cake_layer & adakv_layer
            union = cake_layer | adakv_layer
            
            if len(union) == 0:
                overlap_ratio = 0.0
            else:
                overlap_ratio = len(intersection) / len(union)
            
            if overlap_ratio >= self.config.anchor_merge_threshold:
                # 高重叠：优先使用交集
                unified_anchors[layer_idx] = list(intersection)
                if not unified_anchors[layer_idx]:
                    # 交集为空，使用并集
                    unified_anchors[layer_idx] = list(union)
            else:
                # 低重叠：智能选择
                unified_anchors[layer_idx] = self._smart_anchor_selection(
                    cake_layer, adakv_layer, layer_idx
                )
        
        return unified_anchors
    
    def _smart_anchor_selection(
        self,
        cake_anchors: set,
        adakv_anchors: set,
        layer_idx: int
    ) -> List[int]:
        """智能锚点选择"""
        # 简化版本：交替选择
        cake_list = sorted(list(cake_anchors))
        adakv_list = sorted(list(adakv_anchors))
        
        selected = []
        max_len = max(len(cake_list), len(adakv_list))
        
        for i in range(max_len):
            if i < len(cake_list):
                selected.append(cake_list[i])
            if i < len(adakv_list) and adakv_list[i] not in selected:
                selected.append(adakv_list[i])
        
        return selected
    
    def handle_memory_overflow(
        self,
        layer_idx: int,
        requested_budget: int,
        available_budget: int,
        priority_scores: List[float]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        处理内存溢出
        
        Args:
            layer_idx: 层索引
            requested_budget: 请求的预算
            available_budget: 可用预算
            priority_scores: 优先级评分
            
        Returns:
            (adjusted_budgets, overflow_info): 调整后的预算和溢出信息
        """
        if requested_budget <= available_budget:
            # 无溢出
            return [available_budget], {'overflow': False}
        
        overflow_ratio = requested_budget / available_budget
        self.overflow_history[layer_idx] = overflow_ratio
        
        strategy = self.config.overflow_strategy
        
        if strategy == "conservative":
            return self._conservative_overflow_handling(
                available_budget, priority_scores
            )
        elif strategy == "aggressive":
            return self._aggressive_overflow_handling(
                available_budget, priority_scores
            )
        else:  # adaptive
            return self._adaptive_overflow_handling(
                layer_idx, available_budget, priority_scores, overflow_ratio
            )
    
    def _conservative_overflow_handling(
        self,
        available_budget: int,
        priority_scores: List[float]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """保守的溢出处理：均匀削减"""
        num_heads = len(priority_scores)
        base_budget = available_budget // num_heads
        remainder = available_budget % num_heads
        
        budgets = [base_budget] * num_heads
        for i in range(remainder):
            budgets[i] += 1
        
        return budgets, {
            'overflow': True,
            'strategy': 'conservative',
            'reduction_method': 'uniform'
        }
    
    def _aggressive_overflow_handling(
        self,
        available_budget: int,
        priority_scores: List[float]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """激进的溢出处理：按优先级分配"""
        scores = np.array(priority_scores)
        total_score = np.sum(scores)
        
        if total_score <= 0:
            return self._conservative_overflow_handling(available_budget, priority_scores)
        
        # 按优先级比例分配
        budgets = []
        for score in scores:
            budget = int(score * available_budget / total_score)
            budgets.append(max(1, budget))  # 至少分配1
        
        # 调整总和
        current_total = sum(budgets)
        diff = available_budget - current_total
        
        if diff > 0:
            # 分配剩余预算给最高优先级
            sorted_indices = np.argsort(scores)[::-1]
            for i in range(min(diff, len(sorted_indices))):
                budgets[sorted_indices[i]] += 1
        elif diff < 0:
            # 削减预算从最低优先级开始
            sorted_indices = np.argsort(scores)
            for i in range(min(-diff, len(sorted_indices))):
                if budgets[sorted_indices[i]] > 1:
                    budgets[sorted_indices[i]] -= 1
        
        return budgets, {
            'overflow': True,
            'strategy': 'aggressive',
            'reduction_method': 'priority_based'
        }
    
    def _adaptive_overflow_handling(
        self,
        layer_idx: int,
        available_budget: int,
        priority_scores: List[float],
        overflow_ratio: float
    ) -> Tuple[List[int], Dict[str, Any]]:
        """自适应溢出处理"""
        # 根据溢出程度选择策略
        if overflow_ratio < 1.2:  # 轻微溢出
            return self._conservative_overflow_handling(available_budget, priority_scores)
        elif overflow_ratio < 2.0:  # 中等溢出
            # 混合策略
            conservative_budgets, _ = self._conservative_overflow_handling(
                available_budget, priority_scores
            )
            aggressive_budgets, _ = self._aggressive_overflow_handling(
                available_budget, priority_scores
            )
            
            # 50-50混合
            mixed_budgets = [
                int(0.5 * c + 0.5 * a)
                for c, a in zip(conservative_budgets, aggressive_budgets)
            ]
            
            # 确保总和正确
            current_total = sum(mixed_budgets)
            diff = available_budget - current_total
            
            if diff != 0:
                # 简单调整：加到第一个或从第一个减
                if diff > 0:
                    mixed_budgets[0] += diff
                else:
                    mixed_budgets[0] = max(1, mixed_budgets[0] + diff)
            
            return mixed_budgets, {
                'overflow': True,
                'strategy': 'adaptive_mixed',
                'overflow_ratio': overflow_ratio
            }
        else:  # 严重溢出
            return self._aggressive_overflow_handling(available_budget, priority_scores)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.overflow_history:
            return {'no_overflow_recorded': True}
        
        overflow_ratios = list(self.overflow_history.values())
        return {
            'total_overflow_events': len(self.overflow_history),
            'avg_overflow_ratio': np.mean(overflow_ratios),
            'max_overflow_ratio': np.max(overflow_ratios),
            'layers_with_overflow': list(self.overflow_history.keys()),
            'config': {
                'overflow_strategy': self.config.overflow_strategy,
                'anchor_coordination': self.config.enable_anchor_coordination
            }
        }