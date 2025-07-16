"""
统一分配器模块

这是我们的核心创新：CAKE-AdaKV统一集成框架
独立于原始CAKE和AdaKV代码，提供完整的层级-头级协同优化。

核心创新：
1. 统一warmup机制：单次前向传播收集两种算法所需信息
2. 智能策略选择：基于H/V指标自动选择最优分配策略
3. 严格预算守恒：数学保证层级预算完全分配到头级
4. 稳健异常处理：完整的边界情况和错误恢复机制
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings

# 导入我们的核心组件
try:
    from indicator_normalizer import IndicatorNormalizer, BudgetNormalizer, NormalizationConfig
    from strategy_selector import StrategySelector, RobustKeyHeadDetector, AllocationStrategy, StrategyConfig
except ImportError:
    from .indicator_normalizer import IndicatorNormalizer, BudgetNormalizer, NormalizationConfig
    from .strategy_selector import StrategySelector, RobustKeyHeadDetector, AllocationStrategy, StrategyConfig


@dataclass 
class UnifiedCacheConfig:
    """统一缓存配置"""
    # CAKE相关配置
    total_cache_size: int = 4096  # 总缓存大小
    tau1: float = 0.1  # H指标温度参数
    tau2: float = 0.1  # V指标温度参数
    
    # AdaKV相关配置  
    window_size: int = 32  # 窗口大小
    base_capacity: int = 512  # 基础容量
    kernel_size: int = 7  # 卷积核大小
    floor_alpha: float = 0.5  # 下限系数
    beta: int = 20  # AdaKV的beta参数
    
    # 统一框架配置
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)
    
    # 系统配置
    enable_fallback: bool = True  # 启用回退机制
    strict_budget_conservation: bool = True  # 严格预算守恒
    enable_monitoring: bool = True  # 启用监控
    
    # 性能配置
    batch_processing: bool = True  # 批处理模式
    memory_efficient: bool = True  # 内存高效模式
    
    # 新增：V指标计算方式，可选 'var' | 'scaled_var' | 'std' | 'entropy'
    v_metric: str = "var"  # V指标计算方式


class UnifiedWarmupManager:
    """
    统一预热管理器
    
    核心创新：单次前向传播同时收集CAKE和AdaKV所需的所有信息
    避免重复计算，提高效率
    """
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.warmup_completed = False
        self.layer_statistics = {}  # 层级统计信息
        
    def collect_layer_info(
        self, 
        attention_weights: np.ndarray,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        收集层级信息（用于CAKE算法）
        
        Args:
            attention_weights: 注意力权重 [batch, heads, seq_len, seq_len]
            layer_idx: 层索引
            head_idx: 头索引（可选）
            
        Returns:
            包含H/V指标的字典
        """
        try:
            if attention_weights.ndim != 4:
                raise ValueError(f"注意力权重维度应为4，得到: {attention_weights.ndim}")
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # 计算H指标（空间注意力分散度）
            h_values = []
            for b in range(batch_size):
                for h in range(num_heads):
                    # 计算每个位置的注意力分布熵
                    attn_dist = attention_weights[b, h]  # [seq_len, seq_len]
                    
                    # 避免log(0)
                    attn_dist = np.clip(attn_dist, 1e-8, 1.0)
                    
                    # 计算熵
                    entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-8), axis=-1)
                    h_value = np.mean(entropy)
                    h_values.append(h_value)
            
            # 计算V指标（时间注意力变化）
            v_values = []
            for b in range(batch_size):
                for h in range(num_heads):
                    attn_seq = attention_weights[b, h]  # [seq_len, seq_len]
                    
                    # 根据配置选择不同的V指标计算方式
                    if self.config.v_metric == "std":
                        # 标准差版本（对方差开根号）
                        temporal_measure = np.std(attn_seq, axis=-2)
                    elif self.config.v_metric == "entropy":
                        # 信息熵版本：衡量注意力随时间的分散程度
                        # 先添加小的epsilon避免log(0)，然后归一化成有效的概率分布
                        attn_safe = attn_seq + 1e-8
                        attn_normalized = attn_safe / np.sum(attn_safe, axis=-2, keepdims=True)
                        temporal_measure = -np.sum(attn_normalized * np.log(attn_normalized), axis=-2)

                    elif self.config.v_metric == "scaled_var":
                        # 放大版方差：乘以序列长度，缓解过小的问题
                        temporal_measure = np.var(attn_seq, axis=-2) * seq_len
                    else:  # 默认或 'var'
                        temporal_measure = np.var(attn_seq, axis=-2)
                    
                    v_value = np.mean(temporal_measure)
                    v_values.append(v_value)
            
            layer_info = {
                'h_indicator': np.mean(h_values),
                'v_indicator': np.mean(v_values),
                'h_values': h_values,
                'v_values': v_values,
                'num_heads': num_heads,
                'seq_len': seq_len,
                'layer_idx': layer_idx
            }
            
            # 存储层级统计
            self.layer_statistics[layer_idx] = layer_info
            
            return layer_info
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"层{layer_idx}信息收集失败，使用默认值: {e}")
                return {
                    'h_indicator': 0.5,
                    'v_indicator': 0.5,
                    'h_values': [0.5],
                    'v_values': [0.5],
                    'num_heads': attention_weights.shape[1] if attention_weights.ndim >= 2 else 1,
                    'seq_len': attention_weights.shape[-1] if attention_weights.ndim >= 1 else 1,
                    'layer_idx': layer_idx
                }
            else:
                raise e
    
    def collect_head_info(
        self,
        attention_weights: np.ndarray,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        收集头级信息（用于AdaKV算法）
        
        Args:
            attention_weights: 注意力权重
            layer_idx: 层索引
            
        Returns:
            包含集中度评分的字典
        """
        try:
            if attention_weights.ndim != 4:
                raise ValueError(f"注意力权重维度应为4，得到: {attention_weights.ndim}")
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # 计算每个头的注意力集中度
            concentration_scores = []
            
            for h in range(num_heads):
                head_scores = []
                for b in range(batch_size):
                    attn_matrix = attention_weights[b, h]  # [seq_len, seq_len]
                    
                    # 计算注意力集中度（使用方差作为集中度的逆指标）
                    # 方差越小，注意力越集中
                    attention_variance = np.var(attn_matrix, axis=-1)  # 每行的方差
                    concentration = 1.0 / (1.0 + np.mean(attention_variance))  # 转换为集中度
                    head_scores.append(concentration)
                
                # 头级平均集中度
                avg_concentration = np.mean(head_scores)
                concentration_scores.append(avg_concentration)
            
            head_info = {
                'concentration_scores': concentration_scores,
                'num_heads': num_heads,
                'layer_idx': layer_idx,
                'avg_concentration': np.mean(concentration_scores)
            }
            
            return head_info
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"层{layer_idx}头级信息收集失败，使用默认值: {e}")
                num_heads = attention_weights.shape[1] if attention_weights.ndim >= 2 else 1
                return {
                    'concentration_scores': [0.5] * num_heads,
                    'num_heads': num_heads,
                    'layer_idx': layer_idx,
                    'avg_concentration': 0.5
                }
            else:
                raise e
    
    def unified_warmup(
        self, 
        attention_weights_list: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        统一预热：单次遍历收集所有必要信息
        
        Args:
            attention_weights_list: 所有层的注意力权重列表
            
        Returns:
            完整的预热信息
        """
        warmup_info = {
            'layer_info': {},
            'head_info': {},
            'global_stats': {}
        }
        
        all_h_values = []
        all_v_values = []
        
        try:
            for layer_idx, attention_weights in enumerate(attention_weights_list):
                # 同时收集层级和头级信息
                layer_info = self.collect_layer_info(attention_weights, layer_idx)
                head_info = self.collect_head_info(attention_weights, layer_idx)
                
                warmup_info['layer_info'][layer_idx] = layer_info
                warmup_info['head_info'][layer_idx] = head_info
                
                # 累积全局统计
                all_h_values.extend(layer_info['h_values'])
                all_v_values.extend(layer_info['v_values'])
            
            # 计算全局统计
            warmup_info['global_stats'] = {
                'total_layers': len(attention_weights_list),
                'avg_h_indicator': np.mean(all_h_values) if all_h_values else 0.5,
                'avg_v_indicator': np.mean(all_v_values) if all_v_values else 0.5,
                'h_std': np.std(all_h_values) if len(all_h_values) > 1 else 0.0,
                'v_std': np.std(all_v_values) if len(all_v_values) > 1 else 0.0
            }
            
            self.warmup_completed = True
            return warmup_info
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"统一预热失败，使用默认配置: {e}")
                # 返回默认配置
                num_layers = len(attention_weights_list)
                for i in range(num_layers):
                    warmup_info['layer_info'][i] = {
                        'h_indicator': 0.5, 'v_indicator': 0.5,
                        'h_values': [0.5], 'v_values': [0.5],
                        'num_heads': 32, 'seq_len': 512, 'layer_idx': i
                    }
                    warmup_info['head_info'][i] = {
                        'concentration_scores': [0.5] * 32,
                        'num_heads': 32, 'layer_idx': i, 'avg_concentration': 0.5
                    }
                
                warmup_info['global_stats'] = {
                    'total_layers': num_layers,
                    'avg_h_indicator': 0.5, 'avg_v_indicator': 0.5,
                    'h_std': 0.0, 'v_std': 0.0
                }
                self.warmup_completed = True
                return warmup_info
            else:
                raise e


class UnifiedCakeAdaKVAllocator:
    """
    CAKE-AdaKV统一分配器
    
    这是我们的核心创新：将CAKE的层级优化与AdaKV的头级优化
    无缝集成，实现端到端的层级-头级协同优化。
    
    主要创新点：
    1. 统一warmup：避免重复计算
    2. 指标归一化：确保跨域一致性  
    3. 智能策略选择：基于H/V指标自动选择最优策略
    4. 严格预算守恒：数学保证预算完全分配
    5. 稳健异常处理：完整的边界情况处理
    """
    
    def __init__(self, config: UnifiedCacheConfig = None):
        self.config = config or UnifiedCacheConfig()
        
        # 初始化核心组件
        self.indicator_normalizer = IndicatorNormalizer(self.config.normalization_config)
        self.strategy_selector = StrategySelector(self.config.strategy_config)
        self.key_head_detector = RobustKeyHeadDetector(self.config.strategy_config)
        self.warmup_manager = UnifiedWarmupManager(self.config)
        
        # 状态管理
        self.is_warmed_up = False
        self.allocation_history = {}  # 分配历史
        self.performance_stats = {}  # 性能统计
        
    def warmup(self, attention_weights_list: List[np.ndarray]) -> None:
        """
        预热阶段：收集必要的统计信息
        
        Args:
            attention_weights_list: 所有层的注意力权重
        """
        try:
            # 统一预热
            warmup_info = self.warmup_manager.unified_warmup(attention_weights_list)
            
            # 提取H/V指标用于归一化器初始化
            all_h_values = []
            all_v_values = []
            
            for layer_info in warmup_info['layer_info'].values():
                all_h_values.extend(layer_info['h_values'])
                all_v_values.extend(layer_info['v_values'])
            
            # 更新指标归一化器
            if all_h_values and all_v_values:
                self.indicator_normalizer.update_stats(all_h_values, all_v_values)
            
            self.is_warmed_up = True
            
            if self.config.enable_monitoring:
                print(f"预热完成: {len(attention_weights_list)}层, "
                      f"平均H={warmup_info['global_stats']['avg_h_indicator']:.3f}, "
                      f"平均V={warmup_info['global_stats']['avg_v_indicator']:.3f}")
                
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"预热失败，使用默认配置: {e}")
                self.is_warmed_up = True  # 标记为已预热，使用默认值
            else:
                raise e
    
    def allocate_layer_budgets(
        self, 
        attention_weights_list: List[np.ndarray]
    ) -> List[int]:
        """
        CAKE算法：分配层级预算
        
        Args:
            attention_weights_list: 所有层的注意力权重
            
        Returns:
            每层的预算分配
        """
        try:
            if not self.is_warmed_up:
                self.warmup(attention_weights_list)
            
            num_layers = len(attention_weights_list)
            layer_preferences = []
            
            # 计算每层的偏好评分
            for layer_idx, attention_weights in enumerate(attention_weights_list):
                layer_info = self.warmup_manager.collect_layer_info(attention_weights, layer_idx)
                
                h_indicator = layer_info['h_indicator']
                v_indicator = layer_info['v_indicator']
                
                # CAKE偏好评分计算
                pref_score = (h_indicator ** (1/self.config.tau1)) * (v_indicator ** (1/self.config.tau2))
                layer_preferences.append(pref_score)
            
            # 分配层级预算
            total_pref = sum(layer_preferences)
            if total_pref <= 0:
                # 均匀分配
                base_budget = self.config.total_cache_size // num_layers
                remainder = self.config.total_cache_size % num_layers
                layer_budgets = [base_budget] * num_layers
                for i in range(remainder):
                    layer_budgets[i] += 1
            else:
                # 按偏好比例分配
                raw_budgets = [
                    int(pref * self.config.total_cache_size / total_pref)
                    for pref in layer_preferences
                ]
                
                # 严格预算守恒
                if self.config.strict_budget_conservation:
                    layer_budgets = BudgetNormalizer.normalize_to_budget(
                        raw_budgets, self.config.total_cache_size
                    )
                else:
                    layer_budgets = raw_budgets
            
            return layer_budgets
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"层级预算分配失败，使用均匀分配: {e}")
                base_budget = self.config.total_cache_size // len(attention_weights_list)
                remainder = self.config.total_cache_size % len(attention_weights_list)
                layer_budgets = [base_budget] * len(attention_weights_list)
                for i in range(remainder):
                    layer_budgets[i] += 1
                return layer_budgets
            else:
                raise e
    
    def allocate_head_budgets(
        self,
        attention_weights: np.ndarray,
        layer_budget: int,
        layer_idx: int
    ) -> List[int]:
        """
        AdaKV算法：分配头级预算
        
        Args:
            attention_weights: 当前层的注意力权重
            layer_budget: 层级预算
            layer_idx: 层索引
            
        Returns:
            每个头的预算分配
        """
        try:
            # 收集头级信息
            head_info = self.warmup_manager.collect_head_info(attention_weights, layer_idx)
            layer_info = self.warmup_manager.collect_layer_info(attention_weights, layer_idx)
            
            concentration_scores = head_info['concentration_scores']
            num_heads = head_info['num_heads']
            
            # 归一化H/V指标
            h_norm, v_norm = self.indicator_normalizer.normalize(
                layer_info['h_indicator'], 
                layer_info['v_indicator']
            )
            
            # 策略选择
            strategy, strategy_params = self.strategy_selector.select_strategy(
                h_norm, v_norm, layer_idx
            )
            
            # 根据策略分配预算
            head_budgets = self._allocate_by_strategy(
                concentration_scores, layer_budget, strategy, strategy_params, layer_idx
            )
            
            # 记录分配历史
            self.allocation_history[layer_idx] = {
                'strategy': strategy,
                'h_norm': h_norm,
                'v_norm': v_norm,
                'head_budgets': head_budgets,
                'layer_budget': layer_budget
            }
            
            return head_budgets
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"头级预算分配失败，使用均匀分配: {e}")
                num_heads = attention_weights.shape[1] if attention_weights.ndim >= 2 else 1
                base_budget = layer_budget // num_heads
                remainder = layer_budget % num_heads
                head_budgets = [base_budget] * num_heads
                for i in range(remainder):
                    head_budgets[i] += 1
                return head_budgets
            else:
                raise e
    
    def _allocate_by_strategy(
        self,
        concentration_scores: List[float],
        layer_budget: int,
        strategy: AllocationStrategy,
        strategy_params: Dict,
        layer_idx: int
    ) -> List[int]:
        """
        根据策略分配头级预算
        """
        num_heads = len(concentration_scores)
        # 调整最小预算约束：确保总约束不超过层级预算
        min_budget_ratio = strategy_params.get('min_budget_ratio', 0.02)
        theoretical_min_budget = max(1, int(layer_budget * min_budget_ratio))
        
        # 如果理论最小预算会导致约束无法满足，则动态调整
        if theoretical_min_budget * num_heads > layer_budget:
            # 使用能够满足约束的最大最小预算
            min_budget = max(1, layer_budget // num_heads)
            if min_budget * num_heads > layer_budget:
                # 极端情况：层级预算太小，无法为每个头分配至少1个token
                min_budget = 0  # 允许某些头分配0个token
        else:
            min_budget = theoretical_min_budget
        
        if strategy == AllocationStrategy.STANDARD:
            return self._standard_allocation(concentration_scores, layer_budget, min_budget)
        
        elif strategy == AllocationStrategy.UNIFORM_GUIDED:
            return self._uniform_guided_allocation(
                concentration_scores, layer_budget, strategy_params, min_budget
            )
        
        elif strategy == AllocationStrategy.AGGRESSIVE_ADAPTIVE:
            return self._aggressive_adaptive_allocation(
                concentration_scores, layer_budget, strategy_params, min_budget
            )
        
        elif strategy == AllocationStrategy.HIGHLY_ADAPTIVE:
            return self._highly_adaptive_allocation(
                concentration_scores, layer_budget, strategy_params, min_budget, layer_idx
            )
        
        else:
            # 回退到标准分配
            return self._standard_allocation(concentration_scores, layer_budget, min_budget)
    
    def _standard_allocation(
        self, 
        concentration_scores: List[float], 
        layer_budget: int, 
        min_budget: int
    ) -> List[int]:
        """标准集中度分配"""
        scores = np.array(concentration_scores)
        
        # 避免除零
        total_score = np.sum(scores)
        if total_score <= 0:
            # 均匀分配
            base = layer_budget // len(scores)
            remainder = layer_budget % len(scores)
            budgets = [base] * len(scores)
            for i in range(remainder):
                budgets[i] += 1
            return budgets
        
        # 按集中度比例分配
        raw_budgets = [int(score * layer_budget / total_score) for score in scores]
        
        # 预算守恒
        return BudgetNormalizer.normalize_to_budget(raw_budgets, layer_budget, min_budget)
    
    def _uniform_guided_allocation(
        self,
        concentration_scores: List[float],
        layer_budget: int,
        strategy_params: Dict,
        min_budget: int
    ) -> List[int]:
        """均匀引导分配"""
        num_heads = len(concentration_scores)
        uniformity_weight = strategy_params.get('uniformity_weight', 0.5)
        adjustment_strength = strategy_params.get('adjustment_strength', 0.5)
        
        # 基础均匀分配
        uniform_budget = layer_budget // num_heads
        uniform_budgets = [uniform_budget] * num_heads
        
        # 按集中度调整
        scores = np.array(concentration_scores)
        if np.sum(scores) > 0:
            score_weights = scores / np.sum(scores)
            adjustments = (score_weights - 1/num_heads) * layer_budget * adjustment_strength
            
            # 混合均匀和调整
            mixed_budgets = [
                uniform_budget + adj * (1 - uniformity_weight)
                for adj in adjustments
            ]
        else:
            mixed_budgets = uniform_budgets
        
        # 确保非负并取整
        mixed_budgets = [max(min_budget, int(b)) for b in mixed_budgets]
        
        # 预算守恒
        return BudgetNormalizer.normalize_to_budget(mixed_budgets, layer_budget, min_budget)
    
    def _aggressive_adaptive_allocation(
        self,
        concentration_scores: List[float],
        layer_budget: int,
        strategy_params: Dict,
        min_budget: int
    ) -> List[int]:
        """激进自适应分配"""
        sharpness_factor = strategy_params.get('sharpness_factor', 2.0)
        
        scores = np.array(concentration_scores)
        
        # 增强分配差异
        if np.sum(scores) > 0:
            # 使用幂函数增强差异
            enhanced_scores = scores ** sharpness_factor
            total_enhanced = np.sum(enhanced_scores)
            
            if total_enhanced > 0:
                raw_budgets = [
                    int(score * layer_budget / total_enhanced)
                    for score in enhanced_scores
                ]
            else:
                raw_budgets = [layer_budget // len(scores)] * len(scores)
        else:
            raw_budgets = [layer_budget // len(scores)] * len(scores)
        
        # 预算守恒
        return BudgetNormalizer.normalize_to_budget(raw_budgets, layer_budget, min_budget)
    
    def _highly_adaptive_allocation(
        self,
        concentration_scores: List[float],
        layer_budget: int,
        strategy_params: Dict,
        min_budget: int,
        layer_idx: int
    ) -> List[int]:
        """
        高度自适应分配（两级分配）
        - 识别关键头和非关键头
        - 将大部分预算优先分配给关键头
        - 剩余预算在非关键头中分配
        """
        scores = np.array(concentration_scores, dtype=np.float32)
        num_heads = len(scores)

        if num_heads == 0:
            return []

        # 1. 识别关键头
        key_head_mask = self.key_head_detector.detect_key_heads(scores, layer_idx)
        key_head_indices = np.where(key_head_mask)[0]
        non_key_head_indices = np.where(~key_head_mask)[0]
        
        num_key_heads = len(key_head_indices)

        # 边界情况：如果没有或所有都是关键头，退化为激进自适应策略
        if num_key_heads == 0 or num_key_heads == num_heads:
            return self._aggressive_adaptive_allocation(
                scores.tolist(), layer_budget, strategy_params, min_budget
            )

        # 2. 预算两级划分
        key_budget_ratio_base = strategy_params.get('key_budget_ratio', 0.7)
        
        key_total_budget = int(layer_budget * key_budget_ratio_base)
        
        non_key_min_total_budget = len(non_key_head_indices) * min_budget
        key_total_budget = min(key_total_budget, layer_budget - non_key_min_total_budget)
        key_total_budget = max(key_total_budget, num_key_heads * min_budget)

        non_key_total_budget = layer_budget - key_total_budget

        # 3. 分别对两组头进行预算分配
        key_scores = scores[key_head_indices]
        non_key_scores = scores[non_key_head_indices]

        key_budgets_raw = self._standard_allocation(key_scores.tolist(), key_total_budget, min_budget)
        
        non_key_strategy_params = {'sharpness_factor': 0.5}
        non_key_budgets_raw = self._aggressive_adaptive_allocation(
            non_key_scores.tolist(),
            non_key_total_budget,
            non_key_strategy_params,
            min_budget
        )

        # 4. 合并结果
        final_budgets = np.zeros(num_heads, dtype=int)
        final_budgets[key_head_indices] = key_budgets_raw
        final_budgets[non_key_head_indices] = non_key_budgets_raw
        
        # 5. 使用BudgetNormalizer做最终的严格预算守恒
        return BudgetNormalizer.normalize_to_budget(
            final_budgets.tolist(), layer_budget, min_budget
        )

    def unified_allocate(
        self, 
        attention_weights_list: List[np.ndarray]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        统一分配：完整的层级-头级预算分配
        
        Args:
            attention_weights_list: 所有层的注意力权重
            
        Returns:
            (layer_budgets, head_budgets_list): 层级预算和头级预算列表
        """
        try:
            # 1. 层级预算分配（CAKE）
            layer_budgets = self.allocate_layer_budgets(attention_weights_list)
            
            # 2. 头级预算分配（AdaKV）
            head_budgets_list = []
            for layer_idx, (attention_weights, layer_budget) in enumerate(
                zip(attention_weights_list, layer_budgets)
            ):
                head_budgets = self.allocate_head_budgets(
                    attention_weights, layer_budget, layer_idx
                )
                head_budgets_list.append(head_budgets)
            
            # 3. 验证预算守恒
            if self.config.strict_budget_conservation:
                total_allocated = sum(sum(head_budgets) for head_budgets in head_budgets_list)
                if total_allocated != self.config.total_cache_size:
                    warnings.warn(
                        f"预算守恒验证失败: 期望={self.config.total_cache_size}, "
                        f"实际={total_allocated}"
                    )
            
            return layer_budgets, head_budgets_list
            
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"统一分配失败，使用均匀分配: {e}")
                
                # 均匀分配回退
                num_layers = len(attention_weights_list)
                layer_base = self.config.total_cache_size // num_layers
                layer_remainder = self.config.total_cache_size % num_layers
                
                layer_budgets = [layer_base] * num_layers
                for i in range(layer_remainder):
                    layer_budgets[i] += 1
                
                head_budgets_list = []
                for layer_idx, (attention_weights, layer_budget) in enumerate(
                    zip(attention_weights_list, layer_budgets)
                ):
                    num_heads = attention_weights.shape[1] if attention_weights.ndim >= 2 else 1
                    head_base = layer_budget // num_heads
                    head_remainder = layer_budget % num_heads
                    
                    head_budgets = [head_base] * num_heads
                    for i in range(head_remainder):
                        head_budgets[i] += 1
                    
                    head_budgets_list.append(head_budgets)
                
                return layer_budgets, head_budgets_list
            else:
                raise e
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配摘要"""
        if not self.allocation_history:
            return {}
        
        strategy_counts = {}
        for info in self.allocation_history.values():
            strategy = info['strategy']
            strategy_counts[strategy.value] = strategy_counts.get(strategy.value, 0) + 1
        
        return {
            'total_layers': len(self.allocation_history),
            'strategy_distribution': strategy_counts,
            'normalizer_stats': self.indicator_normalizer.get_stats_summary(),
            'is_warmed_up': self.is_warmed_up,
            'config_summary': {
                'total_cache_size': self.config.total_cache_size,
                'strict_budget_conservation': self.config.strict_budget_conservation,
                'enable_fallback': self.config.enable_fallback
            }
        }