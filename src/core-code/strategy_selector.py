"""
策略选择器模块

核心创新：
1. 基于H/V指标的自适应策略选择
2. 稳健的关键头检测（MAD/TopK/Percentile）
3. EMA历史平滑和多策略融合

这是我们独立开发的创新组件，不依赖原始CAKE/AdaKV代码。
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class DetectionMethod(Enum):
    """关键头检测方法"""
    MAD = "mad"  # Median Absolute Deviation
    TOPK = "topk"  # Top-K selection
    PERCENTILE = "percentile"  # Percentile-based


class AllocationStrategy(Enum):
    """分配策略类型"""
    STANDARD = "standard"  # 标准AdaKV分配
    UNIFORM_GUIDED = "uniform_guided"  # 均匀引导分配
    AGGRESSIVE_ADAPTIVE = "aggressive_adaptive"  # 激进自适应分配
    HIGHLY_ADAPTIVE = "highly_adaptive"  # 高度自适应分配


@dataclass
class StrategyConfig:
    """策略配置"""
    # 关键头检测配置
    key_head_method: DetectionMethod = DetectionMethod.MAD
    key_head_ratio: float = 0.2  # 关键头比例
    ema_decay: float = 0.9  # EMA衰减率
    
    # 策略选择阈值
    high_dispersion_threshold: float = 0.7  # H指标高分散阈值
    high_dynamics_threshold: float = 0.5   # V指标高动态阈值
    
    # 分配参数
    min_budget_ratio: float = 0.02  # 最小预算比例
    key_head_priority: float = 0.6  # 关键头优先级基础值
    
    # 稳健性配置
    enable_fallback: bool = True
    mad_sigma_factor: float = 2.0  # MAD的sigma倍数


class RobustKeyHeadDetector:
    """
    稳健的关键头检测器
    
    核心创新：
    - MAD替代标准差：对重尾分布更稳健
    - EMA历史平滑：减少单步波动影响
    - 多策略可选：MAD/TopK/Percentile三种方法
    - 自适应阈值：根据数据分布动态调整
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.head_importance_history: Dict[int, np.ndarray] = {}  # 层级 -> EMA重要性
        
    def detect_key_heads(
        self, 
        concentration_scores: Union[List[float], np.ndarray], 
        layer_idx: int,
        method: Optional[DetectionMethod] = None
    ) -> np.ndarray:
        """
        检测关键头
        
        Args:
            concentration_scores: 集中度评分
            layer_idx: 层索引
            method: 检测方法，None时使用配置中的方法
            
        Returns:
            key_head_mask: bool数组，True表示关键头
        """
        method = method or self.config.key_head_method
        
        # 转换为numpy数组
        if isinstance(concentration_scores, list):
            scores = np.array(concentration_scores, dtype=np.float32)
        else:
            scores = concentration_scores.astype(np.float32)
        
        # 处理无效值
        if len(scores) == 0 or not np.any(np.isfinite(scores)):
            return np.zeros(len(scores), dtype=bool)
        
        # 用有限值的中位数填充无效值
        finite_mask = np.isfinite(scores)
        if not np.all(finite_mask):
            median_val = np.median(scores[finite_mask]) if np.any(finite_mask) else 0.0
            scores[~finite_mask] = median_val
        
        # EMA平滑历史重要性
        if layer_idx in self.head_importance_history:
            ema_scores = (self.config.ema_decay * self.head_importance_history[layer_idx] + 
                         (1 - self.config.ema_decay) * scores)
        else:
            ema_scores = scores.copy()
        
        self.head_importance_history[layer_idx] = ema_scores
        
        # 根据方法检测关键头
        try:
            if method == DetectionMethod.MAD:
                return self._detect_by_mad(ema_scores)
            elif method == DetectionMethod.TOPK:
                return self._detect_by_topk(ema_scores)
            elif method == DetectionMethod.PERCENTILE:
                return self._detect_by_percentile(ema_scores)
            else:
                raise ValueError(f"未知的检测方法: {method}")
        except Exception as e:
            if self.config.enable_fallback:
                # 回退到TopK方法
                return self._detect_by_topk(ema_scores)
            else:
                raise e
    
    def _detect_by_mad(self, scores: np.ndarray) -> np.ndarray:
        """基于MAD的检测"""
        if len(scores) < 3:
            # 样本太少，使用TopK
            return self._detect_by_topk(scores)
        
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        
        if mad < 1e-8:
            # MAD太小，说明分布很集中，使用TopK
            return self._detect_by_topk(scores)
        
        threshold = median + self.config.mad_sigma_factor * mad
        return scores > threshold
    
    def _detect_by_topk(self, scores: np.ndarray) -> np.ndarray:
        """基于Top-K的检测"""
        k = max(1, int(len(scores) * self.config.key_head_ratio))
        k = min(k, len(scores))  # 确保不超过总数
        
        if k >= len(scores):
            return np.ones(len(scores), dtype=bool)
        
        # 使用argpartition提高效率
        top_k_indices = np.argpartition(scores, -k)[-k:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_k_indices] = True
        return mask
    
    def _detect_by_percentile(self, scores: np.ndarray) -> np.ndarray:
        """基于百分位数的检测"""
        if len(scores) < 2:
            return np.ones(len(scores), dtype=bool)
        
        percentile = (1 - self.config.key_head_ratio) * 100
        try:
            threshold = np.percentile(scores, percentile)
            return scores > threshold
        except:
            # 回退到TopK
            return self._detect_by_topk(scores)
    
    def get_detection_stats(self, layer_idx: int) -> Dict:
        """获取检测统计信息"""
        if layer_idx not in self.head_importance_history:
            return {}
        
        scores = self.head_importance_history[layer_idx]
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'mad': float(np.median(np.abs(scores - np.median(scores)))),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }


class StrategySelector:
    """
    策略选择器
    
    核心创新：
    - H/V指标指导策略选择：基于层级特征自动选择分配策略
    - 多模式分配：4种不同的分配模式适应不同场景
    - 平滑策略切换：避免策略跳变导致的性能波动
    - 自适应参数调整：根据指标动态调整分配参数
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.strategy_history: Dict[int, AllocationStrategy] = {}  # 层级策略历史
        
    def select_strategy(
        self, 
        h_normalized: float, 
        v_normalized: float,
        layer_idx: int,
        smooth_transition: bool = True
    ) -> Tuple[AllocationStrategy, Dict]:
        """
        基于H/V指标选择分配策略
        
        Args:
            h_normalized: 归一化的H指标
            v_normalized: 归一化的V指标
            layer_idx: 层索引
            smooth_transition: 是否启用平滑过渡
            
        Returns:
            (strategy, params): 选择的策略和参数
        """
        # 输入验证和清理
        h_norm = np.clip(float(h_normalized), 0.0, 1.0)
        v_norm = np.clip(float(v_normalized), 0.0, 1.0)
        
        # 策略选择逻辑
        if h_norm > self.config.high_dispersion_threshold and v_norm > self.config.high_dynamics_threshold:
            # 高分散+高动态：需要精细的头级差异化
            strategy = AllocationStrategy.HIGHLY_ADAPTIVE
            params = self._get_highly_adaptive_params(h_norm, v_norm)
            
        elif h_norm > self.config.high_dispersion_threshold:
            # 高分散：偏向均匀但保留微调
            strategy = AllocationStrategy.UNIFORM_GUIDED
            params = self._get_uniform_guided_params(h_norm, v_norm)
            
        elif v_norm > self.config.high_dynamics_threshold:
            # 高动态：激进的自适应分配
            strategy = AllocationStrategy.AGGRESSIVE_ADAPTIVE
            params = self._get_aggressive_adaptive_params(h_norm, v_norm)
            
        else:
            # 低分散+低动态：标准AdaKV分配
            strategy = AllocationStrategy.STANDARD
            params = self._get_standard_params(h_norm, v_norm)
        
        # 平滑过渡处理
        if smooth_transition and layer_idx in self.strategy_history:
            previous_strategy = self.strategy_history[layer_idx]
            if previous_strategy != strategy:
                # 策略发生变化，进行平滑处理
                params = self._smooth_strategy_transition(
                    previous_strategy, strategy, params, h_norm, v_norm
                )
        
        # 更新历史
        self.strategy_history[layer_idx] = strategy
        
        return strategy, params
    
    def _get_standard_params(self, h_norm: float, v_norm: float) -> Dict:
        """标准分配参数"""
        return {
            'allocation_mode': 'concentration_based',
            'sharpness_factor': 1.0,
            'uniformity_weight': 0.0,
            'min_budget_ratio': self.config.min_budget_ratio
        }
    
    def _get_uniform_guided_params(self, h_norm: float, v_norm: float) -> Dict:
        """均匀引导分配参数"""
        # H越大，越倾向于均匀分配
        uniformity_weight = h_norm * 0.8
        adjustment_strength = 1.0 - h_norm
        
        return {
            'allocation_mode': 'uniform_guided',
            'uniformity_weight': uniformity_weight,
            'adjustment_strength': adjustment_strength,
            'adjustment_ratio': 0.2,
            'min_budget_ratio': self.config.min_budget_ratio
        }
    
    def _get_aggressive_adaptive_params(self, h_norm: float, v_norm: float) -> Dict:
        """激进自适应分配参数"""
        # V越大，分配差异越大
        sharpness_factor = 1.0 + 2.0 * v_norm
        
        return {
            'allocation_mode': 'aggressive_adaptive',
            'sharpness_factor': sharpness_factor,
            'concentration_power': sharpness_factor,
            'min_budget_ratio': self.config.min_budget_ratio
        }
    
    def _get_highly_adaptive_params(self, h_norm: float, v_norm: float) -> Dict:
        """高度自适应分配参数"""
        # 关键头优先级随V指标增加
        key_head_priority = self.config.key_head_priority + 0.2 * v_norm
        
        # 两级分配的预算比例
        key_budget_ratio = np.clip(key_head_priority, 0.5, 0.8)
        
        return {
            'allocation_mode': 'highly_adaptive',
            'key_budget_ratio': key_budget_ratio,
            'use_key_head_detection': True,
            'key_head_method': self.config.key_head_method,
            'normal_head_uniformity': 0.8,  # 普通头的均匀程度
            'min_budget_ratio': self.config.min_budget_ratio
        }
    
    def _smooth_strategy_transition(
        self,
        prev_strategy: AllocationStrategy,
        new_strategy: AllocationStrategy,
        new_params: Dict,
        h_norm: float,
        v_norm: float
    ) -> Dict:
        """
        平滑策略过渡
        
        在策略切换时，通过参数插值减少突变
        """
        # 简化的平滑处理：调整关键参数的激进程度
        smoothed_params = new_params.copy()
        
        # 如果从保守策略切换到激进策略，适当降低激进程度
        if (prev_strategy in [AllocationStrategy.STANDARD, AllocationStrategy.UNIFORM_GUIDED] and
            new_strategy in [AllocationStrategy.AGGRESSIVE_ADAPTIVE, AllocationStrategy.HIGHLY_ADAPTIVE]):
            
            if 'sharpness_factor' in smoothed_params:
                smoothed_params['sharpness_factor'] *= 0.8  # 降低20%的激进程度
            
            if 'key_budget_ratio' in smoothed_params:
                smoothed_params['key_budget_ratio'] = (
                    smoothed_params['key_budget_ratio'] * 0.7 + 0.6 * 0.3
                )  # 向保守值插值
        
        return smoothed_params
    
    def get_strategy_distribution(self) -> Dict[AllocationStrategy, int]:
        """获取策略分布统计"""
        if not self.strategy_history:
            return {}
        
        distribution = {}
        for strategy in self.strategy_history.values():
            distribution[strategy] = distribution.get(strategy, 0) + 1
        
        return distribution
    
    def reset_history(self) -> None:
        """重置策略历史"""
        self.strategy_history.clear()


def create_strategy_selector(
    key_head_method: str = "mad",
    key_head_ratio: float = 0.2,
    high_h_threshold: float = 0.7,
    high_v_threshold: float = 0.5
) -> StrategySelector:
    """
    创建策略选择器的便捷函数
    
    Args:
        key_head_method: 关键头检测方法
        key_head_ratio: 关键头比例
        high_h_threshold: H指标高值阈值
        high_v_threshold: V指标高值阈值
        
    Returns:
        配置好的策略选择器
    """
    config = StrategyConfig(
        key_head_method=DetectionMethod(key_head_method),
        key_head_ratio=key_head_ratio,
        high_dispersion_threshold=high_h_threshold,
        high_dynamics_threshold=high_v_threshold
    )
    
    return StrategySelector(config)