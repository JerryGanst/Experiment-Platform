"""
指标归一化器模块

核心创新：
1. H/V指标的滑动分位数归一化
2. 严格的预算守恒机制
3. 极值处理和EMA平滑

这是我们独立开发的创新组件，不依赖原始CAKE/AdaKV代码。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """归一化配置"""
    ema_decay: float = 0.9  # EMA衰减率
    winsorize_ratio: float = 0.02  # 极值截断比例
    min_samples: int = 10  # 最小样本数
    enable_fallback: bool = True  # 启用回退机制


class IndicatorNormalizer:
    """
    H/V指标归一化器
    
    核心创新：
    - 滑动分位数归一化：使用p05-p95而非min-max，更稳健
    - 极值截断（Winsorize）：防止异常值污染统计
    - EMA自适应：新域自动适配，无需重新标定
    - 跨模型一致性：归一化后的指标具有可比性
    """
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
        
        # H指标统计
        self.h_stats = {
            'p05': float('inf'), 
            'p95': float('-inf'),
            'count': 0
        }
        
        # V指标统计  
        self.v_stats = {
            'p05': float('inf'),
            'p95': float('-inf'), 
            'count': 0
        }
        
        self.initialized = False
        self.update_count = 0
        
    def update_stats(self, h_values: List[float], v_values: List[float]) -> None:
        """
        更新H/V统计信息
        
        Args:
            h_values: H指标值列表
            v_values: V指标值列表
        """
        if not h_values or not v_values:
            return
            
        h_array = np.array(h_values, dtype=np.float32)
        v_array = np.array(v_values, dtype=np.float32)
        
        # 过滤无效值
        h_valid = h_array[np.isfinite(h_array)]
        v_valid = v_array[np.isfinite(v_array)]
        
        if len(h_valid) == 0 or len(v_valid) == 0:
            return
        
        # Winsorize处理极值
        h_winsorized = self._winsorize(h_valid)
        v_winsorized = self._winsorize(v_valid)
        
        # 计算分位数
        h_p05, h_p95 = np.percentile(h_winsorized, [5, 95])
        v_p05, v_p95 = np.percentile(v_winsorized, [5, 95])
        
        if not self.initialized:
            # 首次初始化
            self.h_stats = {'p05': h_p05, 'p95': h_p95, 'count': len(h_valid)}
            self.v_stats = {'p05': v_p05, 'p95': v_p95, 'count': len(v_valid)}
            self.initialized = True
        else:
            # EMA更新
            decay = self.config.ema_decay
            self.h_stats['p05'] = decay * self.h_stats['p05'] + (1-decay) * h_p05
            self.h_stats['p95'] = decay * self.h_stats['p95'] + (1-decay) * h_p95
            self.h_stats['count'] += len(h_valid)
            
            self.v_stats['p05'] = decay * self.v_stats['p05'] + (1-decay) * v_p05
            self.v_stats['p95'] = decay * self.v_stats['p95'] + (1-decay) * v_p95
            self.v_stats['count'] += len(v_valid)
        
        self.update_count += 1
    
    def _winsorize(self, values: np.ndarray) -> np.ndarray:
        """
        极值截断处理
        
        Args:
            values: 输入数值数组
            
        Returns:
            截断后的数组
        """
        if len(values) < self.config.min_samples:
            return values
            
        p_low = self.config.winsorize_ratio * 100
        p_high = (1 - self.config.winsorize_ratio) * 100
        
        try:
            low, high = np.percentile(values, [p_low, p_high])
            return np.clip(values, low, high)
        except:
            # 回退：返回原值
            return values
    
    def normalize(self, h_value: float, v_value: float) -> Tuple[float, float]:
        """
        归一化H/V值到[0,1]区间
        
        Args:
            h_value: H指标原始值
            v_value: V指标原始值
            
        Returns:
            (h_normalized, v_normalized): 归一化后的值
        """
        if not self.initialized:
            if self.config.enable_fallback:
                # 未初始化时的回退策略：返回中性值
                return 0.5, 0.5
            else:
                return h_value, v_value
        
        # 处理无效输入
        if not np.isfinite(h_value) or not np.isfinite(v_value):
            return 0.5, 0.5
            
        # Min-max归一化到[0,1]
        h_range = self.h_stats['p95'] - self.h_stats['p05'] + 1e-8
        v_range = self.v_stats['p95'] - self.v_stats['p05'] + 1e-8
        
        h_norm = (h_value - self.h_stats['p05']) / h_range
        v_norm = (v_value - self.v_stats['p05']) / v_range
        
        # 确保在[0,1]范围内
        h_norm = np.clip(h_norm, 0.0, 1.0)
        v_norm = np.clip(v_norm, 0.0, 1.0)
        
        return float(h_norm), float(v_norm)
    
    def get_stats_summary(self) -> dict:
        """获取统计信息摘要"""
        return {
            'initialized': self.initialized,
            'update_count': self.update_count,
            'h_stats': self.h_stats.copy(),
            'v_stats': self.v_stats.copy()
        }
    
    def reset(self) -> None:
        """重置归一化器状态"""
        self.h_stats = {'p05': float('inf'), 'p95': float('-inf'), 'count': 0}
        self.v_stats = {'p05': float('inf'), 'p95': float('-inf'), 'count': 0}
        self.initialized = False
        self.update_count = 0


class BudgetNormalizer:
    """
    严格的预算守恒器
    
    核心创新：
    - 数学严格性：保证 Σ(head_budgets) = layer_budget
    - 智能分配：按小数部分优先级分配剩余预算
    - 超额处理：从非关键位置智能回收超额预算
    - 防御性编程：完整的错误检查和异常处理
    """
    
    @staticmethod
    def normalize_to_budget(
        raw_budgets: List[int], 
        total_budget: int,
        min_budget: int = 1,
        validate: bool = True
    ) -> List[int]:
        """
        严格保证预算守恒的归一化
        
        Args:
            raw_budgets: 原始预算分配列表
            total_budget: 总预算
            min_budget: 最小预算（每个头至少分配的token数）
            validate: 是否进行严格验证
            
        Returns:
            归一化后的预算分配，保证和等于total_budget
            
        Raises:
            ValueError: 当输入参数无效时
        """
        # 输入验证
        if not raw_budgets:
            return []
        
        if total_budget <= 0:
            raise ValueError(f"总预算必须为正数，得到: {total_budget}")
        
        if min_budget < 0:
            raise ValueError(f"最小预算不能为负数，得到: {min_budget}")
        
        num_heads = len(raw_budgets)
        if min_budget * num_heads > total_budget:
            raise ValueError(
                f"最小预算约束无法满足: {min_budget} × {num_heads} = "
                f"{min_budget * num_heads} > {total_budget}"
            )
        
        # 转为numpy数组便于操作
        try:
            raw_array = np.array(raw_budgets, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法转换预算列表为数组: {e}")
        
        # 确保所有值≥最小预算
        raw_array = np.maximum(raw_array, min_budget)
        
        # 检查是否已经满足约束
        if raw_array.sum() == total_budget:
            return raw_array.astype(int).tolist()
        
        # 比例缩放
        current_sum = raw_array.sum()
        if current_sum <= 0:
            # 极端情况：均匀分配
            base_budget = total_budget // num_heads
            remainder = total_budget % num_heads
            result = [base_budget] * num_heads
            for i in range(remainder):
                result[i] += 1
            return result
        
        scale_factor = total_budget / current_sum
        scaled = raw_array * scale_factor
        
        # 向下取整
        rounded = np.floor(scaled).astype(int)
        
        # 确保最小预算约束
        rounded = np.maximum(rounded, min_budget)
        
        # 计算差额
        deficit = total_budget - rounded.sum()
        
        if deficit > 0:
            # 预算不足：按小数部分从大到小分配剩余预算
            fractional_parts = scaled - np.floor(scaled)
            # 找到可以增加预算的位置（避免超出合理范围）
            candidates = np.arange(len(rounded))
            # 按小数部分排序
            sorted_indices = candidates[np.argsort(-fractional_parts)]
            
            # 分配剩余预算
            for i in range(min(deficit, len(sorted_indices))):
                rounded[sorted_indices[i]] += 1
                
        elif deficit < 0:
            # 预算超额：从非最小预算位置回收
            excess = -deficit
            # 找到可以减少预算的位置
            candidates = np.where(rounded > min_budget)[0]
            
            if len(candidates) >= excess:
                # 优先从预算较大的地方回收
                sorted_candidates = candidates[np.argsort(-rounded[candidates])]
                for i in range(excess):
                    if rounded[sorted_candidates[i]] > min_budget:
                        rounded[sorted_candidates[i]] -= 1
            else:
                # 无法完全回收，尽力而为
                for candidate in candidates:
                    if excess <= 0:
                        break
                    if rounded[candidate] > min_budget:
                        rounded[candidate] -= 1
                        excess -= 1
        
        # 最终验证
        final_sum = rounded.sum()
        if validate and final_sum != total_budget:
            # 最后的补偿机制（确定性实现，避免随机性）
            diff = total_budget - final_sum
            if diff == 0:
                pass  # 无需调整
            elif diff > 0:
                # 预算仍有剩余：按当前预算从小到大依次分配，保证确定性
                sorted_indices = np.argsort(rounded)  # 较小的预算优先获得补偿
                for i in range(diff):
                    rounded[sorted_indices[i % num_heads]] += 1
            else:  # diff < 0
                # 预算仍然超出：按当前预算从大到小依次回收，保证确定性
                excess = -diff
                candidates = np.where(rounded > min_budget)[0]
                if len(candidates) == 0:
                    # 所有头都已达到最小预算仍超额，无法处理
                    raise RuntimeError(
                        "预算守恒失败且无法回收预算：所有头均已达到最小预算"
                    )
                sorted_indices = candidates[np.argsort(-rounded[candidates])]  # 较大的预算优先被回收

                # 逐个头循环，直到成功回收全部超额预算
                for idx in sorted_indices:
                    if excess <= 0:
                        break
                    # 当前头最多可以回收的额度
                    reducible = min(rounded[idx] - min_budget, excess)
                    if reducible <= 0:
                        continue  # 已无法再回收
                    rounded[idx] -= reducible
                    excess -= reducible

                # 如果仍有剩余超额，说明无法在不违反最小预算约束的情况下完成回收
                if excess > 0:
                    raise RuntimeError(
                        "预算守恒失败且无法回收预算：剩余超额 {}，所有可回收预算已用尽".format(excess)
                    )
            
            # 检查最小预算约束
            if np.any(rounded < min_budget):
                min_actual = rounded.min()
                raise RuntimeError(
                    f"最小预算约束违反: 期望≥{min_budget}, 实际最小={min_actual}"
                )
        
        return rounded.tolist()
    
    @staticmethod
    def validate_budget_allocation(budgets: List[int], total_budget: int, min_budget: int = 1) -> bool:
        """
        验证预算分配的正确性
        
        Args:
            budgets: 预算分配
            total_budget: 总预算
            min_budget: 最小预算
            
        Returns:
            是否通过验证
        """
        try:
            # 检查和
            if sum(budgets) != total_budget:
                return False
            
            # 检查最小约束
            if any(b < min_budget for b in budgets):
                return False
            
            # 检查非负
            if any(b < 0 for b in budgets):
                return False
                
            return True
        except:
            return False