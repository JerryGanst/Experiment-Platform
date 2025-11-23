"""
CAKE-AdaKV集成框架

这是我们创新方法的主入口，提供完整的层级-头级协同优化解决方案。
独立于原始CAKE和AdaKV代码，实现端到端的统一优化。

核心特性：
1. 一键式集成：简单的API接口
2. 自动配置：智能参数调优
3. 性能监控：全面的性能分析
4. 可扩展性：支持自定义策略和配置
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import warnings

# 导入我们的核心组件
try:
    from unified_allocator import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig
    from indicator_normalizer import IndicatorNormalizer, BudgetNormalizer
    from strategy_selector import StrategySelector, AllocationStrategy
    from memory_manager import UnifiedMemoryManager, MemoryConfig
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from .unified_allocator import UnifiedCakeAdaKVAllocator, UnifiedCacheConfig
    from .indicator_normalizer import IndicatorNormalizer, BudgetNormalizer
    from .strategy_selector import StrategySelector, AllocationStrategy
    from .memory_manager import UnifiedMemoryManager, MemoryConfig


@dataclass
class IntegrationConfig:
    """集成配置"""
    # 核心配置
    total_cache_size: int = 4096
    enable_monitoring: bool = True
    enable_fallback: bool = True
    
    # 性能配置
    warmup_samples: int = 5  # 预热样本数
    enable_auto_tuning: bool = True  # 自动调优
    performance_tracking: bool = True  # 性能跟踪
    
    # 实验配置
    experiment_mode: bool = False  # 实验模式
    detailed_logging: bool = False  # 详细日志
    
    # 自定义配置
    custom_strategies: Optional[Dict[str, Any]] = None
    custom_thresholds: Optional[Dict[str, float]] = None


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.timing_history = []
        self.allocation_history = []
        
    def start_timing(self, operation: str) -> None:
        """开始计时"""
        if not hasattr(self, '_start_times'):
            self._start_times = {}
        self._start_times[operation] = time.time()
    
    def end_timing(self, operation: str) -> float:
        """结束计时"""
        if not hasattr(self, '_start_times') or operation not in self._start_times:
            return 0.0
        
        duration = time.time() - self._start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
        return duration
    
    def record_allocation(self, allocation_info: Dict[str, Any]) -> None:
        """记录分配信息"""
        self.allocation_history.append({
            'timestamp': time.time(),
            **allocation_info
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_calls': len(times)
                }
        
        return {
            'timing_metrics': summary,
            'total_allocations': len(self.allocation_history),
            'monitoring_active': True
        }


class CakeAdaKVIntegration:
    """
    CAKE-AdaKV集成框架
    
    这是我们创新方法的主要接口，提供完整的层级-头级协同优化。
    
    使用示例：
    ```python
    # 基本使用
    integration = CakeAdaKVIntegration()
    layer_budgets, head_budgets = integration.optimize_cache(attention_weights_list)
    
    # 高级配置
    config = IntegrationConfig(total_cache_size=8192, enable_auto_tuning=True)
    integration = CakeAdaKVIntegration(config)
    ```
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        
        # 构建统一配置
        unified_config = self._build_unified_config()
        
        # 初始化核心组件
        self.allocator = UnifiedCakeAdaKVAllocator(unified_config)
        self.memory_manager = UnifiedMemoryManager()
        
        # 性能监控
        if self.config.enable_monitoring:
            self.monitor = PerformanceMonitor()
        else:
            self.monitor = None
        
        # 状态管理
        self.is_initialized = False
        self.optimization_history = []
        
    def _build_unified_config(self) -> UnifiedCacheConfig:
        """构建统一配置"""
        unified_config = UnifiedCacheConfig()
        
        # 基本配置
        unified_config.total_cache_size = self.config.total_cache_size
        unified_config.enable_fallback = self.config.enable_fallback
        unified_config.enable_monitoring = self.config.enable_monitoring
        
        # 应用自定义阈值
        if self.config.custom_thresholds:
            for key, value in self.config.custom_thresholds.items():
                if hasattr(unified_config.strategy_config, key):
                    setattr(unified_config.strategy_config, key, value)
        
        return unified_config
    
    def optimize_cache(
        self, 
        attention_weights_list: List[Any],
        return_detailed_info: bool = False
    ) -> Union[Tuple[List[int], List[List[int]]], Dict[str, Any]]:
        """
        优化缓存分配
        
        Args:
            attention_weights_list: 注意力权重列表
            return_detailed_info: 是否返回详细信息
            
        Returns:
            如果return_detailed_info=False: (layer_budgets, head_budgets_list)
            如果return_detailed_info=True: 包含详细信息的字典
        """
        if self.monitor:
            self.monitor.start_timing('total_optimization')
        
        try:
            # 输入验证
            if not attention_weights_list:
                raise ValueError("注意力权重列表不能为空")
            
            # 转换输入格式（如果需要）
            processed_weights = self._preprocess_attention_weights(attention_weights_list)
            
            if self.monitor:
                self.monitor.start_timing('unified_allocation')
            
            # 统一分配
            layer_budgets, head_budgets_list = self.allocator.unified_allocate(processed_weights)
            
            if self.monitor:
                self.monitor.end_timing('unified_allocation')
            
            # 记录优化历史
            optimization_info = {
                'timestamp': time.time(),
                'num_layers': len(attention_weights_list),
                'total_cache_size': self.config.total_cache_size,
                'layer_budgets': layer_budgets,
                'head_budgets_count': [len(hb) for hb in head_budgets_list]
            }
            
            self.optimization_history.append(optimization_info)
            
            if self.monitor:
                self.monitor.record_allocation(optimization_info)
                self.monitor.end_timing('total_optimization')
            
            if return_detailed_info:
                return self._build_detailed_result(
                    layer_budgets, head_budgets_list, optimization_info
                )
            else:
                return layer_budgets, head_budgets_list
                
        except Exception as e:
            if self.config.enable_fallback:
                warnings.warn(f"优化失败，使用回退策略: {e}")
                return self._fallback_allocation(attention_weights_list, return_detailed_info)
            else:
                raise e
    
    def _preprocess_attention_weights(self, attention_weights_list: List[Any]) -> List[Any]:
        """预处理注意力权重"""
        # 这里可以添加格式转换、验证等逻辑
        # 目前简单返回原始输入
        return attention_weights_list
    
    def _build_detailed_result(
        self,
        layer_budgets: List[int],
        head_budgets_list: List[List[int]],
        optimization_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建详细结果"""
        result = {
            'layer_budgets': layer_budgets,
            'head_budgets_list': head_budgets_list,
            'optimization_info': optimization_info,
            'allocator_summary': self.allocator.get_allocation_summary()
        }
        
        if self.monitor:
            result['performance_metrics'] = self.monitor.get_performance_summary()
        
        if self.memory_manager:
            result['memory_stats'] = self.memory_manager.get_memory_stats()
        
        return result
    
    def _fallback_allocation(
        self,
        attention_weights_list: List[Any],
        return_detailed_info: bool
    ) -> Union[Tuple[List[int], List[List[int]]], Dict[str, Any]]:
        """回退分配策略"""
        num_layers = len(attention_weights_list)
        
        # 均匀层级分配
        layer_base = self.config.total_cache_size // num_layers
        layer_remainder = self.config.total_cache_size % num_layers
        layer_budgets = [layer_base] * num_layers
        for i in range(layer_remainder):
            layer_budgets[i] += 1
        
        # 均匀头级分配
        head_budgets_list = []
        for i, attention_weights in enumerate(attention_weights_list):
            # 尝试获取头数，回退到默认值
            try:
                if hasattr(attention_weights, 'shape') and len(attention_weights.shape) >= 2:
                    num_heads = attention_weights.shape[1]
                else:
                    num_heads = 32  # 默认头数
            except:
                num_heads = 32
            
            layer_budget = layer_budgets[i]
            head_base = layer_budget // num_heads
            head_remainder = layer_budget % num_heads
            
            head_budgets = [head_base] * num_heads
            for j in range(head_remainder):
                head_budgets[j] += 1
            
            head_budgets_list.append(head_budgets)
        
        if return_detailed_info:
            return {
                'layer_budgets': layer_budgets,
                'head_budgets_list': head_budgets_list,
                'optimization_info': {
                    'strategy': 'fallback_uniform',
                    'timestamp': time.time(),
                    'num_layers': num_layers
                },
                'fallback_used': True
            }
        else:
            return layer_budgets, head_budgets_list
    
    def auto_tune(
        self,
        sample_attention_weights: List[List[Any]],
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        自动调优
        
        Args:
            sample_attention_weights: 样本注意力权重
            target_metrics: 目标性能指标
            
        Returns:
            调优结果
        """
        if not self.config.enable_auto_tuning:
            return {'auto_tuning': 'disabled'}
        
        # 简化的自动调优逻辑
        tuning_results = {}
        
        # 测试不同的缓存大小
        cache_sizes = [2048, 4096, 6144, 8192]
        best_size = self.config.total_cache_size
        best_performance = float('inf')
        
        for cache_size in cache_sizes:
            if len(sample_attention_weights) == 0:
                break
                
            # 临时配置
            temp_config = IntegrationConfig(
                total_cache_size=cache_size,
                enable_monitoring=True
            )
            temp_integration = CakeAdaKVIntegration(temp_config)
            
            # 测试性能
            total_time = 0
            for sample in sample_attention_weights[:self.config.warmup_samples]:
                start_time = time.time()
                try:
                    temp_integration.optimize_cache(sample)
                    total_time += time.time() - start_time
                except:
                    total_time += float('inf')  # 惩罚失败的配置
            
            avg_time = total_time / min(len(sample_attention_weights), self.config.warmup_samples)
            
            if avg_time < best_performance:
                best_performance = avg_time
                best_size = cache_size
        
        # 更新配置
        if best_size != self.config.total_cache_size:
            self.config.total_cache_size = best_size
            # 重新构建分配器
            unified_config = self._build_unified_config()
            self.allocator = UnifiedCakeAdaKVAllocator(unified_config)
        
        tuning_results = {
            'best_cache_size': best_size,
            'best_performance': best_performance,
            'tested_sizes': cache_sizes,
            'tuning_completed': True
        }
        
        return tuning_results
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_history.copy()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'integration_config': {
                'total_cache_size': self.config.total_cache_size,
                'enable_monitoring': self.config.enable_monitoring,
                'enable_auto_tuning': self.config.enable_auto_tuning
            },
            'optimization_count': len(self.optimization_history),
            'allocator_initialized': self.allocator.is_warmed_up
        }
        
        if self.monitor:
            report['performance_metrics'] = self.monitor.get_performance_summary()
        
        if hasattr(self.allocator, 'get_allocation_summary'):
            report['allocation_summary'] = self.allocator.get_allocation_summary()
        
        return report
    
    def reset(self) -> None:
        """重置集成器状态"""
        # 重新构建分配器
        unified_config = self._build_unified_config()
        self.allocator = UnifiedCakeAdaKVAllocator(unified_config)
        
        # 重置监控器
        if self.monitor:
            self.monitor = PerformanceMonitor()
        
        # 清空历史
        self.optimization_history.clear()
        self.is_initialized = False


def create_integration(
    cache_size: int = 4096,
    enable_monitoring: bool = True,
    enable_auto_tuning: bool = False,
    **kwargs
) -> CakeAdaKVIntegration:
    """
    创建集成器的便捷函数
    
    Args:
        cache_size: 缓存大小
        enable_monitoring: 启用监控
        enable_auto_tuning: 启用自动调优
        **kwargs: 其他配置参数
        
    Returns:
        配置好的集成器
    """
    config = IntegrationConfig(
        total_cache_size=cache_size,
        enable_monitoring=enable_monitoring,
        enable_auto_tuning=enable_auto_tuning,
        **kwargs
    )
    
    return CakeAdaKVIntegration(config)


# 导出主要接口
__all__ = [
    'CakeAdaKVIntegration',
    'IntegrationConfig',
    'PerformanceMonitor',
    'create_integration'
]