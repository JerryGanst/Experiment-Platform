"""
CAKE-AdaKV 统一预热和记忆管理机制 (改进版)

解决的核心问题：
1. 统一输入格式，避免双重预热
2. 记忆锚点的协同管理
3. 溢出记忆的智能处理
4. 指标归一化和预算守恒
5. 稳健的关键头检测
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from collections import deque


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
    floor_alpha: float = 0.02  # 最小分配比例（改为百分比）
    beta: float = 20.0  # 集中度敏感参数
    kernel_size: int = 7  # 平滑核大小
    
    # 统一预热配置
    warmup_steps: int = 10  # 预热步数
    warmup_mode: str = "unified"  # unified, cake_only, adakv_only
    
    # 归一化配置
    normalize_indicators: bool = True  # 是否归一化H/V指标
    ema_decay: float = 0.9  # 指标EMA衰减率
    winsorize_ratio: float = 0.02  # 极值截断比例
    
    # 稳健性配置
    key_head_method: str = "mad"  # mad, topk, percentile
    key_head_ratio: float = 0.2  # 关键头比例
    min_budget_tokens: int = 8  # 最小预算token数


class IndicatorNormalizer:
    """H/V指标归一化器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.h_stats = {'min': float('inf'), 'p95': float('-inf'), 'p05': float('inf')}
        self.v_stats = {'min': float('inf'), 'p95': float('-inf'), 'p05': float('inf')}
        self.initialized = False
        self.update_count = 0
        
    def update_stats(self, h_values: List[float], v_values: List[float]):
        """更新H/V统计信息"""
        if not h_values or not v_values:
            return
            
        h_array = np.array(h_values)
        v_array = np.array(v_values)
        
        # Winsorize处理极值
        h_winsorized = self._winsorize(h_array)
        v_winsorized = self._winsorize(v_array)
        
        # 计算分位数
        h_p05, h_p95 = np.percentile(h_winsorized, [5, 95])
        v_p05, v_p95 = np.percentile(v_winsorized, [5, 95])
        
        if not self.initialized:
            # 首次初始化
            self.h_stats = {'min': h_p05, 'p95': h_p95, 'p05': h_p05}
            self.v_stats = {'min': v_p05, 'p95': v_p95, 'p05': v_p05}
            self.initialized = True
        else:
            # EMA更新
            decay = self.config.ema_decay
            self.h_stats['p05'] = decay * self.h_stats['p05'] + (1-decay) * h_p05
            self.h_stats['p95'] = decay * self.h_stats['p95'] + (1-decay) * h_p95
            self.v_stats['p05'] = decay * self.v_stats['p05'] + (1-decay) * v_p05
            self.v_stats['p95'] = decay * self.v_stats['p95'] + (1-decay) * v_p95
        
        self.update_count += 1
    
    def _winsorize(self, values: np.ndarray) -> np.ndarray:
        """截断极值"""
        if len(values) < 10:  # 样本太少不截断
            return values
            
        p_low = self.config.winsorize_ratio * 100
        p_high = (1 - self.config.winsorize_ratio) * 100
        low, high = np.percentile(values, [p_low, p_high])
        return np.clip(values, low, high)
    
    def normalize(self, h_value: float, v_value: float) -> Tuple[float, float]:
        """归一化H/V值"""
        if not self.initialized:
            return h_value, v_value  # 未初始化时返回原值
            
        # Min-max归一化到[0,1]
        h_range = self.h_stats['p95'] - self.h_stats['p05'] + 1e-6
        v_range = self.v_stats['p95'] - self.v_stats['p05'] + 1e-6
        
        h_norm = np.clip((h_value - self.h_stats['p05']) / h_range, 0, 1)
        v_norm = np.clip((v_value - self.v_stats['p05']) / v_range, 0, 1)
        
        return float(h_norm), float(v_norm)


class BudgetNormalizer:
    """严格的预算守恒器"""
    
    @staticmethod
    def normalize_to_budget(raw_budgets: List[int], total_budget: int) -> List[int]:
        """
        严格保证预算守恒的归一化
        
        Args:
            raw_budgets: 原始预算分配
            total_budget: 总预算
            
        Returns:
            normalized_budgets: 归一化后的预算，保证和等于total_budget
        """
        if not raw_budgets or total_budget <= 0:
            return [0] * len(raw_budgets)
            
        # 转为numpy数组便于操作
        raw_array = np.array(raw_budgets, dtype=float)
        
        # 确保所有值>0
        raw_array = np.maximum(raw_array, 1.0)
        
        # 比例缩放
        scale_factor = total_budget / raw_array.sum()
        scaled = raw_array * scale_factor
        
        # 向下取整
        rounded = np.floor(scaled).astype(int)
        
        # 计算差额
        deficit = total_budget - rounded.sum()
        
        if deficit > 0:
            # 按小数部分从大到小分配剩余预算
            fractional_parts = scaled - rounded
            top_indices = np.argsort(-fractional_parts)[:deficit]
            rounded[top_indices] += 1
        elif deficit < 0:
            # 超额时从非关键位置回收
            excess = -deficit
            # 优先从预算较大的地方回收
            candidates = np.where(rounded > 1)[0]
            if len(candidates) >= excess:
                top_candidates = candidates[np.argsort(-rounded[candidates])[:excess]]
                rounded[top_candidates] -= 1
        
        # 最终检查
        assert rounded.sum() == total_budget, f"预算不守恒: {rounded.sum()} != {total_budget}"
        
        return rounded.tolist()


class RobustKeyHeadDetector:
    """稳健的关键头检测器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.head_importance_history = {}  # 层级 -> EMA重要性
        
    def detect_key_heads(
        self, 
        concentration_scores: torch.Tensor, 
        layer_idx: int,
        method: str = None
    ) -> torch.Tensor:
        """
        检测关键头
        
        Args:
            concentration_scores: [num_heads] 集中度评分
            layer_idx: 层索引
            method: 检测方法 mad/topk/percentile
            
        Returns:
            key_head_mask: [num_heads] bool mask
        """
        method = method or self.config.key_head_method
        scores = concentration_scores.detach().cpu().numpy()
        
        # EMA平滑历史重要性
        if layer_idx in self.head_importance_history:
            ema_scores = (self.config.ema_decay * self.head_importance_history[layer_idx] + 
                         (1 - self.config.ema_decay) * scores)
        else:
            ema_scores = scores
        self.head_importance_history[layer_idx] = ema_scores
        
        # 根据方法检测关键头
        if method == "mad":
            return self._detect_by_mad(ema_scores)
        elif method == "topk":
            return self._detect_by_topk(ema_scores)
        elif method == "percentile":
            return self._detect_by_percentile(ema_scores)
        else:
            raise ValueError(f"未知的检测方法: {method}")
    
    def _detect_by_mad(self, scores: np.ndarray) -> torch.Tensor:
        """基于MAD的检测"""
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        threshold = median + 2.0 * mad  # 2-sigma rule
        return torch.tensor(scores > threshold)
    
    def _detect_by_topk(self, scores: np.ndarray) -> torch.Tensor:
        """基于Top-K的检测"""
        k = max(1, int(len(scores) * self.config.key_head_ratio))
        threshold_idx = np.argsort(scores)[-k]
        threshold = scores[threshold_idx]
        return torch.tensor(scores >= threshold)
    
    def _detect_by_percentile(self, scores: np.ndarray) -> torch.Tensor:
        """基于百分位数的检测"""
        percentile = (1 - self.config.key_head_ratio) * 100
        threshold = np.percentile(scores, percentile)
        return torch.tensor(scores > threshold)


class UnifiedMemoryAnchor:
    """统一的记忆锚点管理器（改进版）"""
    
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
        识别记忆锚点（改进版）
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 防御性检查：异常序列处理
        if seq_len <= 1 or attention_weights.sum() < 1e-6:
            # 异常情况：返回最后几个token作为锚点
            anchor_size = min(self.config.window_size, seq_len)
            return torch.arange(seq_len - anchor_size, seq_len)
        
        if method == "cake" or method == "hybrid":
            # CAKE方法：基于注意力均值+方差
            attn_mean = attention_weights.mean(dim=-2)
            attn_var = attention_weights.var(dim=-2)
            cake_importance = attn_mean + self.config.gamma * attn_var
            
        if method == "adakv" or method == "hybrid":
            # AdaKV方法：基于注意力集中度
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
        anchor_size = min(self.config.window_size, seq_len)
        _, anchor_indices = torch.topk(
            token_importance, 
            anchor_size, 
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
        """保护记忆锚点不被驱逐"""
        if layer_idx not in self.anchor_tokens:
            return eviction_scores
            
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


class EnhancedAdaKVAllocator:
    """增强的AdaKV分配器"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.key_head_detector = RobustKeyHeadDetector(config)
        
    def allocate_heads(
        self,
        layer_budget: int,
        concentration_scores: torch.Tensor,
        h_indicator: float,
        v_indicator: float,
        layer_idx: int
    ) -> List[int]:
        """
        基于H/V指标的头级分配
        
        Args:
            layer_budget: 该层的总预算
            concentration_scores: 每个头的集中度
            h_indicator: 归一化后的H指标
            v_indicator: 归一化后的V指标
            layer_idx: 层索引
            
        Returns:
            head_budgets: 每个头的预算分配
        """
        num_heads = len(concentration_scores)
        
        # 计算最小预算
        min_budget = max(
            self.config.min_budget_tokens,
            int(self.config.floor_alpha * layer_budget)
        )
        
        # 策略选择
        if h_indicator > 0.7 and v_indicator > 0.5:
            # 高分散+高动态：混合策略
            head_budgets = self._highly_adaptive_allocation(
                layer_budget, concentration_scores, h_indicator, v_indicator, layer_idx
            )
        elif h_indicator > 0.7:
            # 高分散：偏向均匀
            head_budgets = self._guided_uniform_allocation(
                layer_budget, concentration_scores, h_indicator
            )
        elif v_indicator > 0.5:
            # 高动态：激进自适应
            head_budgets = self._aggressive_adaptive_allocation(
                layer_budget, concentration_scores, v_indicator
            )
        else:
            # 低分散+低动态：标准分配
            head_budgets = self._standard_allocation(
                layer_budget, concentration_scores
            )
        
        # 确保最小预算
        head_budgets = [max(b, min_budget) for b in head_budgets]
        
        # 严格预算守恒
        head_budgets = BudgetNormalizer.normalize_to_budget(head_budgets, layer_budget)
        
        return head_budgets
    
    def _guided_uniform_allocation(
        self, 
        layer_budget: int, 
        concentrations: torch.Tensor, 
        h_indicator: float
    ) -> List[int]:
        """高分散场景的分配"""
        num_heads = len(concentrations)
        base_per_head = layer_budget // num_heads
        
        # H越大，调整幅度越小
        adjustment_strength = 1.0 - h_indicator
        
        concentrations_np = concentrations.detach().cpu().numpy()
        normalized_conc = concentrations_np / concentrations_np.sum()
        
        head_budgets = []
        for h in range(num_heads):
            # 均匀为主，集中度微调
            adjustment = int(
                adjustment_strength * layer_budget * 0.2 * 
                (normalized_conc[h] - 1/num_heads)
            )
            budget = base_per_head + adjustment
            head_budgets.append(budget)
        
        return head_budgets
    
    def _aggressive_adaptive_allocation(
        self, 
        layer_budget: int, 
        concentrations: torch.Tensor, 
        v_indicator: float
    ) -> List[int]:
        """高动态场景的分配"""
        # V越大，分配差异越大
        sharpness = 1.0 + 2.0 * v_indicator
        
        concentrations_np = concentrations.detach().cpu().numpy()
        
        # 使用幂函数增强差异
        sharpened_scores = concentrations_np ** sharpness
        allocation_weights = sharpened_scores / sharpened_scores.sum()
        
        # 分配预算
        head_budgets = [int(w * layer_budget) for w in allocation_weights]
        
        return head_budgets
    
    def _highly_adaptive_allocation(
        self,
        layer_budget: int,
        concentrations: torch.Tensor,
        h_indicator: float,
        v_indicator: float,
        layer_idx: int
    ) -> List[int]:
        """混合场景的复杂策略"""
        # 检测关键头
        key_heads = self.key_head_detector.detect_key_heads(
            concentrations, layer_idx
        )
        
        concentrations_np = concentrations.detach().cpu().numpy()
        num_heads = len(concentrations_np)
        
        # 两级分配策略
        key_head_ratio = 0.6 + 0.2 * v_indicator  # V越大，关键头越重要
        key_budget = int(layer_budget * key_head_ratio)
        normal_budget = layer_budget - key_budget
        
        num_key = key_heads.sum().item()
        num_normal = num_heads - num_key
        
        head_budgets = []
        key_concentrations = concentrations_np[key_heads.numpy()]
        
        for h, is_key in enumerate(key_heads):
            if is_key and num_key > 0:
                # 关键头内部差异化分配
                weight = concentrations_np[h] / key_concentrations.sum()
                budget = int(key_budget * weight)
            elif num_normal > 0:
                # 普通头相对均匀
                budget = normal_budget // num_normal
            else:
                budget = layer_budget // num_heads
            head_budgets.append(budget)
        
        return head_budgets
    
    def _standard_allocation(
        self, 
        layer_budget: int, 
        concentrations: torch.Tensor
    ) -> List[int]:
        """标准AdaKV分配"""
        concentrations_np = concentrations.detach().cpu().numpy()
        allocation_weights = concentrations_np / concentrations_np.sum()
        head_budgets = [int(w * layer_budget) for w in allocation_weights]
        return head_budgets


class UnifiedWarmupManager:
    """统一的预热管理器（改进版）"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.normalizer = IndicatorNormalizer(config)
        self.warmup_cache = {}
        self.warmup_stats = {}
        self.current_step = 0
        
    def unified_warmup(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """统一预热过程，避免重复计算"""
        warmup_info = {
            'layer_stats': [],
            'head_stats': [],
            'anchor_info': {},
            'overflow_strategy': {},
            'normalization_stats': {}
        }
        
        with torch.no_grad():
            # 防御性检查
            if input_ids.numel() == 0:
                return self._create_fallback_warmup_info(model)
            
            # 单次前向传播收集所有信息
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=True
                )
            except Exception as e:
                print(f"预热过程中出现错误: {e}")
                return self._create_fallback_warmup_info(model)
            
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                return self._create_fallback_warmup_info(model)
            
            attentions = outputs.attentions
            h_values = []
            v_values = []
            
            # 统一分析每层的注意力模式
            for layer_idx, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                    
                # 计算CAKE指标
                h_indicator = self._compute_h_indicator(layer_attn)
                v_indicator = self._compute_v_indicator(layer_attn)
                
                h_values.append(h_indicator)
                v_values.append(v_indicator)
                
                # 计算AdaKV指标
                concentration_scores = self._compute_concentration(layer_attn)
                
                # 保存原始统计
                warmup_info['layer_stats'].append({
                    'h_indicator': h_indicator,
                    'v_indicator': v_indicator,
                    'mean_concentration': concentration_scores.mean().item()
                })
                
                warmup_info['head_stats'].append(concentration_scores)
            
            # 更新归一化统计
            if h_values and v_values:
                self.normalizer.update_stats(h_values, v_values)
                
                # 归一化指标
                normalized_stats = []
                for h_raw, v_raw in zip(h_values, v_values):
                    h_norm, v_norm = self.normalizer.normalize(h_raw, v_raw)
                    normalized_stats.append({
                        'h_normalized': h_norm,
                        'v_normalized': v_norm
                    })
                
                warmup_info['normalization_stats'] = normalized_stats
                
                # 基于归一化结果决定策略
                warmup_info['overflow_strategy'] = self._determine_overflow_strategy(
                    normalized_stats
                )
        
        self.warmup_cache = warmup_info
        return warmup_info
    
    def _create_fallback_warmup_info(self, model) -> Dict:
        """创建回退的预热信息"""
        num_layers = getattr(model.config, 'num_hidden_layers', 32)
        
        fallback_info = {
            'layer_stats': [
                {'h_indicator': 0.5, 'v_indicator': 0.5, 'mean_concentration': 0.5}
                for _ in range(num_layers)
            ],
            'head_stats': [
                torch.ones(getattr(model.config, 'num_attention_heads', 32)) * 0.5
                for _ in range(num_layers)
            ],
            'anchor_info': {},
            'overflow_strategy': {
                i: {'method': 'balanced', 'cake_weight': 0.5, 'adakv_weight': 0.5, 'anchor_protection': False}
                for i in range(num_layers)
            },
            'normalization_stats': [
                {'h_normalized': 0.5, 'v_normalized': 0.5}
                for _ in range(num_layers)
            ]
        }
        return fallback_info
    
    def _compute_h_indicator(self, attention_weights: torch.Tensor) -> float:
        """计算H指标（空间分散度）"""
        try:
            # 防止log(0)
            attention_weights = torch.clamp(attention_weights, min=1e-10)
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights), 
                dim=-1
            )
            return entropy.mean().item()
        except:
            return 0.5  # 回退值
    
    def _compute_v_indicator(self, attention_weights: torch.Tensor) -> float:
        """计算V指标（时间变化度）"""
        try:
            variance = torch.var(attention_weights, dim=-2)
            return variance.mean().item()
        except:
            return 0.5  # 回退值
    
    def _compute_concentration(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算每个头的集中度"""
        try:
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # 计算每个头的熵
            entropy_per_head = []
            for h in range(num_heads):
                attn_h = attention_weights[:, h, :, :]
                attn_h = torch.clamp(attn_h, min=1e-10)
                entropy = -torch.sum(attn_h * torch.log(attn_h), dim=-1)
                entropy_per_head.append(entropy.mean())
            
            entropy_per_head = torch.stack(entropy_per_head)
            
            # 转换为集中度
            max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))
            concentration_scores = 1.0 - (entropy_per_head / max_entropy)
            
            return torch.clamp(concentration_scores, 0, 1)
        except:
            # 回退到均匀分布
            num_heads = attention_weights.shape[1] if attention_weights.dim() >= 2 else 32
            return torch.ones(num_heads) * 0.5
    
    def _determine_overflow_strategy(self, normalized_stats: List[Dict]) -> Dict:
        """基于归一化统计决定溢出处理策略"""
        strategy = {}
        
        for layer_idx, stats in enumerate(normalized_stats):
            h_val = stats['h_normalized']
            v_val = stats['v_normalized']
            
            # 基于归一化后的指标决定策略
            if h_val > 0.7 and v_val > 0.5:
                strategy[layer_idx] = {
                    'method': 'hybrid',
                    'cake_weight': 0.6,
                    'adakv_weight': 0.4,
                    'anchor_protection': True
                }
            elif h_val > 0.7:
                strategy[layer_idx] = {
                    'method': 'cake_dominant',
                    'cake_weight': 0.8,
                    'adakv_weight': 0.2,
                    'anchor_protection': True
                }
            elif v_val > 0.5:
                strategy[layer_idx] = {
                    'method': 'adakv_dominant',
                    'cake_weight': 0.3,
                    'adakv_weight': 0.7,
                    'anchor_protection': True
                }
            else:
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
    """统一的CAKE-AdaKV实现（改进版）"""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        self.warmup_manager = UnifiedWarmupManager(config)
        self.memory_anchor = UnifiedMemoryAnchor(config)
        self.adakv_allocator = EnhancedAdaKVAllocator(config)
        self.overflow_handler = UnifiedOverflowHandler(config)
        
        # 缓存预热结果，避免重复计算
        self.warmup_completed = False
        self.warmup_results = None
        
    def initialize(self, model, sample_input):
        """统一初始化过程"""
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
        """统一处理KV缓存"""
        if not self.warmup_completed or self.warmup_results is None:
            return past_key_values
            
        # 1. 识别记忆锚点
        anchor_indices = self.memory_anchor.identify_anchors(
            attention_weights, 
            layer_idx,
            method="hybrid"
        )
        
        # 2. 获取该层的策略和归一化指标
        strategy = self.warmup_results['overflow_strategy'].get(layer_idx, {
            'method': 'balanced', 'anchor_protection': False
        })
        
        norm_stats = self.warmup_results['normalization_stats'][layer_idx]
        h_norm = norm_stats['h_normalized']
        v_norm = norm_stats['v_normalized']
        
        # 3. 计算层级预算（基于CAKE的层级分配）
        layer_budget = self._compute_layer_budget(layer_idx, h_norm, v_norm)
        
        # 4. 头级分配
        if layer_idx < len(self.warmup_results['head_stats']):
            concentration_scores = self.warmup_results['head_stats'][layer_idx]
            head_budgets = self.adakv_allocator.allocate_heads(
                layer_budget, concentration_scores, h_norm, v_norm, layer_idx
            )
        else:
            # 回退到均匀分配
            num_heads = past_key_values.key_cache[layer_idx].shape[1]
            head_budgets = [layer_budget // num_heads] * num_heads
            head_budgets = BudgetNormalizer.normalize_to_budget(head_budgets, layer_budget)
        
        # 5. 处理溢出
        compressed_keys, compressed_values = self.overflow_handler.handle_overflow(
            past_key_values.key_cache[layer_idx],
            past_key_values.value_cache[layer_idx],
            layer_idx,
            strategy,
            layer_budget
        )
        
        # 6. 更新缓存
        past_key_values.key_cache[layer_idx] = compressed_keys
        past_key_values.value_cache[layer_idx] = compressed_values
        
        return past_key_values
    
    def _compute_layer_budget(self, layer_idx: int, h_norm: float, v_norm: float) -> int:
        """计算层级预算"""
        # 简化的预算计算
        base_budget = self.config.cache_size - self.config.window_size
        
        # 基于归一化的H和V指标调整
        h_factor = 1.0 + h_norm * 0.2
        v_factor = 1.0 + v_norm * 0.1
        
        adjusted_budget = int(base_budget * h_factor * v_factor)
        
        return min(adjusted_budget, self.config.cache_size)


# 使用示例
def create_unified_cache_system(model_config):
    """创建统一的缓存系统"""
    config = UnifiedCacheConfig(
        window_size=32,
        cache_size=512,
        tau1=1.0,
        tau2=1.0,
        gamma=0.5,
        floor_alpha=0.02,  # 改为百分比
        beta=20.0,
        warmup_mode="unified",
        normalize_indicators=True,
        key_head_method="mad",
        key_head_ratio=0.2
    )
    
    return UnifiedCakeAdaKV(config)