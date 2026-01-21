"""
VLLM 集成工具模块

解决 VLLM 部署的工程难题：
1. 显存生命周期管理 - 串行加载策略
2. Token-to-Block 预算对齐
3. VLLM 特性冲突检测
4. 请求级分配器上下文
5. Tokenizer 对齐验证

工程约束：
- VLLM PagedAttention 的 fused kernel 无法直接导出 attention scores
- 必须采用"外部预热"模式：HF模型分析 -> 释放显存 -> VLLM推理
- 初期仅支持 TP=1，Prefix Caching 必须禁用
"""

import logging
import gc
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import torch

logger = logging.getLogger(__name__)


# ============== 1. 显存生命周期管理 ==============

class MemoryManager:
    """
    显存生命周期管理器

    解决 HF 模型（用于 Warmup）和 VLLM 引擎无法在同一进程内共存的问题。
    采用串行加载策略：
    1. 加载 HF 模型 -> 收集 Attention -> 计算 Budget
    2. 彻底释放 HF 模型显存
    3. 初始化 VLLM 引擎
    """

    @staticmethod
    def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
        """获取所有 GPU 的显存信息"""
        info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                info[i] = {
                    "total_gb": props.total_memory / (1024**3),
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "free_gb": (props.total_memory - reserved) / (1024**3),
                }
        return info

    @staticmethod
    def force_cleanup():
        """强制清理所有 GPU 显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("GPU memory force cleaned")

    @staticmethod
    def estimate_model_memory(
        num_params_billions: float,
        precision: str = "fp16",
        include_kv_cache: bool = True,
        max_seq_len: int = 4096,
        batch_size: int = 1,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128
    ) -> float:
        """
        估算模型显存占用 (GB)

        Args:
            num_params_billions: 模型参数量（十亿）
            precision: 精度 (fp16/bf16/fp32/int8/int4)
            include_kv_cache: 是否包含 KV Cache 估算
            max_seq_len: 最大序列长度（用于 KV Cache 估算）
            batch_size: 批大小
            num_layers: 模型层数
            num_heads: 注意力头数
            head_dim: 每个头的维度

        Returns:
            预估显存占用 (GB)
        """
        bytes_per_param = {
            "fp16": 2, "bf16": 2, "fp32": 4,
            "int8": 1, "int4": 0.5, "auto": 2
        }
        bpp = bytes_per_param.get(precision, 2)

        # 模型参数显存
        param_memory_gb = (num_params_billions * 1e9 * bpp) / (1024**3)

        # CUDA 内核和激活值开销（约 10-15%）
        activation_overhead = param_memory_gb * 0.15

        total = param_memory_gb + activation_overhead

        if include_kv_cache:
            # KV Cache 显存估算
            # 每层每个 token 需要: 2 (K+V) * num_heads * head_dim * bytes_per_param
            kv_per_token_per_layer = 2 * num_heads * head_dim * bpp
            kv_cache_bytes = batch_size * max_seq_len * num_layers * kv_per_token_per_layer
            kv_cache_gb = kv_cache_bytes / (1024**3)
            total += kv_cache_gb

        return total

    @staticmethod
    def calculate_vllm_gpu_utilization(
        total_gpu_memory_gb: float,
        model_memory_gb: float,
        reserve_for_hf_gb: float = 0.0
    ) -> float:
        """
        计算 VLLM 应设置的 gpu_memory_utilization

        Args:
            total_gpu_memory_gb: GPU 总显存
            model_memory_gb: 模型预估显存
            reserve_for_hf_gb: 为 HF 模型预留的显存（如果需要共存）

        Returns:
            推荐的 gpu_memory_utilization 值
        """
        available = total_gpu_memory_gb - reserve_for_hf_gb
        # VLLM 需要额外空间用于 KV Cache
        recommended = min(0.95, (available - 2) / total_gpu_memory_gb)
        return max(0.5, recommended)


# ============== 2. Token-to-Block 预算对齐 ==============

@dataclass
class AlignedBudget:
    """对齐后的预算结构"""
    layer_token_budgets: List[int]  # 原始 token 级预算
    layer_block_budgets: List[int]  # 对齐后的 block 级预算
    head_token_budgets: List[List[int]]  # 原始头级 token 预算
    block_size: int
    total_tokens_requested: int
    total_tokens_aligned: int
    alignment_loss_ratio: float  # 对齐损失比例


class BudgetAligner:
    """
    Token-to-Block 预算对齐器

    将 CAKE/AdaKV 的 Token 级精细预算转换为 VLLM PagedAttention
    的 Block 级预算，同时尽量保持预算守恒和层间重要性分布。
    """

    def __init__(self, block_size: int = 16):
        """
        Args:
            block_size: VLLM PagedAttention 的块大小
        """
        self.block_size = block_size

    def align_layer_budgets(
        self,
        layer_token_budgets: List[int],
        priority_scores: Optional[List[float]] = None,
        rounding_strategy: str = "importance_weighted"
    ) -> List[int]:
        """
        将层级 token 预算对齐到 block 整数倍

        Args:
            layer_token_budgets: 每层的 token 预算
            priority_scores: 每层的重要性评分（用于余数分配）
            rounding_strategy: 取整策略
                - "floor": 全部向下取整
                - "ceil": 全部向上取整
                - "round": 四舍五入
                - "importance_weighted": 按重要性分配余数（推荐）

        Returns:
            对齐后的 block 预算列表
        """
        num_layers = len(layer_token_budgets)

        if rounding_strategy == "floor":
            return [budget // self.block_size for budget in layer_token_budgets]

        elif rounding_strategy == "ceil":
            return [
                (budget + self.block_size - 1) // self.block_size
                for budget in layer_token_budgets
            ]

        elif rounding_strategy == "round":
            return [
                (budget + self.block_size // 2) // self.block_size
                for budget in layer_token_budgets
            ]

        elif rounding_strategy == "importance_weighted":
            # 除零保护：如果所有预算都为0，直接返回全0
            total_requested_tokens = sum(layer_token_budgets)
            if total_requested_tokens == 0:
                logger.warning("All layer budgets are zero, returning zero blocks")
                return [0] * num_layers

            # 计算基础 block 数和余数
            base_blocks = [budget // self.block_size for budget in layer_token_budgets]
            remainders = [budget % self.block_size for budget in layer_token_budgets]

            # 计算可分配的额外 block 数
            total_base_tokens = sum(b * self.block_size for b in base_blocks)
            extra_tokens_available = total_requested_tokens - total_base_tokens

            # 按重要性分配额外 blocks
            if priority_scores is None:
                # 默认按余数大小作为优先级
                priority_scores = remainders

            # 创建 (层索引, 优先级) 对并排序
            layer_priorities = list(enumerate(priority_scores))
            layer_priorities.sort(key=lambda x: x[1], reverse=True)

            # 分配额外 blocks
            extra_blocks_to_give = extra_tokens_available // self.block_size
            result = base_blocks.copy()

            for i in range(min(extra_blocks_to_give, num_layers)):
                layer_idx = layer_priorities[i][0]
                result[layer_idx] += 1

            return result

        else:
            raise ValueError(f"Unknown rounding strategy: {rounding_strategy}")

    def compute_aligned_budget(
        self,
        layer_token_budgets: List[int],
        head_token_budgets: List[List[int]],
        layer_priority_scores: Optional[List[float]] = None
    ) -> AlignedBudget:
        """
        计算完整的对齐预算

        Args:
            layer_token_budgets: 层级 token 预算
            head_token_budgets: 头级 token 预算
            layer_priority_scores: 层重要性评分

        Returns:
            AlignedBudget 结构
        """
        # 对齐层级预算
        layer_block_budgets = self.align_layer_budgets(
            layer_token_budgets,
            layer_priority_scores,
            rounding_strategy="importance_weighted"
        )

        # 计算对齐损失
        total_requested = sum(layer_token_budgets)
        total_aligned = sum(b * self.block_size for b in layer_block_budgets)
        loss_ratio = abs(total_aligned - total_requested) / total_requested if total_requested > 0 else 0

        if loss_ratio > 0.1:
            logger.warning(
                f"Budget alignment loss is high: {loss_ratio:.1%}. "
                f"Requested: {total_requested}, Aligned: {total_aligned}"
            )

        return AlignedBudget(
            layer_token_budgets=layer_token_budgets,
            layer_block_budgets=layer_block_budgets,
            head_token_budgets=head_token_budgets,
            block_size=self.block_size,
            total_tokens_requested=total_requested,
            total_tokens_aligned=total_aligned,
            alignment_loss_ratio=loss_ratio,
        )


# ============== 3. VLLM 特性冲突检测 ==============

@dataclass
class VLLMCompatibilityReport:
    """VLLM 兼容性检查报告"""
    is_compatible: bool
    warnings: List[str]
    errors: List[str]
    recommendations: Dict[str, Any]


def check_vllm_compatibility(vllm_config: Dict[str, Any]) -> VLLMCompatibilityReport:
    """
    检查 VLLM 配置与 CAKE/AdaKV 的兼容性

    当前已知冲突：
    - Prefix Caching: 与动态 KV 策略冲突，必须禁用
    - Tensor Parallel (TP > 1): Attention 数据聚合复杂，暂不支持
    - Chunked Prefill: 可能影响 Attention 模式，建议禁用
    """
    warnings = []
    errors = []
    recommendations = {}

    # 检查 Prefix Caching
    if vllm_config.get("enable_prefix_caching", False):
        errors.append(
            "Prefix Caching 必须禁用！CAKE/AdaKV 对不同请求可能应用不同的 "
            "Layer/Head 裁剪策略，会导致缓存命中错误。"
        )
        recommendations["enable_prefix_caching"] = False

    # 检查 Tensor Parallel
    tp_size = vllm_config.get("tensor_parallel_size", 1)
    if tp_size > 1:
        errors.append(
            f"Tensor Parallel (TP={tp_size}) 暂不支持！多卡 TP 模式下 Attention Heads "
            "被切分到不同卡，汇总 Attention 数据的逻辑尚未实现。"
        )
        recommendations["tensor_parallel_size"] = 1

    # 检查 Pipeline Parallel
    pp_size = vllm_config.get("pipeline_parallel_size", 1)
    if pp_size > 1:
        warnings.append(
            f"Pipeline Parallel (PP={pp_size}) 可能导致 Layer 级统计不完整，"
            "建议设置为 1。"
        )
        recommendations["pipeline_parallel_size"] = 1

    # 检查 enforce_eager
    if not vllm_config.get("enforce_eager", False):
        warnings.append(
            "建议启用 enforce_eager=True 以便于调试。CUDA Graph 可能会"
            "掩盖某些内存问题。"
        )

    # 检查 GPU Memory Utilization
    gpu_util = vllm_config.get("gpu_memory_utilization", 0.90)
    if gpu_util > 0.95:
        warnings.append(
            f"gpu_memory_utilization={gpu_util} 过高，留给 KV Cache 动态分配的"
            "空间可能不足，建议设为 0.85-0.90。"
        )
        recommendations["gpu_memory_utilization"] = 0.90

    is_compatible = len(errors) == 0

    return VLLMCompatibilityReport(
        is_compatible=is_compatible,
        warnings=warnings,
        errors=errors,
        recommendations=recommendations,
    )


def get_safe_vllm_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取与 CAKE/AdaKV 兼容的安全 VLLM 配置

    强制覆盖不兼容的设置项。
    """
    safe_config = base_config.copy()

    # 强制禁用不兼容特性
    safe_config["enable_prefix_caching"] = False
    safe_config["tensor_parallel_size"] = 1
    safe_config["pipeline_parallel_size"] = 1

    # 推荐设置
    if safe_config.get("gpu_memory_utilization", 0.9) > 0.92:
        safe_config["gpu_memory_utilization"] = 0.90

    return safe_config


# ============== 4. 请求级分配器上下文 ==============

@dataclass
class RequestBudgetContext:
    """
    请求级预算上下文

    在高并发异步环境中，每个请求需要独立的预算上下文，
    避免共享状态导致的数据污染。
    """
    request_id: str
    layer_budgets: List[int]
    head_budgets: List[List[int]]
    aligned_budgets: Optional[AlignedBudget] = None
    # 元数据
    creation_time: float = 0.0
    input_length: int = 0
    # 分配策略信息
    allocation_strategy: str = "unified"
    strategy_params: Dict[str, Any] = field(default_factory=dict)


class RequestScopedAllocator:
    """
    请求级分配器

    为每个请求创建独立的预算上下文，避免并发冲突。
    核心原则：无共享可变状态。
    """

    def __init__(
        self,
        attention_data: "AttentionData",
        block_size: int = 16,
        allocator_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            attention_data: 预热阶段收集的注意力数据（只读）
            block_size: VLLM block 大小
            allocator_config: 分配器配置
        """
        # 只读数据
        self._attention_data = attention_data
        self._block_size = block_size
        self._allocator_config = allocator_config or {}

        # 预计算不变的基础预算（可被所有请求共享）
        self._base_layer_budgets = None
        self._base_head_budgets = None
        self._precompute_base_budgets()

    def _precompute_base_budgets(self):
        """预计算基础预算（只做一次）"""
        # 尝试多个可能的导入路径
        UnifiedCakeAdaKVAllocator = None
        UnifiedCacheConfig = None

        import_paths = [
            "src.hace.core.unified_allocator",        # 新路径
            "src.core_code.unified_allocator",        # 旧路径兼容
            "core_code.unified_allocator",            # 相对导入兼容
        ]

        for path in import_paths:
            try:
                module = __import__(path, fromlist=["UnifiedCakeAdaKVAllocator", "UnifiedCacheConfig"])
                UnifiedCakeAdaKVAllocator = getattr(module, "UnifiedCakeAdaKVAllocator", None)
                UnifiedCacheConfig = getattr(module, "UnifiedCacheConfig", None)
                if UnifiedCakeAdaKVAllocator and UnifiedCacheConfig:
                    logger.debug(f"Successfully imported allocator from: {path}")
                    break
            except (ImportError, ModuleNotFoundError):
                continue

        if UnifiedCakeAdaKVAllocator and UnifiedCacheConfig:
            try:
                total_cache_size = self._allocator_config.get("total_cache_size", 4096)
                config = UnifiedCacheConfig(total_cache_size=total_cache_size)

                allocator = UnifiedCakeAdaKVAllocator(config)
                self._base_layer_budgets, self._base_head_budgets = allocator.unified_allocate(
                    self._attention_data.attention_weights_list
                )

                logger.info(
                    f"Pre-computed base budgets: {len(self._base_layer_budgets)} layers, "
                    f"total={sum(self._base_layer_budgets)}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to use UnifiedCakeAdaKVAllocator: {e}")

        # Fallback: 使用均匀分配
        logger.warning("UnifiedCakeAdaKVAllocator not available, using uniform allocation")
        num_layers = self._attention_data.num_layers
        num_heads = self._attention_data.num_heads
        total = self._allocator_config.get("total_cache_size", 4096)

        # 防止除零
        if num_layers == 0:
            logger.error("num_layers is 0, cannot compute budgets")
            self._base_layer_budgets = []
            self._base_head_budgets = []
            return

        layer_budget = total // num_layers
        self._base_layer_budgets = [layer_budget] * num_layers

        if num_heads == 0:
            logger.warning("num_heads is 0, setting empty head budgets")
            self._base_head_budgets = [[] for _ in range(num_layers)]
        else:
            head_budget = layer_budget // num_heads
            self._base_head_budgets = [[head_budget] * num_heads for _ in range(num_layers)]

    def create_context(
        self,
        request_id: str,
        input_length: Optional[int] = None,
        custom_total_cache: Optional[int] = None
    ) -> RequestBudgetContext:
        """
        为请求创建独立的预算上下文

        Args:
            request_id: 请求唯一标识
            input_length: 输入长度（可用于动态调整）
            custom_total_cache: 自定义总缓存大小

        Returns:
            RequestBudgetContext 实例
        """
        import time

        # 复制基础预算（避免修改共享数据）
        layer_budgets = self._base_layer_budgets.copy()
        head_budgets = [heads.copy() for heads in self._base_head_budgets]

        # 如果指定了自定义总缓存，按比例缩放
        if custom_total_cache is not None:
            base_total = sum(self._base_layer_budgets)
            scale = custom_total_cache / base_total
            layer_budgets = [int(b * scale) for b in layer_budgets]
            # 头级预算也相应缩放
            head_budgets = [
                [int(h * scale) for h in heads]
                for heads in head_budgets
            ]

        # 对齐到 block
        aligner = BudgetAligner(self._block_size)
        aligned = aligner.compute_aligned_budget(layer_budgets, head_budgets)

        return RequestBudgetContext(
            request_id=request_id,
            layer_budgets=layer_budgets,
            head_budgets=head_budgets,
            aligned_budgets=aligned,
            creation_time=time.time(),
            input_length=input_length or 0,
            allocation_strategy="unified",
        )


# ============== 5. Tokenizer 对齐验证 ==============

class TokenizerAlignmentChecker:
    """
    Tokenizer 对齐检查器

    确保 HF Tokenizer 和 VLLM 使用的 Tokenizer 行为一致，
    避免 Token 下标错位导致的 Budget 应用错误。
    """

    @staticmethod
    def check_alignment(
        hf_tokenizer,
        vllm_tokenizer,
        test_texts: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        检查两个 tokenizer 是否对齐

        Args:
            hf_tokenizer: HuggingFace tokenizer
            vllm_tokenizer: VLLM 使用的 tokenizer
            test_texts: 测试文本列表

        Returns:
            (is_aligned, mismatches): 是否对齐及不匹配列表
        """
        if test_texts is None:
            test_texts = [
                "Hello, world!",
                "What is machine learning?",
                "The quick brown fox jumps over the lazy dog.",
                "1 + 1 = 2",
                "",  # 空字符串
            ]

        mismatches = []

        for text in test_texts:
            try:
                hf_ids = hf_tokenizer.encode(text, add_special_tokens=True)
                vllm_ids = vllm_tokenizer.encode(text)

                if hf_ids != vllm_ids:
                    mismatches.append(
                        f"Text: '{text[:30]}...' | HF: {hf_ids[:5]}... | VLLM: {vllm_ids[:5]}..."
                    )
            except Exception as e:
                mismatches.append(f"Error tokenizing '{text[:30]}': {e}")

        is_aligned = len(mismatches) == 0

        if not is_aligned:
            logger.warning(
                f"Tokenizer alignment check failed with {len(mismatches)} mismatches. "
                "This may cause incorrect budget application!"
            )

        return is_aligned, mismatches

    @staticmethod
    def get_special_token_info(tokenizer) -> Dict[str, Any]:
        """获取 tokenizer 的特殊 token 信息"""
        return {
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": tokenizer.vocab_size,
        }


# ============== 6. 串行执行管道 ==============

class SerialInferencePipeline:
    """
    串行推理管道

    严格按照以下顺序执行：
    1. 加载 HF 模型
    2. 收集 Attention 数据
    3. 计算预算
    4. 释放 HF 模型显存
    5. 初始化 VLLM 引擎
    6. 执行推理

    这是解决显存共存问题的唯一可靠方案。
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        vllm_config: Dict[str, Any],
        allocator_config: Optional[Dict[str, Any]] = None
    ):
        self.model_config = model_config
        self.vllm_config = get_safe_vllm_config(vllm_config)
        self.allocator_config = allocator_config or {}

        # 状态
        self._attention_data = None
        self._request_allocator = None
        self._vllm_backend = None
        self._last_request_context = None

    def phase1_collect_attention(
        self,
        sample_prompts: List[str],
        cache_file: Optional[str] = None
    ) -> "AttentionData":
        """
        阶段1：收集注意力数据

        完成后自动释放 HF 模型显存。
        """
        from .attention_collector import (
            AttentionCollector,
            AttentionCollectionConfig,
            AttentionData,
        )

        # 检查缓存
        if cache_file:
            from pathlib import Path
            if Path(cache_file).exists():
                logger.info(f"Loading cached attention data: {cache_file}")
                self._attention_data = AttentionData.load(cache_file)
                return self._attention_data

        # 收集注意力
        logger.info("Phase 1: Collecting attention data with HF model...")

        config = AttentionCollectionConfig(
            warmup_samples=len(sample_prompts),
            cache_file=cache_file,
        )
        collector = AttentionCollector(config)
        self._attention_data = collector.collect_from_hf_model(
            self.model_config,
            sample_prompts
        )

        # 强制清理显存
        MemoryManager.force_cleanup()
        logger.info("Phase 1 complete. HF model memory released.")

        return self._attention_data

    def phase2_compute_budgets(self) -> RequestScopedAllocator:
        """
        阶段2：计算预算

        创建请求级分配器。
        """
        if self._attention_data is None:
            raise RuntimeError("Must run phase1_collect_attention first")

        logger.info("Phase 2: Computing budgets...")

        block_size = self.vllm_config.get("block_size", 16)
        self._request_allocator = RequestScopedAllocator(
            attention_data=self._attention_data,
            block_size=block_size,
            allocator_config=self.allocator_config,
        )

        logger.info("Phase 2 complete. Request allocator ready.")
        return self._request_allocator

    def phase3_initialize_vllm(self):
        """
        阶段3：初始化 VLLM 引擎

        必须在 HF 模型释放后执行。
        """
        # 检查兼容性
        report = check_vllm_compatibility(self.vllm_config)
        if not report.is_compatible:
            raise RuntimeError(
                f"VLLM config not compatible with CAKE/AdaKV:\n"
                + "\n".join(report.errors)
            )

        for warning in report.warnings:
            logger.warning(warning)

        logger.info("Phase 3: Initializing VLLM engine...")

        from .inference_backend import VLLMBackend

        self.model_config["inference_backend"] = "vllm"
        self._vllm_backend = VLLMBackend(self.model_config, self.vllm_config)
        self._vllm_backend.initialize()

        logger.info("Phase 3 complete. VLLM engine ready.")
        return self._vllm_backend

    def generate(
        self,
        prompts: List[str],
        request_id: Optional[str] = None,
        config=None
    ):
        """
        执行推理

        Args:
            prompts: 输入 prompts
            request_id: 请求 ID（用于预算上下文）
            config: 可选的生成配置，传递给 VLLM 后端

        Returns:
            生成结果（预算上下文会被内部记录以供调试）
        """
        if self._vllm_backend is None:
            raise RuntimeError("Must run phase3_initialize_vllm first")

        # 创建请求上下文
        context = None
        if self._request_allocator and request_id:
            context = self._request_allocator.create_context(request_id)
            logger.debug(
                f"Request {request_id}: aligned budget = "
                f"{context.aligned_budgets.total_tokens_aligned} tokens"
            )

        result = self._vllm_backend.generate(prompts, config=config)

        # 始终返回生成结果，保持与以往 API 兼容；上下文可用于调试
        self._last_request_context = context
        return result

    def cleanup(self):
        """清理所有资源"""
        if self._vllm_backend:
            self._vllm_backend.cleanup()
            self._vllm_backend = None
        MemoryManager.force_cleanup()
        logger.info("Pipeline cleaned up")
