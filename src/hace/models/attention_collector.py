"""
注意力数据收集器

为 VLLM 部署模式提供注意力权重收集功能。
由于 VLLM 默认不返回注意力权重，需要通过以下方式获取：

1. external_warmup 模式：使用 HuggingFace 模型进行预热采样，
   收集注意力数据后用于 VLLM 推理时的 KV 缓存策略配置。

2. vllm_hook 模式：（实验性）在 VLLM 内部插桩收集注意力数据。
   需要修改 VLLM 源码或使用自定义模型。

核心设计：
- 注意力数据与推理引擎解耦
- 支持注意力数据的缓存和复用
- 为 UnifiedCakeAdaKVAllocator 提供标准化输入
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np

import torch

logger = logging.getLogger(__name__)


@dataclass
class AttentionCollectionConfig:
    """注意力收集配置"""
    mode: str = "external_warmup"  # "external_warmup" 或 "vllm_hook"
    warmup_samples: int = 10  # 预热采样数量
    max_seq_length: int = 2048  # 最大序列长度
    cache_file: Optional[str] = None  # 缓存文件路径
    use_cache: bool = True  # 是否使用缓存
    precision: str = "fp16"  # 收集时的精度


@dataclass
class AttentionData:
    """注意力数据结构"""
    # 每层的注意力权重列表 [layer_idx][batch, heads, seq, seq]
    attention_weights_list: List[np.ndarray]
    # 元数据
    num_layers: int
    num_heads: int
    seq_length: int
    model_name: str
    collection_mode: str
    # 可选：额外统计信息
    layer_statistics: Optional[Dict[str, Any]] = None
    head_statistics: Optional[Dict[str, Any]] = None

    def save(self, path: str) -> None:
        """保存注意力数据到文件"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "attention_weights_list": [arr.tolist() for arr in self.attention_weights_list],
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "seq_length": self.seq_length,
            "model_name": self.model_name,
            "collection_mode": self.collection_mode,
            "layer_statistics": self.layer_statistics,
            "head_statistics": self.head_statistics,
        }

        # 根据文件扩展名选择格式
        if str(path).endswith(".pkl") or str(path).endswith(".pickle"):
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
        else:
            # 默认JSON（注意：大数据可能会很慢）
            with open(save_path, "w") as f:
                json.dump(data, f)

        logger.info(f"Attention data saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "AttentionData":
        """从文件加载注意力数据"""
        if str(path).endswith(".pkl") or str(path).endswith(".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            with open(path, "r") as f:
                data = json.load(f)

        return cls(
            attention_weights_list=[np.array(arr) for arr in data["attention_weights_list"]],
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            seq_length=data["seq_length"],
            model_name=data["model_name"],
            collection_mode=data["collection_mode"],
            layer_statistics=data.get("layer_statistics"),
            head_statistics=data.get("head_statistics"),
        )


class AttentionCollector:
    """
    注意力数据收集器

    为 CAKE/AdaKV 等 KV 缓存优化策略收集注意力权重数据。
    """

    def __init__(self, config: AttentionCollectionConfig):
        self.config = config
        self._hf_model = None
        self._hf_tokenizer = None

    def collect_from_hf_model(
        self,
        model_config: Dict[str, Any],
        sample_prompts: List[str],
    ) -> AttentionData:
        """
        使用 HuggingFace 模型收集注意力数据

        Args:
            model_config: 模型配置
            sample_prompts: 用于预热的样本prompts

        Returns:
            AttentionData: 收集到的注意力数据
        """
        logger.info(f"Collecting attention data using HuggingFace model (samples: {len(sample_prompts)})")

        # 检查缓存
        if self.config.use_cache and self.config.cache_file:
            cache_path = Path(self.config.cache_file)
            if cache_path.exists():
                logger.info(f"Loading attention data from cache: {cache_path}")
                return AttentionData.load(str(cache_path))

        # 加载HF模型
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = model_config["model_name_or_path"]

        # 确定数据类型
        torch_dtype = torch.float16
        if self.config.precision == "bf16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif self.config.precision == "fp32":
            torch_dtype = torch.float32

        logger.info(f"Loading HuggingFace model for attention collection: {model_path}")

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            output_attentions=True,  # 启用注意力输出
        )

        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        # 收集注意力数据
        all_attention_weights = []

        self._hf_model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(sample_prompts[:self.config.warmup_samples]):
                logger.info(f"Processing sample {i + 1}/{min(len(sample_prompts), self.config.warmup_samples)}")

                # 编码输入
                inputs = self._hf_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(self._hf_model.device)

                # 前向传播获取注意力
                outputs = self._hf_model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

                # 提取注意力权重
                # outputs.attentions: tuple of (batch, heads, seq, seq) for each layer
                attentions = outputs.attentions

                if attentions is not None:
                    sample_attentions = [
                        attn.cpu().numpy() for attn in attentions
                    ]
                    all_attention_weights.append(sample_attentions)

        # 聚合注意力数据（取平均）
        num_layers = len(all_attention_weights[0]) if all_attention_weights else 0
        num_heads = all_attention_weights[0][0].shape[1] if all_attention_weights and num_layers > 0 else 0

        aggregated_attentions = []
        for layer_idx in range(num_layers):
            layer_attentions = []
            for sample_attn in all_attention_weights:
                layer_attentions.append(sample_attn[layer_idx])

            # 取平均（需要处理不同序列长度）
            # 这里简化处理：取第一个样本的注意力作为代表
            # 实际应用中可以根据需求选择更复杂的聚合策略
            aggregated_attentions.append(layer_attentions[0])

        # 获取模型信息
        seq_length = all_attention_weights[0][0].shape[-1] if all_attention_weights else 0

        attention_data = AttentionData(
            attention_weights_list=aggregated_attentions,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_length=seq_length,
            model_name=model_path,
            collection_mode="external_warmup",
        )

        # 保存缓存
        if self.config.use_cache and self.config.cache_file:
            attention_data.save(self.config.cache_file)

        # 清理HF模型
        self._cleanup_hf_model()

        return attention_data

    def collect_from_hf_backend(
        self,
        hf_backend: "HuggingFaceBackend",
        sample_prompts: List[str],
    ) -> AttentionData:
        """
        使用已有的 HuggingFace 后端收集注意力数据

        Args:
            hf_backend: HuggingFaceBackend 实例
            sample_prompts: 用于预热的样本prompts

        Returns:
            AttentionData: 收集到的注意力数据
        """
        logger.info(f"Collecting attention data from HuggingFace backend (samples: {len(sample_prompts)})")

        # 检查缓存
        if self.config.use_cache and self.config.cache_file:
            cache_path = Path(self.config.cache_file)
            if cache_path.exists():
                logger.info(f"Loading attention data from cache: {cache_path}")
                return AttentionData.load(str(cache_path))

        # 确保后端已初始化
        if not hf_backend.is_initialized:
            hf_backend.initialize()

        model = hf_backend.get_model()
        tokenizer = hf_backend.get_tokenizer()

        # 收集注意力数据
        all_attention_weights = []

        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(sample_prompts[:self.config.warmup_samples]):
                logger.info(f"Processing sample {i + 1}/{min(len(sample_prompts), self.config.warmup_samples)}")

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(model.device)

                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

                attentions = outputs.attentions
                if attentions is not None:
                    sample_attentions = [attn.cpu().numpy() for attn in attentions]
                    all_attention_weights.append(sample_attentions)

        # 聚合处理
        num_layers = len(all_attention_weights[0]) if all_attention_weights else 0
        num_heads = all_attention_weights[0][0].shape[1] if all_attention_weights and num_layers > 0 else 0

        aggregated_attentions = []
        for layer_idx in range(num_layers):
            # 简化：取第一个样本
            aggregated_attentions.append(all_attention_weights[0][layer_idx])

        seq_length = all_attention_weights[0][0].shape[-1] if all_attention_weights else 0

        attention_data = AttentionData(
            attention_weights_list=aggregated_attentions,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_length=seq_length,
            model_name=hf_backend.model_config.get("model_name_or_path", "unknown"),
            collection_mode="external_warmup",
        )

        if self.config.use_cache and self.config.cache_file:
            attention_data.save(self.config.cache_file)

        return attention_data

    def _cleanup_hf_model(self) -> None:
        """清理HF模型资源"""
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HuggingFace model for attention collection cleaned up")


class VLLMAttentionAdapter:
    """
    VLLM 注意力适配器

    将预热收集的注意力数据转换为 VLLM 可用的 KV 缓存配置。
    这是 CAKE/AdaKV 策略与 VLLM 集成的桥梁。
    """

    def __init__(self, attention_data: AttentionData):
        self.attention_data = attention_data

    def get_attention_weights_for_allocator(self) -> List[np.ndarray]:
        """
        获取用于 UnifiedCakeAdaKVAllocator 的注意力权重

        Returns:
            注意力权重列表，格式为 [layer][batch, heads, seq, seq]
        """
        return self.attention_data.attention_weights_list

    def compute_layer_budgets(
        self,
        total_cache_size: int,
        allocator_config: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        使用注意力数据计算层级 KV 缓存预算

        Args:
            total_cache_size: 总缓存大小
            allocator_config: 分配器配置

        Returns:
            每层的缓存预算列表
        """
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
                    break
            except (ImportError, ModuleNotFoundError):
                continue

        if not UnifiedCakeAdaKVAllocator or not UnifiedCacheConfig:
            logger.warning("UnifiedCakeAdaKVAllocator not available, using uniform allocation")
            num_layers = self.attention_data.num_layers
            if num_layers == 0:
                logger.error("num_layers is 0, returning empty budgets")
                return []
            base_budget = total_cache_size // num_layers
            remainder = total_cache_size % num_layers
            budgets = [base_budget] * num_layers
            for i in range(remainder):
                budgets[i] += 1
            return budgets

        try:
            # 配置分配器
            config = UnifiedCacheConfig(total_cache_size=total_cache_size)
            if allocator_config:
                for key, value in allocator_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            allocator = UnifiedCakeAdaKVAllocator(config)

            # 分配层级预算
            layer_budgets = allocator.allocate_layer_budgets(
                self.attention_data.attention_weights_list
            )
            return layer_budgets
        except Exception as e:
            logger.warning(f"Failed to use UnifiedCakeAdaKVAllocator: {e}, falling back to uniform allocation")
            num_layers = self.attention_data.num_layers
            if num_layers == 0:
                return []
            base_budget = total_cache_size // num_layers
            remainder = total_cache_size % num_layers
            budgets = [base_budget] * num_layers
            for i in range(remainder):
                budgets[i] += 1
            return budgets

    def compute_head_budgets(
        self,
        layer_budgets: List[int],
        allocator_config: Optional[Dict[str, Any]] = None
    ) -> List[List[int]]:
        """
        使用注意力数据计算头级 KV 缓存预算

        Args:
            layer_budgets: 层级预算列表
            allocator_config: 分配器配置

        Returns:
            每层每头的缓存预算列表
        """
        num_heads = self.attention_data.num_heads

        # 防止除零
        if num_heads == 0:
            logger.warning("num_heads is 0, returning empty head budgets")
            return [[] for _ in layer_budgets]

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
                    break
            except (ImportError, ModuleNotFoundError):
                continue

        if not UnifiedCakeAdaKVAllocator or not UnifiedCacheConfig:
            logger.warning("UnifiedCakeAdaKVAllocator not available, using uniform allocation")
            head_budgets_list = []
            for layer_budget in layer_budgets:
                base_budget = layer_budget // num_heads
                remainder = layer_budget % num_heads
                head_budgets = [base_budget] * num_heads
                for i in range(remainder):
                    head_budgets[i] += 1
                head_budgets_list.append(head_budgets)
            return head_budgets_list

        try:
            total_cache_size = sum(layer_budgets)
            config = UnifiedCacheConfig(total_cache_size=total_cache_size)
            if allocator_config:
                for key, value in allocator_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            allocator = UnifiedCakeAdaKVAllocator(config)
            allocator.warmup(self.attention_data.attention_weights_list)

            head_budgets_list = []
            for layer_idx, (attn_weights, layer_budget) in enumerate(
                zip(self.attention_data.attention_weights_list, layer_budgets)
            ):
                head_budgets = allocator.allocate_head_budgets(
                    attn_weights, layer_budget, layer_idx
                )
                head_budgets_list.append(head_budgets)

            return head_budgets_list
        except Exception as e:
            logger.warning(f"Failed to use UnifiedCakeAdaKVAllocator: {e}, falling back to uniform allocation")
            head_budgets_list = []
            for layer_budget in layer_budgets:
                base_budget = layer_budget // num_heads
                remainder = layer_budget % num_heads
                head_budgets = [base_budget] * num_heads
                for i in range(remainder):
                    head_budgets[i] += 1
                head_budgets_list.append(head_budgets)
            return head_budgets_list

    def to_vllm_kv_config(
        self,
        total_cache_size: int,
        block_size: int = 16
    ) -> Dict[str, Any]:
        """
        将预算转换为 VLLM KV 缓存配置格式

        这是将 CAKE/AdaKV 预算映射到 VLLM PagedAttention
        block 管理的桥接接口。

        Args:
            total_cache_size: 总缓存大小
            block_size: VLLM block 大小

        Returns:
            VLLM KV 缓存配置字典
        """
        layer_budgets = self.compute_layer_budgets(total_cache_size)
        head_budgets = self.compute_head_budgets(layer_budgets)

        # 将 token 级预算转换为 block 级配置
        layer_blocks = [budget // block_size for budget in layer_budgets]

        return {
            "layer_budgets": layer_budgets,
            "head_budgets": head_budgets,
            "layer_blocks": layer_blocks,
            "block_size": block_size,
            "total_cache_size": total_cache_size,
            "num_layers": self.attention_data.num_layers,
            "num_heads": self.attention_data.num_heads,
        }


def create_attention_collector(
    vllm_config: Dict[str, Any]
) -> AttentionCollector:
    """
    工厂函数：创建注意力收集器

    Args:
        vllm_config: VLLM 配置字典

    Returns:
        AttentionCollector 实例
    """
    attn_config = vllm_config.get("attention_collection", {})

    config = AttentionCollectionConfig(
        mode=attn_config.get("mode", "external_warmup"),
        warmup_samples=attn_config.get("warmup_samples", 10),
        cache_file=attn_config.get("cache_attention_file"),
        use_cache=attn_config.get("cache_attention_file") is not None,
    )

    return AttentionCollector(config)
