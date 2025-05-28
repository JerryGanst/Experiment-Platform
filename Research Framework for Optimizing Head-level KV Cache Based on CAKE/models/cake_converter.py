"""
CAKE模型转换模块

基于CAKE方法提供层级感知的KV缓存管理
"""
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LayerAnalyzer:
    """层级注意力模式分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化层级分析器。

        Args:
            config: 层分析相关的配置字典，例如：
                    {
                        "attention_pattern_analysis": True,
                        "layer_importance_scoring": True,
                        "dynamic_allocation": True
                        # ... 其他分析参数
                    }
        """
        self.config = config
        self.layer_patterns: Dict[int, Any] = {}  # 存储每层的分析结果
        self.attention_statistics: Dict[int, Any] = {} # 存储每层的注意力统计信息
        self.layer_importance_scores: Dict[int, float] = {} # 存储每层的重要性分数
        logger.info(f"层级分析器已初始化，配置: {self.config}")

    def analyze_layer_attention_patterns(self, model: torch.nn.Module, sample_inputs: Dict[str, torch.Tensor]):
        """
        分析模型每一层的注意力模式特征。
        这个方法会通过运行模型（或部分层）并收集每层的注意力分布、
        稀疏度等统计信息。

        Args:
            model: 要分析的PyTorch模型。
            sample_inputs: 用于模型前向传播以收集注意力数据的样本输入。
                           例如: {"input_ids": ..., "attention_mask": ...}
        
        Note:
            具体的实现会依赖于模型架构 (例如 Transformer)。
            需要注册钩子 (hooks) 到注意力模块来捕获注意力权重。
        """
        if not self.config.get("attention_pattern_analysis", False):
            logger.info("注意力模式分析未启用，跳过分析。")
            return

        logger.info(f"开始分析模型 {model.__class__.__name__} 的层级注意力模式...")
        # 示例：这里需要实际的逻辑来注册钩子并运行模型以收集注意力数据
        # 以下为伪代码占位符
        # for layer_idx, layer in enumerate(model.model.layers): # 假设是Llama类模型结构
        #     # 注册钩子到 self_attn 模块
        #     hook = layer.self_attn.register_forward_hook(self._attention_hook_fn(layer_idx))
        #     # 运行一小部分数据通过模型
        #     # model_output = model(**sample_inputs) # 简化示例，实际可能更复杂
        #     # hook.remove() # 移除钩子
        #     logger.debug(f"已分析层 {layer_idx} 的注意力模式。")
        # self.attention_statistics = ... # 基于收集到的数据填充

        # 此处应填充实际的注意力分析逻辑
        logger.warning("`analyze_layer_attention_patterns` 方法需要实现实际的注意力数据收集和分析逻辑。")
        pass

    def _attention_hook_fn(self, layer_idx: int):
        """
        内部辅助函数，用于创建注意力钩子。
        """
        def hook(module, input, output):
            # output[1] 通常是注意力权重 (对于HuggingFace Transformers)
            # 注意：这取决于具体的模型实现
            attention_weights = output[1] 
            if attention_weights is not None:
                # 对注意力权重进行统计或存储
                # 例如计算均值、方差、稀疏度等
                self.attention_statistics.setdefault(layer_idx, []).append(attention_weights.detach().cpu())
                logger.debug(f"层 {layer_idx} 捕获到注意力权重，形状: {attention_weights.shape}")
        return hook

    def compute_layer_importance_scores(self, strategy: str = "default") -> Dict[int, float]:
        """
        根据收集到的注意力统计信息或其他分析结果，计算每层的重要性分数。
        这些分数将用于CAKE的缓存分配策略。

        Args:
            strategy: 计算重要性分数的策略 (例如："attention_entropy", "gradient_based", "random")。

        Returns:
            一个字典，键是层索引，值是该层的重要性分数。
        """
        if not self.config.get("layer_importance_scoring", False):
            logger.info("层重要性评分未启用，跳过计算。")
            return {}

        logger.info(f"开始计算层重要性分数，策略: {strategy}...")
        # 示例：这里需要基于 this.attention_statistics 来计算分数
        # 以下为伪代码占位符
        num_layers = len(self.attention_statistics) # 假设已经收集了统计数据
        if num_layers == 0 and strategy != "random": # 如果没有统计数据，且不是随机策略
             logger.warning("没有可用的注意力统计数据来计算层重要性分数。请先运行 `analyze_layer_attention_patterns`。")
             # 可以考虑返回一个默认的均匀分布或引发错误
             # for layer_idx in range(model_num_layers): # 需要知道模型的总层数
             #    self.layer_importance_scores[layer_idx] = 1.0 / model_num_layers


        for layer_idx in self.attention_statistics.keys():
            # 这是一个非常简化的示例，实际策略会更复杂
            # 例如，可以基于注意力权重的熵、稀疏度或梯度信息
            # score = torch.stack(self.attention_statistics[layer_idx]).mean().item() 
            # self.layer_importance_scores[layer_idx] = score
            self.layer_importance_scores[layer_idx] = 1.0 # 占位符
        
        # 标准化分数 (可选)
        # total_score = sum(self.layer_importance_scores.values())
        # if total_score > 0:
        #     for layer_idx in self.layer_importance_scores:
        #         self.layer_importance_scores[layer_idx] /= total_score
        
        logger.info(f"层重要性分数计算完成: {self.layer_importance_scores}")
        # 此处应填充实际的重要性分数计算逻辑
        logger.warning("`compute_layer_importance_scores` 方法需要实现实际的分数计算逻辑。")
        return self.layer_importance_scores


class CAKECacheManager:
    """
    CAKE缓存管理器。
    负责根据层级重要性分数和总预算来分配和管理KV缓存。
    """
    
    def __init__(self, model_config: Dict[str, Any], cake_specific_config: Dict[str, Any], layer_importance_scores: Dict[int, float], num_layers: int):
        """
        初始化CAKE缓存管理器。

        Args:
            model_config: 模型的通用配置 (例如 Transformers model.config)。
            cake_specific_config: CAKE相关的配置，例如：
                {
                    "allocation_strategy": "adaptive" | "uniform" | "attention_based",
                    "cache_budget": 0.8, # 占原始缓存大小的比例
                    "dynamic_allocation": True,
                    # ... 其他CAKE参数
                }
            layer_importance_scores: 每层的重要性分数。
            num_layers: 模型的总层数。
        """
        self.model_config = model_config
        self.config = cake_specific_config
        self.layer_importance_scores = layer_importance_scores
        self.num_layers = num_layers
        
        self.total_budget_ratio = self.config.get("cache_budget", 0.8)
        self.allocation_strategy = self.config.get("allocation_strategy", "adaptive")
        
        self.layer_budgets_abs: Dict[int, int] = {} # 每层绝对的KV缓存条目数
        self.original_kv_cache_len_per_layer: Optional[int] = None # 原始情况下每层的KV缓存长度 (如果统一)

        logger.info(f"CAKE缓存管理器已初始化。策略: {self.allocation_strategy}, 总预算比例: {self.total_budget_ratio}")
        self.allocate_layer_budgets()

    def allocate_layer_budgets(self):
        """
        根据指定的分配策略和层级重要性分数，为每一层分配KV缓存的预算。
        预算可以是绝对的条目数，或者是相对于原始缓存大小的比例。
        """
        logger.info(f"开始为 {self.num_layers} 层分配缓存预算，策略: {self.allocation_strategy}...")

        # 假设原始情况下，所有层的KV缓存长度是相同的，并且可以通过模型配置获取
        # 例如，从 model_config.max_position_embeddings 或类似参数
        # 这里我们暂时假设一个原始长度，实际应从模型配置读取
        self.original_kv_cache_len_per_layer = self.model_config.get("max_position_embeddings", 2048) # 示例值
        
        total_kv_entries_original = self.num_layers * self.original_kv_cache_len_per_layer
        total_kv_entries_budgeted = int(total_kv_entries_original * self.total_budget_ratio)

        logger.info(f"原始总KV条目数: {total_kv_entries_original}, 预算总KV条目数: {total_kv_entries_budgeted}")

        if self.allocation_strategy == "uniform":
            budget_per_layer = total_kv_entries_budgeted // self.num_layers
            for i in range(self.num_layers):
                self.layer_budgets_abs[i] = budget_per_layer
        elif self.allocation_strategy == "adaptive" or self.allocation_strategy == "attention_based":
            if not self.layer_importance_scores:
                logger.warning(f"无法使用 '{self.allocation_strategy}' 策略，因为层重要性分数为空。将回退到 'uniform' 策略。")
                # 回退到 uniform
                budget_per_layer = total_kv_entries_budgeted // self.num_layers
                for i in range(self.num_layers):
                    self.layer_budgets_abs[i] = budget_per_layer
            else:
                total_score = sum(self.layer_importance_scores.values())
                if total_score == 0: # 避免除以零
                    logger.warning("总重要性分数为零，无法按比例分配。将回退到 'uniform' 策略。")
                    budget_per_layer = total_kv_entries_budgeted // self.num_layers
                    for i in range(self.num_layers):
                        self.layer_budgets_abs[i] = budget_per_layer
                else:
                    for i in range(self.num_layers):
                        # 分数越高，预算越多
                        score = self.layer_importance_scores.get(i, 0) # 如果某层没有分数，则预算为0或平均值
                        proportional_budget = int(total_kv_entries_budgeted * (score / total_score))
                        self.layer_budgets_abs[i] = proportional_budget
        else:
            logger.error(f"未知的分配策略: {self.allocation_strategy}。将使用 'uniform' 策略。")
            budget_per_layer = total_kv_entries_budgeted // self.num_layers
            for i in range(self.num_layers):
                self.layer_budgets_abs[i] = budget_per_layer
        
        # 调整以确保总预算不超过限制 (由于取整可能导致总和略有偏差)
        current_total_allocated = sum(self.layer_budgets_abs.values())
        if current_total_allocated > total_kv_entries_budgeted:
            # 按比例缩减，或从预算最多的层扣除
            logger.debug(f"分配的总预算 {current_total_allocated} 超出目标 {total_kv_entries_budgeted}，进行调整。")
            # 简单调整：从最大的开始扣除，直到满足预算
            diff = current_total_allocated - total_kv_entries_budgeted
            sorted_layers = sorted(self.layer_budgets_abs.items(), key=lambda item: item[1], reverse=True)
            for layer_idx, budget in sorted_layers:
                if diff <= 0: break
                can_reduce = min(budget, diff)
                self.layer_budgets_abs[layer_idx] -= can_reduce
                diff -= can_reduce
        elif current_total_allocated < total_kv_entries_budgeted and self.num_layers > 0 :
             logger.debug(f"分配的总预算 {current_total_allocated} 低于目标 {total_kv_entries_budgeted}，进行调整。")
             diff = total_kv_entries_budgeted - current_total_allocated
             # 简单调整：平均分配给所有层，或分配给得分最高的层
             add_per_layer = diff // self.num_layers
             remainder = diff % self.num_layers
             for i in range(self.num_layers):
                 self.layer_budgets_abs[i] += add_per_layer
             # 将余数分配给得分最高的层或前几层
             sorted_scores = sorted(self.layer_importance_scores.items(), key=lambda item: item[1], reverse=True)
             for i in range(remainder):
                 if i < len(sorted_scores):
                    layer_to_add_idx = sorted_scores[i][0]
                    self.layer_budgets_abs[layer_to_add_idx] +=1


        logger.info(f"层级缓存预算分配完成: {self.layer_budgets_abs}")
        logger.info(f"最终分配的总KV条目数: {sum(self.layer_budgets_abs.values())} (目标: {total_kv_entries_budgeted})")


    def get_cache_budget_for_layer(self, layer_idx: int) -> int:
        """
        获取指定层的KV缓存预算（条目数）。

        Args:
            layer_idx: 层索引。

        Returns:
            该层的KV缓存条目数。
        """
        return self.layer_budgets_abs.get(layer_idx, 0)

    def update_cache_for_layer(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, past_key_value: Optional[tuple]):
        """
        为特定层更新（或创建）KV缓存，需要遵循该层分配的预算。
        这通常在模型的注意力层内部被调用。

        Args:
            layer_idx: 当前层的索引。
            key_states: 新计算得到的key状态。
            value_states: 新计算得到的value状态。
            past_key_value: 来自前一个时间步的 (past_key, past_value) 元组。

        Returns:
            更新后的 (key, value) 元组，其长度受限于该层的预算。
        
        Note:
            这部分逻辑非常关键，需要与模型（例如LlamaAttention）的KV缓存处理方式紧密集成。
            可能需要修改模型代码以调用此方法，或者通过钩子注入此逻辑。
            如果预算小于当前缓存+新token，则需要实现驱逐策略。
        """
        budget_for_layer = self.get_cache_budget_for_layer(layer_idx)
        # logger.debug(f"层 {layer_idx}: 更新KV缓存，预算 {budget_for_layer} 条目。输入K形状: {key_states.shape}")

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=-2)
            value_states = torch.cat([past_value, value_states], dim=-2)

        # 应用驱逐策略如果超出预算
        # seq_len_dim typically -2 for KV cache (batch_size, num_heads, seq_len, head_dim)
        current_seq_len = key_states.shape[-2] 
        if current_seq_len > budget_for_layer:
            # 实现一个简单的驱逐策略：保留最新的 (FIFO for old tokens)
            key_states = key_states[..., -budget_for_layer:, :]
            value_states = value_states[..., -budget_for_layer:, :]
            # logger.debug(f"层 {layer_idx}: KV缓存超出预算 ({current_seq_len}/{budget_for_layer})，已裁剪。")
        
        return (key_states, value_states)


def apply_cake_to_model(
    model: torch.nn.Module, 
    model_config_hf: Dict[str, Any], # 来自HuggingFace model.config.to_dict()
    cake_experiment_config: Dict[str, Any], # 来自EXPERIMENT_CONFIG中CAKE相关的部分
    cake_model_specific_config: Dict[str, Any] # 来自CAKE_MODEL_CONFIG
    ) -> torch.nn.Module:
    """
    将CAKE缓存管理应用于给定的PyTorch模型。
    这个函数会：
    1. 初始化LayerAnalyzer并（可选地）分析模型的层级特性。
    2. 基于分析结果和配置初始化CAKECacheManager。
    3. 修改模型的注意力层（或其KV缓存处理逻辑）以使用CAKECacheManager进行层级感知的缓存管理。

    Args:
        model: 要应用CAKE的PyTorch模型 (通常是HuggingFace Transformers模型)。
        model_config_hf: 模型的HuggingFace配置字典。
        cake_experiment_config: 实验相关的CAKE配置。
        cake_model_specific_config: 模型准备相关的CAKE配置。

    Returns:
        修改后的模型，其KV缓存由CAKE管理。
    """
    logger.info(f"开始将CAKE应用于模型: {model_config_hf.get('model_type', '未知类型')}")
    logger.info(f"实验CAKE配置: {cake_experiment_config}")
    logger.info(f"模型特定CAKE配置: {cake_model_specific_config}")

    num_hidden_layers = model_config_hf.get("num_hidden_layers", model_config_hf.get("n_layer")) # Llama uses num_hidden_layers, GPTNeoX uses n_layer
    if num_hidden_layers is None:
        logger.error("无法从模型配置中确定层数 (num_hidden_layers 或 n_layer)。")
        raise ValueError("模型配置中缺少层数信息。")

    # 1. 初始化层分析器
    # LayerAnalyzer的配置可以从 cake_experiment_config["layer_analysis_configs"] 获取
    layer_analyzer_config = cake_experiment_config.get("layer_analysis_configs", {})
    analyzer = LayerAnalyzer(config=layer_analyzer_config)

    # (可选) 执行层级分析 - 这可能需要在实际数据上运行模型
    # if layer_analyzer_config.get("attention_pattern_analysis", False):
    #     logger.info("执行层级注意力模式分析...")
    #     # 注意：这里需要一个 sample_inputs。在实际流程中，这可能来自校准数据集。
    #     # dummy_input_ids = torch.randint(0, model_config_hf.get("vocab_size", 32000), (1, 10), device=model.device)
    #     # sample_inputs_for_analysis = {"input_ids": dummy_input_ids}
    #     # analyzer.analyze_layer_attention_patterns(model, sample_inputs_for_analysis)
    #     logger.warning("层级分析的样本输入部分需要实际实现。")
    
    # 计算层重要性分数
    # 分数计算策略可以从 cake_model_specific_config 或 cake_experiment_config 获取
    importance_strategy = cake_model_specific_config.get("importance_computation_strategy", "default")
    layer_importance_scores = analyzer.compute_layer_importance_scores(strategy=importance_strategy)
    
    # 如果分数为空 (例如分析未执行或未成功)，则使用默认均匀分数
    if not layer_importance_scores and num_hidden_layers > 0 :
        logger.info("层重要性分数为空，将为所有层使用默认的均匀重要性分数。")
        default_score = 1.0 / num_hidden_layers
        for i in range(num_hidden_layers):
            layer_importance_scores[i] = default_score


    # 2. 初始化CAKE缓存管理器
    # CAKECacheManager的配置主要来自 cake_experiment_config 和 cake_model_specific_config
    # 例如: allocation_strategy, cache_budget
    cache_manager_config = {
        "allocation_strategy": cake_experiment_config.get("layer_allocation_strategies", ["adaptive"])[0], # 取第一个作为示例
        "cache_budget": cake_experiment_config.get("cache_budgets", [0.8])[0], # 取第一个作为示例
        "dynamic_allocation": cake_model_specific_config.get("dynamic_allocation_default", True)
    }
    cake_manager = CAKECacheManager(
        model_config=model_config_hf, 
        cake_specific_config=cache_manager_config, 
        layer_importance_scores=layer_importance_scores,
        num_layers=num_hidden_layers
    )

    # 3. 修改模型的注意力层以使用CAKE缓存管理器
    # 这是最复杂的部分，高度依赖于模型架构 (Llama, Mistral, Falcon等)
    # 我们需要遍历模型的每一层，并替换或包装其注意力模块中的KV缓存逻辑。
    
    # 假设模型是常见的HuggingFace Transformer结构，例如Llama
    # model.model.layers 是一个 ModuleList
    model_type = model_config_hf.get("model_type", "").lower()
    logger.info(f"检测到模型类型: {model_type}，开始修改注意力层...")

    if "llama" in model_type or "mistral" in model_type: # Mistral 架构与Llama类似
        # 遍历LlamaDecoderLayer
        for layer_idx, decoder_layer in enumerate(model.model.layers):
            original_attention_module = decoder_layer.self_attn
            
            # 创建一个包装函数或修改 forward 方法
            # 这个新的forward方法会在计算K,V之后，但在注意力计算之前，
            # 使用 cake_manager.update_cache_for_layer 来处理KV缓存
            
            # 方式一：替换整个注意力模块 (如果它的forward可以很容易地被包装)
            # new_attention_module = PatchedLlamaAttention(original_attention_module, cake_manager, layer_idx)
            # decoder_layer.self_attn = new_attention_module
            
            # 方式二：直接修改forward方法 (更具侵入性，但可能更灵活)
            # 或者通过钩子注入
            # 这是非常核心的修改，需要确保与原始逻辑兼容
            
            # 以下是一个简化的概念演示，实际实现需要更鲁棒
            # 我们需要确保 `past_key_value` 被正确传递和管理
            # 并且与模型的 `use_cache` 和 `output_attentions` 等参数兼容

            # 为LlamaAttention (或等效模块)的forward方法打补丁
            # 目的是在 `apply_rotary_pos_emb` 之后，实际注意力计算之前，以及 `past_key_value` 更新时介入
            
            # 存储原始的forward方法
            original_forward = original_attention_module.forward

            def cake_aware_llama_attention_forward(self, *args, **kwargs):
                # 从kwargs中获取past_key_value
                # 注意: LlamaAttention的forward签名是 (hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
                # 我们需要正确地捕获和传递这些参数
                
                # 调用原始的forward，但可能只到KV计算和旋转编码完成
                # 或者，如果原始forward能够返回KV而不仅仅是注意力输出，那会更好
                # 另一种方式是，在原始forward之后，如果它返回了更新后的KV缓存，
                # 我们再对这个KV缓存应用预算限制。

                # 假设我们能拿到 past_key_value (它是输入参数)
                # 以及新计算的 key_states, value_states (它们在原始forward内部计算)

                # **简化方案：在原始forward调用 *之后* 裁剪 past_key_value**
                # 这要求原始的forward调用会返回 past_key_value (如果 use_cache=True)
                
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                # past_key_value_input = args[3] if len(args) > 3 else kwargs.get("past_key_value") # (key, value)
                past_key_value_arg_index = 3 # 根据LlamaAttention的签名
                
                # 确保我们能拿到past_key_value
                current_past_key_value = None
                if len(args) > past_key_value_arg_index:
                    current_past_key_value = args[past_key_value_arg_index]
                elif "past_key_value" in kwargs:
                    current_past_key_value = kwargs["past_key_value"]


                # 调用原始的注意力前向传播
                # output = (attn_output, attn_weights_reshaped, past_key_value)
                attn_output, attn_weights, new_past_key_value = original_forward(*args, **kwargs)
                
                use_cache = kwargs.get("use_cache", False)
                if use_cache and new_past_key_value is not None:
                    # new_past_key_value 是 (key, value)
                    # cake_manager 需要接收的是当前时间步生成的 key_states, value_states
                    # 以及来自上一个时间步的 past_key_value (即 current_past_key_value)
                    
                    # 从 new_past_key_value 中分离出当前生成的 K, V
                    # new_past_key_value 包含了历史和当前
                    # 我们需要的是当前新生成的 K,V 部分来传递给 update_cache_for_layer
                    # 或者，让 update_cache_for_layer 直接处理合并后的 new_past_key_value
                    
                    # 策略：直接让cake_manager处理合并后的 new_past_key_value
                    # cake_manager.update_cache_for_layer 内部会处理拼接和裁剪
                    # 这要求 update_cache_for_layer 的 past_key_value 参数是None，
                    # 然后它只使用传入的 key_states 和 value_states 进行裁剪。
                    # 我们需要调整 update_cache_for_layer 的逻辑或这里的调用方式。

                    # 方案调整:
                    # update_cache_for_layer 接收的是当前步的K,V和来自上一时间步的past_k_v
                    # 它返回的是合并和裁剪后的 K,V
                    # 所以，我们需要在调用原始forward之前或者修改其内部逻辑

                    # **更稳健的方案: 直接修改 Attention 模块的 KV Cache 更新部分**
                    # 这通常在 forward 方法的末尾，当 use_cache=True 时。
                    # 如果无法直接修改，则下面的方法是一个近似：
                    
                    # 假设 new_past_key_value 是 (updated_k, updated_v)
                    updated_k, updated_v = new_past_key_value
                    
                    # 应用CAKE的预算 (直接裁剪完整的updated_k, updated_v)
                    budget_for_this_layer = cake_manager.get_cache_budget_for_layer(layer_idx)
                    
                    if updated_k.shape[-2] > budget_for_this_layer: # seq_len dim
                        final_k = updated_k[..., -budget_for_this_layer:, :]
                        final_v = updated_v[..., -budget_for_this_layer:, :]
                        new_past_key_value = (final_k, final_v)
                        # logger.debug(f"Layer {layer_idx} (CAKE): KV cache裁剪至 {budget_for_this_layer}. Orig shape: {updated_k.shape}")

                return attn_output, attn_weights, new_past_key_value

            # 使用 functools.partial 来绑定 self (即 original_attention_module) 和其他参数
            import functools
            # 将 original_attention_module 的 forward 方法替换掉
            # 注意：这里直接替换方法可能在多线程或序列化时产生问题，需要谨慎测试
            # 更安全的方式是创建一个新的子类或包装类
            original_attention_module.forward = functools.partial(cake_aware_llama_attention_forward, original_attention_module)
            logger.info(f"已为Llama层 {layer_idx} 的自注意力模块打上CAKE补丁。")
            
    elif "falcon" in model_type:
        # Falcon模型的注意力机制 (例如 FalconAttention) 可能不同
        # 需要针对性地修改
        logger.warning(f"Falcon模型的CAKE集成尚未完全实现。需要针对 FalconAttention 进行适配。")
        # for layer_idx, decoder_layer in enumerate(model.transformer.h): # Falcon结构
        #    pass
        pass
    elif "gptneox" in model_type:
        logger.warning(f"GPTNeoX模型的CAKE集成尚未完全实现。需要针对 GPTNeoXAttention 进行适配。")
        # for layer_idx, decoder_layer in enumerate(model.gpt_neox.layers):
        #   pass
        pass
    else:
        logger.error(f"不支持的模型类型: {model_type}。无法应用CAKE。")
        raise NotImplementedError(f"模型类型 {model_type} 的CAKE集成尚未实现。")

    model.is_cake_enabled = True # 添加一个标记
    model.cake_manager = cake_manager # 可以将管理器附加到模型上，方便调试或动态调整
    logger.info(f"CAKE已成功应用于模型。is_cake_enabled: {hasattr(model, 'is_cake_enabled')}")
    return model

def convert_model_to_cake(model, model_config_hf, cake_exp_config, cake_model_config):
    """
    高级封装函数，用于将标准HuggingFace模型转换为启用了CAKE缓存的模型。
    """
    # 可以在这里添加更多检查，例如模型是否已经被转换等
    if hasattr(model, 'is_cake_enabled') and model.is_cake_enabled:
        logger.info("模型已经被转换为CAKE，跳过重复转换。")
        return model
        
    return apply_cake_to_model(model, model_config_hf, cake_exp_config, cake_model_config)

if __name__ == '__main__':
    # 用于模块独立测试的示例代码
    logging.basicConfig(level=logging.DEBUG)
    logger.info("测试CAKE转换器模块...")

    # 伪造一些配置进行测试
    dummy_model_config = {"num_hidden_layers": 2, "max_position_embeddings": 128, "model_type": "llama"}
    dummy_cake_exp_config = {
        "layer_analysis_configs": {"attention_pattern_analysis": False, "layer_importance_scoring": True},
        "layer_allocation_strategies": ["adaptive"],
        "cache_budgets": [0.5] 
    }
    dummy_cake_model_config = {"dynamic_allocation_default": True}

    # 1. 测试 LayerAnalyzer
    analyzer_cfg = dummy_cake_exp_config["layer_analysis_configs"]
    analyzer = LayerAnalyzer(analyzer_cfg)
    # 假设有一些伪造的注意力统计数据 (正常情况下应通过分析真实模型获得)
    analyzer.attention_statistics = {0: [torch.rand(4,4)], 1: [torch.rand(4,4) * 0.5]} # layer 0 更重要
    scores = analyzer.compute_layer_importance_scores()
    logger.info(f"计算得到的重要性分数: {scores}")
    assert 0 in scores and 1 in scores

    # 2. 测试 CAKECacheManager
    manager_cfg = {
        "allocation_strategy": dummy_cake_exp_config["layer_allocation_strategies"][0],
        "cache_budget": dummy_cake_exp_config["cache_budgets"][0]
    }
    cake_manager = CAKECacheManager(dummy_model_config, manager_cfg, scores, num_layers=dummy_model_config["num_hidden_layers"])
    logger.info(f"层0的预算: {cake_manager.get_cache_budget_for_layer(0)}")
    logger.info(f"层1的预算: {cake_manager.get_cache_budget_for_layer(1)}")
    assert cake_manager.get_cache_budget_for_layer(0) > 0
    assert sum(cake_manager.layer_budgets_abs.values()) <= int(dummy_model_config["num_hidden_layers"] * dummy_model_config["max_position_embeddings"] * manager_cfg["cache_budget"])


    # 3. 测试 KV 缓存更新逻辑 (概念性)
    # 假设层0的预算是 10 个token
    cake_manager.layer_budgets_abs[0] = 10 
    # (batch_size, num_heads, seq_len, head_dim)
    current_k = torch.randn(1, 2, 5, 8) # 当前有5个token
    current_v = torch.randn(1, 2, 5, 8)
    
    # 新进入3个token
    new_k_step1 = torch.randn(1, 2, 3, 8)
    new_v_step1 = torch.randn(1, 2, 3, 8)
    
    # 第一次更新 (5+3=8 <= 10)
    updated_kv_step1 = cake_manager.update_cache_for_layer(0, new_k_step1, new_v_step1, past_key_value=(current_k, current_v))
    logger.info(f"Step 1 KV cache K shape: {updated_kv_step1[0].shape}") # 应该 (1,2,8,8)
    assert updated_kv_step1[0].shape[-2] == 8

    # 再新进入4个token (8+4=12 > 10, 会裁剪)
    new_k_step2 = torch.randn(1, 2, 4, 8)
    new_v_step2 = torch.randn(1, 2, 4, 8)
    updated_kv_step2 = cake_manager.update_cache_for_layer(0, new_k_step2, new_v_step2, past_key_value=updated_kv_step1)
    logger.info(f"Step 2 KV cache K shape: {updated_kv_step2[0].shape}") # 应该 (1,2,10,8)
    assert updated_kv_step2[0].shape[-2] == 10
    
    logger.info("CAKE转换器模块基本测试完成。")
    # 注意：apply_cake_to_model 的完整测试需要一个真实的HuggingFace模型实例。 