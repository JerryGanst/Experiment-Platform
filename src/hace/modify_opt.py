import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTAttention

__all__ = ['convert_kvcache_opt_heavy_recent', 'OPTAttention_heavy_hitter']

class HeavyRecentCacheLayer:
    def __init__(self, config):
        self.config = config
        self.heavy_ratio = config.heavy_ratio
        self.recent_ratio = config.recent_ratio
        self.cache = None
        self.attention_masks = None
        self.previous_scores = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None

    def init_cache(self):
        self.cache = None
        self.attention_masks = None
        self.previous_scores = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None

    def update_cache(self, key_states, value_states):
        if self.cache is None:
            self.cache = (key_states, value_states)
        else:
            # 计算heavy hitter和recent token的预算
            seq_len = key_states.shape[2]
            if self.heavy_budget is None:
                self.heavy_budget = int(seq_len * self.heavy_ratio)
                self.recent_budget = int(seq_len * self.recent_ratio)
                self.cache_budget = self.heavy_budget + self.recent_budget

            # 更新缓存
            old_keys, old_values = self.cache
            new_keys = torch.cat([old_keys, key_states], dim=2)
            new_values = torch.cat([old_values, value_states], dim=2)

            # 如果超过预算，只保留最近的token
            if new_keys.shape[2] > self.cache_budget:
                new_keys = new_keys[:, :, -self.cache_budget:]
                new_values = new_values[:, :, -self.cache_budget:]

            self.cache = (new_keys, new_values)
            return self.cache

    def get_cache(self):
        return self.cache

class OPTAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.layer_idx = None
        self._cache_layer = None

    def set_cache_layer(self, cache_layer):
        self._cache_layer = cache_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_head_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 使用本地缓存层
        if self._cache_layer is not None:
            if past_key_value is not None:
                key_states, value_states = self._cache_layer.update_cache(key_states, value_states)
            elif use_cache:
                past_key_value = self._cache_layer.get_cache()
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)

        kv_seq_len = key_states.shape[-2]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, (key_states, value_states) if use_cache else None

def convert_kvcache_opt_heavy_recent(model, config):
    # 创建缓存层列表
    layers_cache = [HeavyRecentCacheLayer(config) for _ in range(config.num_hidden_layers)]
    
    def init_cache():
        for layer in layers_cache:
            layer.init_cache()
    
    # 将init_cache方法附加到模型实例
    model.init_cache = init_cache
    
    # 获取所有注意力层
    attention_layers = []
    for module in model.modules():
        if isinstance(module, OPTAttention):
            attention_layers.append(module)
    
    # 替换每个注意力层
    for idx, old_attention in enumerate(attention_layers):
        attention_layer = OPTAttention_heavy_hitter(config)
        attention_layer.layer_idx = idx
        attention_layer.set_cache_layer(layers_cache[idx])
        
        # 复制权重
        attention_layer.q_proj.weight.data = old_attention.q_proj.weight.data
        attention_layer.k_proj.weight.data = old_attention.k_proj.weight.data
        attention_layer.v_proj.weight.data = old_attention.v_proj.weight.data
        attention_layer.out_proj.weight.data = old_attention.out_proj.weight.data
        
        # 替换原始模块
        for name, module in model.named_modules():
            if module is old_attention:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, attention_layer)
                break
    
    # 保存缓存层列表
    model.layers_cache = layers_cache
    return model 