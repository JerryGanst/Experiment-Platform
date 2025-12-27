import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
from cake.utils import adjust_budgets

class CakeCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.pref_scores = []
        self.evict_scores = []
        self.layer_budget = []
        self.head_concentrations = []  # HACE: Per-layer head concentration scores
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `CakeCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_score(
        self,
        pref_score: torch.Tensor,
        evict_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        head_concentration: Optional[torch.Tensor] = None,  # HACE: [num_kv_heads]
    ):
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)
        if head_concentration is not None:
            self.head_concentrations.append(head_concentration)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. CakeCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `CakeCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "CakeCache":
        """Converts a cache in the legacy cache format into an equivalent `CakeCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    @classmethod
    def from_hybrid_cache(cls, past_key_values: Optional[HybridCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["CakeCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = CakeCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.pref_scores = self.pref_scores
            current_split.evict_scores = self.evict_scores
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["CakeCache"]) -> "CakeCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
            cache.pref_scores = self.pref_scores
            cache.evict_scores = self.evict_scores
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]



class CakeprefillKVCache:
    def __init__(
        self,
        cache_size=512,
        window_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads = 32,
        num_layers = 32,
        use_cascading = False,
        use_head_adaptive = False,  # HACE: Enable head-level adaptive allocation
    ):

        self.window_size = window_size
        self.total_size = (cache_size-window_size) * num_layers
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cascading = use_cascading  # If true, ensure high attention precision
        self.use_head_adaptive = use_head_adaptive  # HACE: head-level optimization

        # print(f"CakeprefillKVCache: {self.total_size}, {self.window_size}")
        
    def __call__(self, past_key_values, seq_len):
        if seq_len<=self.cache_size+self.window_size:
            return past_key_values

        pref_scores = past_key_values.pref_scores

        layer_budgets = [pref_score/sum(pref_scores)*self.total_size for pref_score in pref_scores]

        layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)

        # HACE: Get head-level mode
        import os
        head_mode = os.environ.get("HACE_HEAD_MODE", "").strip().lower()
        # Supported modes: adakv (Ada-KV algorithm), lh1, lh2 (simple concentration-based)
        use_head_alloc = self.use_head_adaptive and head_mode in ("adakv", "lh1", "lh2")

        if self.use_cascading:
            layer_idx = 0
            print(layer_budgets)
            for budget in layer_budgets:
                if budget>= seq_len-self.window_size:
                    budget = seq_len-self.window_size

                head_budgets = None
                if use_head_alloc:
                    if head_mode == "adakv":
                        # Ada-KV: use hh_score for counting-based allocation
                        head_budgets = self._compute_head_budgets_adakv(
                            past_key_values.evict_scores[layer_idx],
                            budget
                        )
                    elif layer_idx < len(past_key_values.head_concentrations):
                        # lh1/lh2: use concentration-based allocation
                        head_budgets = self._compute_head_budgets_simple(
                            past_key_values.head_concentrations[layer_idx],
                            budget,
                            head_mode
                        )

                past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget, head_budgets)
                past_key_values.layer_budget[layer_idx]=budget
                layer_idx +=1
        else:
            layer_idx = 0
            if len(layer_budgets) ==self.num_layers:
                for budget in layer_budgets:
                    if budget>= seq_len-self.window_size:
                        budget = seq_len-self.window_size

                    head_budgets = None
                    if use_head_alloc:
                        if head_mode == "adakv":
                            # Ada-KV: use hh_score for counting-based allocation
                            head_budgets = self._compute_head_budgets_adakv(
                                past_key_values.evict_scores[layer_idx],
                                budget
                            )
                        elif layer_idx < len(past_key_values.head_concentrations):
                            # lh1/lh2: use concentration-based allocation
                            head_budgets = self._compute_head_budgets_simple(
                                past_key_values.head_concentrations[layer_idx],
                                budget,
                                head_mode
                            )

                    past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget, head_budgets)
                    past_key_values.layer_budget[layer_idx]=budget
                    layer_idx +=1

        return past_key_values

    def _compute_head_budgets_adakv(self, hh_score, layer_budget, alpha=0.5):
        """
        Compute per-head budgets using Ada-KV's counting-based algorithm.

        Ada-KV Algorithm (Algorithm 2 from paper):
        1. Concatenate attention weights across all heads
        2. Select top-B indices globally
        3. Count frequency of each head in top-B
        4. Allocate budget proportional to frequency
        5. Blend with uniform allocation using alpha

        Args:
            hh_score: Tensor [bsz, num_kv_heads, seq_len] - attention scores per head
            layer_budget: Total budget for this layer
            alpha: Blending factor (default 0.5)
                   final_budget = alpha * adaptive + (1-alpha) * uniform

        Returns:
            Tensor [num_kv_heads] - budget per head
        """
        bsz, num_kv_heads, seq_len = hh_score.shape

        # Step 1: Flatten across heads -> [bsz, num_kv_heads * seq_len]
        flat_scores = hh_score.reshape(bsz, -1)

        # Step 2: Create head indicators
        head_ids = torch.arange(num_kv_heads, device=hh_score.device)
        head_ids = head_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)  # [num_kv_heads * seq_len]

        # Step 3: Select top-B indices globally
        total_budget = int(layer_budget)
        _, top_indices = flat_scores.topk(min(total_budget, flat_scores.shape[-1]), dim=-1)

        # Step 4: Count frequency of each head in top-B
        # top_indices: [bsz, B] -> map to head_ids
        top_head_ids = head_ids[top_indices[0]]  # Take first batch item
        head_counts = torch.zeros(num_kv_heads, device=hh_score.device)
        for h in range(num_kv_heads):
            head_counts[h] = (top_head_ids == h).sum().float()

        # Step 5: Normalize to get adaptive budget
        if head_counts.sum() > 0:
            adaptive_budget = head_counts / head_counts.sum() * layer_budget
        else:
            adaptive_budget = torch.ones(num_kv_heads, device=hh_score.device) * layer_budget / num_kv_heads

        # Step 6: Blend with uniform allocation
        uniform_budget = layer_budget / num_kv_heads
        head_budgets = alpha * adaptive_budget + (1 - alpha) * uniform_budget

        # Ensure minimum budget of 1 per head
        head_budgets = head_budgets.clamp(min=1)

        return head_budgets

    def _compute_head_budgets_simple(self, head_concentration, layer_budget, head_mode):
        """
        Simple head budget allocation based on concentration (our experimental version).

        This is a simplified alternative to Ada-KV for hypothesis testing:
        - lh1: Low concentration heads get more budget (attention is dispersed)
        - lh2: High concentration heads get more budget (attention is focused)

        Args:
            head_concentration: Tensor [num_kv_heads] - concentration score per head
            layer_budget: Total budget for this layer
            head_mode: 'lh1' or 'lh2'

        Returns:
            Tensor [num_kv_heads] - budget per head
        """
        if head_mode == "lh1":
            # Low concentration -> high budget (dispersed attention needs more tokens)
            weights = 1.0 / (head_concentration + 1e-8)
        elif head_mode == "lh2":
            # High concentration -> high budget (focused attention is important)
            weights = head_concentration
        else:
            weights = torch.ones_like(head_concentration)

        weights = weights / weights.sum()
        head_budgets = weights * layer_budget
        head_budgets = head_budgets.clamp(min=1)

        return head_budgets

    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget, head_budgets=None):
        """
        Evict tokens from layer's KV cache.

        Args:
            past_key_values: Cache object
            layer_idx: Layer index
            budget: Total layer budget (used when head_budgets is None)
            head_budgets: Optional per-head budgets tensor [num_kv_heads]
                         If provided, each head gets its own budget allocation
        """
        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape

        num_key_value_groups = self.num_heads // num_key_value_heads
        hh_score = past_key_values.evict_scores[layer_idx]

        if budget > hh_score.shape[-1]:
            budget = hh_score.shape[-1]

        # HACE: Head-level adaptive allocation
        if head_budgets is not None and self.use_head_adaptive:
            # Per-head differentiated eviction
            k_past = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :]
            v_past = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :]

            k_compressed_list = []
            v_compressed_list = []
            hh_compressed_list = []

            for h in range(num_key_value_heads):
                h_budget = int(head_budgets[h].item())
                h_budget = min(h_budget, hh_score.shape[-1])
                h_budget = max(h_budget, 1)  # At least 1 token

                # Get top-k indices for this head
                h_indices = hh_score[:, h:h+1, :].topk(h_budget, dim=-1).indices
                h_hh_compress = hh_score[:, h:h+1, :].gather(dim=2, index=h_indices)

                h_indices_expanded = h_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_h_compress = k_past[:, h:h+1, :, :].gather(dim=2, index=h_indices_expanded)
                v_h_compress = v_past[:, h:h+1, :, :].gather(dim=2, index=h_indices_expanded)

                k_compressed_list.append(k_h_compress)
                v_compressed_list.append(v_h_compress)
                hh_compressed_list.append(h_hh_compress)

            # Pad to max head budget for concatenation
            max_budget = max(k.shape[2] for k in k_compressed_list)

            for i in range(num_key_value_heads):
                curr_len = k_compressed_list[i].shape[2]
                if curr_len < max_budget:
                    pad_len = max_budget - curr_len
                    # Pad with zeros (will be masked during attention)
                    k_compressed_list[i] = F.pad(k_compressed_list[i], (0, 0, 0, pad_len))
                    v_compressed_list[i] = F.pad(v_compressed_list[i], (0, 0, 0, pad_len))
                    hh_compressed_list[i] = F.pad(hh_compressed_list[i], (0, pad_len))

            k_past_compress = torch.cat(k_compressed_list, dim=1)
            v_past_compress = torch.cat(v_compressed_list, dim=1)
            hh_score_compress = torch.cat(hh_compressed_list, dim=1)

            past_key_values.evict_scores[layer_idx] = hh_score_compress
        else:
            # Original: all heads share the same indices
            indices = hh_score.topk(budget, dim=-1).indices
            hh_score_compress = hh_score.gather(dim=2, index=indices)
            past_key_values.evict_scores[layer_idx] = hh_score_compress

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)

        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states

        return past_key_values

class CakeDecodingKVCache_LayerWise:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        # print(f"CakeDecodingKVCache_LayerWise: {hh_size}, {window_size}")
        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, layer_idx):
        num_heads = attn_score_cache.shape[1]
        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        num_key_value_groups = num_heads // num_key_value_heads

        seq_len = past_key_values.key_cache[layer_idx].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim = -2)

        attn_cache = F.avg_pool1d(attn_cache, kernel_size = 5, padding=5//2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)

        attn_cache = attn_cache.mean(dim=-2)

        indices = attn_cache.topk(self.hh_size, dim=-1).indices
        # indices = indices.sort().values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states

        return past_key_values

    def _update_hh_score(self, attn_score_cache, num_key_value_heads):

        bsz,num_heads, num_new_tokens,_ = attn_score_cache.shape
        num_key_value_groups = num_heads //  num_key_value_heads
        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(2) #bsz, total num_heads, seq_len
            self.hh_score = self.hh_score.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            self.hh_score = self.hh_score.mean(dim=-2)
        
        else:
            # print(self.hh_score.shape)
            attn_score_cache = attn_score_cache.sum(2)
            attn_score_cache = attn_score_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_score_cache = attn_score_cache.mean(dim=-2)
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
