import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
from cake.utils import adjust_budgets

# Import diagnostic recorder from modify_qwen2
def get_diagnostic_recorder():
    """Lazy import to avoid circular dependency"""
    from cake.model.modify_qwen2 import DIAGNOSTIC
    return DIAGNOSTIC

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
        self.head_entropies = []  # HACE: Per-layer head entropy scores
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
        head_entropy: Optional[torch.Tensor] = None,
    ):
        """
        Update layer scores for CAKE budget allocation.

        Args:
            pref_score: Layer preference score for layer-level budget allocation (Cake)
            evict_score: Per-head attention scores [bsz, num_kv_heads, seq_len] for:
                        - Token selection (which tokens to keep)
                        - Ada-KV head-level budget allocation (counting in global top-B)
            head_entropy: Optional per-head entropy scores [num_kv_heads] for HACE
        """
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)
        if head_entropy is not None:
            self.head_entropies.append(head_entropy)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        key = self.key_cache[layer_idx]
        if key is None:
            return 0
        return key.shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. CakeCache does not have a maximum length."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # For CakeCache, we don't have a max length, so we just return the current sequence length
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

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
            # Handle both old API (key_cache) and new API (layers)
            if hasattr(past_key_values, 'key_cache'):
                cache.key_cache = past_key_values.key_cache
                cache.value_cache = past_key_values.value_cache
            elif hasattr(past_key_values, 'layers'):
                # New transformers >= 4.57 API
                cache.key_cache = [layer.keys for layer in past_key_values.layers]
                cache.value_cache = [layer.values for layer in past_key_values.layers]
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
        # When use_head_adaptive=True, each head independently selects its own top-k positions
        # (same quantity, different positions - this is the correct Ada-KV approach)
        # The head_mode env var is checked but head_budgets are no longer used
        # (all heads get the same budget quantity, just different positions)

        if self.use_cascading:
            layer_idx = 0
            print(layer_budgets)
            for budget in layer_budgets:
                if budget>= seq_len-self.window_size:
                    budget = seq_len-self.window_size

                # head_budgets is deprecated, pass None
                past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget, None)
                past_key_values.layer_budget[layer_idx]=budget
                layer_idx +=1
        else:
            layer_idx = 0
            if len(layer_budgets) ==self.num_layers:
                for budget in layer_budgets:
                    if budget>= seq_len-self.window_size:
                        budget = seq_len-self.window_size

                    # head_budgets is deprecated, pass None
                    past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget, None)
                    past_key_values.layer_budget[layer_idx]=budget
                    layer_idx +=1

        return past_key_values

    def _compute_head_budgets_adakv(self, attn_weights, layer_budget, alpha=0.2):
        """
        Compute per-head budgets using Ada-KV's counting-based algorithm.

        Ada-KV Algorithm (from paper arXiv:2407.11550):
        1. Concatenate attention weights across all heads
        2. Select top-B indices globally (based on attention weight magnitude)
        3. Count frequency of each head in top-B (fi)
        4. Allocate budget proportional to frequency: B*i = fi
        5. Blend with uniform allocation: final = alpha * adaptive + (1-alpha) * uniform

        Key insight from paper:
        - "Assigns larger budgets to dispersed heads" (attention spread across many tokens)
        - "Conservatively adjusts budgets for sparse heads" (attention focused on few tokens)
        - Dispersed heads have more tokens in global top-B -> higher fi -> larger budget

        Args:
            attn_weights: Tensor [bsz, num_heads, query_len, seq_len] - softmax attention weights
                         OR [bsz, num_kv_heads, seq_len] - aggregated attention scores
            layer_budget: Total budget for this layer (sum across all heads)
            alpha: Blending factor (default 0.2 per paper)
                   final_budget = alpha * adaptive + (1-alpha) * uniform

        Returns:
            Tensor [num_kv_heads] - budget per head
        """
        # Handle different input shapes
        if attn_weights.dim() == 4:
            # [bsz, num_heads, query_len, seq_len] -> aggregate to [bsz, num_heads, seq_len]
            # Use mean across query positions (like SnapKV observation window)
            attn_scores = attn_weights.mean(dim=2)  # [bsz, num_heads, seq_len]
        elif attn_weights.dim() == 3:
            attn_scores = attn_weights  # Already [bsz, num_heads, seq_len]
        else:
            raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

        bsz, num_heads, seq_len = attn_scores.shape

        # Step 1: Flatten across heads -> [bsz, num_heads * seq_len]
        flat_scores = attn_scores.reshape(bsz, -1)

        # Step 2: Create head indicators for each position
        # head_ids[i] tells us which head position i belongs to
        head_ids = torch.arange(num_heads, device=attn_weights.device)
        head_ids = head_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)  # [num_heads * seq_len]

        # Step 3: Select top-B indices globally
        # B = layer_budget (total tokens to keep across all heads)
        total_budget = int(layer_budget)
        k = min(total_budget, flat_scores.shape[-1])
        _, top_indices = flat_scores.topk(k, dim=-1)  # [bsz, B]

        # Step 4: Count frequency of each head in top-B
        # fi = how many tokens from head i appear in global top-B
        # Use first batch item (assume similar distribution across batch)
        top_head_ids = head_ids[top_indices[0]]  # [B]

        # Vectorized counting
        head_counts = torch.zeros(num_heads, device=attn_weights.device)
        for h in range(num_heads):
            head_counts[h] = (top_head_ids == h).sum().float()

        # Step 5: Compute adaptive budget proportional to frequency
        # B*i = fi (from paper Algorithm 1)
        if head_counts.sum() > 0:
            # Normalize to sum to layer_budget
            adaptive_budget = head_counts / head_counts.sum() * layer_budget
        else:
            adaptive_budget = torch.ones(num_heads, device=attn_weights.device) * layer_budget / num_heads

        # Step 6: Blend with uniform allocation (safeguard from paper)
        # "α × {B*i} + (1−α) × (B/h)" where α=0.2 default
        uniform_budget = layer_budget / num_heads
        head_budgets = alpha * adaptive_budget + (1 - alpha) * uniform_budget

        # Ensure minimum budget of 1 per head
        head_budgets = head_budgets.clamp(min=1)

        # Rescale to ensure total budget is preserved
        head_budgets = head_budgets / head_budgets.sum() * layer_budget
        head_budgets = head_budgets.clamp(min=1)

        return head_budgets

    def _compute_head_budgets_entropy(self, head_entropy, layer_budget, head_mode, alpha=0.3):
        """
        Compute per-head budgets based on head entropy.

        HACE Entropy-based allocation:
        - high_entropy: High entropy heads get more budget
          (dispersed attention needs more tokens to preserve information)
        - low_entropy: Low entropy heads get more budget
          (focused attention is more important, needs protection)

        Args:
            head_entropy: Tensor [num_kv_heads] - entropy per head
            layer_budget: Total budget for this layer
            head_mode: 'high_entropy' or 'low_entropy'
            alpha: Blending factor with uniform allocation (default 0.3)

        Returns:
            Tensor [num_kv_heads] - budget per head
        """
        num_heads = head_entropy.shape[0]

        if head_mode == "high_entropy":
            # High entropy -> more budget
            # Normalize entropy to weights
            weights = head_entropy / (head_entropy.sum() + 1e-8)
        elif head_mode == "low_entropy":
            # Low entropy -> more budget (inverse)
            inv_entropy = 1.0 / (head_entropy + 1e-8)
            weights = inv_entropy / (inv_entropy.sum() + 1e-8)
        else:
            weights = torch.ones_like(head_entropy) / num_heads

        # Compute adaptive budget
        adaptive_budget = weights * layer_budget

        # Blend with uniform allocation (safeguard)
        uniform_budget = layer_budget / num_heads
        head_budgets = alpha * adaptive_budget + (1 - alpha) * uniform_budget

        # Ensure minimum budget and rescale
        head_budgets = head_budgets.clamp(min=1)
        head_budgets = head_budgets / head_budgets.sum() * layer_budget
        head_budgets = head_budgets.clamp(min=1)

        return head_budgets

    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget, head_budgets=None):
        """
        Evict tokens from layer's KV cache.

        Args:
            past_key_values: Cache object
            layer_idx: Layer index
            budget: Total layer budget (number of tokens to keep per head)
            head_budgets: Deprecated - no longer used (kept for API compatibility)

        HACE Fix (2024-12):
        - All heads select the SAME NUMBER of tokens (= budget)
        - But each head INDEPENDENTLY chooses which positions to keep
        - This avoids padding issues with shared attention mask

        HACE Entropy Strategy (2024-12):
        - Entropy doesn't change quantity, but affects score weighting
        - high_entropy: boost scores for high-entropy heads -> their tokens are more likely to be selected
        - low_entropy: boost scores for low-entropy heads -> their tokens are more likely to be selected
        """
        import os
        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape

        num_key_value_groups = self.num_heads // num_key_value_heads
        hh_score = past_key_values.evict_scores[layer_idx]

        if budget > hh_score.shape[-1]:
            budget = hh_score.shape[-1]

        # HACE: Per-head INDEPENDENT position selection (same quantity, different positions)
        if self.use_head_adaptive:
            # Each head selects its own top-k positions, but k is the same for all heads
            uniform_budget = int(budget)

            k_past = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :]
            v_past = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :]

            # HACE: Apply entropy-based score weighting
            head_mode = os.environ.get("HACE_HEAD_MODE", "").strip().lower()

            # Get head entropy if available
            head_entropy = None
            if hasattr(past_key_values, 'head_entropies') and len(past_key_values.head_entropies) > layer_idx:
                head_entropy = past_key_values.head_entropies[layer_idx]

            # Compute entropy-weighted scores
            weighted_hh_score = hh_score.clone()
            if head_entropy is not None and head_mode in ("high_entropy", "low_entropy"):
                # Normalize entropy to [0, 1] range per layer
                entropy_min = head_entropy.min()
                entropy_max = head_entropy.max()
                if entropy_max > entropy_min:
                    normalized_entropy = (head_entropy - entropy_min) / (entropy_max - entropy_min + 1e-8)
                else:
                    normalized_entropy = torch.ones_like(head_entropy) * 0.5

                # Apply entropy weighting: boost certain heads' scores
                # alpha controls the strength of entropy influence (0.5 = moderate)
                alpha = 0.5

                if head_mode == "high_entropy":
                    # High entropy heads get score boost: more dispersed attention -> need more tokens
                    # weight = 1 + α * normalized_entropy (range: [1, 1+α])
                    weights = 1.0 + alpha * normalized_entropy
                elif head_mode == "low_entropy":
                    # Low entropy heads get score boost: focused attention is important
                    # weight = 1 + α * (1 - normalized_entropy) (range: [1, 1+α])
                    weights = 1.0 + alpha * (1.0 - normalized_entropy)

                # Apply weights to each head's scores: [num_heads] -> [1, num_heads, 1]
                weights = weights.view(1, -1, 1).to(hh_score.device)
                weighted_hh_score = hh_score * weights

                if layer_idx == 0:
                    print(f"[HACE] Layer {layer_idx}: head_entropy range [{entropy_min:.3f}, {entropy_max:.3f}], mode={head_mode}")

            k_compressed_list = []
            v_compressed_list = []
            hh_compressed_list = []
            
            # HACE Diagnostic: Get recorder
            diagnostic = get_diagnostic_recorder()

            for h in range(num_key_value_heads):
                # Each head uses its (weighted) score to select its own top-k positions
                h_indices = weighted_hh_score[:, h:h+1, :].topk(uniform_budget, dim=-1).indices
                # Store original (unweighted) scores for future use
                h_hh_compress = hh_score[:, h:h+1, :].gather(dim=2, index=h_indices)

                h_indices_expanded = h_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_h_compress = k_past[:, h:h+1, :, :].gather(dim=2, index=h_indices_expanded)
                v_h_compress = v_past[:, h:h+1, :, :].gather(dim=2, index=h_indices_expanded)

                k_compressed_list.append(k_h_compress)
                v_compressed_list.append(v_h_compress)
                hh_compressed_list.append(h_hh_compress)
                
                # HACE Diagnostic: Record kept indices for this head
                if diagnostic.enabled:
                    kept_idx_list = sorted(h_indices[0, 0, :].cpu().tolist())
                    h_entropy = 0.0
                    if head_entropy is not None:
                        h_entropy = head_entropy[h].item() if h < len(head_entropy) else 0.0
                    diagnostic.record(
                        layer_idx=layer_idx,
                        head_idx=h,
                        kept_indices=kept_idx_list,
                        head_budget=uniform_budget,
                        head_entropy=h_entropy,
                        total_seq_len=seq_len - self.window_size  # Only count non-window tokens
                    )

            # All heads have the same length, direct concat without padding!
            k_past_compress = torch.cat(k_compressed_list, dim=1)
            v_past_compress = torch.cat(v_compressed_list, dim=1)
            hh_score_compress = torch.cat(hh_compressed_list, dim=1)

            past_key_values.evict_scores[layer_idx] = hh_score_compress
        else:
            # Original: all heads share the same indices
            indices = hh_score.topk(budget, dim=-1).indices
            hh_score_compress = hh_score.gather(dim=2, index=indices)
            past_key_values.evict_scores[layer_idx] = hh_score_compress
            
            # HACE Diagnostic: Record for baseline (all heads share same indices)
            diagnostic = get_diagnostic_recorder()
            if diagnostic.enabled:
                # All heads share the same indices in baseline
                shared_indices = sorted(indices[0, 0, :].cpu().tolist())
                for h in range(num_key_value_heads):
                    diagnostic.record(
                        layer_idx=layer_idx,
                        head_idx=h,
                        kept_indices=shared_indices,
                        head_budget=int(budget),
                        head_entropy=0.0,  # No per-head entropy in baseline
                        total_seq_len=seq_len - self.window_size
                    )

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
