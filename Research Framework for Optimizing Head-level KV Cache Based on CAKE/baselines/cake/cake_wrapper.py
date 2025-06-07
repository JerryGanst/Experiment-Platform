from ..base_method import BaseKVCacheMethod

try:
    from cake.cake_cache import CakeCache, CakeprefillKVCache
except Exception as e:  # noqa: E722
    CakeCache = None
    CakeprefillKVCache = None

class CAKEBaseline(BaseKVCacheMethod):
    """Wrapper for CAKE KV cache handling."""

    def __init__(self, num_layers: int, num_heads: int, cache_size: int = 512,
                 window_size: int = 32, use_cascading: bool = False):
        if CakeCache is None or CakeprefillKVCache is None:
            raise ImportError("CAKE modules are not available")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cache_size = cache_size
        self.window_size = window_size
        self.use_cascading = use_cascading
        self.cache = None
        self.prefill_manager = CakeprefillKVCache(
            cache_size=cache_size,
            window_size=window_size,
            num_heads=num_heads,
            num_layers=num_layers,
            use_cascading=use_cascading,
        )

    def initialize_cache(self):
        """Initialize a fresh CAKE cache instance."""
        self.cache = CakeCache()
        return self.cache

    def update_cache(self, key_states, value_states, layer_idx):
        """Update cache for a given layer and apply prefill eviction."""
        if self.cache is None:
            raise ValueError("Cache not initialized")
        self.cache.update(key_states, value_states, layer_idx)
        seq_len = self.cache.get_seq_length(layer_idx)
        self.prefill_manager(self.cache, seq_len)
        return self.cache
