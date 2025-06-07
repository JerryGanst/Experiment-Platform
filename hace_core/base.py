class BaseKVCacheMethod:
    """Base class for KV cache compression strategies."""

    def __init__(self, **kwargs):
        # Store arbitrary configuration parameters
        self.config = kwargs

    def initialize_cache(self, *args, **kwargs):
        """Initialize cache structures."""
        raise NotImplementedError

    def update_cache(self, *args, **kwargs):
        """Update cache contents during generation."""
        raise NotImplementedError

    def get_method_specific_metrics(self) -> dict:
        """Return metrics specific to the method."""
        return {}
