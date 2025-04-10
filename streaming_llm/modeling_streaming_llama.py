import torch
from torch import nn
from .h2o_block import H2OBlock

class StreamingLlama(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            H2OBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)
        logits = self.output_proj(x)
        return logits
