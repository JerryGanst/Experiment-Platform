import torch
from torch import nn
from .h2o_llama_attention import H2OLlamaAttention

class H2OBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = H2OLlamaAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output = self.attention(self.layernorm1(x))
        x = x + attn_output
        ff_output = self.feed_forward(self.layernorm2(x))
        x = x + ff_output
        return x
