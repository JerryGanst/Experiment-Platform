import torch
from torch import nn

class H2OLlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return self.out_proj(output)
