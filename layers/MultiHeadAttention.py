import torch
from torch import nn

from layers.Head import Head
from utils import ModelSpecs

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, specs:ModelSpecs):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, specs) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, specs.N_EMBD)
        self.dropout = nn.Dropout(specs.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
