import torch
import torch.nn as nn
from layers.MLA_Head import MLAHead
from utils import ModelSpecs

class MLAMultiHeadAttention(nn.Module):
    """ MLA: Multi-head attention with shared Key/Value projections """

    def __init__(self, num_heads, head_size, specs: ModelSpecs):
        super().__init__()
        # Shared key/value projections
        self.shared_k = nn.Linear(specs.N_EMBD, head_size, bias=False)
        self.shared_v = nn.Linear(specs.N_EMBD, head_size, bias=False)

        # Per-head queries
        self.heads = nn.ModuleList([
            MLAHead(head_size, specs, self.shared_k, self.shared_v)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, specs.N_EMBD)
        self.dropout = nn.Dropout(specs.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
