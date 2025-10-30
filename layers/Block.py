import torch
from torch import nn

from layers.FeedForward import FeedFoward
from layers.MultiHeadAttention import MultiHeadAttention
from utils import ModelSpecs


class Block(nn.Module):
    """Transformer block with optional DeepSeek MLA attention."""

    def __init__(self, specs: ModelSpecs, use_mla: bool = True):
        super().__init__()

        n_embd = specs.N_EMBD
        n_head = specs.N_HEAD
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size, specs, use_mla=use_mla)
        self.ffwd = FeedFoward(specs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.use_mla = use_mla
        self.block_size = specs.BLOCK_SIZE

        # ---- Precompute causal mask once ----
        mask = torch.tril(torch.ones(self.block_size, self.block_size))
        # Register as buffer so it moves automatically with .to(device)
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))  # (1, 1, T, T)

    def forward(self, x):
        # ---- Slice only required part ----
        seq_len = x.size(1)
        mask = self.mask[:, :, :seq_len, :seq_len]

        # ---- Forward pass ----
        x = x + self.sa(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x
