import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

from utils import ModelSpecs

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, specs: ModelSpecs):
        super().__init__()
        n_embd = specs.N_EMBD
        block_size = specs.BLOCK_SIZE
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(specs.DROPOUT)
        self.specs = specs

    def forward(self, x : torch.Tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k : torch.Tensor= self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        # # compute attention scores ("affinities")
        # wei :torch.Tensor = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # wei = F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)
        # out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        return out
