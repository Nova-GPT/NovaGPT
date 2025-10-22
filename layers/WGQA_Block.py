from torch import nn

from layers.FeedForward import FeedFoward
# from layers.MultiHeadAttention import MultiHeadAttention
from layers.WGQAAttention import WGQAAttention
from utils import ModelSpecs

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, specs : ModelSpecs):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        n_embd = specs.N_EMBD
        n_head = specs.N_HEAD
        kv_heads = specs.KV_HEADS
        # assert n_embd % n_head == 0, "N_EMBD must be divisible by N_HEAD"
        # assert hasattr(specs, "KV_HEADS"), "Add KV_HEADS to ModelSpecs (e.g., 8 or 4)."


        super().__init__()
        head_size = n_embd // n_head
        self.sa = WGQAAttention(n_embd, n_head, kv_heads, dropout=specs.DROPOUT)
        self.ffwd = FeedFoward(specs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
