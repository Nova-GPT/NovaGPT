from torch import nn

from layers.FeedForward import FeedFoward
from layers.MultiHeadAttention import MultiHeadAttention
from utils import ModelSpecs

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, specs : ModelSpecs):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        n_embd = specs.N_EMBD
        n_head = specs.N_HEAD

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, specs)
        self.ffwd = FeedFoward(specs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
