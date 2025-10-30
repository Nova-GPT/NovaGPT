import torch.nn as nn
from layers.MLAMultiHeadAttention import MLAMultiHeadAttention
from layers.FeedForward import FeedFoward
from utils import ModelSpecs

class MLABlock(nn.Module):
    """ Transformer block with MLA instead of standard MHA """

    def __init__(self, specs: ModelSpecs):
        super().__init__()
        n_embd = specs.N_EMBD
        n_head = specs.N_HEAD
        head_size = n_embd // n_head

        self.sa = MLAMultiHeadAttention(n_head, head_size, specs)
        self.ffwd = FeedFoward(specs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
