from torch import nn
import torch

from utils import ModelSpecs
from layers.SwiGLU import SwiGLU

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity 
      Linear -> ReLU -> Linear -> Dropout"""

    def __init__(self, specs : ModelSpecs):
        super().__init__()
        n_embd = specs.N_EMBD
        d_ffn = int(2.67 * n_embd)  # Reduced ratio for SwiGLU, can tune as needed
        self.net = nn.Sequential(
            SwiGLU(n_embd, d_ffn),
            nn.Dropout(specs.DROPOUT),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)