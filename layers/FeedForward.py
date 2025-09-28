from torch import nn
import torch

from utils import ModelSpecs

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 
      Linear -> ReLU -> Linear -> Dropout"""

    def __init__(self, specs : ModelSpecs):
        super().__init__()
        n_embd = specs.N_EMBD
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(specs.DROPOUT),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)