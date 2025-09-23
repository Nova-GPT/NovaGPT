from torch import nn
import torch

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 
      Linear -> ReLU -> Linear -> Dropout"""

    def __init__(self, n_embd : int, dropout : float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x : torch.Tensor['batch_size', 'n_embed']) -> torch.Tensor['batch_size', 'n_embed']:
        return self.net(x)