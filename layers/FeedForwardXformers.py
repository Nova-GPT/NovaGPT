from torch import nn
import torch
from xformers.components.feedforward import build_feedforward

from utils import ModelSpecs


class FeedForwardSwiGLU(nn.Module):
    """
    Feedforward layer using xformers SwiGLU feedforward.
    """

    def __init__(self, specs: ModelSpecs):
        super().__init__()
        d_model = specs.N_EMBD
        d_ffn = int(2.67 * d_model)  # recommended reduced ratio for SwiGLU

        # Build xformers SwiGLU feedforward with dropout
        self.net = build_feedforward({
            "name": "swi_glu",
            "dim_model": d_model,
            "dim_feedforward": d_ffn,
            "dropout": specs.DROPOUT,
            "activation": "swi_glu",
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Example usage:
# specs = ModelSpecs(N_EMBD=512, DROPOUT=0.1)
# ffn = FeedForwardSwiGLU(specs)
# x = torch.randn(2, 128, specs.N_EMBD)
# out = ffn(x)
# print(out.shape)  # (2, 128, 512)
