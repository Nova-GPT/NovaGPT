from torch import nn
from layers.FeedForward import FeedFoward
from utils import ModelSpecs
from layers.MultiHeadLatentAttention import MultiHeadLatentAttention

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, specs: ModelSpecs):
        n_embd = specs.N_EMBD
        n_head = specs.N_HEAD
        super().__init__()

        self.sa = MultiHeadLatentAttention(
            d_model=n_embd,
            num_head=n_head,
            d_embed=n_embd,           # same as model width
            d_c=n_embd // 8,          # latent KV dim
            d_c1=n_embd // 8,         # latent Q dim
            d_rotate=n_embd // 16,    # latent dims with RoPE
            dropout=specs.DROPOUT,
        )

        self.ffwd = FeedFoward(specs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # Static sanity checks that can be run without input
        assert hasattr(self.sa, "num_head"), "MultiHeadLatentAttention must expose num_head" 
        assert hasattr(self.sa, "d_model"), "MultiHeadLatentAttention must expose d_model" 
        assert self.sa.d_model == n_embd, "Attention must preserve embedding dim (d_model == N_EMBD)" 

        # Optional: if the attention exposes head dims, validate consistency
        if hasattr(self.sa, "d_head") and hasattr(self.sa, "num_head"):
            assert self.sa.num_head * self.sa.d_head == n_embd, "num_head * d_head must equal d_model" 

        # Optional: if using separate QK and V head dims, validate downstream merge size
        if hasattr(self.sa, "d_qk") and hasattr(self.sa, "d_v"):
            # Q/K head size only used for score scaling, V head size defines output width per head
            expected_out = self.sa.num_head * getattr(self.sa, "d_v")
            assert expected_out == n_embd, "num_head * d_v must equal d_model when merging heads" 

    def forward(self, x):
        # Runtime shape checks on the fly (cheap and safe)
        B, S, C = x.shape
        assert C == self.sa.d_model, "Input channel must equal d_model" 

        y = self.sa(self.ln1(x))
        # Ensure attention returns same embedding width
        assert y.shape == (B, S, C), "Attention output shape must be (B, S, d_model)" 

        x = x + y

        z = self.ffwd(self.ln2(x))
        # FeedForward should preserve width
        assert z.shape == (B, S, C), "FeedForward output shape must be (B, S, d_model)" 

        x = x + z
        return x
