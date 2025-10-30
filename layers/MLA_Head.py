import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentHead(nn.Module):
    def __init__(self, head_size, specs):
        super().__init__()
        self.d_model = specs.N_EMBD
        self.head_size = head_size
        self.dropout = nn.Dropout(specs.DROPOUT)

        # Attention projections
        self.key = nn.Linear(self.d_model, head_size, bias=False)
        self.query = nn.Linear(self.d_model, head_size, bias=False)
        self.value = nn.Linear(self.d_model, head_size, bias=False)

        # Register a causal mask buffer (not trainable)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(specs.BLOCK_SIZE, specs.BLOCK_SIZE)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.size()

        # project key, query, value
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # compute attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)  # (B, T, T)

        # add causal mask
        causal_mask = self.causal_mask[:, :, :T, :T]  # (1,1,T,T)
        attn = attn.unsqueeze(1)  # (B,1,T,T) -> easier broadcast
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        attn = attn.squeeze(1)    # back to (B,T,T)

        # softmax + dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # weighted aggregation
        out = attn @ v  # (B, T, head_size)
        return out
