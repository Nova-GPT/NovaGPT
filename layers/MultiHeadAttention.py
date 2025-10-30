import torch
import torch.nn as nn
from utils import ModelSpecs
from layers.MultiHeadLatentAttention import MLA, DeepseekConfig  # assumes MLA code is in layers/Deepseek_MLA.py


class DeepseekMLAHead(nn.Module):
    """Wrapper for the Deepseek MLA attention head, keeping your interface consistent."""

    def __init__(self, specs: ModelSpecs):
        super().__init__()

        # Map your existing ModelSpecs to DeepseekConfig
        self.config = DeepseekConfig(
            hidden_size=specs.N_EMBD,
            num_heads=specs.N_HEAD,
            max_position_embeddings=specs.BLOCK_SIZE,
            rope_theta=128000,              # can be tuned
            attention_dropout=specs.DROPOUT,
            q_lora_rank=int(specs.N_EMBD // 3),  # compression rank
            qk_rope_head_dim=64,
            kv_lora_rank=int(specs.N_EMBD // 8),
            v_head_dim=int(specs.N_EMBD // specs.N_HEAD),
            qk_nope_head_dim=64,
            attention_bias=False,
        )

        self.mla = MLA(self.config)

    def forward(self, x, position_ids=None, mask=None):
        if position_ids is None:
            # default incremental sequence
            position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out, _ = self.mla(x, position_ids, attention_mask=mask)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel (Deepseek MLA version)."""

    def __init__(self, num_heads, head_size, specs: ModelSpecs, use_mla=True):
        super().__init__()
        self.use_mla = use_mla

        if use_mla:
            self.attn = DeepseekMLAHead(specs)
            self.proj = nn.Identity()  # MLA already projects internally
        else:
            from layers.MLA_Head import LatentHead
            self.heads = nn.ModuleList([LatentHead(head_size, specs) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, specs.N_EMBD)

        self.dropout = nn.Dropout(specs.DROPOUT)

    def forward(self, x, mask=None):
        if self.use_mla:
            out = self.attn(x, mask=mask)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.proj(out)
        out = self.dropout(out)
        return out
