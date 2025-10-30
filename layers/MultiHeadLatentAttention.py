import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from dataclasses import dataclass

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype=input_dtype).to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states - hidden_states * torch.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len = max_position_embeddings,
            device = self.inv_freq.device,
            dtype = torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class DeepseekConfig:
    hidden_size: int
    num_heads: int
    max_position_embeddings: int # rope parameter
    rope_theta: float # frequency, usually large
    attention_dropout: float
    q_lora_rank: int # latent shape, usually >10k
    qk_rope_head_dim: int # 64
    kv_lora_rank: int # 512
    v_head_dim: int # 128
    qk_nope_head_dim: int
    attention_bias: bool

class MLA(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        # 1. MHA
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        self.out_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False
        )

        # 2. MLA compression
        # Correctly store both head-dim parts
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # down projections
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias = config.attention_bias,
        )
        self.q_down_norm = DeepseekV2RMSNorm(self.q_lora_rank)

        # kv down proj: produce compressed kv + small rope chunk
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,  # note: rope dim appended
            bias=config.attention_bias,
        )
        self.kv_down_norm = DeepseekV2RMSNorm(self.kv_lora_rank)

        # up projections
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=config.attention_bias,
        )

        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (
                (self.q_head_dim - self.qk_rope_head_dim) + self.v_head_dim
            ),
            bias = config.attention_bias,
        )

        # rotary should be sized for the ROPE head dim
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        # hidden_states (b, seq_len, hidden_dim)
        bsz, q_len, _ = hidden_states.size()

        # 1. q compression
        q = self.q_down_proj(hidden_states)
        q = self.q_down_norm(q)
        q = self.q_up_proj(q)
        # q shape: (b, seq_len, num_heads * q_head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # (b, num_head, seq_len, q_head_dim)

        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1
        )

        # kv part
        c_kv = self.kv_down_proj(hidden_states)
        # split compressed kv and small rope chunk (k_rope)
        c_kv, k_rope = torch.split(
            c_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        ) # k_rope shape: (b, seq_len, qk_rope_head_dim)

        # reshape k_rope to (b, 1, seq_len, qk_rope_head_dim) for broadcasting to heads
        k_rope = k_rope.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_down_norm(c_kv)
        kv = self.kv_up_proj(kv)
        # (b, seq_len, num_head * (qk_nope_head_dim + v_head_dim))

        kv = kv.view(
            bsz, q_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        # (b, num_head, seq_len, qk_nope + v_head_dim)

        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        )
        # k_nope: (b, num_head, seq_len, qk_nope_head_dim)
        # value_states: (b, num_head, seq_len, v_head_dim)

        # apply position encoding (RoPE) to q_rope and k_rope
        kv_seq_len = value_states.size(2)  # seq_len dimension
        cos, sin = self.rotary_emb(value_states, seq_len = kv_seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(
            q_rope, k_rope, cos, sin, position_ids,
        )

        # MHA: reassemble
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)
        # shapes: (b, num_head, q_len, head_dim)

        # compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            # attention_mask expected broadcastable to attn_weights shape
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim = -1).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training = self.training)

        output = torch.matmul(attn_weights, value_states)
        output = output.transpose(1, 2).reshape(bsz, q_len, -1)
        output = self.out_proj(output)
        return output, attn_weights
