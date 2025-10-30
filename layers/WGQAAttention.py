import torch
import torch.nn as nn
import torch.nn.functional as F

class WGQAAttention(nn.Module):
    def __init__(self, d_model : int, n_heads : int, kv_heads : int, dropout=0.0, bias=False):
        super().__init__()
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # Per-KV-head learnable weights (WGQA). Two common variants:
        # 1) Logit reweight: scale attention logits per KV head
        # 2) Value reweight: scale values per KV head
        # Provide both; enable one via flags as needed.
        self.weight_logits = nn.Parameter(torch.ones(kv_heads))   # for logits scaling
        self.weight_values = nn.Parameter(torch.ones(kv_heads))   # for value scaling

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, use_logit_weight=True, use_value_weight=False):
        """
        x: (B, T, d_model)
        attn_mask: optional additive mask (B, 1, T_q, T_k) or (1, 1, T, T)
        """
        B, T, _ = x.size()
        H = self.n_heads
        G = self.kv_heads
        D = self.head_dim
        group_size = H // G

        # Compute Q, K, V
        q = self.W_q(x)  # (B, T, d_model)
        k = self.W_k(x)  # (B, T, G*D)
        v = self.W_v(x)  # (B, T, G*D)

        # Reshape
        q = q.view(B, T, H, D).transpose(1, 2)            # (B, H, T, D)
        k = k.view(B, T, G, D).transpose(1, 2)            # (B, G, T, D)
        v = v.view(B, T, G, D).transpose(1, 2)            # (B, G, T, D)

        # Map each query head to a KV head group
        # For head h, group index g = h // group_size
        # Expand k,v to per-query-head by indexing (no data copy if using gather)
        # Build index mapping H -> G
        device = x.device
        head_to_group = torch.arange(H, device=device) // group_size  # (H,)
        # Shape for gathering: repeat per batch, per time
        # Expand k,v to (B, H, T, D) via gather along head dimension
        k_gather = k.index_select(1, head_to_group)                   # (B, H, T, D)
        v_gather = v.index_select(1, head_to_group)                   # (B, H, T, D)

        # Optional WGQA reweighting
        if use_logit_weight:
            # Scale attention logits per KV group for each query head
            logit_scale = self.weight_logits[head_to_group]           # (H,)
            logit_scale = logit_scale.view(1, H, 1, 1)                # (1,H,1,1)
        else:
            logit_scale = 1.0

        if use_value_weight:
            # Scale values per KV group
            value_scale = self.weight_values[head_to_group]           # (H,)
            value_scale = value_scale.view(1, H, 1, 1)                # (1,H,1,1)
            v_gather = v_gather * value_scale

        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k_gather.transpose(-2, -1)) / (D ** 0.5)
        if use_logit_weight is True:
            attn_logits = attn_logits * logit_scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(~attn_mask, float('-inf'))
            else:
                attn_logits = attn_logits + attn_mask  # additive mask

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v_gather)                               # (B, H, T, D)

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)           # (B, T, d_model)
        y = self.W_o(y)
        return y
