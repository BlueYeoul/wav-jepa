import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import drop_path


# ---------------------------------------------------------------------------
# 1-D RoPE rotation helper
# ---------------------------------------------------------------------------

def rotate_queries_or_keys(x, pos, n_registers=0, has_cls_first=False):
    """
    Apply 1-D RoPE to queries or keys.

    x   : (B, H, N, D)
    pos : (N_ctx,)  or  (B, H, N_ctx)  – sequence positions for context tokens
    """
    B, num_heads, N, D = x.size()
    assert D % 2 == 0

    n_cls = 1 if has_cls_first else 0
    start = n_cls
    end   = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls     else None
    x_ctx = x[..., start:end, :]
    x_reg = x[..., end:, :]    if n_registers else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (D / 2.0)))

    freq = torch.einsum("..., f -> ... f", pos.float(), omega)  # (..., N_ctx, D//2)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)           # (..., N_ctx, D)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)

    out = x_ctx * emb_cos + y * emb_sin

    parts = []
    if n_cls:       parts.append(x_cls)
    parts.append(out)
    if n_registers: parts.append(x_reg)
    return torch.cat(parts, dim=-2)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (used when act_layer=nn.SiLU)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        h = hidden_features or in_features
        if wide_silu:
            h = int(2 * h / 3)
            h = (h + 7) // 8 * 8   # align to 8
        self.fc1 = nn.Linear(in_features, h)
        self.fc2 = nn.Linear(in_features, h)
        self.fc3 = nn.Linear(h, out_features)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    """Self-attention with 1-D RoPE (sequence position only)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False,
                 n_registers=0, has_cls_first=False):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.scale        = qk_scale or self.head_dim ** -0.5
        self.qkv          = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop    = nn.Dropout(attn_drop)
        self.proj         = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop    = nn.Dropout(proj_drop)
        self.use_sdpa     = use_sdpa
        self.is_causal    = is_causal
        self.n_registers  = n_registers
        self.has_cls_first = has_cls_first

    def forward(self, x, mask=None, return_attn=False):
        """
        x    : (B, N, C)
        mask : (B, N) – actual token indices (used as positions for RoPE)
                        or None for full sequence
        """
        B, N, C = x.size()
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, D_head)

        if mask is not None:
            # mask holds actual patch indices → use as 1D RoPE positions
            pos = mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # (B, H, N)
        else:
            n_cls = 1 if self.has_cls_first else 0
            n_ctx = N - self.n_registers - n_cls
            pos = torch.arange(n_ctx, device=x.device)             # (N_ctx,)

        q = rotate_queries_or_keys(q, pos, self.n_registers, self.has_cls_first)
        k = rotate_queries_or_keys(k, pos, self.n_registers, self.has_cls_first)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = self.proj(self.proj_drop(x.transpose(1, 2).reshape(B, N, C)))
        return x, (attn if return_attn else None)


class Attention(nn.Module):
    """Standard self-attention (no positional encoding)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa  = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
            )
        else:
            attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
            x = attn @ v

        return self.proj(x.transpose(1, 2).reshape(B, N, C))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, wide_silu=True, norm_layer=nn.LayerNorm,
                 use_sdpa=True, is_causal=False,
                 use_rope=False, n_registers=0, has_cls_first=False, **kwargs):
        super().__init__()
        self.norm1    = norm_layer(dim)
        self.use_rope = use_rope

        if use_rope:
            self.attn = RoPEAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, use_sdpa=use_sdpa,
                is_causal=is_causal, n_registers=n_registers, has_cls_first=has_cls_first,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, use_sdpa=use_sdpa, is_causal=is_causal,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(dim, hidden_features=mlp_hidden_dim,
                                  act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attn=False):
        if self.use_rope:
            y, attn = self.attn(self.norm1(x), mask=mask, return_attn=return_attn)
        else:
            y    = self.attn(self.norm1(x))
            attn = None

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
