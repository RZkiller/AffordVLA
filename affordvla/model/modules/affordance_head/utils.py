"""Shared building blocks for the Affordance-VLA affordance head."""

import numpy as np
import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Generate MAE-style 2D sin-cos positional embeddings.

    Args:
        embed_dim: Total embedding dimension. Must be divisible by 4.
        grid_h: Patch grid height.
        grid_w: Patch grid width.

    Returns:
        Array with shape ``[grid_h * grid_w, embed_dim]``.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4")

    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float64) / (half_dim // 2)
    omega = 1.0 / (10000.0**omega)

    grid_y = np.arange(grid_h, dtype=np.float64)
    grid_x = np.arange(grid_w, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(grid_y, grid_x, indexing="ij")

    out_y = np.einsum("hw,d->hwd", grid_y, omega).reshape(grid_h * grid_w, -1)
    out_x = np.einsum("hw,d->hwd", grid_x, omega).reshape(grid_h * grid_w, -1)

    return np.concatenate(
        [np.sin(out_y), np.cos(out_y), np.sin(out_x), np.cos(out_x)],
        axis=1,
    )


class Mlp(nn.Module):
    """Feed-forward block: Linear, GELU, dropout, Linear, dropout."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Batch-first scaled dot-product multi-head attention."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Query tensor with shape ``[B, Nq, C]``.
            k: Key tensor with shape ``[B, Nk, C]``.
            v: Value tensor with shape ``[B, Nk, C]``.

        Returns:
            Attention output with shape ``[B, Nq, C]``.
        """
        batch_size, num_queries, channels = q.shape
        num_keys = k.shape[1]

        q = self.q_proj(q).reshape(batch_size, num_queries, self.num_heads, self.head_dim)
        k = self.k_proj(k).reshape(batch_size, num_keys, self.num_heads, self.head_dim)
        v = self.v_proj(v).reshape(batch_size, num_keys, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_queries, channels)
        return self.out_proj(out)
