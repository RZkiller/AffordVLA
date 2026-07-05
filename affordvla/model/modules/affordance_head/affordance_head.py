"""
This module exposes a single public class, ``AffordanceHead``. It implements
the patch-conditioned two-way attention decoder used by Affordance-VLA:

1. Project AFF hidden states and pre-projector vision patch features.
2. Refine both query tokens and patch tokens with two-way attention blocks.
3. Predict one instruction-conditioned mask per view with a dot-product mask
   embedding head.

Output logits have shape ``[B, V, H_p, W_p]`` and are intentionally returned
without sigmoid or thresholding.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from affordvla.model.modules.affordance_head.utils import (
    Mlp,
    MultiHeadAttention,
    get_2d_sincos_pos_embed,
)


class TwoWayAttentionBlock(nn.Module):
    """
    SAM-style two-way attention block.

    Each block updates AFF query tokens and vision patch tokens:
      1. query self-attention
      2. query-to-patch cross-attention
      3. query feed-forward network
      4. patch-to-query cross-attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.q_norm_self = nn.LayerNorm(dim)
        self.q_self_attn = MultiHeadAttention(dim, num_heads, drop=drop)

        self.q_norm_cross_q = nn.LayerNorm(dim)
        self.q_norm_cross_kv = nn.LayerNorm(dim)
        self.q_cross_attn = MultiHeadAttention(dim, num_heads, drop=drop)

        self.q_norm_mlp = nn.LayerNorm(dim)
        self.q_mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        self.p_norm_cross_q = nn.LayerNorm(dim)
        self.p_norm_cross_kv = nn.LayerNorm(dim)
        self.p_cross_attn = MultiHeadAttention(dim, num_heads, drop=drop)

    def forward(
        self,
        query_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_tokens: ``[B*V, K, C_dec]`` AFF query projections.
            patch_tokens: ``[B*V, Np, C_dec]`` vision patch projections.

        Returns:
            Refined ``query_tokens`` and ``patch_tokens`` with the same shapes.
        """
        q_norm = self.q_norm_self(query_tokens)
        query_tokens = query_tokens + self.q_self_attn(q_norm, q_norm, q_norm)

        q_q = self.q_norm_cross_q(query_tokens)
        q_kv = self.q_norm_cross_kv(patch_tokens)
        query_tokens = query_tokens + self.q_cross_attn(q_q, q_kv, q_kv)

        query_tokens = query_tokens + self.q_mlp(self.q_norm_mlp(query_tokens))

        p_q = self.p_norm_cross_q(patch_tokens)
        p_kv = self.p_norm_cross_kv(query_tokens)
        patch_tokens = patch_tokens + self.p_cross_attn(p_q, p_kv, p_kv)

        return query_tokens, patch_tokens


class AffordanceHead(nn.Module):
    """
    Two-way attention affordance mask decoder with dot-product prediction.
    """

    def __init__(
        self,
        llm_hidden_dim: int = 2048,
        vision_hidden_dim: int = 1280,
        decoder_dim: int = 256,
        num_decoder_layers: int = 2,
        num_decoder_heads: int = 8,
        num_aff_queries_per_view: int = 4,
        mlp_ratio: float = 4.0,
        base_grid_size: Tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.num_aff_queries_per_view = num_aff_queries_per_view
        self.base_grid_size = base_grid_size
        h_base, w_base = base_grid_size

        self.query_proj = nn.Linear(llm_hidden_dim, decoder_dim)
        self.patch_proj = nn.Linear(vision_hidden_dim, decoder_dim)

        pos_embed_np = get_2d_sincos_pos_embed(decoder_dim, h_base, w_base)
        self.register_buffer(
            "base_spatial_pos_embed",
            torch.tensor(pos_embed_np, dtype=torch.float32).unsqueeze(0),
        )

        self.two_way_blocks = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    dim=decoder_dim,
                    num_heads=num_decoder_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.final_q_norm = nn.LayerNorm(decoder_dim)
        self.final_p_norm = nn.LayerNorm(decoder_dim)
        self.final_attn_q_to_p = MultiHeadAttention(decoder_dim, num_decoder_heads)

        self.query_out_norm = nn.LayerNorm(decoder_dim)
        self.spatial_out_norm = nn.LayerNorm(decoder_dim)
        self.mask_embed_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        h_base: int,
        w_base: int,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        """Bilinearly interpolate ``[1, H*W, C]`` positional embeddings."""
        channels = pos_embed.shape[-1]
        pos_2d = pos_embed.reshape(1, h_base, w_base, channels).permute(0, 3, 1, 2)
        pos_2d = F.interpolate(pos_2d, size=(h_p, w_p), mode="bilinear", align_corners=False)
        return pos_2d.permute(0, 2, 3, 1).reshape(1, h_p * w_p, channels)

    def _get_spatial_pos_embed(self, h_p: int, w_p: int) -> torch.Tensor:
        """Return spatial positional embeddings for the current patch grid."""
        h_base, w_base = self.base_grid_size
        if h_p == h_base and w_p == w_base:
            return self.base_spatial_pos_embed
        return self._interpolate_pos_embed(self.base_spatial_pos_embed, h_base, w_base, h_p, w_p)

    def forward(
        self,
        aff_hidden: torch.Tensor,
        num_views: int,
        patch_grid_hw: Tuple[int, int],
        vision_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode AFF hidden states into per-view affordance mask logits.

        Args:
            aff_hidden: ``[B, K_per_view * V, C_llm]`` hidden states from AFF tokens.
            num_views: Number of camera views per sample.
            patch_grid_hw: Spatial patch grid ``(H_p, W_p)`` for each view.
            vision_patches: ``[B, V, H_p*W_p, C_vis]`` pre-projector vision features.

        Returns:
            Raw mask logits with shape ``[B, V, H_p, W_p]``.
        """
        if vision_patches is None:
            raise ValueError(
                "AffordanceHead requires vision_patches from the VLM vision encoder."
            )

        batch_size = aff_hidden.shape[0]
        num_patches_h, num_patches_w = patch_grid_hw
        num_patches = num_patches_h * num_patches_w
        queries_per_view = self.num_aff_queries_per_view

        expected_aff_tokens = queries_per_view * num_views
        if aff_hidden.shape[1] != expected_aff_tokens:
            raise ValueError(
                f"aff_hidden has {aff_hidden.shape[1]} AFF tokens, expected "
                f"{queries_per_view} * {num_views} = {expected_aff_tokens}."
            )

        expected_patch_shape = (batch_size, num_views, num_patches)
        if vision_patches.shape[:3] != expected_patch_shape:
            raise ValueError(
                f"vision_patches shape {tuple(vision_patches.shape)} is incompatible with "
                f"(B={batch_size}, V={num_views}, Np={num_patches})."
            )

        aff_folded = aff_hidden.reshape(batch_size, num_views, queries_per_view, -1)
        aff_folded = aff_folded.reshape(batch_size * num_views, queries_per_view, -1)
        aff_folded = aff_folded.to(dtype=self.query_proj.weight.dtype)
        query_tokens = self.query_proj(aff_folded)

        vis_folded = vision_patches.reshape(batch_size * num_views, num_patches, -1)
        vis_folded = vis_folded.to(dtype=self.patch_proj.weight.dtype)
        spatial_pos = self._get_spatial_pos_embed(num_patches_h, num_patches_w)
        spatial_pos = spatial_pos.to(dtype=query_tokens.dtype, device=query_tokens.device)
        patch_tokens = self.patch_proj(vis_folded) + spatial_pos

        for block in self.two_way_blocks:
            query_tokens, patch_tokens = block(query_tokens, patch_tokens)

        q_final = self.final_q_norm(query_tokens)
        p_final = self.final_p_norm(patch_tokens)
        query_tokens = query_tokens + self.final_attn_q_to_p(q_final, p_final, p_final)

        query_out = self.query_out_norm(query_tokens)
        mask_embed = query_out.mean(dim=1)
        mask_embed = self.mask_embed_mlp(mask_embed)

        spatial_out = self.spatial_out_norm(patch_tokens)
        mask_logits = torch.einsum("bc,bnc->bn", mask_embed, spatial_out)
        return mask_logits.reshape(batch_size, num_views, num_patches_h, num_patches_w)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = AffordanceHead(
        llm_hidden_dim=2048,
        vision_hidden_dim=1280,
        decoder_dim=256,
        num_decoder_layers=2,
        num_decoder_heads=8,
        num_aff_queries_per_view=4,
        base_grid_size=(16, 16),
    ).to(device)

    bsz, views, queries = 2, 2, 4
    aff = torch.randn(bsz, views * queries, 2048, device=device)
    patches = torch.randn(bsz, views, 16 * 16, 1280, device=device)
    logits = head(aff, num_views=views, patch_grid_hw=(16, 16), vision_patches=patches)
    assert logits.shape == (bsz, views, 16, 16), logits.shape
    print(f"AffordanceHead smoke test passed: {tuple(logits.shape)}")
