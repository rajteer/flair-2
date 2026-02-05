"""Mid-level multimodal fusion with cross-attention between aerial and Sentinel features.

This module implements a mid-level fusion architecture that applies cross-attention
between UNetFormer encoder features (aerial) and TSViT temporal encoder features
(Sentinel-2), enabling richer feature interaction before decoding.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn


logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Cross-attention module for fusing aerial and Sentinel features.

    Aerial features serve as Query, Sentinel temporal tokens serve as Key/Value.
    This allows the aerial branch to selectively attend to relevant temporal
    information from Sentinel data.

    Spatial position embeddings are added to sentinel tokens to preserve
    their spatial reference during cross-attention.

    Args:
        aerial_channels: Number of channels in aerial feature maps.
        sentinel_dim: Embedding dimension of Sentinel temporal tokens.
        num_heads: Number of attention heads.
        dropout: Dropout rate for attention weights.
        max_sentinel_patches: Maximum number of Sentinel patches (for position embeddings).

    """

    def __init__(
        self,
        aerial_channels: int,
        sentinel_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sentinel_patches: int = 16,  # Default: 4x4 patches from 40x40 with patch_size=10
    ) -> None:
        """Initialize cross-attention fusion module."""
        super().__init__()

        self.aerial_channels = aerial_channels
        self.sentinel_dim = sentinel_dim
        self.num_heads = num_heads
        self.head_dim = aerial_channels // num_heads
        self.max_sentinel_patches = max_sentinel_patches

        if aerial_channels % num_heads != 0:
            msg = (
                f"aerial_channels ({aerial_channels}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Project aerial features to Q
        self.q_proj = nn.Conv2d(aerial_channels, aerial_channels, kernel_size=1)

        # Project Sentinel tokens to K, V (match aerial channel dim)
        self.k_proj = nn.Linear(sentinel_dim, aerial_channels)
        self.v_proj = nn.Linear(sentinel_dim, aerial_channels)

        # Learnable spatial position embeddings for Sentinel patches
        # These help the model understand WHERE each Sentinel token came from
        self.sentinel_spatial_pos = nn.Parameter(
            torch.randn(1, max_sentinel_patches, sentinel_dim) * 0.02
        )

        # Output projection
        self.out_proj = nn.Conv2d(aerial_channels, aerial_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(aerial_channels)

        # Learnable gating for residual connection
        # Initialized to -5.0 so sigmoid(-5) ≈ 0.007, making model start as pure aerial
        # This "zero-init residual" pattern improves training stability (ResNet-v2, GPT-2)
        self.gate = nn.Parameter(torch.tensor([-5.0]))

    def forward(
        self,
        aerial_features: torch.Tensor,
        sentinel_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention from aerial features to Sentinel tokens.

        Args:
            aerial_features: Aerial encoder features of shape (B, C, H, W).
            sentinel_tokens: Sentinel temporal tokens of shape (B, K, P, dim)
                where K = num_classes, P = num_patches.

        Returns:
            Fused features of shape (B, C, H, W).

        """
        batch, channels, height, width = aerial_features.shape
        _, num_classes, num_patches, sentinel_dim = sentinel_tokens.shape

        # Add spatial position embeddings to sentinel tokens BEFORE flattening
        # This preserves the spatial reference (which patch each token came from)
        if num_patches <= self.max_sentinel_patches:
            spatial_pos = self.sentinel_spatial_pos[:, :num_patches, :]
        else:
            # Interpolate position embeddings if needed
            spatial_pos = nn.functional.interpolate(
                self.sentinel_spatial_pos.transpose(1, 2),
                size=num_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # sentinel_tokens: (B, K, P, dim) + spatial_pos: (1, P, dim) -> broadcast over K
        sentinel_with_pos = sentinel_tokens + spatial_pos.unsqueeze(1)

        # Flatten sentinel tokens: (B, K*P, dim)
        sentinel_flat = sentinel_with_pos.view(batch, num_classes * num_patches, sentinel_dim)

        # Project to Q (from aerial), K/V (from sentinel)
        # Q: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        q = self.q_proj(aerial_features)
        q = q.view(batch, channels, height * width).permute(0, 2, 1)

        # K, V: (B, K*P, dim) -> (B, K*P, C)
        k = self.k_proj(sentinel_flat)
        v = self.v_proj(sentinel_flat)

        # Multi-head attention
        # Reshape for multi-head: (B, N, num_heads, head_dim)
        q = q.view(batch, height * width, self.num_heads, self.head_dim)
        k = k.view(batch, num_classes * num_patches, self.num_heads, self.head_dim)
        v = v.view(batch, num_classes * num_patches, self.num_heads, self.head_dim)

        # Transpose to (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention: (B, heads, H*W, K*P)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        # Reshape back: (B, heads, H*W, head_dim) -> (B, H*W, C)
        out = out.permute(0, 2, 1, 3).reshape(batch, height * width, channels)

        # Layer norm after attention
        out = self.norm(out)

        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        out = out.permute(0, 2, 1).view(batch, channels, height, width)
        out = self.out_proj(out)

        # Gated residual connection (starts at 0, learns optimal blend)
        return aerial_features + torch.sigmoid(self.gate) * out


class MultimodalMidFusion(nn.Module):
    """Mid-level fusion model combining aerial and Sentinel at feature level.

    Uses cross-attention to fuse UNetFormer encoder features with TSViT temporal
    tokens before passing to the decoder.

    Args:
        aerial_backbone: UNetFormer encoder backbone (timm model).
        aerial_decoder: UNetFormer decoder module.
        sentinel_encoder: TSViT model for temporal encoding.
        cross_attention: CrossAttentionFusion module.
        num_classes: Number of output segmentation classes.
        freeze_encoders: Whether to freeze pre-trained encoder weights.

    """

    def __init__(
        self,
        aerial_backbone: nn.Module,
        aerial_decoder: nn.Module,
        sentinel_encoder: nn.Module,
        cross_attention: CrossAttentionFusion,
        num_classes: int,
        *,
        freeze_encoders: bool = True,
    ) -> None:
        """Initialize mid-level fusion model."""
        super().__init__()

        self.aerial_backbone = aerial_backbone
        self.aerial_decoder = aerial_decoder
        self.sentinel_encoder = sentinel_encoder
        self.cross_attention = cross_attention
        self.num_classes = num_classes

        if freeze_encoders:
            self._freeze_model(self.aerial_backbone)
            self._freeze_model(self.sentinel_encoder)
            logger.info("Froze aerial backbone and Sentinel encoder weights")

    @staticmethod
    def _freeze_model(model: nn.Module) -> None:
        """Freeze all parameters in a model."""
        for param in model.parameters():
            param.requires_grad = False

    def forward(
        self,
        aerial_input: torch.Tensor,
        sentinel_input: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with mid-level cross-attention fusion.

        Args:
            aerial_input: Aerial imagery tensor of shape (B, C, H, W).
            sentinel_input: Sentinel-2 time series of shape (B, T, C, H, W).
            batch_positions: Temporal positions of shape (B, T) for Sentinel model.
            pad_mask: Padding mask of shape (B, T) indicating valid timesteps.

        Returns:
            Segmentation predictions of shape (B, num_classes, H, W).

        """
        h, w = aerial_input.shape[-2:]

        # Extract encoder features from aerial backbone
        res1, res2, res3, res4 = self.aerial_backbone(aerial_input)

        # Get temporal tokens from Sentinel encoder
        sentinel_tokens = self.sentinel_encoder.encode_temporal(
            sentinel_input,
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )

        # Apply cross-attention fusion at deepest level
        fused_res4 = self.cross_attention(res4, sentinel_tokens)

        # Decode with fused features
        return self.aerial_decoder(res1, res2, res3, fused_res4, h, w)


def build_multimodal_mid_fusion(  # noqa: PLR0913
    aerial_backbone: nn.Module,
    aerial_decoder: nn.Module,
    sentinel_encoder: nn.Module,
    num_classes: int,
    *,
    sentinel_dim: int = 128,
    aerial_channels: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    freeze_encoders: bool = True,
    max_sentinel_patches: int = 16,
) -> MultimodalMidFusion:
    """Factory function to build MultimodalMidFusion model.

    Args:
        aerial_backbone: UNetFormer encoder backbone.
        aerial_decoder: UNetFormer decoder module.
        sentinel_encoder: TSViT model for temporal encoding.
        num_classes: Number of output segmentation classes.
        sentinel_dim: Embedding dimension of Sentinel tokens.
        aerial_channels: Number of channels at deepest encoder level.
        num_heads: Number of cross-attention heads.
        dropout: Dropout rate for attention.
        freeze_encoders: Whether to freeze encoder weights.
        max_sentinel_patches: Maximum Sentinel patches for spatial position embeddings.

    Returns:
        Configured MultimodalMidFusion model.

    """
    cross_attention = CrossAttentionFusion(
        aerial_channels=aerial_channels,
        sentinel_dim=sentinel_dim,
        num_heads=num_heads,
        dropout=dropout,
        max_sentinel_patches=max_sentinel_patches,
    )

    return MultimodalMidFusion(
        aerial_backbone=aerial_backbone,
        aerial_decoder=aerial_decoder,
        sentinel_encoder=sentinel_encoder,
        cross_attention=cross_attention,
        num_classes=num_classes,
        freeze_encoders=freeze_encoders,
    )
