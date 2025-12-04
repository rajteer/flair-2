"""Temporal-Spatial Vision Transformer (TSViT) implementation.

This module adapts the TSViT architecture from
"ViTs for SITS: Vision Transformers for Satellite Image Time Series"
(CVPR 2023, Tarasiou et al.) to the FLAIR-2 Sentinel-2 only scenario. The
implementation follows the original design (temporal transformer followed
by spatial transformer over patch tokens) while relaxing assumptions about
sequence length and date encodings so it can ingest the monthly averaged
sentinel stacks produced by this repository.
"""

from __future__ import annotations

import math

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as torch_functional

TEMPORAL_INPUT_NDIM = 5


class MultiHeadSelfAttention(nn.Module):
    """Simplified multi-head self-attention with optional padding mask."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = "Embedding dimension must be divisible by number of heads"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional boolean mask of shape (batch, seq_len)
                where True indicates positions to ignore

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        """
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return (
                t.view(batch, seq_len, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x S
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block with attention and feedforward.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional boolean mask for attention

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        """
        attn_out = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks with shared padding mask."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)],
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply stack of transformer blocks.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            key_padding_mask: Optional boolean mask for attention

        Returns:
            Output tensor of shape (batch, seq_len, dim) after layer normalization

        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class TSViT(nn.Module):
    """Temporal-Spatial Vision Transformer tailored to Sentinel-2 patches."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        max_seq_len: int,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        temporal_metadata_channels: int = 0,
    ) -> None:
        """Initialize the TSViT module.

        Args:
            image_size: Height/width (in Sentinel pixels) of the cropped patch.
            patch_size: Edge length of each Vision Transformer patch.
            in_channels: Number of spectral channels produced by the dataset.
            num_classes: Number of segmentation categories.
            max_seq_len: Maximum temporal length for one-hot position encoding.
                Use 12 for month-of-year encoding (0-11), 366 for day-of-year encoding (0-365).
                Position indices are one-hot encoded and projected to embedding dimension.
            dim: Embedding dimension of token representations.
            temporal_depth: Number of transformer blocks in the temporal encoder.
            spatial_depth: Number of transformer blocks in the spatial encoder.
            num_heads: Number of attention heads for both encoders.
            mlp_dim: Hidden size of the feed-forward sublayers.
            dropout: Dropout applied inside the transformer blocks.
            emb_dropout: Dropout applied after adding positional embeddings.
            temporal_metadata_channels: Optional number of metadata channels reserved
                at the end of the spectral dimension (e.g., timestamps).

        """
        super().__init__()
        if image_size % patch_size != 0:
            msg = "image_size must be divisible by patch_size"
            raise ValueError(msg)
        if in_channels <= temporal_metadata_channels:
            msg = "in_channels must be larger than temporal_metadata_channels"
            raise ValueError(msg)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.temporal_metadata_channels = temporal_metadata_channels

        self.num_patches_1d = image_size // patch_size
        self.num_patches = self.num_patches_1d**2
        patch_dim = (in_channels - temporal_metadata_channels) * (patch_size**2)

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )
        # Use one-hot encoding + linear projection for temporal positions (as in paper)
        # max_seq_len should be 366 for day-of-year or 12 for month-of-year
        self.to_temporal_embedding_input = nn.Linear(max_seq_len, dim)
        self.temporal_dropout = nn.Dropout(emb_dropout)
        self.temporal_cls_tokens = nn.Parameter(torch.randn(1, num_classes, dim))
        self.temporal_encoder = TransformerEncoder(
            dim=dim,
            depth=temporal_depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.space_dropout = nn.Dropout(emb_dropout)
        self.space_encoder = TransformerEncoder(
            dim=dim,
            depth=spatial_depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_size**2))

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch_positions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the temporal-spatial transformer over a Sentinel-2 sequence."""
        if x.ndim != TEMPORAL_INPUT_NDIM:
            msg = "TSViT expects input of shape (B, T, C, H, W)"
            raise ValueError(msg)

        batch_size, time, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            msg = (
                "Input spatial resolution does not match configured image_size: "
                f"expected {self.image_size}, got {(height, width)}"
            )
            raise ValueError(msg)

        clean_x, metadata = self._split_metadata(x)
        temporal_pos = self._prepare_positions(batch_positions, batch_size, time, x.device)
        effective_mask = self._resolve_pad_mask(pad_mask, metadata, clean_x.device)

        temporal_tokens = self._encode_temporal(
            clean_x,
            temporal_pos,
            effective_mask,
            batch_size,
            time,
        )

        return self._decode_spatial(temporal_tokens, batch_size)

    def _split_metadata(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.temporal_metadata_channels:
            return x, None

        metadata = x[:, :, -self.temporal_metadata_channels :, :, :]
        clean = x[:, :, : (x.shape[2] - self.temporal_metadata_channels), :, :]
        return clean, metadata

    def _prepare_positions(
        self,
        batch_positions: torch.Tensor | None,
        batch_size: int,
        time: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Prepare temporal position encodings using one-hot + linear projection.

        Args:
            batch_positions: Optional (B, T) tensor with position indices (DOY or month).
                Negative values are treated as padding and mapped to a dedicated
                pad index (``max_seq_len - 1``).
            batch_size: Batch size.
            time: Sequence length.
            device: Device to create tensors on.

        Returns:
            One-hot encoded and projected position embeddings (B, T, dim).

        """
        if batch_positions is None:
            batch_positions = torch.arange(time, device=device).unsqueeze(0)
            batch_positions = batch_positions.repeat(batch_size, 1)

        pad_index = self.max_seq_len - 1
        batch_positions = batch_positions.long()

        is_pad = batch_positions < 0
        batch_positions = batch_positions.clamp(min=0, max=pad_index - 1)
        batch_positions[is_pad] = pad_index

        # One-hot encode the positions (B, T, max_seq_len)
        positions_one_hot = torch_functional.one_hot(
            batch_positions,
            num_classes=self.max_seq_len,
        ).float()

        # Project to embedding dimension (B, T, dim)
        return self.to_temporal_embedding_input(
            positions_one_hot.view(-1, self.max_seq_len),
        ).view(batch_size, time, self.dim)

    def _resolve_pad_mask(
        self,
        pad_mask: torch.Tensor | None,
        metadata: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if pad_mask is not None:
            return pad_mask.to(dtype=torch.bool, device=device)
        if metadata is None:
            return None
        return metadata.sum(dim=(2, 3, 4)) == 0

    def _encode_temporal(
        self,
        x: torch.Tensor,
        temporal_pos: torch.Tensor,
        pad_mask: torch.Tensor | None,
        batch_size: int,
        time: int,
    ) -> torch.Tensor:
        # temporal_pos is already (B, T, dim) from one-hot encoding
        x = self.to_patch_embedding(x)
        x = x.view(batch_size, self.num_patches, time, self.dim)
        # Add temporal position to each spatial patch
        x = self.temporal_dropout(x + temporal_pos.unsqueeze(1))

        repeated_mask = None
        if pad_mask is not None:
            repeated_mask = pad_mask.unsqueeze(1).repeat(1, self.num_patches, 1)
            repeated_mask = repeated_mask.reshape(batch_size * self.num_patches, time)

        temporal_tokens = x.view(batch_size * self.num_patches, time, self.dim)
        cls_tokens = repeat(
            self.temporal_cls_tokens,
            "() n d -> b n d",
            b=batch_size * self.num_patches,
        )
        tokens = torch.cat([cls_tokens, temporal_tokens], dim=1)

        if repeated_mask is not None:
            cls_padding = torch.zeros(
                repeated_mask.size(0),
                self.num_classes,
                dtype=torch.bool,
                device=repeated_mask.device,
            )
            temporal_padding = torch.cat([cls_padding, repeated_mask], dim=1)
        else:
            temporal_padding = None

        tokens = self.temporal_encoder(tokens, key_padding_mask=temporal_padding)
        tokens = tokens[:, : self.num_classes, :]
        tokens = tokens.view(batch_size, self.num_patches, self.num_classes, self.dim)
        return tokens.permute(0, 2, 1, 3).contiguous()

    def _decode_spatial(
        self,
        tokens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        spatial_tokens = tokens.view(batch_size * self.num_classes, self.num_patches, self.dim)
        spatial_tokens = spatial_tokens + self.space_pos_embedding.expand_as(spatial_tokens)
        spatial_tokens = self.space_dropout(spatial_tokens)
        spatial_tokens = self.space_encoder(spatial_tokens)

        logits = self.head(spatial_tokens.view(-1, self.dim))
        logits = logits.view(
            batch_size,
            self.num_classes,
            self.num_patches,
            self.patch_size**2,
        )
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(
            batch_size,
            self.num_patches_1d,
            self.num_patches_1d,
            self.patch_size,
            self.patch_size,
            self.num_classes,
        )
        logits = logits.permute(0, 5, 1, 3, 2, 4)
        return logits.reshape(batch_size, self.num_classes, self.image_size, self.image_size)
