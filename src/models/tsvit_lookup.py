"""TSViT with lookup-based temporal position embeddings."""

from __future__ import annotations

import warnings

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Simplified multi-head self-attention with optional padding mask."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = "Embedding dimension must be divisible by number of heads"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert key_padding_mask to attention mask format for SDPA
        # SDPA expects: True = attend, False = ignore (opposite of key_padding_mask)
        attn_mask = None
        if key_padding_mask is not None:
            # Expand to (batch, 1, 1, seq_len) for broadcasting
            attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Use PyTorch's fused scaled_dot_product_attention (FlashAttention when available)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
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
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class TSViTLookup(nn.Module):
    """Temporal-Spatial ViT with lookup-based temporal position embeddings.

    This is the most advanced temporal encoding approach from the paper.
    It learns a separate position embedding for each unique date seen during training,
    then uses linear interpolation for unseen dates during inference.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        train_dates: list[int] | torch.Tensor,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        temporal_metadata_channels: int = 0,
        date_range: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the TSViT module with lookup temporal embeddings.

        Args:
            image_size: Height/width (in Sentinel pixels) of the cropped patch.
            patch_size: Edge length of each Vision Transformer patch.
            in_channels: Number of spectral channels produced by the dataset.
            num_classes: Number of segmentation categories.
            train_dates: List/tensor of unique dates (e.g., day-of-year 1-365) seen in training.
                During inference, dates not in this list are interpolated.
                IMPORTANT: All positions passed to forward() must use the same indexing scheme.
                For day-of-year: use 1-365 (not 0-364).
                For months: use 1-12 (not 0-11) or pass custom date_range.
            dim: Embedding dimension of token representations.
            temporal_depth: Number of transformer blocks in the temporal encoder.
            spatial_depth: Number of transformer blocks in the spatial encoder.
            num_heads: Number of attention heads for both encoders.
            mlp_dim: Hidden size of the feed-forward sublayers.
            dropout: Dropout applied inside the transformer blocks.
            emb_dropout: Dropout applied after adding positional embeddings.
            temporal_metadata_channels: Optional number of metadata channels reserved
                at the end of the spectral dimension (e.g., timestamps).
            date_range: Optional (min, max) tuple defining the range of valid dates for
                interpolation during inference. Defaults to (1, 365) for day-of-year.
                For 0-indexed DOY use (0, 364), for months use (1, 12) or (0, 11).

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
        self.dim = dim
        self.temporal_metadata_channels = temporal_metadata_channels

        # Register train dates as buffer (non-trainable parameter)
        if isinstance(train_dates, list):
            train_dates = torch.tensor(sorted(train_dates), dtype=torch.long)
        else:
            train_dates = torch.sort(train_dates.long())[0]

        # For inference, we'll interpolate over all possible dates in the specified range
        if date_range is None:
            date_range = (1, 365)  # Default to 1-indexed day-of-year
        self.date_range = date_range

        # Validate train_dates are within date_range
        if train_dates.min() < date_range[0] or train_dates.max() > date_range[1]:
            msg = (
                f"train_dates must be within date_range [{date_range[0]}, {date_range[1]}]. "
                f"Got train_dates range: [{train_dates.min()}, {train_dates.max()}]"
            )
            raise ValueError(msg)

        self.register_buffer("train_dates", train_dates)
        self.register_buffer(
            "eval_dates",
            torch.arange(date_range[0], date_range[1] + 1, dtype=torch.long),
        )

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

        # Learnable position embedding for each training date
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(len(train_dates), dim),
            requires_grad=True,
        )

        # Pre-compute interpolated embeddings for evaluation (lazy evaluation)
        num_eval_dates = date_range[1] - date_range[0] + 1
        self.register_buffer(
            "inference_temporal_pos_embedding",
            torch.zeros(num_eval_dates, dim),
        )
        self._inference_embeddings_stale = True  # Mark for lazy computation

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

    def train(self, mode: bool = True) -> TSViTLookup:
        """Mark inference embeddings as stale when switching to eval mode.

        Args:
            mode: If True, sets to training mode; if False, sets to eval mode.

        Returns:
            Self for method chaining.

        """
        result = super().train(mode)
        if not mode:  # Switching to eval mode
            self._inference_embeddings_stale = True
        return result

    def _update_inference_temporal_position_embeddings(self) -> None:
        """Pre-compute interpolated position embeddings for inference (lazy evaluation).

        For dates not seen during training, we use linear interpolation between
        the nearest training dates. For dates outside the training range, we use
        the nearest boundary embedding.

        Only recomputes if embeddings are marked as stale.
        """
        if not self._inference_embeddings_stale:
            return  # Skip if already up-to-date

        train_dates_idx = torch.arange(self.train_dates.shape[0])

        # Handle edge case where 0 might be in training dates (use next date as min)
        min_val = torch.min(self.train_dates).item()
        min_idx = torch.argmin(self.train_dates).item()
        if min_val == 0:
            # Find the minimum value excluding 0, and its correct index
            non_zero_dates = self.train_dates[self.train_dates > 0]
            if len(non_zero_dates) == 0:
                # All training dates are 0 - this is probably a user error
                msg = (
                    "All train_dates are 0. This is likely incorrect. "
                    "Please verify your date indices."
                )
                raise ValueError(msg)
            min_val = torch.min(non_zero_dates).item()
            min_idx = (self.train_dates == min_val).nonzero(as_tuple=True)[0].item()

        max_val = torch.max(self.train_dates).item()
        max_idx = torch.argmax(self.train_dates).item()

        pos_eval = torch.zeros(self.eval_dates.shape[0], self.dim)

        for i, evdate in enumerate(self.eval_dates):
            if evdate < min_val:
                # Before training range: use minimum embedding
                pos_eval[i] = self.temporal_pos_embedding[min_idx]
            elif evdate > max_val:
                # After training range: use maximum embedding
                pos_eval[i] = self.temporal_pos_embedding[max_idx]
            else:
                # Within training range
                dist = evdate - self.train_dates
                if 0 in dist:
                    # Exact match with training date
                    pos_eval[i] = self.temporal_pos_embedding[dist == 0]
                else:
                    # Linear interpolation between nearest dates
                    lower_idx = train_dates_idx[dist >= 0].max().item()
                    upper_idx = train_dates_idx[dist <= 0].min().item()
                    lower_date = self.train_dates[lower_idx].item()
                    upper_date = self.train_dates[upper_idx].item()

                    # Weighted average based on distance
                    weight_lower = (upper_date - evdate) / (upper_date - lower_date)
                    weight_upper = (evdate - lower_date) / (upper_date - lower_date)

                    pos_eval[i] = (
                        weight_lower * self.temporal_pos_embedding[lower_idx]
                        + weight_upper * self.temporal_pos_embedding[upper_idx]
                    )

        self.inference_temporal_pos_embedding.copy_(pos_eval)
        self._inference_embeddings_stale = False  # Mark as fresh

    def _get_temporal_position_embeddings(
        self,
        positions: torch.Tensor,
        inference: bool = False,
    ) -> torch.Tensor:
        """Get temporal position embeddings via lookup or interpolation.

        Args:
            positions: (B, T) tensor of date indices
            inference: If True, use interpolated embeddings; else use training lookup

        Returns:
            (B, T, dim) position embeddings

        """
        B, T = positions.shape

        if inference:
            # Use pre-computed interpolated embeddings
            positions_flat = positions.ravel()

            # Warn about out-of-range positions
            out_of_range = (positions_flat < self.date_range[0]) | (
                positions_flat > self.date_range[1]
            )
            if out_of_range.any():
                warnings.warn(
                    f"Some inference positions are outside date_range "
                    f"[{self.date_range[0]}, {self.date_range[1]}]. "
                    f"Out-of-range values: {positions_flat[out_of_range].unique().tolist()}. "
                    f"These will use boundary embeddings.",
                    UserWarning,
                    stacklevel=2,
                )

            # Bucketize finds the insertion index for each position
            index = torch.bucketize(positions_flat, self.eval_dates)
            index = index.clamp(max=len(self.eval_dates) - 1)
            return self.inference_temporal_pos_embedding[index].reshape(B, T, self.dim)
        # Direct lookup from training embeddings - require exact matches for valid positions
        positions_flat = positions.ravel()

        # Mask out padding positions (negative values like -100)
        valid_mask = positions_flat >= 0

        index = torch.searchsorted(self.train_dates, positions_flat.clamp(min=0))
        index = index.clamp(max=len(self.train_dates) - 1)

        # Verify valid (non-padding) positions exactly match train_dates
        if valid_mask.any():
            valid_positions = positions_flat[valid_mask]
            valid_indices = index[valid_mask]
            matched_dates = self.train_dates[valid_indices]
            if not torch.all(matched_dates == valid_positions):
                unmatched = valid_positions[matched_dates != valid_positions]
                msg = (
                    f"Training positions must exactly match train_dates. "
                    f"Found unmatched dates: {unmatched.unique().tolist()}. "
                    f"Available train_dates: {self.train_dates.tolist()}"
                )
                raise ValueError(msg)

        pos_embeddings = self.temporal_pos_embedding[index].reshape(B, T, self.dim)
        # Ensure padding positions (negative indices) do not receive a valid date embedding
        if not valid_mask.all():
            valid_mask_reshaped = valid_mask.view(B, T)
            pos_embeddings = pos_embeddings.clone()
            pos_embeddings[~valid_mask_reshaped] = 0.0
        return pos_embeddings

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch_positions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        inference: bool | None = None,
    ) -> torch.Tensor:
        """Run the temporal-spatial transformer over a Sentinel-2 sequence.

        Args:
            x: Input tensor of shape (B, T, C, H, W)
            batch_positions: Optional (B, T) tensor with date indices (e.g., day-of-year)
            pad_mask: Optional (B, T) boolean mask (True = padded/invalid)
            inference: If True, use interpolated embeddings; if False, use direct lookup.
                If None (default), auto-detects based on model.training
                (True when model.eval(), False when model.train()).

        Returns:
            Output logits of shape (B, num_classes, H, W)

        """
        # Auto-detect inference mode if not specified
        if inference is None:
            inference = not self.training
        if x.ndim != 5:
            msg = "TSViTLookup expects input of shape (B, T, C, H, W)"
            raise ValueError(msg)

        batch_size, time, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            msg = (
                f"Input spatial resolution {(height, width)} does not match "
                f"configured image_size {self.image_size}"
            )
            raise ValueError(msg)

        # Update interpolated embeddings if in inference mode
        if inference:
            self._update_inference_temporal_position_embeddings()

        clean_x, metadata = self._split_metadata(x)

        # Get temporal positions (default to sequential if not provided)
        if batch_positions is None:
            # Use sequential dates starting from the minimum of the date range
            batch_positions = (
                torch.arange(
                    self.date_range[0],
                    self.date_range[0] + time,
                    device=x.device,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

        temporal_pos = self._get_temporal_position_embeddings(
            batch_positions,
            inference=inference,
        )

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
        # temporal_pos is already (B, T, dim) from lookup/interpolation
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


__all__ = ["TSViTLookup"]
