"""Multimodal Late Fusion model combining aerial and Sentinel-2 modalities.

This module implements a late fusion architecture that combines predictions from
a pre-trained aerial model (e.g., UNetFormer) and a pre-trained Sentinel-2 temporal
model (e.g., TSViT) using learnable per-class modality weights.
"""

from __future__ import annotations

import logging

import torch
from torch.nn import functional
from torch import nn

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_STATE_KEYS: tuple[str, ...] = (
    "model_state_dict",
    "state_dict",
    "model",
    "net",
    "weights",
)


class MultiScaleChannelAttention(nn.Module):
    """Multi-Scale Channel Attention Module (MS-CAM).

    Based on 'Attentional Feature Fusion' (Dai et al., 2021).
    Fuses features by considering both global context (GAP) and local context
    through pointwise 1x1 convolutions.

    Args:
        channels: Number of input/output channels.
        r: Reduction ratio for the bottleneck.

    """

    def __init__(self, channels: int, r: int = 16) -> None:
        """Initialize the MS-CAM module."""
        super().__init__()
        inter_channels = max(channels // r, 1)

        # Local Path
        self.local_path = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Global Path
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial-varying gating weights."""
        l_feat = self.local_path(x)
        g_feat = self.global_path(x)
        return self.sigmoid(l_feat + g_feat)


class MultimodalLateFusion(nn.Module):
    """Late fusion model combining aerial and Sentinel-2 predictions.

    This model fuses predictions from two pre-trained modality-specific models:
    - An aerial model (e.g., UNetFormer) for high-resolution imagery
    - A Sentinel model (e.g., TSViT) for temporal satellite data

    The fusion uses learnable per-class weights to determine each modality's
    contribution for each semantic class.

    Args:
        aerial_model: Pre-trained model for aerial imagery.
        sentinel_model: Pre-trained model for Sentinel-2 time series.
        num_classes: Number of output segmentation classes.
        freeze_encoders: Whether to freeze pre-trained encoder weights.
        fusion_mode: Fusion strategy - 'weighted' (per-class weights),
            'gated' (content-aware spatial gates), 'concat' (channel concatenation),
            or 'average'.
        aerial_resolution: Tuple (H, W) for aerial model output resolution.
        sentinel_resolution: Tuple (H, W) for Sentinel model output resolution.
        use_cloud_uncertainty: Whether to use cloud coverage as input to gated fusion.

    """

    def __init__(
        self,
        aerial_model: nn.Module,
        sentinel_model: nn.Module,
        num_classes: int,
        *,
        freeze_encoders: bool = True,
        fusion_mode: str = "weighted",
        aerial_resolution: tuple[int, int] = (512, 512),
        sentinel_resolution: tuple[int, int] = (10, 10),
        use_cloud_uncertainty: bool = False,
    ) -> None:
        """Initialize the multimodal late fusion model."""
        super().__init__()

        self.aerial_model = aerial_model
        self.sentinel_model = sentinel_model
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.aerial_resolution = aerial_resolution
        self.sentinel_resolution = sentinel_resolution
        self.use_cloud_uncertainty = use_cloud_uncertainty

        # Freeze encoder weights if requested
        if freeze_encoders:
            self._freeze_model(self.aerial_model)
            self._freeze_model(self.sentinel_model)
            logger.info("Froze aerial and Sentinel encoder weights")

        # Per-class modality weights: (num_classes, 2) for [aerial, sentinel]
        # Initialize to zeros so softmax gives equal weights (0.5, 0.5)
        if fusion_mode == "weighted":
            self.class_weights = nn.Parameter(torch.zeros(num_classes, 2))
        elif fusion_mode == "gated":
            # Gated fusion: learn spatially-varying weights from logits
            # Input: aerial_logits (K) + sentinel_logits (K) + optional cloud (1)
            gate_input_channels = num_classes * 2 + (1 if use_cloud_uncertainty else 0)
            self.gate_network = nn.Sequential(
                nn.Conv2d(gate_input_channels, num_classes, kernel_size=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes, num_classes, kernel_size=1),
                nn.Sigmoid(),  # Gate values in [0, 1]
            )
        elif fusion_mode == "concat":
            # Fusion head that takes concatenated logits
            self.fusion_head = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes * 2, kernel_size=1),
                nn.BatchNorm2d(num_classes * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=1),
            )
        elif fusion_mode == "attentional":
            # Multi-Scale Channel Attention based fusion (SOTA)
            # Operates on the sum of modality features
            self.ms_cam = MultiScaleChannelAttention(num_classes)
        else:
            # 'average' mode requires no extra parameters
            self.class_weights = None  # type: ignore[assignment]

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
        cloud_coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass combining both modalities.

        Args:
            aerial_input: Aerial imagery tensor of shape (B, C, H, W).
            sentinel_input: Sentinel-2 time series of shape (B, T, C, H, W).
            batch_positions: Temporal positions of shape (B, T) for Sentinel model.
            pad_mask: Boolean padding mask of shape (B, T) where True indicates a padded
                (invalid) timestep to be ignored by the Sentinel model.
            cloud_coverage: Cloud coverage tensor for gated fusion, shape (B, 1, H, W)
                at Sentinel resolution. Will be upsampled to aerial resolution.

        Returns:
            Fused predictions of shape (B, num_classes, H, W) at aerial resolution.

        """
        # Get predictions from aerial model
        # Handle models that return tuple (main_out, aux_out) during training
        aerial_out = self.aerial_model(aerial_input)
        aerial_logits = aerial_out[0] if isinstance(aerial_out, tuple) else aerial_out

        # Get predictions from Sentinel model
        sentinel_logits = self.sentinel_model(
            sentinel_input,
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )

        # Upsample Sentinel logits to aerial resolution
        if sentinel_logits.shape[-2:] != aerial_logits.shape[-2:]:
            sentinel_logits_up = functional.interpolate(
                sentinel_logits,
                size=aerial_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            sentinel_logits_up = sentinel_logits

        # Apply fusion strategy
        if self.fusion_mode == "weighted":
            return self._weighted_fusion(aerial_logits, sentinel_logits_up)
        if self.fusion_mode == "gated":
            return self._gated_fusion(aerial_logits, sentinel_logits_up, cloud_coverage)
        if self.fusion_mode == "concat":
            return self._concat_fusion(aerial_logits, sentinel_logits_up)
        if self.fusion_mode == "average":
            return (aerial_logits + sentinel_logits_up) / 2
        if self.fusion_mode == "attentional":
            return self._attentional_fusion(aerial_logits, sentinel_logits_up)

        msg = f"Unknown fusion_mode: {self.fusion_mode}"
        raise ValueError(msg)

    def _weighted_fusion(
        self,
        aerial_logits: torch.Tensor,
        sentinel_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-class weighted fusion.

        Args:
            aerial_logits: Aerial predictions (B, K, H, W).
            sentinel_logits: Sentinel predictions (B, K, H, W).

        Returns:
            Weighted combination (B, K, H, W).

        """
        # Normalize weights per class using softmax
        weights = functional.softmax(self.class_weights, dim=1)  # (K, 2)

        # Reshape for broadcasting: (1, K, 1, 1)
        w_aerial = weights[:, 0].view(1, -1, 1, 1)
        w_sentinel = weights[:, 1].view(1, -1, 1, 1)

        return w_aerial * aerial_logits + w_sentinel * sentinel_logits

    def _gated_fusion(
        self,
        aerial_logits: torch.Tensor,
        sentinel_logits: torch.Tensor,
        cloud_coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply gated fusion with content-aware spatial weights.

        The gate network learns to weight modalities based on the logits
        and optionally the cloud coverage of the Sentinel data.

        Args:
            aerial_logits: Aerial predictions (B, K, H, W).
            sentinel_logits: Sentinel predictions (B, K, H, W).
            cloud_coverage: Optional cloud coverage (B, 1, H, W) at Sentinel resolution.
                Values in [0, 1] where higher = more cloudy.

        Returns:
            Gated combination (B, K, H, W) where gate controls aerial contribution.

        """
        # Build gate input
        gate_inputs = [aerial_logits, sentinel_logits]

        if self.use_cloud_uncertainty:
            if cloud_coverage is None:
                # Default to 0.5 (neutral) if not provided
                cloud_up = torch.full(
                    (aerial_logits.shape[0], 1, *aerial_logits.shape[-2:]),
                    0.5,
                    device=aerial_logits.device,
                    dtype=aerial_logits.dtype,
                )
            else:
                # Upsample cloud coverage to aerial resolution
                cloud_up = functional.interpolate(
                    cloud_coverage,
                    size=aerial_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate_inputs.append(cloud_up)

        gate_input = torch.cat(gate_inputs, dim=1)

        # Compute gate: high values = trust aerial more
        gate = self.gate_network(gate_input)  # (B, K, H, W), values in [0, 1]

        # Fuse: output = gate * aerial + (1 - gate) * sentinel
        return gate * aerial_logits + (1 - gate) * sentinel_logits

    def _concat_fusion(
        self,
        aerial_logits: torch.Tensor,
        sentinel_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply concatenation-based fusion.

        Args:
            aerial_logits: Aerial predictions (B, K, H, W).
            sentinel_logits: Sentinel predictions (B, K, H, W).

        Returns:
            Fused output (B, K, H, W).

        """
        concat = torch.cat([aerial_logits, sentinel_logits], dim=1)
        return self.fusion_head(concat)

    def _attentional_fusion(
        self,
        aerial_logits: torch.Tensor,
        sentinel_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SOTA attentional fusion using MS-CAM.

        Args:
            aerial_logits: Aerial predictions (B, K, H, W).
            sentinel_logits: Sentinel predictions (B, K, H, W).

        Returns:
            Fused output (B, K, H, W).
        """
        # Sum of features for attention computation
        sum_feat = aerial_logits + sentinel_logits
        att = self.ms_cam(sum_feat)

        # Weighted combination based on learned multi-scale attention
        return att * aerial_logits + (1 - att) * sentinel_logits

    def get_fusion_weights(self) -> dict[str, torch.Tensor]:
        """Get the current per-class fusion weights.

        Returns:
            Dictionary with 'raw' unnormalized weights and 'normalized' softmax weights.

        """
        if self.fusion_mode != "weighted":
            return {}

        with torch.no_grad():
            normalized = functional.softmax(self.class_weights, dim=1)

        return {
            "raw": self.class_weights.detach().clone(),
            "normalized": normalized,
            "aerial_weights": normalized[:, 0],
            "sentinel_weights": normalized[:, 1],
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable fusion parameters (not frozen encoders).

        Returns:
            List of trainable parameters.

        """
        if self.fusion_mode == "weighted":
            return [self.class_weights]
        if self.fusion_mode == "gated":
            return list(self.gate_network.parameters())
        if self.fusion_mode == "concat":
            return list(self.fusion_head.parameters())
        if self.fusion_mode == "attentional":
            return list(self.ms_cam.parameters())
        return []


def load_pretrained_multimodal(
    aerial_checkpoint: str | None,
    sentinel_checkpoint: str | None,
    aerial_model: nn.Module,
    sentinel_model: nn.Module,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
    strip_prefixes: list[str] | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Load pre-trained weights into aerial and Sentinel models.

    Args:
        aerial_checkpoint: Path to aerial model checkpoint, or None to skip.
        sentinel_checkpoint: Path to Sentinel model checkpoint, or None to skip.
        aerial_model: Aerial model instance to load weights into.
        sentinel_model: Sentinel model instance to load weights into.
        device: Device to load checkpoints to.
        strict: Whether to enforce that checkpoint keys match the model exactly.
        strip_prefixes: Optional list of prefixes to strip from checkpoint keys
            (e.g., ['module.'] for DataParallel checkpoints). Prefix stripping is only
            applied when *all* keys share the prefix.

    Returns:
        Tuple of (aerial_model, sentinel_model) with loaded weights.

    """
    if strip_prefixes is None:
        strip_prefixes = ["module."]

    def _is_state_dict(candidate: object) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not candidate:
            return False
        return all(isinstance(k, str) for k in candidate)

    def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
        """Extract a model state_dict from common checkpoint formats."""
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()

        if _is_state_dict(checkpoint):
            # Heuristic: raw mapping of parameter-name -> tensor
            if any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint  # type: ignore[return-value]

            # Nested formats: {'model_state_dict': {...}}, {'state_dict': {...}}, {'model': {...}}, ...
            for key in _DEFAULT_CHECKPOINT_STATE_KEYS:
                nested = checkpoint.get(key)  # type: ignore[union-attr]
                if isinstance(nested, nn.Module):
                    return nested.state_dict()
                if _is_state_dict(nested) and any(torch.is_tensor(v) for v in nested.values()):
                    return nested  # type: ignore[return-value]

        msg = (
            "Unsupported checkpoint format. Expected a state_dict mapping or a dict containing one of "
            f"{list(_DEFAULT_CHECKPOINT_STATE_KEYS)}."
        )
        raise ValueError(msg)

    def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if not state_dict:
            return state_dict
        if not all(k.startswith(prefix) for k in state_dict):
            return state_dict
        return {k[len(prefix) :]: v for k, v in state_dict.items()}

    def _prepare_state_dict(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state_dict = raw_state_dict
        for prefix in strip_prefixes or []:
            state_dict = _strip_prefix(state_dict, prefix)
        return state_dict

    def _load(model: nn.Module, checkpoint_path: str, label: str) -> None:
        logger.info("Loading %s model from %s", label, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_state_dict = _extract_state_dict(checkpoint)
        state_dict = _prepare_state_dict(raw_state_dict)
        incompatible = model.load_state_dict(state_dict, strict=strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "%s checkpoint load had missing=%d unexpected=%d (strict=%s)",
                label,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
                strict,
            )
            if incompatible.missing_keys:
                logger.debug("%s missing keys: %s", label, incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.debug("%s unexpected keys: %s", label, incompatible.unexpected_keys)

    if aerial_checkpoint:
        _load(aerial_model, aerial_checkpoint, "aerial")

    if sentinel_checkpoint:
        _load(sentinel_model, sentinel_checkpoint, "sentinel")

    return aerial_model, sentinel_model
