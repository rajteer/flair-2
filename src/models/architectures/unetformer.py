"""Implementation of the UNetFormer architecture.

This file is adapted from the SSRS repository:
https://github.com/sstary/SSRS/blob/main/RS3Mamba/model/UNetFormer.py

Original paper: UNetFormer: A UNet-like transformer for efficient semantic
segmentation of remote sensing urban scene imagery
https://arxiv.org/abs/2109.08937
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
from torch import nn
from torch.nn import functional

from src.models.common_blocks import (
    WF,
    Block,
    Conv,
    ConvBN,
    ConvBNReLU,
    FeatureRefinementHead,
)

try:
    from src.models.encoders.samba_encoder import SambaEncoder
except ImportError:
    SambaEncoder = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class AuxHead(nn.Module):
    """Auxiliary head for deep supervision."""

    def __init__(self, in_channels: int = 64, num_classes: int = 8) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Produce auxiliary segmentation logits upsampled to (h, w).

        Args:
            x: Feature tensor from the decoder.
            h: Target output height.
            w: Target output width.

        Returns:
            Logits tensor of shape (B, num_classes, h, w).

        """
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = functional.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return feat


class Decoder(nn.Module):
    """UNetFormer decoder with Global-Local Attention blocks."""

    def __init__(
        self,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512),
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 6,
    ) -> None:
        super().__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1),
        )
        self.init_weight()

    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        res3: torch.Tensor,
        res4: torch.Tensor,
        h: int,
        w: int,
        *,
        return_aux_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode multi-scale encoder features into segmentation logits.

        Args:
            res1: Stage-1 encoder features (highest resolution).
            res2: Stage-2 encoder features.
            res3: Stage-3 encoder features.
            res4: Stage-4 encoder features (lowest resolution).
            h: Target output height.
            w: Target output width.
            return_aux_features: If ``True``, also return pre-head features
                for the auxiliary loss.

        Returns:
            Segmentation logits of shape (B, num_classes, h, w), or a tuple
            of (logits, aux_features) when *return_aux_features* is ``True``.

        """
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        # Store features for auxiliary head before segmentation
        aux_features = x

        x = self.segmentation_head(x)
        x = functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if return_aux_features:
            return x, aux_features
        return x

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetFormer(nn.Module):
    """UNetFormer: A UNet-like Transformer for Semantic Segmentation.

    This model uses a CNN backbone (from timm) or Samba encoder for feature
    extraction and a transformer-style decoder with Global-Local Attention.

    Args:
        decode_channels: Number of decoder channels. Default: 64
        dropout: Dropout rate in decoder. Default: 0.1
        backbone_name: Name of timm backbone. Default: 'swsl_resnet18'
        pretrained: Whether to use pretrained backbone. Default: True
        window_size: Window size for attention. Default: 8
        num_classes: Number of output classes. Default: 6
        in_channels: Number of input channels. Default: 3
        use_aux_head: Whether to use auxiliary head for deep supervision. Default: False
        encoder_type: Type of encoder: 'timm' or 'samba'. Default: 'timm'
        samba_config: Configuration dict for Samba encoder when encoder_type='samba'

    """

    def __init__(
        self,
        decode_channels: int = 64,
        dropout: float = 0.1,
        backbone_name: str = "swsl_resnet18",
        pretrained: bool = True,
        window_size: int = 8,
        num_classes: int = 6,
        in_channels: int = 3,
        img_size: int | tuple[int, int] | None = None,
        *,
        use_aux_head: bool = False,
        encoder_type: str = "timm",
        samba_config: dict[str, Any] | None = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_aux_head = use_aux_head
        self.encoder_type = encoder_type
        self._warned_nhwc_features = False

        if encoder_type == "samba":
            if SambaEncoder is None:
                msg = "SambaEncoder could not be imported. Check dependencies (e.g. mamba_ssm)."
                raise ImportError(msg)

            samba_config = samba_config or {}
            self.backbone = SambaEncoder(
                in_channels=in_channels,
                stem_hidden_dim=samba_config.get("stem_hidden_dim", 32),
                embed_dims=samba_config.get("embed_dims", [64, 128, 320, 448]),
                mlp_ratios=samba_config.get("mlp_ratios", [8, 8, 4, 4]),
                drop_path_rate=samba_config.get("drop_path_rate", 0.0),
                depths=samba_config.get("depths", [3, 4, 6, 3]),
            )
            encoder_channels = tuple(self.backbone.get_channels())
        else:
            # Select out_indices based on encoder architecture
            # ConvNeXt family has 4 stages (0-3), ResNet family has 5 stages (1-4)
            backbone_lower = backbone_name.lower()
            if "convnext" in backbone_lower or "swin" in backbone_lower:
                out_indices = (0, 1, 2, 3)
            else:
                out_indices = (1, 2, 3, 4)

            timm_kwargs: dict[str, Any] = {
                "features_only": True,
                "out_indices": out_indices,
                "pretrained": pretrained,
                "in_chans": in_channels,
                "drop_path_rate": drop_path_rate,
            }
            if img_size is not None:
                timm_kwargs["img_size"] = img_size

            try:
                self.backbone = timm.create_model(backbone_name, **timm_kwargs)
            except TypeError:
                if "img_size" not in timm_kwargs:
                    raise
                logger.warning(
                    "Backbone '%s' does not accept img_size=%s. Falling back to default size.",
                    backbone_name,
                    img_size,
                )
                timm_kwargs.pop("img_size", None)
                self.backbone = timm.create_model(backbone_name, **timm_kwargs)
            encoder_channels = tuple(self.backbone.feature_info.channels())
        self.encoder_channels = encoder_channels

        self.decoder = Decoder(
            encoder_channels,
            decode_channels,
            dropout,
            window_size,
            num_classes,
        )

        # Auxiliary head for deep supervision (as per paper)
        if use_aux_head:
            self.aux_head = AuxHead(
                in_channels=decode_channels,
                num_classes=num_classes,
            )

    def _ensure_nchw(self, feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        """Convert NHWC features from timm backbones into NCHW expected by decoder."""
        if feat.ndim != 4:  # noqa: PLR2004
            return feat
        if int(feat.shape[1]) == expected_channels:
            return feat
        if int(feat.shape[-1]) == expected_channels:
            if not self._warned_nhwc_features:
                logger.warning(
                    "Backbone returned NHWC features; converting to NCHW for UNetFormer decoder.",
                )
                self._warned_nhwc_features = True
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the full encoder-decoder forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W). During
            training with ``use_aux_head=True``, returns a tuple of
            (main_logits, aux_logits).

        """
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        res1 = self._ensure_nchw(res1, int(self.encoder_channels[0]))
        res2 = self._ensure_nchw(res2, int(self.encoder_channels[1]))
        res3 = self._ensure_nchw(res3, int(self.encoder_channels[2]))
        res4 = self._ensure_nchw(res4, int(self.encoder_channels[3]))

        if self.use_aux_head and self.training:
            main_out, aux_features = self.decoder(
                res1,
                res2,
                res3,
                res4,
                h,
                w,
                return_aux_features=True,
            )
            aux_out = self.aux_head(aux_features, h, w)
            return main_out, aux_out

        return self.decoder(res1, res2, res3, res4, h, w)
