"""RS3Mamba: Visual State Space Model for Remote Sensing Semantic Segmentation.

This file is adapted from the SSRS repository:
https://github.com/sstary/SSRS/blob/main/RS3Mamba/model/RS3Mamba.py

Original paper: RS3Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation
https://arxiv.org/abs/2404.02457
"""

from __future__ import annotations

import logging
import re

import timm
import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional

from src.models.vssm_encoder import VSSMEncoder

logger = logging.getLogger(__name__)


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    """Conv2d + BatchNorm2d block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    """Simple Conv2d block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
        )


class SeparableConvBN(nn.Sequential):
    """Depthwise separable convolution with BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    """MLP block with 1x1 convolutions."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    """Global-Local Attention module combining window attention with local convolutions."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(
            kernel_size=(window_size, 1),
            stride=1,
            padding=(window_size // 2 - 1, 0),
        )
        self.attn_y = nn.AvgPool2d(
            kernel_size=(1, window_size),
            stride=1,
            padding=(0, window_size // 2 - 1),
        )

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            )

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def pad(self, x: torch.Tensor, ps: int) -> torch.Tensor:
        _, _, H, W = x.size()
        if W % ps != 0:
            x = functional.pad(x, (0, ps - W % ps), mode="reflect")
        if H % ps != 0:
            x = functional.pad(x, (0, 0, 0, ps - H % ps), mode="reflect")
        return x

    def pad_out(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.pad(x, pad=(0, 1, 0, 1), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        from einops import rearrange

        q, k, v = rearrange(
            qkv,
            "b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(
            attn,
            "(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )

        attn = attn[:, :, :H, :W]

        out = self.attn_x(functional.pad(attn, pad=(0, 0, 0, 1), mode="reflect"))
        out = self.attn_y(functional.pad(out, pad=(0, 1, 0, 0), mode="reflect"))

        out = out + local
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    """Transformer-style block with Global-Local Attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.ReLU6,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SoftPool2d(nn.Module):
    """Soft pooling operation."""

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x_weighted = self.avgpool(x_exp * x)
        return x_weighted / x_exp_pool


class ChannelAttention(nn.Module):
    """Channel attention module with avg, max, and soft pooling."""

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 2,
        pool_types: list[str] | None = None,
    ) -> None:
        super().__init__()
        if pool_types is None:
            pool_types = ["avg", "max", "soft"]
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = functional.adaptive_avg_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = functional.adaptive_max_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "soft":
                soft_pool = SoftPool2d(x.size(2), stride=x.size(2))
                soft_pool_out = soft_pool(x)
                channel_att_raw = self.mlp(soft_pool_out)
            else:
                continue

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class FusionAttention(nn.Module):
    """Attention module for fusing CNN and Mamba features."""

    def __init__(
        self,
        dim: int = 256,
        ssm_dims: int = 256,
        num_heads: int = 16,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(ssm_dims, dim, kernel_size=3)
        self.local2 = ConvBN(ssm_dims, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(
            kernel_size=(window_size, 1),
            stride=1,
            padding=(window_size // 2 - 1, 0),
        )
        self.attn_y = nn.AvgPool2d(
            kernel_size=(1, window_size),
            stride=1,
            padding=(0, window_size // 2 - 1),
        )

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            )

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def pad(self, x: torch.Tensor, ps: int) -> torch.Tensor:
        _, _, H, W = x.size()
        if W % ps != 0:
            x = functional.pad(x, (0, ps - W % ps), mode="reflect")
        if H % ps != 0:
            x = functional.pad(x, (0, 0, 0, ps - H % ps), mode="reflect")
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuse CNN features (x) with Mamba features (y)."""
        B, C, H, W = x.shape

        # Local features from Mamba branch
        local = self.local2(y) + self.local1(y)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        from einops import rearrange

        q, k, v = rearrange(
            qkv,
            "b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(
            attn,
            "(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )

        attn = attn[:, :, :H, :W]

        out = self.attn_x(functional.pad(attn, pad=(0, 0, 0, 1), mode="reflect"))
        out = self.attn_y(functional.pad(out, pad=(0, 1, 0, 0), mode="reflect"))

        out = out + local
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class FusionBlock(nn.Module):
    """Block for fusing CNN and Mamba features with attention and MLP."""

    def __init__(
        self,
        dim: int = 256,
        ssm_dims: int = 256,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.ReLU6,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssm_dims)
        self.attn = FusionAttention(
            dim,
            ssm_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.normx(x), self.normy(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    """Weighted Feature fusion module."""

    def __init__(
        self,
        in_channels: int = 128,
        decode_channels: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Upsample and fuse decoder features with residual features.

        Args:
            x: Decoder feature map to be upsampled.
            res: Residual feature map from encoder to fuse.

        Returns:
            Fused feature map after convolution and activation.

        """
        up = functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        out = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * up
        return self.post_conv(out)


class FeatureRefinementHead(nn.Module):
    """Feature Refinement Head for final feature processing."""

    def __init__(
        self,
        in_channels: int = 64,
        decode_channels: int = 64,
    ) -> None:
        """Initialize the Feature Refinement Head.

        Args:
            in_channels: Number of channels in the input feature map.
            decode_channels: Number of channels to decode to.

        """
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(
                decode_channels,
                decode_channels,
                kernel_size=3,
                padding=1,
                groups=decode_channels,
            ),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Refine fused features using positional and channel attention.

        Args:
            x: Decoder feature map to be upsampled.
            res: Residual feature map from encoder to fuse.

        Returns:
            Refined feature map.

        """
        up = functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        out = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * up
        out = self.post_conv(out)

        shortcut = self.shortcut(out)
        pa = self.pa(out) * out
        ca = self.ca(out) * out
        merged = pa + ca
        return self.act(self.proj(merged) + shortcut)


class Decoder(nn.Module):
    """UNetFormer-style decoder with Global-Local Attention."""

    def __init__(
        self,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512),
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 6,
    ) -> None:
        """Initialize the decoder used to upsample and produce segmentation maps.

        Args:
            encoder_channels: Tuple with channels from encoder stages.
            decode_channels: Number of decoder channels.
            dropout: Dropout probability in segmentation head.
            window_size: Window size used by attention blocks.
            num_classes: Number of segmentation classes.

        """
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
    ) -> torch.Tensor:
        """Run the decoder to produce segmentation logits at size (h, w).

        Args:
            res1..res4: Encoder feature maps from shallow to deep.
            h: Target output height.
            w: Target output width.

        Returns:
            Segmentation logits resized to (h, w).

        """
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        seg = self.segmentation_head(x)
        return functional.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)

    def init_weight(self) -> None:
        """Initialize Conv2d weights using Kaiming normalization.

        This initializes weights for convolutional layers in the decoder.
        """
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class RS3Mamba(nn.Module):
    """RS³Mamba: Visual State Space Model for Remote Sensing Semantic Segmentation.

    This model combines a CNN backbone (ResNet) with a VMamba encoder for
    capturing both local and global features in remote sensing images.

    Args:
        decode_channels: Number of decoder channels. Default: 64
        dropout: Dropout rate in decoder. Default: 0.1
        backbone_name: Name of timm backbone. Default: 'swsl_resnet18'
        pretrained: Whether to use pretrained backbone. Default: True
        window_size: Window size for attention. Default: 8
        num_classes: Number of output classes. Default: 6
        in_channels: Number of input channels. Default: 3

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
    ) -> None:
        """Initialize the RS3Mamba model.

        Args:
            decode_channels: Number of decoder channels.
            dropout: Dropout rate used in decoder.
            backbone_name: Name of the backbone model from timm.
            pretrained: Whether to load pretrained backbone weights.
            window_size: Window size used for attention modules.
            num_classes: Number of output classes.
            in_channels: Number of input image channels.

        """
        super().__init__()

        # CNN backbone (ResNet)
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
            in_chans=in_channels,
        )
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.act1 = self.backbone.act1
        self.maxpool = self.backbone.maxpool
        self.layers = nn.ModuleList()
        self.layers.append(self.backbone.layer1)
        self.layers.append(self.backbone.layer2)
        self.layers.append(self.backbone.layer3)
        self.layers.append(self.backbone.layer4)

        # VMamba encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=48)
        encoder_channels = self.backbone.feature_info.channels()
        ssm_dims = [96, 192, 384, 768]

        # Fusion blocks
        self.Fuse = nn.ModuleList()
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
        for i in range(4):
            fuse = FusionBlock(encoder_channels[i], ssm_dims[i])
            self.Fuse.append(fuse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining VMamba encoder and CNN backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits resized to the input spatial size.

        """
        h, w = x.size()[-2:]

        # VMamba branch
        ssmx = self.stem(x)
        vss_outs = self.vssm_encoder(ssmx)  # 48*128*128, 96*64*64, 192*32*32, 384*16*16, 768*8*8

        # CNN branch with fusion
        ress = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.Fuse[i](x, vss_outs[i + 1])
            ress.append(x)

        # Decoder
        return self.decoder(ress[0], ress[1], ress[2], ress[3], h, w)


def load_pretrained_ckpt(
    model: RS3Mamba,
    ckpt_path: str = "./pretrain/vmamba_tiny_e292.pth",
) -> RS3Mamba:
    """Load pretrained VMamba weights into RS3Mamba model.

    Args:
        model: RS3Mamba model instance
        ckpt_path: Path to VMamba pretrained weights

    Returns:
        Model with loaded weights

    """
    logger.info("Loading weights from: %s", ckpt_path)
    skip_params = [
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "patch_embed.norm.weight",
        "patch_embed.norm.weight",
    ]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()

    for k, v in ckpt["model"].items():
        if k in skip_params:
            logger.info("Skipping weights: %s", k)
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict
        if kr in model_dict:
            if model_dict[kr].shape == v.shape:
                model_dict[kr] = v
            else:
                logger.warning(
                    "Shape mismatch for %s: %s vs %s",
                    kr,
                    tuple(model_dict[kr].shape),
                    tuple(v.shape),
                )

    model.load_state_dict(model_dict, strict=False)
    return model
