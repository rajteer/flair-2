"""Implementation of the UNetFormer architecture.

This file is adapted from the SSRS repository:
https://github.com/sstary/SSRS/blob/main/RS3Mamba/model/UNetFormer.py

Original paper: UNetFormer: A UNet-like transformer for efficient semantic
segmentation of remote sensing urban scene imagery
https://arxiv.org/abs/2109.08937
"""

from __future__ import annotations

import timm
import torch
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU6 block."""

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


class SeparableConvBNReLU(nn.Sequential):
    """Depthwise separable convolution with BatchNorm and ReLU6."""

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
            nn.ReLU6(),
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


class SeparableConv(nn.Sequential):
    """Depthwise separable convolution without normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
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
    """Global-Local Attention module.

    Combines window-based global attention with local convolutional features.
    Uses relative position embeddings for the attention mechanism.
    """

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
            # Define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            )

            # Get pair-wise relative position index for each token inside the window
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

        # Local branch
        local = self.local2(x) + self.local1(x)

        # Pad for window attention
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        # Rearrange to windows
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

        # Rearrange back
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
        out = out + self.attn_y(functional.pad(attn, pad=(0, 1, 0, 0), mode="reflect"))

        out = out + local
        out = self.pad_out(out)
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


class WF(nn.Module):
    """Weighted Feature fusion module for skip connections."""

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
        x = functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    """Feature Refinement Head with positional and channel attention."""

    def __init__(
        self,
        in_channels: int = 64,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        # Positional attention
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
        # Channel attention
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
        x = functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)

        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):
    """Auxiliary head for deep supervision."""

    def __init__(self, in_channels: int = 64, num_classes: int = 8) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
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
    ) -> torch.Tensor:
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        return x

    def init_weight(self) -> None:
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetFormer(nn.Module):
    """UNetFormer: A UNet-like Transformer for Semantic Segmentation.

    This model uses a CNN backbone (from timm) for feature extraction and
    a transformer-style decoder with Global-Local Attention.

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
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
            in_chans=in_channels,
        )
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x
