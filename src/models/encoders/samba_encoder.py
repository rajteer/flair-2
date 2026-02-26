"""Samba Encoder for semantic segmentation.

Adapted from the Samba repository:
https://github.com/zhuqinfeng1999/Samba

Original paper: Samba: Semantic Segmentation of Remotely Sensed Images
with State Space Model
https://doi.org/10.1016/j.heliyon.2024.e38495
"""

from __future__ import annotations

import math

import torch
from mamba_ssm import Mamba
from timm.layers import DropPath, trunc_normal_
from torch import nn


class DWConv(nn.Module):
    """Depth-wise convolution for spatial mixing in FFN."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(nn.Module):
    """Feed-forward network with depth-wise convolution (PVTv2 style)."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaLayer(nn.Module):
    """Single Mamba layer with LayerNorm."""

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba


class SambaBlock(nn.Module):
    """Samba block: Mamba + FFN with residual connections."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class DownSamples(nn.Module):
    """Downsampling layer between stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Stem(nn.Module):
    """Stem module for initial feature extraction."""

    def __init__(
        self,
        in_channels: int,
        stem_hidden_dim: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.conv(x)
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class SambaEncoder(nn.Module):
    """Samba Encoder: Hierarchical Mamba-based encoder for segmentation.

    Produces 4-stage hierarchical features compatible with UNet-style decoders.

    Args:
        in_channels: Number of input image channels. Default: 3
        stem_hidden_dim: Hidden dimension in stem convolutions. Default: 32
        embed_dims: Feature dimensions at each stage. Default: [64, 128, 320, 448]
        mlp_ratios: MLP expansion ratios per stage. Default: [8, 8, 4, 4]
        drop_path_rate: Stochastic depth rate. Default: 0.0
        depths: Number of blocks per stage. Default: [3, 4, 6, 3]
        num_stages: Number of encoder stages. Default: 4

    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_hidden_dim: int = 32,
        embed_dims: list[int] | None = None,
        mlp_ratios: list[float] | None = None,
        drop_path_rate: float = 0.0,
        depths: list[int] | None = None,
        num_stages: int = 4,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [64, 128, 320, 448]
        if mlp_ratios is None:
            mlp_ratios = [8, 8, 4, 4]
        if depths is None:
            depths = [3, 4, 6, 3]

        self.num_stages = num_stages
        self.depths = depths
        self.embed_dims = embed_dims

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            # Patch embedding / downsampling
            if i == 0:
                patch_embed = Stem(in_channels, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            # Samba blocks for this stage
            block = nn.ModuleList(
                [
                    SambaBlock(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )

            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning hierarchical features.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of 4 feature tensors at different scales:
            - Stage 1: (B, embed_dims[0], H/4, W/4)
            - Stage 2: (B, embed_dims[1], H/8, W/8)
            - Stage 3: (B, embed_dims[2], H/16, W/16)
            - Stage 4: (B, embed_dims[3], H/32, W/32)

        """
        b = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, h, w = patch_embed(x)

            for blk in block:
                x = blk(x, h, w)

            x = norm(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def get_channels(self) -> list[int]:
        """Return output channels for each stage."""
        return list(self.embed_dims)
