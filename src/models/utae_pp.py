"""U-TAE++ Implementation - Modernized U-TAE with ConvNeXt blocks and Flash Attention.

Based on U-TAE by Vivien Sainte Fare Garnot (github/VSainteuf)
Improvements:
- ConvNeXt-style blocks with 7x7 depthwise conv
- Flash Attention (PyTorch 2.0+)
- Stochastic Depth (DropPath)
- CBAM attention in decoder
- Deep supervision
- Layer Scale
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
import torch.nn.functional as F
from timm.layers import DropPath
from torch import nn


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True

        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)
        return sinusoid_table


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention with Flash Attention support."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_p = attn_dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        return_comp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
        q: Query tensor (N, d_k)
        k: Key tensor (N, T, d_k)
        v: Value tensor (N, T, d_v)
        pad_mask: Padding mask (N, T)
        return_comp: Whether to return attention compatibility scores

        """
        q = q.unsqueeze(1)  # (N, 1, d_k)

        if return_comp:
            attn = torch.matmul(q, k.transpose(1, 2)) / self.temperature
            if pad_mask is not None:
                attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e9)
            comp = attn
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            output = torch.matmul(attn, v)
            return output, attn, comp
        # Flash Attention path (PyTorch 2.0+)
        attn_mask = None
        if pad_mask is not None:
            attn_mask = pad_mask.unsqueeze(1).float() * -1e9

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=1.0 / self.temperature,
        )
        # Return dummy attention weights for compatibility
        attn = torch.zeros(q.size(0), 1, k.size(1), device=q.device, dtype=q.dtype)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with learnable query."""

    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # Learnable query (shared across positions)
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=math.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=math.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))

    def forward(
        self,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        return_comp: bool = False,
    ):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = self.Q.unsqueeze(1).expand(-1, sz_b, -1).reshape(-1, d_k)

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)

        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=True)
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask)

        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        return output, attn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder for image time series."""

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp: list[int] | None = None,
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ):
        super().__init__()
        if mlp is None:
            mlp = [256, 128]

        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.LayerNorm(self.mlp[i + 1]),  # LayerNorm instead of BatchNorm1d
                    nn.GELU(),
                ],
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        return_comp: bool = False,
    ):
        sz_b, seq_len, d, h, w = x.shape

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))

        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)

        if self.return_att:
            return out, attn
        return out


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise conv, inverted bottleneck, and layer scale."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim,
        )
        self.norm = nn.GroupNorm(1, dim)  # LayerNorm equivalent without permute
        self.pwconv1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer scale
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1)) if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return shortcut + self.drop_path(x)


class ConvLayer(nn.Module):
    """Basic convolution layer with norm and activation."""

    def __init__(
        self,
        nkernels: list[int],
        norm: Literal["batch", "group", "instance", "layer"] = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__()

        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda c: nn.GroupNorm(num_channels=c, num_groups=n_groups)
        elif norm == "layer":
            nl = lambda c: nn.GroupNorm(num_channels=c, num_groups=1)
        else:
            nl = None

        layers = []
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    nkernels[i],
                    nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                ),
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if last_relu or i < len(nkernels) - 2:
                layers.append(nn.GELU())

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporallySharedBlock(nn.Module):
    """Base class for blocks that are shared across temporal dimension."""

    def __init__(self, pad_value: float | None = None):
        super().__init__()
        self.pad_value = pad_value
        self.out_shape = None

    def smart_forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 4:
            return self.forward(input)

        b, t, c, h, w = input.shape

        if self.pad_value is not None:
            dummy = torch.zeros(input.shape, device=input.device).float()
            self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

        out = input.view(b * t, c, h, w)

        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                temp = (
                    torch.ones(self.out_shape, device=input.device, requires_grad=False)
                    * self.pad_value
                )
                temp[~pad_mask] = self.forward(out[~pad_mask])
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)

        _, c, h, w = out.shape
        return out.view(b, t, c, h, w)


class ConvBlock(TemporallySharedBlock):
    """Convolutional block with temporal sharing."""

    def __init__(
        self,
        nkernels: list[int],
        pad_value: float | None = None,
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    """Downsampling block with ConvNeXt-style processing."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        pad_value: float | None = None,
        norm: str = "batch",
        padding_mode: str = "reflect",
        drop_path: float = 0.0,
        use_convnext: bool = True,
    ):
        super().__init__(pad_value=pad_value)

        # Strided conv for downsampling
        self.down = nn.Sequential(
            nn.Conv2d(d_in, d_in, kernel_size=k, stride=s, padding=p, padding_mode=padding_mode),
            nn.GroupNorm(1, d_in) if use_convnext else nn.BatchNorm2d(d_in),
        )

        # Channel projection
        self.proj = nn.Conv2d(d_in, d_out, kernel_size=1)

        if use_convnext:
            self.conv1 = ConvNeXtBlock(d_out, drop_path=drop_path)
            self.conv2 = ConvNeXtBlock(d_out, drop_path=drop_path)
        else:
            self.conv1 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)
            self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down(x)
        out = self.proj(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention from CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CoordinateAttention(nn.Module):
    """Coordinate Attention module (Hou et al., CVPR 2021).

    Unlike CBAM which loses spatial information via global pooling,
    Coordinate Attention encodes channel relationships while preserving
    precise positional information via 1D horizontal and vertical pooling.

    Reference: https://arxiv.org/abs/2103.02907
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Encode spatial info along each axis
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W) -> permute to (B, C, W, 1)

        # Concatenate and reduce
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # Apply attention
        return x * a_h * a_w


def build_attention(attention_type: str, channels: int, reduction: int = 16) -> nn.Module:
    """Factory function to build attention modules.

    Args:
        attention_type: Type of attention ('cbam', 'coord', or 'none')
        channels: Number of input channels
        reduction: Channel reduction ratio

    Returns:
        Attention module or identity-like fallback
    """
    if attention_type == "cbam":
        return CBAM(channels, reduction)
    elif attention_type == "coord":
        return CoordinateAttention(channels, reduction)
    else:  # 'none' or any other value
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )


class UpConvBlock(nn.Module):
    """Upsampling block with configurable attention (CBAM, Coordinate, or none)."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        norm: str = "batch",
        d_skip: int | None = None,
        padding_mode: str = "reflect",
        attention_type: str = "coord",
        use_convnext: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip

        self.skip_attn = build_attention(attention_type, d)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(1, d_out) if use_convnext else nn.BatchNorm2d(d_out),
            nn.GELU(),
        )

        if use_convnext:
            self.conv1 = ConvNeXtBlock(d_out + d, drop_path=drop_path)
            self.proj = nn.Conv2d(d_out + d, d_out, kernel_size=1)
            self.conv2 = ConvNeXtBlock(d_out, drop_path=drop_path)
        else:
            self.conv1 = ConvLayer(
                nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode,
            )
            self.proj = nn.Identity()
            self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        skip = self.skip_attn(skip)
        out = torch.cat([out, skip], dim=1)
        out = self.conv1(out)
        out = self.proj(out)
        out = out + self.conv2(out)
        return out


class TemporalAggregator(nn.Module):
    """Aggregates temporal features using attention masks."""

    def __init__(self, mode: Literal["att_group", "att_mean", "mean"] = "mean"):
        super().__init__()
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = F.interpolate(
                        attn, size=x.shape[-2:], mode="bilinear", align_corners=False,
                    )
                else:
                    attn = F.avg_pool2d(attn, kernel_size=w // x.shape[-2])

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)
                out = torch.cat([group for group in out], dim=1)
                return out

            if self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)
                attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out

            if self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        elif self.mode == "att_group":
            n_heads, b, t, h, w = attn_mask.shape
            attn = attn_mask.view(n_heads * b, t, h, w)

            if x.shape[-2] > w:
                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False,
                )
            else:
                attn = F.avg_pool2d(attn, kernel_size=w // x.shape[-2])

            attn = attn.view(n_heads, b, t, *x.shape[-2:])
            out = torch.stack(x.chunk(n_heads, dim=2))
            out = attn[:, :, :, None, :, :] * out
            out = out.sum(dim=2)
            out = torch.cat([group for group in out], dim=1)
            return out

        elif self.mode == "att_mean":
            attn = attn_mask.mean(dim=0)
            attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
            out = (x * attn[:, :, None, :, :]).sum(dim=1)
            return out

        elif self.mode == "mean":
            return x.mean(dim=1)


class UTAE(nn.Module):
    """U-TAE++ - Modernized U-TAE with ConvNeXt blocks and Flash Attention.

    Args:
        input_dim: Number of input channels
        encoder_widths: Channel widths for each encoder stage
        decoder_widths: Channel widths for each decoder stage
        out_conv: Output convolution channels [hidden, n_classes]
        str_conv_k: Kernel size for strided convolutions
        str_conv_s: Stride for strided convolutions
        str_conv_p: Padding for strided convolutions
        agg_mode: Temporal aggregation mode ('att_group', 'att_mean', 'mean')
        encoder_norm: Normalization type ('group', 'batch', 'instance')
        n_head: Number of attention heads in L-TAE
        d_model: Model dimension for L-TAE
        d_k: Key/query dimension for attention
        encoder: If True, return feature maps instead of predictions
        return_maps: If True, also return intermediate feature maps
        pad_value: Padding value for temporal sequences
        padding_mode: Padding mode for convolutions
        use_convnext: Use ConvNeXt-style blocks
        attention_type: Decoder attention type ('coord', 'cbam', or 'none')
        drop_path_rate: Stochastic depth rate
        deep_supervision: Enable auxiliary outputs for deep supervision

    """

    def __init__(
        self,
        input_dim: int,
        encoder_widths: list[int] | None = None,
        decoder_widths: list[int] | None = None,
        out_conv: list[int] | None = None,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder: bool = False,
        return_maps: bool = False,
        pad_value: float = 0,
        padding_mode: str = "reflect",
        use_convnext: bool = True,
        attention_type: str = "coord",
        drop_path_rate: float = 0.1,
        deep_supervision: bool = False,
    ):
        super().__init__()

        if encoder_widths is None:
            encoder_widths = [64, 64, 64, 128]
        if decoder_widths is None:
            decoder_widths = [32, 32, 64, 128]
        if out_conv is None:
            out_conv = [32, 20]

        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        self.use_convnext = use_convnext
        self.deep_supervision = deep_supervision

        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        # Stochastic depth
        total_blocks = 2 * (self.n_stages - 1)  # 2 ConvNeXt blocks per stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Input conv
        self.in_conv = ConvBlock(
            nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )

        # Encoder
        self.down_blocks = nn.ModuleList(
            [
                DownConvBlock(
                    d_in=encoder_widths[i],
                    d_out=encoder_widths[i + 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    pad_value=pad_value,
                    norm=encoder_norm,
                    padding_mode=padding_mode,
                    drop_path=dpr[i * 2],
                    use_convnext=use_convnext,
                )
                for i in range(self.n_stages - 1)
            ],
        )

        # Decoder
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm="batch",
                    padding_mode=padding_mode,
                    attention_type=attention_type,
                    use_convnext=use_convnext,
                    drop_path=dpr[-(i * 2 + 1)] if i < len(dpr) // 2 else 0.0,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ],
        )

        # Temporal encoder
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)

        # Output
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv,
            padding_mode=padding_mode,
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv2d(decoder_widths[i - 1], out_conv[-1], kernel_size=1)
                    for i in range(self.n_stages - 1, 1, -1)
                ],
            )
        else:
            self.aux_heads = None

    def forward(
        self,
        input: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        return_att: bool = False,
    ) -> torch.Tensor | tuple:
        """Args:
            input: Input tensor (B, T, C, H, W)
            batch_positions: Temporal positions (B, T)
            pad_mask: Boolean padding mask (B, T) where True indicates padded timesteps.
                If not provided, computed from input using self.pad_value.
            return_att: Return attention maps

        Returns:
            Segmentation output and optionally attention/auxiliary outputs

        """
        # Use external pad_mask if provided, otherwise compute from input
        if pad_mask is None:
            pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

        # Input convolution
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]

        # Spatial encoder
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask,
        )

        # Spatial decoder
        if self.return_maps:
            maps = [out]

        aux_outputs = []
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att,
            )
            out = self.up_blocks[i](out, skip)

            if self.return_maps:
                maps.append(out)

            # Deep supervision
            if self.training and self.aux_heads is not None and i < len(self.aux_heads):
                aux_out = self.aux_heads[i](out)
                aux_out = F.interpolate(
                    aux_out, size=input.shape[-2:], mode="bilinear", align_corners=False,
                )
                aux_outputs.append(aux_out)

        # Final output
        if self.encoder:
            return out, maps

        out = self.out_conv(out)

        # Return format
        if self.training and self.deep_supervision and aux_outputs:
            if return_att:
                return out, att, aux_outputs
            return out, aux_outputs

        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out


# Alias for backward compatibility
UTAE_PP = UTAE
