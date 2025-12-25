"""Visual State Space Model (VSSM) Encoder for RS3Mamba.

This module contains the core Mamba components for 2D vision tasks.
The implementation follows VMamba/SwinUMamba but uses mamba-ssm primitives.

Original source: https://github.com/sstary/SSRS/tree/main/RS3Mamba
Paper: RS3Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation
"""

import math
import re
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.utils import checkpoint


class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding.

    Args:
        patch_size: Patch token size. Default: 4.
        in_chans: Number of input image channels. Default: 3.
        embed_dim: Number of linear projection output channels. Default: 96.
        norm_layer: Normalization layer. Default: None

    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Patch Merging Layer for downsampling.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: nn.LayerNorm

    """

    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # Handle odd dimensions (local shape fix like original implementation)
        shape_fix = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            shape_fix[0] = H // 2
            shape_fix[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if shape_fix[0] > 0:
            x0 = x0[:, : shape_fix[0], : shape_fix[1], :]
            x1 = x1[:, : shape_fix[0], : shape_fix[1], :]
            x2 = x2[:, : shape_fix[0], : shape_fix[1], :]
            x3 = x3[:, : shape_fix[0], : shape_fix[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    """Selective Scan 2D - Core Mamba operation for 2D images.

    Implements bidirectional scanning in 4 directions for capturing
    long-range dependencies in 2D feature maps.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # x_proj for 4 directions
        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0),
        )  # (K=4, N, inner)
        del self.x_proj

        # dt_proj for 4 directions
        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0),
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0),
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize dt proj
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(
        d_state: int,
        d_inner: int,
        copies: int = 1,
        device: torch.device | None = None,
        merge: bool = True,
    ) -> nn.Parameter:
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(
        d_inner: int,
        copies: int = 1,
        device: torch.device | None = None,
        *,
        merge: bool = True,
    ) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, C, h, w = x.shape
        l = h * w
        k = 4

        x_hwwh = torch.stack(
            [x.view(b, -1, l), torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, -1, l)],
            dim=1,
        ).view(b, 2, -1, l)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)  # (b, k * d, l)
        dts = dts.contiguous().float().view(b, -1, l)  # (b, k * d, l)
        Bs = Bs.float().view(b, k, -1, l)  # (b, k, d_state, l)
        Cs = Cs.float().view(b, k, -1, l)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, l)
        wh_y = (
            torch.transpose(out_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3)
            .contiguous()
            .view(b, -1, l)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3)
            .contiguous()
            .view(b, -1, l)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Visual State Space Block."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """A layer containing multiple VSSBlocks.

    Args:
        dim: Number of input channels.
        depth: Number of blocks.
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Stochastic depth rate. Default: 0.0
        norm_layer: Normalization layer. Default: nn.LayerNorm
        downsample: Downsample layer at the end. Default: None
        use_checkpoint: Whether to use checkpointing. Default: False
        d_state: State dimension for Mamba. Default: 16

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ],
        )

        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSMEncoder(nn.Module):
    """Visual State Space Model Encoder.

    Hierarchical encoder based on VMamba architecture.

    Args:
        patch_size: Patch embedding size. Default: 4
        in_chans: Number of input channels. Default: 3
        depths: Depth of each stage. Default: [2, 2, 9, 2]
        dims: Dimensions at each stage. Default: [96, 192, 384, 768]
        d_state: State dimension for Mamba. Default: 16
        drop_rate: Dropout rate. Default: 0.0
        attn_drop_rate: Attention dropout rate. Default: 0.0
        drop_path_rate: Stochastic depth rate. Default: 0.2
        norm_layer: Normalization layer. Default: nn.LayerNorm
        patch_norm: Whether to apply norm after patch embedding. Default: True
        use_checkpoint: Whether to use checkpointing. Default: False

    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if depths is None:
            depths = [2, 2, 9, 2]
        if dims is None:
            dims = [96, 192, 384, 768]

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, *self.patches_resolution, self.embed_dim),
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                attn_drop=attn_drop_rate,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"relative_position_bias_table"}

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


def load_vssm_pretrained_ckpt(
    model: nn.Module,
    ckpt_path: str = "./pretrain/vmamba_tiny_e292.pth",
) -> nn.Module:
    """Load pretrained VMamba weights into VSSMEncoder.

    Args:
        model: Model containing vssm_encoder attribute
        ckpt_path: Path to pretrained weights

    Returns:
        Model with loaded weights

    """
    print(f"Loading VSSM weights from: {ckpt_path}")
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
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict:
            if model_dict[kr].shape == v.shape:
                model_dict[kr] = v
            else:
                print(f"Shape mismatch for {kr}: {model_dict[kr].shape} vs {v.shape}")

    model.load_state_dict(model_dict, strict=False)
    return model
