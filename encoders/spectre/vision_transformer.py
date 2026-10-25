"""
Simplified 3D Vision Transformer for SPECTRE.

Consolidates PatchEmbed, RotaryPositionEmbedding, Attention (MHA only),
Block, and VisionTransformer into one file with inlined timm components
(DropPath, LayerScale, Mlp). No external timm dependency required.

All nn.Module attribute names are preserved for strict=True checkpoint loading.
"""
from __future__ import annotations

import math
from enum import Enum
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Literal, Optional, Tuple, Type, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ntuple(n: int):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_3tuple = _ntuple(3)


class Format(str, Enum):
    NCHWD = "NCHWD"
    NHWDC = "NHWDC"


def nchwd_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWDC:
        x = x.permute(0, 2, 3, 4, 1)
    return x


# ---------------------------------------------------------------------------
# Inlined timm components
# ---------------------------------------------------------------------------

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = None,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# 3D Rotary Position Embedding (axial, no mixing across axes)
# ---------------------------------------------------------------------------

def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    """3D axial RoPE.  Requires ``head_dim % 6 == 0``."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 1000.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        assert head_dim % 6 == 0

        self.base = base
        self.head_dim = head_dim
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        device, dtype = self.periods.device, self.dtype
        self.periods.data = self.base ** (
            2 * torch.arange(self.head_dim // 6, device=device, dtype=dtype)
            / (self.head_dim // 3)
        )

    def forward(self, *, H: int, W: int, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = self.periods.device, self.dtype
        dd = dict(device=device, dtype=dtype)

        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords_d = torch.arange(0.5, D, **dd) / D

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"), dim=-1
        )  # (H, W, D, 3)
        coords = coords.flatten(0, 2)          # (HWD, 3)
        coords = 2.0 * coords - 1.0            # [-1, +1]

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)   # (HWD, head_dim)

        return torch.sin(angles), torch.cos(angles)


# ---------------------------------------------------------------------------
# 3D Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (128, 128, 64),
        patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.img_size = to_3tuple(img_size)
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = int(math.prod(self.grid_size))

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHWD

        self.strict_img_size = strict_img_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size, bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        if self.strict_img_size:
            assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], (
                f"Input size ({H},{W},{D}) doesn't match expected ({self.img_size})"
            )
        x = self.proj(x)  # (B, embed_dim, gH, gW, gD)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        else:
            x = nchwd_to(x, self.output_fmt)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Multi-Head Attention (MHA only)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    fused_attn: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, "scaled_dot_product_attention")

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def compute_qkv(self, x: torch.Tensor):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ):
        sin, cos = rope
        # sin/cos: (HWD, head_dim) — broadcast over batch and heads
        if sin.ndim == 2:
            sin = sin.unsqueeze(0).unsqueeze(0)
            cos = cos.unsqueeze(0).unsqueeze(0)
        elif sin.ndim == 3:
            sin = sin.unsqueeze(1)
            cos = cos.unsqueeze(1)

        # prefix tokens (CLS etc.) don't get RoPE
        num_prefix = q.shape[2] - sin.shape[2]
        if num_prefix > 0:
            q_prefix, q_spatial = q[:, :, :num_prefix], q[:, :, num_prefix:]
            k_prefix, k_spatial = k[:, :, :num_prefix], k[:, :, num_prefix:]
            q_spatial = rope_apply(q_spatial, sin, cos)
            k_spatial = rope_apply(k_spatial, sin, cos)
            q = torch.cat([q_prefix, q_spatial], dim=2)
            k = torch.cat([k_prefix, k_spatial], dim=2)
        else:
            q = rope_apply(q, sin, cos)
            k = rope_apply(k, sin, cos)
        return q, k

    def compute_attention(self, q, k, v):
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim)
        return x

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        q, k, v = self.compute_qkv(x)
        if rope is not None:
            q, k = self.apply_rotary_pos_emb(q, k, rope)
        x = self.compute_attention(q, k, v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x
    if pool_type == "token":
        x = x[:, 0]
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
    return x


# ---------------------------------------------------------------------------
# 3D Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Simplified 3D ViT backbone. Supports RoPE or no positional embedding."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (128, 128, 64),
        patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
        in_chans: int = 1,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "rope",
        rope_kwargs: Optional[dict] = None,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[Union[Callable, Type[nn.Module]]] = None,
        act_layer: Optional[Union[Callable, Type[nn.Module]]] = None,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "rope")

        rope_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        rope_kwargs.setdefault("dtype", torch.float32)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        use_fc_norm = global_pool in ("avg", "avgmax", "max")

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.has_class_token = class_token

        # Patch embedding
        embed_args = {}
        if pos_embed == "rope":
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWDC"))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
            **embed_args,
        )

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        # Positional encoding
        self.pos_embed = None
        self.rope = None
        if pos_embed == "rope":
            self.rope = RotaryPositionEmbedding(
                embed_dim=embed_dim, num_heads=num_heads, **rope_kwargs,
            )

        # Identity placeholders (no parameters, safe for strict=True)
        self.pos_drop = nn.Dropout(p=0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)
        ])

        # Head
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pos_embed(self, x: torch.Tensor):
        if self.rope is None:
            x = x.view(x.shape[0], -1, x.shape[-1])
            if self.cls_token is not None:
                x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            return x, None

        # x arrives in NHWDC format from PatchEmbed
        B = x.shape[0]
        H, W, D = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, -1, x.shape[-1])  # (B, HWD, C)
        rope = self.rope(H=H, W=W, D=D)

        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        return self.pos_drop(x), rope

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, rope = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, rope=rope)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x




class SigLIPProjectionHead(nn.Module):
    """Projection head for SigLIP.

    Whereas SigLIP originally used a single linear layer for the projection
    head, we use a 3-layer MLP to deal with the partially frozen image and
    text backbones. This is similar to the DINO projection head without l2
    normalization (l2 normalization is performed in loss) and with LayerNorm.

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        layer_norm:
            Whether to use layer norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.

    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 512,
        layer_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):  
        super().__init__()

        blocks = [
            (
                input_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (
                hidden_dim,
                hidden_dim,
                nn.LayerNorm(hidden_dim) if layer_norm else None,
                nn.GELU(),
            ),
            (hidden_dim, bottleneck_dim, None, None),
        ]

        layers: List[nn.Module] = []
        for block in blocks:
            in_dim, out_dim, ln, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(ln)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if ln:
                layers.append(ln)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init._no_grad_trunc_normal_(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        x = self.last_layer(x)
        return x
