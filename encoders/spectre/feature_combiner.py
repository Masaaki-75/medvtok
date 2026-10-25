"""
Simplified Feature Vision Transformer (combiner) for SPECTRE.

Accepts pre-extracted patch features and combines them with grid-aware
positional encoding to produce scan-level representations.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Callable, Literal, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from .vision_transformer import (
    Block,
    DropPath,
    LayerScale,
    Mlp,
    RotaryPositionEmbedding,
    global_pool_nlc,
    to_3tuple,
)


class FeatureVisionTransformer(nn.Module):
    """Vision Transformer that accepts flattened patch features as input."""

    def __init__(
        self,
        patch_dim: int = 768,
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
        self.dynamic_grid_size = pos_embed == "rope"

        # Input projection (linear instead of Conv3d PatchEmbed)
        self.patch_proj = nn.Linear(patch_dim, embed_dim, proj_bias)

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        # Positional encoding
        self.pos_embed = None
        self.rope = None
        if pos_embed == "rope":
            self.rope = RotaryPositionEmbedding(
                embed_dim=embed_dim, num_heads=num_heads, **rope_kwargs,
            )

        # Identity placeholders
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

    def _pos_embed(
        self,
        x: torch.Tensor,
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        if self.rope is None:
            if self.cls_token is not None:
                x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            return x, None

        assert grid_size is not None, "grid_size required when using RoPE"
        H, W, D = to_3tuple(grid_size)
        rope = self.rope(H=H, W=W, D=D)

        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        return self.pos_drop(x), rope

    def forward_features(
        self,
        x: torch.Tensor,
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        assert x.ndim == 3, f"Expected (B, N, C), got ndim={x.ndim}"
        x = self.patch_proj(x)
        x, rope = self._pos_embed(x, grid_size)
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

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        x = self.forward_features(x, grid_size)
        x = self.forward_head(x)
        return x
