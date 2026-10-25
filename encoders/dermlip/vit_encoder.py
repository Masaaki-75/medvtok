"""
ViT-B Vision Transformer Encoder
Standard Vision Transformer architecture for DermLIP.
"""
from collections import OrderedDict
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    """Convert scalar to 2-tuple"""
    if isinstance(x, tuple):
        return x
    return (x, x)


def _expand_token(token, batch_size: int):
    """Expand token to batch size"""
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    """Transformer block with multi-head attention and MLP"""
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Stack of transformer blocks"""
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class ViTVisionTransformer(nn.Module):
    """
    Vision Transformer (ViT-B/16) for feature extraction.
    Standard implementation used in CLIP.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        pool_type: str = 'tok',
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
    ):
        """
        Args:
            image_size: Input image size (default: 224)
            patch_size: Patch size for patch embedding (default: 16)
            width: Transformer width/embedding dimension (default: 768)
            layers: Number of transformer layers (default: 12)
            heads: Number of attention heads (default: 12)
            mlp_ratio: MLP hidden dim ratio (default: 4.0)
            output_dim: Output feature dimension (default: 512)
            pool_type: Pooling type, 'tok' for CLS token or 'avg' for average pooling
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        assert pool_type in ('tok', 'avg')

        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim
        self.pool_type = pool_type

        # Patch embedding via convolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        # Class embedding and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, do_proj=True):
        """
        Forward pass for feature extraction.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Features of shape [B, output_dim] (default: [B, 512])
        """
        # Patch embedding
        x = self.conv1(x)  # shape = [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [B, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [B, grid ** 2, width]

        # Add class embedding and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [B, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        # Pooling
        pooled = patch_tokens.mean(dim=1) if self.pool_type == 'avg' else cls_token
        # Project to output dimension
        if do_proj and self.proj is not None:
            pooled = pooled @ self.proj
        
        return {
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
            'proj': pooled
        }


def vit_b_16(output_dim=512, **kwargs):
    """
    Create ViT-B/16 vision transformer.

    Args:
        output_dim: Output feature dimension (default: 512 for DermLIP)
        **kwargs: Additional arguments

    Returns:
        ViTVisionTransformer instance
    """
    model = ViTVisionTransformer(
        heads=12,
        mlp_ratio=4.0,
        output_dim=output_dim,
        pool_type='tok',
        **kwargs
    )
    return model
