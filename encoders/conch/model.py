"""
Core CONCH model components: CoCa model and VisualModel.
Combined from conch/open_clip_custom/coca_model.py and conch/open_clip_custom/vision_tower.py
"""
import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
import math
import torch
import logging
import numpy as np
import collections.abc
import torch.nn as nn
import torch.nn.functional as F

from itertools import repeat
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, Union, List
from torchvision.ops.misc import FrozenBatchNorm2d

from timm.layers import Mlp, to_2tuple
from timm.models.vision_transformer import VisionTransformer
from timm.models import group_parameters, group_modules

from encoders.conch.transformer import (
    AttentionalPooler, UNIAttentionalPooler, TextTransformer, MultimodalTransformer
)




def resample_abs_pos_embed(
    posemb: torch.Tensor,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = 'bicubic',
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.
    If `module` is itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into
    `FrozenBatchNorm2d` and returned. Otherwise, the module is walked recursively and submodules are
    converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


@dataclass
class CoCaVisionCfg:
    """Configuration for the vision encoder."""
    layers: int = 12
    width: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    attentional_pool_contrast: bool = False  # perceiver resampler for contrastive loss
    attentional_pool_caption: bool = False  # perceiver resampler for captioning
    n_queries_contrast: int = 1  # n_queries for contrastive loss
    n_queries_caption: int = 256  # n_queries for captioning
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    """Configuration for the text encoder."""
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


class VisualModel(nn.Module):
    """Vision encoder with optional attentional pooling for contrastive and caption tasks."""

    def __init__(
            self,
            embed_dim_contrast,
            embed_dim_caption,
            trunk,
            image_size=224,
            proj='',
            proj_bias=False,
            drop=0.,
            global_average_pool=False,
            use_attentional_pool_contrast=False,
            use_attentional_pool_caption=False,
            n_queries_contrast=1,
            n_queries_caption=256,
            attn_pooler_heads=8,
            norm_layer=nn.LayerNorm,
            output_tokens=False,
            trunk_kwargs={}
    ):
        super().__init__()

        self.trunk = trunk
        self.trunk_kwargs = trunk_kwargs
        self.image_size = to_2tuple(image_size)
        prev_chs = self.trunk.num_features
        head_layers = OrderedDict()

        # whether to use attentional pooling
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
        self.global_average_pool = global_average_pool
        self.output_tokens = output_tokens
        if use_attentional_pool_contrast:
            scale = prev_chs ** -0.5
            self.attn_pool_contrast = AttentionalPooler(d_model=embed_dim_contrast, context_dim=prev_chs,
                                                        n_head=attn_pooler_heads, n_queries=n_queries_contrast)
            self.ln_contrast = norm_layer(embed_dim_contrast)
            self.proj_contrast = nn.Parameter(scale * torch.randn(embed_dim_contrast, embed_dim_contrast))
        else:
            assert proj, 'projection layer needed if not using attentional pooling.'
            # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
            if proj == 'linear':
                head_layers['drop'] = nn.Dropout(drop)
                head_layers['proj'] = nn.Linear(prev_chs, embed_dim_contrast, bias=proj_bias)
            elif proj == 'mlp':
                head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim_contrast, embed_dim_contrast,
                                        drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(d_model=embed_dim_caption, context_dim=prev_chs,
                                                       n_head=attn_pooler_heads, n_queries=n_queries_caption)
            self.ln_caption = norm_layer(embed_dim_caption)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """
        Lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def _global_pool(self, x):
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward_project(self, x):
        if self.use_attentional_pool_contrast:
            x = x @ self.proj_contrast
            return x
        else:
            x = self.head(x)
            return x

    def forward_attn_pool_caption(self, tokens, attn_mask=None):
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(tokens, attn_mask=attn_mask)
            tokens = self.ln_caption(tokens)
            return tokens
        else:
            raise NotImplementedError

    def forward_no_head(self, x, normalize=False):
        """Forward pass without projection head - for image-only tasks."""
        x = self.trunk(x, **self.trunk_kwargs)
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0]
            pooled = self.ln_contrast(pooled)
        else:
            pooled, _ = self._global_pool(x)
        if normalize:
            pooled = nn.functional.normalize(pooled, dim=-1)
        return pooled

    def forward(self, x):
        """Forward pass with projection head - for retrieval tasks."""
        x = self.trunk(x, **self.trunk_kwargs)
        tokens = None
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0]  # single query
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.head(x)

        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(x)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None

        if self.output_tokens:
            return pooled, tokens
        else:
            return pooled


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CoCaVisionCfg,
        embed_dim_caption: Optional[int] = None
):
    """Build the vision tower with ViT backbone."""
    if isinstance(vision_cfg, dict):
        vision_cfg = CoCaVisionCfg(**vision_cfg)

    trunk = VisionTransformer(
        embed_dim=vision_cfg.width,
        depth=vision_cfg.layers,
        num_heads=vision_cfg.num_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        img_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        num_classes=0,
        dynamic_img_size=True
    )

    trunk_kwargs = {}
    trunk.forward = trunk.forward_features

    visual = VisualModel(
        trunk=trunk,
        trunk_kwargs=trunk_kwargs,
        use_attentional_pool_contrast=vision_cfg.attentional_pool_contrast,
        use_attentional_pool_caption=vision_cfg.attentional_pool_caption,
        n_queries_contrast=vision_cfg.n_queries_contrast,
        n_queries_caption=vision_cfg.n_queries_caption,
        output_tokens=vision_cfg.output_tokens,
        embed_dim_contrast=embed_dim,
        embed_dim_caption=embed_dim_caption,
        image_size=vision_cfg.image_size,
    )
    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg
):
    """Build the text encoder tower."""
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg
):
    """Build the multimodal decoder tower."""
    multimodal_cfg = CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Module):
    """
    CoCa (Contrastive Captioners) model for vision-language learning.

    This model combines:
    - Vision encoder (ViT with attentional pooling)
    - Text encoder (Transformer)
    - Multimodal decoder (optional, for captioning)
    """

    def __init__(
            self,
            embed_dim,
            embed_dim_caption,
            multimodal_cfg: CLIPTextCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CoCaVisionCfg,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CoCaVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg
        )

        vocab_size = text_cfg.vocab_size

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            embed_dim_caption=embed_dim_caption,
            vision_cfg=vision_cfg
        )

        if multimodal_cfg.layers > 0:
            self.text_decoder = _build_text_decoder_tower(
                vocab_size,
                multimodal_cfg=multimodal_cfg
            )
        else:
            # no decoder
            self.text_decoder = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id
        self.context_length = text_cfg.context_length

        self.embed_dim = embed_dim
        self.embed_dim_caption = embed_dim_caption

    def lock_temperature(self):
        self.logit_scale.requires_grad = False

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        if self.text_decoder is not None:
            self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images=None, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text  # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True, proj_contrast=True):
        """
        Encode images to embeddings.

        Args:
            images: Input images tensor [B, C, H, W]
            normalize: Whether to L2 normalize the output
            proj_contrast: If True, use projection head (for retrieval). If False, use pre-projection features (for MIL)

        Returns:
            Image embeddings
        """
        if proj_contrast:
            image_latent, _ = self._encode_image(images, normalize=normalize)
        else:
            image_latent = self.visual.forward_no_head(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        """
        Encode text to embeddings.

        Args:
            text: Tokenized text input
            normalize: Whether to L2 normalize the output
            embed_cls: Whether to use CLS token embedding

        Returns:
            Text embeddings
        """
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self, image, text, embed_cls=True, image_latent=None, image_embs=None):
        """
        Forward pass for training (with contrastive and captioning losses).

        Args:
            image: Input images
            text: Input text tokens
            embed_cls: Whether to use CLS embedding for text
            image_latent: Pre-computed image latents (optional)
            image_embs: Pre-computed image embeddings (optional)

        Returns:
            Dictionary with image_features, text_features, logits, labels, logit_scale
        """
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        labels = text[:, -token_embs.shape[1]:]
        if self.text_decoder is not None:
            logits = self.text_decoder(image_embs, token_embs)
        else:
            logits = torch.empty(text.shape[0], 1, device=text.device)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }



# ============================================================================
# CONCHv1.5 (UNI-based) Architecture
# ============================================================================


class CONCHv15VisualModel(nn.Module):
    """
    Vision encoder for CONCH v1.5 (UNI-based architecture).

    Key differences from original CONCH:
    - Uses ViT-L (1024 width) instead of ViT-B (768 width)
    - Uses UNIAttentionalPooler instead of standard AttentionalPooler
    - No separate projection layer (integrated in pooler)
    - Output dimension is 768 (pooler output)

    Based on implementation from conch_v15.py lines 640-652.
    """

    def __init__(
        self,
        trunk,  # timm VisionTransformer
        pooler_d_model: int = 768,
        pooler_context_dim: int = 1024,
        pooler_n_heads: int = 8,
        pooler_n_queries: int = 1,
    ):
        super().__init__()
        self.trunk = trunk
        self.attn_pool_contrast = UNIAttentionalPooler(
            d_model=pooler_d_model,
            context_dim=pooler_context_dim,
            n_head=pooler_n_heads,
            n_queries=pooler_n_queries
        )
        self.ln_contrast = nn.LayerNorm(pooler_d_model)

    def forward(self, x, **kwargs):
        x = self.trunk.forward_features(x)  # [B, 785, 1024] (785=1+448/16*448/16)
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        proj = self.attn_pool_contrast(x)  # [B, 1, 768]
        proj = proj[:, 0]  # [B, 768]
        proj = self.ln_contrast(proj)
        return {
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
            'proj': proj
        }

    def forward_features(self, x):
        """Get features before pooling (for compatibility)."""
        return self.trunk.forward_features(x)

    def forward_no_head(self, x, normalize=False):
        """
        Forward pass without any projection (same as forward for CONCHv1.5).
        CONCHv1.5 doesn't have a separate projection layer.
        """
        return self.forward(x, normalize=normalize)


def _build_conchv15_vision_tower(
    image_size: int = 448,
    patch_size: int = 16,
    width: int = 1024,
    layers: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    init_values: float = 1.0,
    pooler_d_model: int = 768,
    pooler_n_queries: int = 1,
    pooler_n_heads: int = 8,
):
    """
    Build CONCHv1.5 (UNI-based) vision tower.

    Args:
        image_size: Input image size (default: 448)
        patch_size: Patch size for ViT (default: 16)
        width: ViT embedding dimension (default: 1024 for ViT-L)
        layers: Number of transformer layers (default: 24 for ViT-L)
        num_heads: Number of attention heads (default: 16 for ViT-L)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        init_values: LayerScale initialization value (default: 1.0)
        pooler_d_model: Attentional pooler output dimension (default: 768)
        pooler_n_queries: Number of queries for pooling (default: 1)
        pooler_n_heads: Number of heads in pooler (default: 8)

    Returns:
        CONCHv15VisualModel instance
    """
    # Build ViT-L trunk with LayerScale
    trunk = VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=width,
        depth=layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        init_values=init_values,  # LayerScale initialization
        num_classes=0,
        class_token=True,
        no_embed_class=False,
    )

    # Build visual model with UNI-style pooler
    visual = CONCHv15VisualModel(
        trunk=trunk,
        pooler_d_model=pooler_d_model,
        pooler_context_dim=width,
        pooler_n_heads=pooler_n_heads,
        pooler_n_queries=pooler_n_queries,
    )

    return visual


if __name__ == '__main__':
    x = torch.randn((2, 3, 448, 448))
    model = _build_conchv15_vision_tower()
    with torch.no_grad():
        y = model(x)
