"""
Standalone HLIP inference — no open_clip / huggingface_hub dependency.

Replaces example.py by inlining the open_clip wrapper code.
Remaining dependencies: torch, timm, transformers, safetensors, numpy, torchvision
"""

from pathlib import Path
from collections import OrderedDict
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from timm.models.vision_transformer import VisionTransformer
from transformers import BertConfig, BertModel, AutoTokenizer
import safetensors.torch as st
from torchvision.transforms import Normalize
from local_constants import HLIPV_CKPT


# ==============================================================================
# Constants
# ==============================================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CONTEXT_LENGTH = 256
EMBED_DIM = 768
VISION_EMBED_DIM = 1024
VISION_DEPTH = 24
VISION_HEADS = 16
IMG_SIZE = (48, 224, 224)
PATCH_SIZE = (6, 16, 16)
DROP_PATH_RATE = 0.2
INIT_LOGIT_BIAS = 2.659260036932778

# Zero-shot metadata (brain MRI 5-class) — from hlip/zeroshot_metadata_pubbrain5.py
CLASSNAMES = ["Normal", "Stroke", "Glioma", "Meningioma", "Metastasis"]

PROMPTS = (
    "no significant abnormalities",
    "acute stroke",
    "glioma",
    "meningioma",
    "metastasis",
)

TEMPLATES = (
    lambda c: f"This study shows: {c}.",
    lambda c: f"This study shows: {c} identified." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: {c} noted." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: {c} seen." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: new {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: known {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: prominent {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: likely {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: possibly {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: indicating {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: reflecting {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: representing {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: suggesting {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: indicative of {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: suggestive of {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: related to {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: consistent with {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    lambda c: f"This study shows: compatible with {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
)


# ==============================================================================
# 3D Patch Embedding (from hlip/patch_embed.py)
# ==============================================================================

class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(self, img_size=(224, 224, 224), patch_size=(16, 16, 16),
                 in_chans=1, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert len(list(img_size)) == 3
        assert len(list(patch_size)) == 3
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=kwargs.get("bias", True),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W)
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, D, H, W, -1)
        x = self.norm(x)
        return x


# ==============================================================================
# Position Embeddings (from hlip/pos_embed.py)
# ==============================================================================

def _sincos_1d_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _sincos_1d(embed_dim, seq_len):
    return _sincos_1d_from_grid(embed_dim, np.arange(seq_len, dtype=np.float32))


def _sincos_2d(embed_dim, grid_sizes):
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_sizes[0], dtype=np.float32)
    grid_w = np.arange(grid_sizes[1], dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_sizes[0], grid_sizes[1])
    emb_h = _sincos_1d_from_grid(embed_dim // 2, grid[0])
    emb_w = _sincos_1d_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_w, emb_h], axis=1).reshape(
        grid_sizes[0], grid_sizes[1], embed_dim
    )


def resample_1d_posemb(posemb, num_samples, is_train=True):
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    _max = posemb.shape[1]
    if _max < num_samples:
        assert not is_train
        posemb = F.interpolate(
            posemb.permute(0, 2, 1), size=num_samples, mode='linear'
        ).permute(0, 2, 1)
        return posemb.to(orig_dtype)
    if num_samples <= _max:
        perm = torch.randperm(_max)[:num_samples] if is_train else torch.arange(num_samples)
        return posemb[:, perm, :].to(orig_dtype)


def resample_3d_posemb(posemb, new_size, old_size):
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    if new_size == old_size:
        return posemb.to(orig_dtype)
    posemb = F.interpolate(
        posemb.permute(0, 4, 1, 2, 3),
        size=(new_size[0], new_size[1], new_size[2]),
        mode='trilinear',
    ).permute(0, 2, 3, 4, 1)
    return posemb.to(orig_dtype)


def study_pos_embed(max_num_scans, grid_size, embed_dim):
    d_posemb = torch.from_numpy(_sincos_1d(embed_dim, grid_size[0])).float()
    hw_posemb = torch.from_numpy(
        _sincos_2d(embed_dim, (grid_size[1], grid_size[2]))
    ).float()
    spatial_posemb = d_posemb[None, :, None, None, :] + hw_posemb[None, None, :, :, :]

    sequential_posemb = None
    if max_num_scans > 1:
        sequential_posemb = torch.from_numpy(
            _sincos_1d(embed_dim, max_num_scans)
        ).float()[None, ...]
    return spatial_posemb, sequential_posemb


# ==============================================================================
# HLIP Visual Encoder (from hlip/visual_encoder.py)
# ==============================================================================

class HLIPVisualEncoder(VisionTransformer):
    def __init__(self, **kwargs):
        max_num_scans = kwargs.pop('max_num_scans')
        self.slice_attn_indexes = kwargs.pop('slice_attn_indexes', ())
        self.scan_attn_indexes = kwargs.pop('scan_attn_indexes', ())
        self.study_attn_indexes = kwargs.pop('study_attn_indexes', ())
        self.global_pool_dinotxt = kwargs.pop('global_pool_dinotxt', False)
        super().__init__(**kwargs)

        if self.global_pool_dinotxt:
            self.num_features = 2 * self.embed_dim

        spatial_posemb, sequential_posemb = study_pos_embed(
            max_num_scans=max_num_scans,
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
        )
        self.spatial_posemb = nn.Parameter(spatial_posemb)
        self.spatial_posemb.requires_grad = False
        if sequential_posemb is not None:
            self.sequential_posemb = nn.Parameter(sequential_posemb)
            self.sequential_posemb.requires_grad = False
        else:
            self.sequential_posemb = None

    def _pos_embed(self, x):
        # x: [bs, n, d, h, w, c]
        bs, n, d, h, w, _ = x.shape
        spatial_posemb = resample_3d_posemb(
            self.spatial_posemb, (d, h, w), self.patch_embed.grid_size
        )
        if self.sequential_posemb is not None:
            sequential_posemb = resample_1d_posemb(
                self.sequential_posemb, n, is_train=bs != 1
            )
            pos_embed = (
                sequential_posemb[:, :, None, None, None, :]
                + spatial_posemb[:, None, :, :, :, :]
            )
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)
        else:
            pos_embed = spatial_posemb[:, None, :, :, :, :].expand(
                bs, n, -1, -1, -1, -1
            )

        if 0 in self.slice_attn_indexes:
            pos_embed = pos_embed.flatten(3, 4).flatten(0, 2)
            x = x.flatten(3, 4).flatten(0, 2)
        elif 0 in self.scan_attn_indexes:
            pos_embed = pos_embed.flatten(2, 4).flatten(0, 1)
            x = x.flatten(2, 4).flatten(0, 1)
        elif 0 in self.study_attn_indexes:
            pos_embed = pos_embed.flatten(1, 4)
            x = x.flatten(1, 4)

        x = self.pos_drop(x + pos_embed)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x, n, d

    def _slice2scan(self, x, num_slices):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        BND, L, C = src.shape
        prefix = prefix.view(
            BND // num_slices, num_slices, self.num_prefix_tokens, C
        ).mean(dim=1)
        src = src.view(BND // num_slices, num_slices * L, C)
        return torch.cat([prefix, src], dim=1)

    def _slice2study(self, x, num_scans, num_slices):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        BND, L, C = src.shape
        ND = num_scans * num_slices
        prefix = prefix.view(
            BND // ND, ND, self.num_prefix_tokens, C
        ).mean(dim=1)
        src = src.view(BND // ND, ND * L, C)
        return torch.cat([prefix, src], dim=1)

    def _scan2study(self, x, num_scans):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        BN, L, C = src.shape
        prefix = prefix.view(
            BN // num_scans, num_scans, self.num_prefix_tokens, C
        ).mean(dim=1)
        src = src.view(BN // num_scans, num_scans * L, C)
        return torch.cat([prefix, src], dim=1)

    def _study2slice(self, x, num_scans, num_slices):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        B, NDL, C = src.shape
        prefix = (
            prefix.view(B, 1, 1, self.num_prefix_tokens, C)
            .expand(-1, num_scans, num_slices, -1, -1)
            .contiguous()
        )
        src = src.view(B, num_scans, num_slices, NDL // (num_scans * num_slices), C)
        x = torch.cat([prefix, src], dim=3)
        return x.view(-1, self.num_prefix_tokens + NDL // (num_scans * num_slices), C)

    def _study2scan(self, x, num_scans):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        B, NL, C = src.shape
        prefix = (
            prefix.view(B, 1, self.num_prefix_tokens, C)
            .expand(-1, num_scans, -1, -1)
            .contiguous()
        )
        src = src.view(B, num_scans, NL // num_scans, C)
        x = torch.cat([prefix, src], dim=2)
        return x.view(-1, self.num_prefix_tokens + NL // num_scans, C)

    def _scan2slice(self, x, num_slices):
        prefix, src = (
            x[:, : self.num_prefix_tokens, :].contiguous(),
            x[:, self.num_prefix_tokens :, :].contiguous(),
        )
        BN, DL, C = src.shape
        prefix = (
            prefix.view(BN, 1, self.num_prefix_tokens, C)
            .expand(-1, num_slices, -1, -1)
            .contiguous()
        )
        src = src.view(BN, num_slices, DL // num_slices, C)
        x = torch.cat([prefix, src], dim=2)
        return x.view(-1, self.num_prefix_tokens + DL // num_slices, C)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, num_scans, num_slices = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if idx - 1 in self.slice_attn_indexes and idx in self.study_attn_indexes:
                x = self._slice2study(x, num_scans, num_slices)
            elif idx - 1 in self.slice_attn_indexes and idx in self.scan_attn_indexes:
                x = self._slice2scan(x, num_slices)
            elif idx - 1 in self.scan_attn_indexes and idx in self.study_attn_indexes:
                x = self._scan2study(x, num_scans)
            elif idx - 1 in self.study_attn_indexes and idx in self.slice_attn_indexes:
                x = self._study2slice(x, num_scans, num_slices)
            elif idx - 1 in self.study_attn_indexes and idx in self.scan_attn_indexes:
                x = self._study2scan(x, num_scans)
            elif idx - 1 in self.scan_attn_indexes and idx in self.slice_attn_indexes:
                x = self._scan2slice(x, num_slices)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = grad_checkpoint(blk, x)
            else:
                x = blk(x)

        if len(self.blocks) - 1 in self.scan_attn_indexes:
            x = self._scan2study(x, num_scans)
        if len(self.blocks) - 1 in self.slice_attn_indexes:
            x = self._slice2study(x, num_slices, num_scans)

        return self.norm(x)

    def forward_head_dinotxt(self, x):
        prefix_tokens = x[:, : self.num_prefix_tokens]
        visual_tokens = x[:, self.num_prefix_tokens :].mean(dim=1, keepdim=True)
        visual_tokens = visual_tokens.expand(-1, self.num_prefix_tokens, -1)
        x = torch.cat([prefix_tokens, visual_tokens], dim=-1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)

    def encode_multi_scan(self, x):
        """Standard multi-scan forward pass.

        Args:
            x: [B, N, C, D, H, W]  — a study of N 3-D scans.
               B  – batch size
               N  – number of scans in the study (typically 1–40;
                    the model was trained with multi-scan studies)
               C  – input channels (1 for grayscale medical images)
               D  – depth, i.e. number of slices per scan
                    (default config expects 48; internally patchified
                     into D/patch_d = 48/6 = 8 depth tokens)
               H  – height  (default 224, patchified into 224/16 = 14)
               W  – width   (default 224, patchified into 224/16 = 14)

               Pixel-value range: ImageNet-normalised floats.
               The expected pre-processing pipeline is:
                 raw uint8 [0, 255]
                 → float32 / 255.0            → [0, 1]
                 → pad to square
                 → bilinear resize to 256 × 256
                 → nearest-exact depth resample to D = 48
                 → center-crop to 224 × 224
                 → Normalize(mean=0.449, std=0.226)
               where 0.449 = mean(ImageNet_mean) and
                     0.226 = mean(ImageNet_std).

        Returns:
            When global_pool_dinotxt is True (this checkpoint):
              [B, num_prefix_tokens, 2 * embed_dim]
              = [B, 2, 2048]  for ViT-Large with 1 CLS + 1 REG token.

              Token 0 is the CLS token; token 1 is the REG token.
              Both carry a concatenation of [prefix_feature, mean_visual_feature].

            When global_pool_dinotxt is False:
              [B, num_prefix_tokens, embed_dim]
        """
        x = self.forward_features(x)
        if self.global_pool_dinotxt:
            return self.forward_head_dinotxt(x)
        x = self.forward_head(x)
        if self.num_prefix_tokens == 1:
            x = x.unsqueeze(1)
        return x

    def forward(self, x, do_normalize=False, **kwargs):
        """Convenience method: extract features from a **single** 3-D scan.

        This wraps the scan as a 1-scan "study" and runs the standard
        forward pass.  Study-level attention blocks (5, 11, 17, 23)
        degenerate to scan-level because there is only one scan —
        no cross-scan fusion occurs.

        Args:
            x: [B, C, D, H, W]  — one 3-D volume per batch element.
               C = 1, D = 48, H = W = 224 (after standard pre-processing).
               Pixel values are ImageNet-normalised floats (see ``forward``
               docstring for the full pre-processing pipeline).

        Returns:
            features: [B, 2 * embed_dim]  (= [B, 2048] for ViT-Large)
              The CLS-token feature from the DINO.TXT head.
              To obtain the final CLIP-space embedding, pass through
              ``VisualEncoder.head`` (projects 2048 → 768) and then
              L2-normalise.
        """
        # [B, C, D, H, W] → [B, 1, C, D, H, W]  (single-scan study)
        if len(x.shape) == 5:
            x = x.unsqueeze(1)

        if len(x.shape) != 6:
            raise ValueError(f"Expect tensors with shape [B, C, D, H, W] or [B, num_scans, C, D, H, W], got {x.shape}")
        
        # forward returns [B, num_prefix_tokens, 2*embed_dim]
        out = self.encode_multi_scan(x)
        # return CLS token only: [B, 2*embed_dim]

        cls_token = out[:, 0, :]
        if do_normalize:
            cls_token = F.normalize(cls_token, p=2, dim=1)
        return {'cls_token': cls_token}


# ==============================================================================
# Visual Encoder wrapper  (replaces open_clip.TimmModel)
# ==============================================================================

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.trunk = HLIPVisualEncoder(
            max_num_scans=0,
            slice_attn_indexes=(0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21),
            scan_attn_indexes=(4, 10, 16, 22),
            study_attn_indexes=(5, 11, 17, 23),
            global_pool_dinotxt=True,
            reg_tokens=1,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=1,
            depth=VISION_DEPTH,
            embed_dim=VISION_EMBED_DIM,
            num_heads=VISION_HEADS,
            num_classes=0,
            no_embed_class=True,
            pos_embed='none',
            embed_layer=PatchEmbed3D,
            drop_path_rate=DROP_PATH_RATE,
        )
        self.trunk.reset_classifier(0, global_pool='token')

        head_layers = OrderedDict()
        head_layers['drop'] = nn.Dropout(0.)
        head_layers['proj'] = nn.Linear(
            self.trunk.num_features, embed_dim, bias=False
        )
        self.head = nn.Sequential(head_layers)

    def forward(self, x):
        x = self.trunk.encode_multi_scan(x)
        x = self.head(x)
        return x


# ==============================================================================
# Text Encoder  (replaces open_clip.HFTextEncoder + BiomedBERT)
# ==============================================================================

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=0,
        )
        self.transformer = BertModel(config, add_pooling_layer=False)
        self.proj = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.pad_token_id = config.pad_token_id

    def forward(self, x):
        attn_mask = (x != self.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS token
        return self.proj(pooled)


# ==============================================================================
# HLIP Model  (replaces open_clip.CustomTextCLIP)
# ==============================================================================

class HLIPModel(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, init_logit_bias=INIT_LOGIT_BIAS):
        super().__init__()
        self.visual = VisualEncoder(embed_dim)
        self.text = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

    def encode_image(self, image, normalize=False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image=None, text=None):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias,
        }


# ==============================================================================
# Tokenizer  (replaces open_clip.HFTokenizer)
# ==============================================================================

def get_tokenizer(tokenizer_path, context_length=CONTEXT_LENGTH):
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [" ".join(t.split()) for t in texts]  # whitespace cleaning
        return hf_tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=context_length,
            padding="max_length",
            truncation=True,
        ).input_ids

    return tokenize


# ==============================================================================
# Zero-shot classifier  (replaces open_clip.build_zero_shot_classifier)
# ==============================================================================

def build_zero_shot_classifier(model, tokenizer, classnames, templates, device="cpu"):
    with torch.no_grad():
        texts = [template(c) for c in classnames for template in templates]
        text_tokens = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(text_tokens, normalize=True)
        class_embeddings = class_embeddings.reshape(
            len(classnames), len(templates), -1
        ).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        return class_embeddings.T


# ==============================================================================
# Image loader  (same as example.py)
# ==============================================================================

def loader(study_path: str, num_slices: int):
    """
    study_path: folder containing per-slice tensors saved with torch.save()
                each file is a [C, H, W] or [C, H, W, 1] tensor in [0, 255]
    returns:    image tensor of shape [1, n_scans, 1, D, H, W]
    """
    imgs = []
    for scan in [os.path.join(study_path, p) for p in os.listdir(study_path)]:
        img = torch.load(scan, weights_only=True)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img.float() / 255.0
        _, h, w = img.shape

        size = max(h, w)
        pad_h = size - h
        pad_w = size - w
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = torch.nn.functional.pad(
            img, (left, right, top, bottom), mode="constant", value=0
        )

        img = F.interpolate(img[None, ...], size=(256, 256), mode="bilinear")[0]
        img = F.interpolate(
            img[None, None, ...], size=(num_slices, 256, 256), mode="nearest-exact"
        )[0, 0]
        img = img[:, 16:240, 16:240]

        normalizer = Normalize(
            torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
            torch.as_tensor(IMAGENET_DEFAULT_STD).mean(),
        )
        img = normalizer(img[None, ...])
        imgs.append(img)

    return torch.stack(imgs, dim=0)[None, ...]


# ==============================================================================
# Build pretrained model from local files
# ==============================================================================

def build_model(arch='hlip-vision', image_shape=(224, 224), device='cpu', pretrained_path=None,):
    pretrained_path = HLIPV_CKPT if not pretrained_path else pretrained_path
    model = HLIPVisualEncoder(
        max_num_scans=0,
        slice_attn_indexes=(0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21),
        scan_attn_indexes=(4, 10, 16, 22),
        study_attn_indexes=(5, 11, 17, 23),
        global_pool_dinotxt=True,
        reg_tokens=1,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        depth=VISION_DEPTH,
        embed_dim=VISION_EMBED_DIM,
        num_heads=VISION_HEADS,
        num_classes=0,
        no_embed_class=True,
        pos_embed='none',
        embed_layer=PatchEmbed3D,
        drop_path_rate=DROP_PATH_RATE,
    )
    model.reset_classifier(0, global_pool='token')

    if pretrained_path:
        print(f"Loading weights for HLIPVisionEncoder from {pretrained_path} ...")
        sd = torch.load(pretrained_path, map_location='cpu')
        msg = model.load_state_dict(sd, strict=True)
        print(f"MESSAGE: {msg}")
        
    return model.to(device)


def build_hlip_model(repo_dir, device="cpu"):
    repo_dir = Path(repo_dir)

    model = HLIPModel(embed_dim=EMBED_DIM, init_logit_bias=INIT_LOGIT_BIAS)

    weight_path = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        p = repo_dir / fname
        if p.exists():
            weight_path = p
            break
    assert weight_path is not None, "No weights found."

    if weight_path.suffix == ".safetensors":
        state_dict = st.load_file(str(weight_path))
    else:
        state_dict = torch.load(str(weight_path), map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[OK] weights loaded: {weight_path.name} "
        f"| missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")

    model = model.to(device)

    tokenizer = get_tokenizer(str(repo_dir))

    return model, tokenizer


def get_base_dim(image_shape=(224, 224), *args, **kwargs):
    raw_dim = 2048
    # latent_dim = 768  # latent
    # feature_dim = 512  # the projected feature
    # ds_factor = 32

    # if isinstance(image_shape, (list, tuple)):
    #     feature_shape = [s // ds_factor for s in image_shape]
    # else:
    #     feature_shape = [image_shape // ds_factor, image_shape // ds_factor]

    return {
        'raw_dim': raw_dim,
        # 'latent_dim': latent_dim,
        # 'feature_dim': feature_dim,
        # 'raw_shape': feature_shape,
        # 'latent_shape': feature_shape,
        # 'feature_shape': feature_shape
    }
