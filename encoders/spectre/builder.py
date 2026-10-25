"""
Builder functions for SPECTRE simplified inference.

Provides:
    build_model   — construct and load the pretrained SPECTRE encoder
    build_transform — default MONAI inference preprocessing pipeline
    get_base_dim  — model dimension / shape metadata
"""
import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from typing import Optional

import torch
import torch.nn as nn

from encoders.spectre.vision_transformer import VisionTransformer, SigLIPProjectionHead
from encoders.spectre.feature_combiner import FeatureVisionTransformer
from encoders.spectre.spectre import SpectreImageFeatureExtractor
from local_constants import (
    SPECTRE_BACKBONE_CKPT,
    SPECTRE_COMBINER_CKPT,
    SPECTRE_PROJECTION_CKPT,
)


def build_model(
    arch: str = "spectre_large_pretrained",
    pretrained_backbone_path: Optional[str] = SPECTRE_BACKBONE_CKPT,
    pretrained_combiner_path: Optional[str] = SPECTRE_COMBINER_CKPT,
    pretrained_projection_path: Optional[str] = SPECTRE_PROJECTION_CKPT,
    device: str = "cpu",
) -> SpectreImageFeatureExtractor:
    """Build and return a pretrained SPECTRE vision encoder.

    Args:
        arch: Architecture name (currently only ``"spectre_large_pretrained"``).
        pretrained_backbone_path: Path to backbone checkpoint (.pt).
        pretrained_combiner_path: Path to combiner checkpoint (.pt).
        device: Target device string.

    Returns:
        ``SpectreImageFeatureExtractor`` in eval mode with weights loaded
        via ``strict=True``.
    """
    # --- Backbone: ViT-Large ---
    # NOTE: norm_layer=nn.LayerNorm (eps=1e-5) matches the original factory function
    backbone = VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        in_chans=1,
        embed_dim=1080,
        depth=24,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=0,
        global_pool="",
        pos_embed="rope",
        rope_kwargs={"base": 1000.0},
        init_values=1.0,
        norm_layer=nn.LayerNorm,
    )
    if pretrained_backbone_path is not None:
        sd = torch.load(pretrained_backbone_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(sd, strict=True)

    # --- Feature Combiner: ViT-Large (shallow) ---
    # patch_dim = 2 * embed_dim because backbone outputs CLS + AVG-pooled tokens
    combiner = FeatureVisionTransformer(
        patch_dim=2160,  # 2 * 1080
        embed_dim=1080,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=0,
        global_pool="",
        pos_embed="rope",
        rope_kwargs={"base": 100.0},
        init_values=1.0,
        norm_layer=nn.LayerNorm,
    )
    if pretrained_combiner_path is not None:
        sd = torch.load(pretrained_combiner_path, map_location="cpu", weights_only=True)
        combiner.load_state_dict(sd, strict=True)

    projection = SigLIPProjectionHead(
        input_dim=2160,
        hidden_dim=2048,
        bottleneck_dim=256,
        output_dim=512,
        layer_norm=False,
    )
    if pretrained_projection_path is not None:
        sd = torch.load(pretrained_projection_path, map_location="cpu", weights_only=False)
        projection.load_state_dict(sd, strict=True)
    
    model = SpectreImageFeatureExtractor(
        backbone=backbone, 
        feature_combiner=combiner,
        projection_head=projection,
    )
    model = model.to(device).eval()
    return model


def build_transform():
    """Return a MONAI ``Compose`` pipeline for deterministic CT inference.

    Pipeline:
        1. Load NIfTI
        2. Ensure channel-first
        3. Intensity windowing: [-1000, 1000] HU → [0, 1]
        4. Reorient to RAS
        5. Resample to 0.75 × 0.75 × 1.5 mm
        6. Resize / pad-or-crop to 384 × 384 × 256
        7. Cast to float32
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ResizeWithPadOrCropd,
        ScaleIntensityRanged,
        Spacingd,
    )

    return Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(keys=("image",), pixdim=(0.75, 0.75, 1.5), mode=("bilinear",)),
        ResizeWithPadOrCropd(keys=("image",), spatial_size=(384, 384, 256)),
        EnsureTyped(keys=("image",), dtype=torch.float32),
    ])


def get_base_dim() -> dict:
    """Return a dictionary of model dimension and shape metadata."""
    return {
        "embed_dim": 1080,
        "patch_size": (16, 16, 8),
        "img_size": (128, 128, 64),
        "num_heads": 12,
        "backbone_depth": 24,
        "combiner_depth": 4,
        "head_dim": 90,
        "mlp_ratio": 4,
        "num_backbone_tokens": 513,   # 1 CLS + 512 patch tokens
        "num_patches_per_crop": 512,  # 8×8×8
        "raw_dim": 2160,   # 2 × embed_dim
    }
