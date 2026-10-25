import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from typing import Union, Tuple, List, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as tfs

from encoders.conch.model import VisionTransformer, VisualModel, CONCHv15VisualModel, resample_abs_pos_embed
from local_constants import CONCH_CKPT

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)






def detect_architecture_type(state_dict: dict) -> str:
    """
    Detect whether checkpoint is original CONCH or CONCHv1.5.

    Args:
        state_dict: Model checkpoint state dictionary

    Returns:
        'conch_v15': UNI-based architecture with custom AttentionalPooler
        'conch_original': Original CONCH with standard AttentionalPooler

    Raises:
        ValueError: If architecture cannot be determined from checkpoint keys
    """
    # Determine if checkpoint is vision-only or full CoCa format
    prefix = '' if 'trunk.pos_embed' in state_dict else 'visual.'

    # CONCHv1.5 uses to_q, to_kv, to_out in attentional pooler
    # Original CONCH uses PyTorch MultiheadAttention with q_proj_weight, etc.
    if f'{prefix}attn_pool_contrast.to_q.weight' in state_dict:
        return 'conch_v15'
    elif f'{prefix}attn_pool_contrast.attn.in_proj_weight' in state_dict or \
         f'{prefix}attn_pool_contrast.attn.q_proj_weight' in state_dict:
        return 'conch_original'
    else:
        # Try to give helpful error message
        available_keys = [k for k in state_dict.keys() if 'attn_pool_contrast' in k]
        raise ValueError(
            f"Cannot determine architecture type from checkpoint keys. "
            f"Available attn_pool_contrast keys: {available_keys[:5]}"
        )


class CONCHOriginalVisualModel(VisualModel):
    def forward(self, x):
        tokens = self.trunk.forward_features(x)
        pooled = self.ln_contrast(self.attn_pool_contrast(tokens)[:, 0])
        return {
            'cls_token': tokens[:, 0],
            'patch_tokens': tokens[:, 1:],
            'proj': pooled @ self.proj_contrast,
        }


def _build_conch_original(state_dict, is_vision_only_ckpt):
    visual = state_dict if is_vision_only_ckpt else {
        key.removeprefix('visual.'): value for key, value in state_dict.items()
        if key.startswith('visual.')
    }
    width = visual['trunk.pos_embed'].shape[-1]
    patch_size = visual['trunk.patch_embed.proj.weight'].shape[-1]
    image_size = int(math.sqrt(visual['trunk.pos_embed'].shape[1] - 1)) * patch_size
    depth = len({key.split('.')[2] for key in visual if key.startswith('trunk.blocks.')})
    trunk = VisionTransformer(
        img_size=image_size, patch_size=patch_size, embed_dim=width,
        depth=depth, num_heads=width // 64, num_classes=0,
    )
    caption_query = visual.get('attn_pool_caption.query')
    model = CONCHOriginalVisualModel(
        trunk=trunk, image_size=image_size,
        embed_dim_contrast=visual['attn_pool_contrast.query'].shape[1],
        n_queries_contrast=visual['attn_pool_contrast.query'].shape[0],
        use_attentional_pool_contrast=True,
        use_attentional_pool_caption=caption_query is not None,
        embed_dim_caption=caption_query.shape[1] if caption_query is not None else width,
        n_queries_caption=caption_query.shape[0] if caption_query is not None else 256,
    )
    model.load_state_dict(visual, strict=True)
    return model


def _build_conchv15(state_dict: dict, is_vision_only_ckpt: bool, strict=True):
    """
    Build CONCHv1.5 (UNI-based) model from checkpoint.

    Args:
        state_dict: Model state dictionary
        is_vision_only_ckpt: Whether checkpoint is vision-only format
        strict: Whether to strictly load weights
    Returns:
        model: CONCHv15VisualModel instance
    """
    # Infer architecture parameters from checkpoint
    prefix = '' if is_vision_only_ckpt else 'visual.'
    pos_embed_key = f'{prefix}trunk.pos_embed'
    pos_embed_shape = state_dict[pos_embed_key].shape  # [1, num_patches+1, width]

    width = pos_embed_shape[2]  # 1024
    num_patches = pos_embed_shape[1] - 1
    patch_size = 16
    image_size = int((num_patches ** 0.5) * patch_size)  # 448

    layer_keys = [k for k in state_dict.keys() if k.startswith(f'{prefix}trunk.blocks.')]
    idx_pos = 2 if is_vision_only_ckpt else 3
    num_layers = len(set([
        int(k.split('.')[idx_pos]) for k in layer_keys if k.split('.')[idx_pos].isdigit()
    ]))  # 24

    # Infer num_heads (typical head_dim = 64)
    num_heads = width // 64
    pooler_d_model = state_dict[f'{prefix}attn_pool_contrast.query'].shape[1]
    trunk = VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=width,
        depth=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1.0,
        num_classes=0,
        class_token=True,
        no_embed_class=False,
    )
    model = CONCHv15VisualModel(
        trunk=trunk,
        pooler_d_model=pooler_d_model,
        pooler_context_dim=width,
        pooler_n_heads=8,
        pooler_n_queries=1,
    )

    if is_vision_only_ckpt:
        vision_state_dict = state_dict
    else:
        vision_state_dict = {k[len('visual.'):]: v for k, v in state_dict.items() if k.startswith('visual.')}

    pos_embed_w = vision_state_dict['trunk.pos_embed']
    if pos_embed_w.shape != model.trunk.pos_embed.shape:
        # see https://github.com/rwightman/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L509
        num_prefix_tokens = 0 if getattr(model.trunk, 'no_embed_class', False) else getattr(
            model.trunk, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(
            pos_embed_w,
            new_size=model.trunk.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation='bilinear',
            antialias=False,
            verbose=True,
        )
        vision_state_dict['trunk.pos_embed'] = pos_embed_w

    missing, unexpected = model.load_state_dict(vision_state_dict, strict=strict)

    if missing:
        print(f"⚠️Missing keys: {missing}")
        if strict:
            raise RuntimeError(f"Strict mode enabled but {len(missing)} keys are missing")
    if unexpected:
        print(f"⚠️Unexpected keys: {unexpected}")
        if strict:
            raise RuntimeError(f"Strict mode enabled but {len(unexpected)} keys are unexpected")

    # Set image normalization metadata on model
    model.image_mean = OPENAI_DATASET_MEAN
    model.image_std = OPENAI_DATASET_STD
    model.image_size = image_size

    return model


def build_transform(image_shape=(448, 448)):
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),
        tfs.Normalize(
            mean=OPENAI_DATASET_MEAN,
            std=OPENAI_DATASET_STD
        )
    ])


def build_model(
    arch='conch-v15',
    image_shape=(448, 448),
    device="cpu",
    pretrained_path=None,
):
    if isinstance(device, str):
        device = torch.device(device)

    pretrained_path = CONCH_CKPT if not pretrained_path else pretrained_path

    if arch == 'conch-v1':
        checkpoint_path = os.path.join(pretrained_path, 'conch_v1/pytorch_model.bin')
    else:
        checkpoint_path = os.path.join(pretrained_path, 'conch_v15/pytorch_model.bin')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    architecture = detect_architecture_type(state_dict)
    is_vision_only_ckpt = 'trunk.pos_embed' in state_dict

    if architecture == 'conch_v15':
        # CONCHv1.5 (UNI-based) architecture
        model = _build_conchv15(
            state_dict=state_dict,
            is_vision_only_ckpt=is_vision_only_ckpt,
            strict=True
        )
    elif architecture == 'conch_original':
        model = _build_conch_original(state_dict, is_vision_only_ckpt)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model.to(device)



def get_base_dim(image_shape=(448, 448), *args, arch='conch-v15', **kwargs):
    raw_dim = 768 if arch == 'conch-v1' else 1024
    latent_dim = raw_dim
    feature_dim = 512 if arch == 'conch-v1' else 768
    ds_factor = 16
    
    if isinstance(image_shape, (list, tuple)):
        feature_shape = [s // ds_factor for s in image_shape]
    else:
        feature_shape = [image_shape // ds_factor, image_shape // ds_factor]

    return {
        'raw_dim': raw_dim,
        'latent_dim': latent_dim,
        'feature_dim': feature_dim,
        'raw_shape': feature_shape,
        'latent_shape': feature_shape,
        'feature_shape': feature_shape
    }



if __name__ == '__main__':
    model = build_model(
        arch='conch-v15',
        image_shape=(448, 448),
        device="cpu",
        pretrained_path=None,
    )
    x = torch.randn((2, 3, 448, 448))
    with torch.no_grad():
        out = model(x)
    for k, v in out.items():
        print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
