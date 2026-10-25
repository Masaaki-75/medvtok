import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
import json
import torch
import torch.nn as nn
from functools import partial
from safetensors.torch import load_file
from torchvision.transforms import v2 as tfs

from encoders.dermlip.panderm_encoder import PanDermVisionTransformer
from encoders.dermlip.vit_encoder import ViTVisionTransformer
from local_constants import DERMLIP_CKPT


def load_state_dict(checkpoint_path: str, device='cpu', weights_only=True):
    """Load state dict from checkpoint file (.safetensors or .bin)"""
    if str(checkpoint_path).endswith(".safetensors"):
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def _find_weight_file(model_dir):
    """Find weight file (.safetensors or .bin) in model directory"""
    model_dir = str(model_dir)
    safetensors_path = os.path.join(model_dir, "open_clip_model.safetensors")
    bin_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")

    if os.path.exists(safetensors_path):
        return str(safetensors_path)
    elif os.path.exists(bin_path):
        return str(bin_path)
    else:
        raise FileNotFoundError(f"No weight file found in {model_dir}. Expected 'open_clip_model.safetensors' or 'open_clip_pytorch_model.bin'")



def build_model(arch='dermlip-panderm', image_shape=(224, 224), device='cpu', pretrained_path=None):
    pretrained_path = DERMLIP_CKPT if not pretrained_path else pretrained_path

    model_folder = "DermLIP_ViT-B-16" if arch == "dermlip-vit" else "DermLIP_PanDerm-base-w-PubMed-256"
    model_dir = os.path.join(pretrained_path, model_folder)
    config_path = os.path.join(model_dir, "open_clip_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    vision_cfg = config['model_cfg']['vision_cfg']
    embed_dim = config['model_cfg']['embed_dim']

    if arch == 'dermlip-vit':
        model_kwargs = dict(
            output_dim=embed_dim,
            image_size=vision_cfg.get('image_size', 224),
            width=vision_cfg.get('width', 768),
            layers=vision_cfg.get('layers', 12),
            patch_size=vision_cfg.get('patch_size', 16),
        )
        print(f"DermLIP-ViT model_kwargs: {model_kwargs}")
        model = ViTVisionTransformer(
            heads=12,
            mlp_ratio=4.0,
            pool_type='tok',
            **model_kwargs,
        )
    else:
        model_kwargs = dict(
            num_classes=embed_dim,
            img_size=vision_cfg.get('image_size', 224),
            embed_dim=vision_cfg.get('width', 768),
            depth=vision_cfg.get('layers', 12),
            patch_size=vision_cfg.get('patch_size', 16),
        )
        print(f"DermLIP-PanDerm model_kwargs: {model_kwargs}")
        model = PanDermVisionTransformer(
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,  # DermLIP PanDerm was trained with init_values=0.1
            **model_kwargs
        )

    weight_path = _find_weight_file(model_dir)
    state_dict = load_state_dict(weight_path, device='cpu')

    # Extract vision encoder weights (remove 'visual.' prefix)
    vision_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('visual.'):
            new_key = k[7:]  # Remove 'visual.' prefix
            vision_state_dict[new_key] = v

    incompatible_keys = model.load_state_dict(vision_state_dict, strict=True)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        print(f"❌Incompatible keys when loading {arch} from {weight_path}: {incompatible_keys}")
    else:
        print(f"✅Successfully loading {arch} from {weight_path}")
    
    return model.to(device)


def build_transform(image_shape=(224, 224)):
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),
        tfs.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def get_base_dim(image_shape=(224, 224), *args, **kwargs):
    raw_dim = 768  # right after the encoder
    latent_dim = 768  # latent
    feature_dim = 512  # the projected feature
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
    model = build_model()
    x = torch.randn((2, 3, 224, 224)).float()
    with torch.no_grad():
        y = model(x,)
    print([f"{k}: {v.shape if hasattr(v, 'shape') else None}" for k, v in y.items()])
