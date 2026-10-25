import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
import math
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision.transforms import v2 as tfs
from torch.utils.checkpoint import checkpoint
from local_constants import RETCLIP_CKPT


MODEL_CONFIGS = {
    "ViT-B-16": {
        "embed_dim": 512,
        "image_resolution": 224,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "vision_head_width": 64,
    },
    "ViT-L-14": {
        "embed_dim": 768,
        "image_resolution": 224,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_patch_size": 14,
        "vision_head_width": 64,
    },
    "ViT-L-14-336": {
        "embed_dim": 768,
        "image_resolution": 336,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_patch_size": 14,
        "vision_head_width": 64,
    },
    "ViT-H-14": {
        "embed_dim": 1024,
        "image_resolution": 224,
        "vision_layers": 32,
        "vision_width": 1280,
        "vision_patch_size": 14,
        "vision_head_width": 80,
    },
}



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    """
    Vision Transformer (ViT) for RET-CLIP.

    Args:
        input_resolution: Input image resolution (assumes square images)
        patch_size: Size of each patch
        width: Transformer width (hidden dimension)
        layers: Number of transformer layers
        heads: Number of attention heads
        output_dim: Output embedding dimension (after projection)
    """
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        self.grid_size = (input_resolution // patch_size, input_resolution // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Patch embedding via convolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor, do_proj=False) -> dict:
        """
        Forward pass of the visual transformer.

        Args:
            x: Input images of shape [B, 3, H, W]

        Returns:
            Dictionary containing:
                - cls_token: Projected CLS token [B, output_dim]
                - patch_tokens: Patch tokens after ln_post [B, num_patches, width]
                - raw_cls: CLS token before projection [B, width]
                - features: Full sequence after transformer [B, num_patches + 1, width]
        """
        # Patch embedding
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
        x = x.permute(0, 2, 1)  # [B, grid**2, width]

        # Prepend CLS token
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # [B, grid**2 + 1, width]

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Pre-LayerNorm
        x = self.ln_pre(x)

        # Transformer (expects LND format)
        x = x.permute(1, 0, 2)  # [L, B, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, width]

        # Post-LayerNorm for CLS token (original behavior)
        cls_token = self.ln_post(x[:, 0, :])  # [B, width]
        
        # Post-LayerNorm for patch tokens
        patch_tokens = self.ln_post(x[:, 1:, :])  # [B, num_patches, width]

        # Project CLS token
        proj = cls_token @ self.proj if do_proj else None

        return {
            'cls_token': cls_token,        # [B, output_dim] - projected CLS token
            'patch_tokens': patch_tokens,  # [B, num_patches, width] - patch features
            'proj': proj,            # [B, width] - CLS before projection
        }

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to get only the projected CLS token.
        This matches the original forward behavior for backward compatibility.

        Args:
            x: Input images of shape [B, 3, H, W]

        Returns:
            Projected CLS token of shape [B, output_dim]
        """
        return self.forward(x)['cls_token']



def build_model(
    arch: str = "retclip-vision-b16",
    image_shape: tuple = (224, 224),
    device: str = "cpu",
    pretrained_path: str = None,
) -> nn.Module:
    """
    Build and load the RET-CLIP vision encoder.

    Args:
        arch: Architecture name (default: "retclip-vision")
        image_shape: Input image shape as (H, W) tuple
        device: Device to load the model on ("cpu" or "cuda")
        pretrained_path: Path to pretrained weights. If None, uses default path.
        vision_model: Vision model architecture name (default: "ViT-B-16")

    Returns:
        Loaded VisualTransformer model
    """
    pretrained_path = RETCLIP_CKPT if not pretrained_path else pretrained_path
        
    if not isinstance(image_shape, (list, tuple)):
        image_shape = (image_shape, image_shape)

    assert len(image_shape) == 2, f"Expected (H, W) tuple for image_shape, got {image_shape}"

    # Get model config
    vision_model = "ViT-B-16" if "b16" in arch else "ViT-L-14"
    if vision_model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown vision model: {vision_model}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[vision_model]

    # For now, we use the model's default resolution
    # Image shape parameter is kept for API compatibility
    input_resolution = config["image_resolution"]

    # Calculate number of attention heads
    vision_heads = config["vision_width"] // config["vision_head_width"]

    # Build model
    model = VisualTransformer(
        input_resolution=input_resolution,
        patch_size=config["vision_patch_size"],
        width=config["vision_width"],
        layers=config["vision_layers"],
        heads=vision_heads,
        output_dim=config["embed_dim"],
    )

    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove "module." prefix if present (from DDP training)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Extract visual encoder weights
        visual_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("visual."):
                new_key = k[7:]  # Remove "visual." prefix
                visual_state_dict[new_key] = v

        # Load weights
        model.load_state_dict(visual_state_dict, strict=True)
        print(f"Successfully loaded {len(visual_state_dict)} parameters with strict=True")

        # Match original model loading behavior: FP16 roundtrip for consistent precision
        # The original code does: create model -> convert_weights (to FP16) -> load_state_dict
        # This ensures identical precision characteristics when comparing outputs
        #model = model.half().float()
        model = model.float()
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}")

    return model.to(device)


def build_transform(image_shape: tuple = (224, 224)):
    """
    Build the default image transform for RET-CLIP.
    Matches the original RET_CLIP/clip/utils.py image_transform() exactly.

    Args:
        image_shape: Target image shape as (H, W) tuple

    Returns:
        Composed transform pipeline
    """

    if not isinstance(image_shape, (list, tuple)):
        image_shape = (image_shape, image_shape)

    # Match original transform: PIL resize -> RGB -> ToTensor -> Normalize
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),
        tfs.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])



def get_base_dim(
    image_shape: tuple = (224, 224),
    arch: str = "retclip-vision-b16",
    vision_model: str = "ViT-B-16",
    **kwargs
) -> dict:
    """
    Get dimension and shape information for the model output features.

    Args:
        image_shape: Input image shape as (H, W) tuple
        vision_model: Vision model architecture name

    Returns:
        Dictionary containing:
            - raw_dim: Feature dimension before projection (vision_width)
            - latent_dim: Same as raw_dim for ViT
            - feature_dim: Output feature dimension (embed_dim after projection)
            - raw_shape: Spatial shape of patch features
            - latent_shape: Same as raw_shape for ViT
            - feature_shape: Same as raw_shape for ViT
            - num_patches: Number of patch tokens
    """
    vision_model = "ViT-B-16" if "b16" in arch else "ViT-B-32"
    if vision_model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown vision model: {vision_model}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[vision_model]

    if not isinstance(image_shape, (list, tuple)):
        image_shape = (image_shape, image_shape)

    # Calculate grid size
    grid_h = config["image_resolution"] // config["vision_patch_size"]
    grid_w = config["image_resolution"] // config["vision_patch_size"]

    return {
        "raw_dim": config["vision_width"],       # 768 for ViT-B-16
        "latent_dim": config["vision_width"],    # 768 for ViT-B-16
        "feature_dim": config["embed_dim"],      # 512 for ViT-B-16 (after projection)
        "raw_shape": (grid_h, grid_w),           # (14, 14) for ViT-B-16 with 224x224 input
        "latent_shape": (grid_h, grid_w),
        "feature_shape": (grid_h, grid_w),
    }


if __name__ == "__main__":
    # Quick test
    print("Building model...")
    model = build_model(device="cpu")
    print(f"Model built: {type(model)}")

    transform = build_transform()
    print(f"Transform built: {transform}")

    dim_info = get_base_dim()
    print(f"Dimension info: {dim_info}")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)

    print(f"\nOutput keys: {output.keys()}")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
