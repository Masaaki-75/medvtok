import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
from typing import Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torchvision.transforms import v2 as tfs
from torchvision import models as models_2d
from local_constants import PESKAVLP_CKPT


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer."""

    def forward(self, x):
        return x


def _create_resnet50(pretrained='imagenet'):
    """
    Create a ResNet-50 backbone with the FC layer replaced by Identity.

    Args:
        pretrained: 'imagenet' for pretrained weights, 'random' for random init

    Returns:
        model: ResNet-50 model with Identity FC
        feature_dims: Output feature dimension (2048 for ResNet-50)
        interm_feature_dim: Intermediate feature dimension (1024)
    """
    if pretrained == 'imagenet':
        model = models_2d.resnet50(weights=models_2d.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models_2d.resnet50(weights=None)

    feature_dims = model.fc.in_features  # 2048
    model.fc = Identity()
    return model, feature_dims, 1024



class ImageEncoder(nn.Module):
    """
    PeskaVLP Vision Encoder.

    Uses ResNet-50 as backbone with a global embedding layer to project
    features to the target dimension.

    Args:
        num_classes: Output embedding dimension (default: 768)
        pretrained: Pretrained weights option ('imagenet' or 'random')
        backbone_name: Backbone architecture (only 'resnet_50' supported)
        img_norm: Whether to L2-normalize the output (default: False)
    """

    def __init__(
        self,
        num_classes: int = 768,
        pretrained: str = None,
        backbone_name: str = 'resnet_50',
        img_norm: bool = False
    ):
        super().__init__()

        if backbone_name != 'resnet_50':
            raise ValueError(f"Only 'resnet_50' backbone is supported, got {backbone_name}")

        self.model, self.feature_dim, self.interm_feature_dim = _create_resnet50(pretrained=pretrained)
        self.global_embedder = nn.Linear(self.feature_dim, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.norm = img_norm

    @property
    def dtype(self):
        return self.model.conv1.weight.dtype

    @property
    def device(self):
        return self.model.conv1.weight.device

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Global embedding tensor of shape [B, num_classes]
        """
        global_ft, local_ft = self._resnet_forward(x)
        global_emb = self.global_embedder(global_ft)

        if self.norm:
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return {
            'cls_token': global_ft,
            'patch_tokens': local_ft,
            'proj': global_emb,
        }

    def _resnet_forward(self, x: torch.Tensor):
        """
        ResNet forward pass extracting global and local features.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            global_features: Pooled features [B, 2048]
            local_features: Feature map before layer4 [B, 1024, H', W']
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        local_features = x

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features


def build_transform(image_shape: Tuple[int, int] = (224, 224)):
    """
    Build the image transform pipeline for PeskaVLP.

    Args:
        image_shape: Target image size (height, width). Default: (224, 224)

    Returns:
        torchvision transform pipeline
    """
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),  # bicubic
        tfs.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def build_model(
    arch: str = 'peskavlp-vision',
    image_shape: Tuple[int, int] = (224, 224),
    device: Union[str, torch.device] = "cpu",
    pretrained_path: str = None,
    *args,
    **kwargs
) -> nn.Module:
    """
    Build PeskaVLP vision encoder model.

    Args:
        arch: Architecture name (only 'peskavlp-vision' supported)
        image_shape: Input image shape (not used, kept for API compatibility)
        device: Device to load model on
        pretrained_path: Path to pretrained weights. If None, uses PESKAVLP_CKPT

    Returns:
        ImageEncoder model loaded with pretrained weights
    """
    pretrained_path = PESKAVLP_CKPT if not pretrained_path else pretrained_path

    model = ImageEncoder(
        num_classes=768,
        pretrained=None,
        backbone_name='resnet_50',
        img_norm=False
    )

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Filter and remap backbone_img weights
    # The full PeskaVLP checkpoint contains both backbone_img.* and backbone_text.* keys
    # We only need backbone_img.* and need to remove the prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone_img.'):
            new_key = key.replace('backbone_img.', '')
            new_state_dict[new_key] = value

    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=True)

    return model.to(device)


def get_base_dim(image_shape: Tuple[int, int] = (224, 224), *args, **kwargs) -> Dict[str, Any]:
    """
    Get base dimension information for PeskaVLP vision encoder.

    Args:
        image_shape: Input image shape

    Returns:
        Dictionary containing dimension information:
        - raw_dim: ResNet-50 output dimension (2048)
        - latent_dim: After global_embedder dimension (768)
        - feature_dim: Final output dimension (768)
        - raw_shape: Feature map shape after ResNet
        - latent_shape: Same as raw_shape
        - feature_shape: Same as raw_shape
    """
    raw_dim = 2048
    latent_dim = 768  # After global_embedder
    feature_dim = 768  # Final output
    ds_factor = 32  # Downsampling factor for ResNet-50

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
    # Test the builder
    model = build_model()
    print(f"Model dtype: {model.dtype}, device: {model.device}")

    x = torch.randn((2, 3, 224, 224)).float()
    with torch.no_grad():
        y = model(x)
    
    if isinstance(y, dict):
        for k, v in y.items():
            print(f"Shape for {k}: {v.shape if hasattr(v, 'shape') else len(v)}")

    dims = get_base_dim()
    print(f"Dimensions: {dims}")
