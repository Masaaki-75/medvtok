"""
Simplified SPECTRE top-level model wrapper.

Combines a backbone VisionTransformer with a FeatureVisionTransformer combiner
to extract and aggregate multi-patch CT scan features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectreImageFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_combiner: nn.Module,
        projection_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_combiner = feature_combiner
        self.projection_head = projection_head

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C, H, W, D) — B batches of N 3D patches.
        Returns:
            (B, N, T, F) — per-patch token features.
        """
        assert x.ndim == 6, "Input must be (B, N, C, H, W, D)"
        B, N, C, H, W, D = x.shape
        x = x.view(B * N, C, H, W, D)
        features = self.backbone(x)
        if features.ndim == 2:  # only CLS token
            features = features.unsqueeze(1)
        features = features.view(B, N, features.shape[1], -1)
        return features

    def combine_features(
        self,
        features: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            features: (B, N, T, F) backbone features.
            grid_size: 3D grid dimensions, product must equal N.
        Returns:
            (B, T', F') combined scan-level features.
        """
        _, N, T, _ = features.shape
        assert N == grid_size[0] * grid_size[1] * grid_size[2]

        if T == 1:
            features = features.squeeze(2)  # (B, N, F)
        else:
            # CLS token + average-pooled patch tokens → 2F dim
            features = torch.cat([
                features[:, :, 0, :],               # CLS: (B, N, F)
                features[:, :, 1:, :].mean(dim=2),  # AVG:  (B, N, F)
            ], dim=-1)  # (B, N, 2F)

        features = self.feature_combiner(features, grid_size)
        return features

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int] | None = None,
        return_proj_only: bool = True,
        do_normalize: bool = True,
        do_proj: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:

        patch_tokens = self.extract_backbone_features(x)
        outputs = {}
        if not return_proj_only:
            outputs["patch_tokens"] = patch_tokens

        assert grid_size is not None, "grid_size required when using feature combiner"
        features = self.combine_features(patch_tokens, grid_size)
        if not return_proj_only:
            outputs["features"] = features

        proj = torch.cat([features[:, 0, :], features[:, 1:, :].mean(1)], dim=1)
        proj = self.projection_head(proj) if do_proj else proj
        outputs["cls_token"] = F.normalize(proj, p=2, dim=-1) if do_normalize else proj

        return outputs
