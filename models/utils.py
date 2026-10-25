from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange



def get_batch_slice(batch_idx: int, seqlens: List[int]) -> slice:
    """
    Get slice for extracting batch_idx from concatenated tensor.

    Args:
        batch_idx: Index of the batch to extract
        seqlens: List of sequence lengths for each batch

    Returns:
        slice object for indexing

    Example:
        >>> seqlens = [256, 1024, 576]
        >>> get_batch_slice(1, seqlens)  # Returns slice(256, 1280)
    """
    start = sum(seqlens[:batch_idx])
    end = start + seqlens[batch_idx]
    return slice(start, end)


def process_single_image(
    image: torch.Tensor,
    patch_size: Tuple[int, int, int],
    normalize: bool = True,
    padding_mode: str = 'zero',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single image/video and convert to patches.

    Args:
        image: [T, H, W, C] tensor (values in [0, 255] or [-1, 1])
        patch_size: (Pt, Ph, Pw) patch dimensions
        normalize: Whether to normalize to [-1, 1]

    Returns:
        feats: [num_patches, patch_dim] patch features
        coords: [num_patches, 4] coordinates [t, x, y, z]
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]

    T, H, W, C = image.shape
    Pt, Ph, Pw = patch_size

    # Ensure T is divisible by Pt (pad if needed)
    if T % Pt != 0:
        target_T = ((T + Pt - 1) // Pt) * Pt  # Round up to nearest multiple
        padding = target_T - T

        if padding_mode == "zero":
            # Pad with zeros on temporal dimension
            pad_tensor = torch.zeros(
                padding, H, W, C,
                dtype=image.dtype,
                device=image.device
            )
        elif padding_mode == "replicate":
            # Final frame
            pad_tensor = image[-1:].expand(padding, -1, -1, -1)
        else:
            raise ValueError(f"Unsupported pad_mode: {padding_mode}")

        image = torch.cat([image, pad_tensor], dim=0)
        T = target_T

    # Make H, W divisible by patch_size
    target_H = (H // Ph) * Ph
    target_W = (W // Pw) * Pw

    # Resize if needed
    if target_H != H or target_W != W:
        image = image.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        image = F.interpolate(
            image.float(),
            size=(target_H, target_W),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

    T, H, W, C = image.shape

    # Verify dimensions
    assert H % Ph == 0, f"Height {H} not divisible by patch size {Ph}"
    assert W % Pw == 0, f"Width {W} not divisible by patch size {Pw}"
    assert T % Pt == 0, f"Time {T} not divisible by patch size {Pt}"

    # Normalize if needed
    if normalize:
        if image.min() >= 0:
            image = image.float() / 127.5 - 1.0

    # Patchify
    feats = rearrange(
        image,
        "(nt pt) (nh ph) (nw pw) c -> nt nh nw (pt ph pw c)",
        # (1 4) (H/Ph Ph) (W/Pw Pw) 3 -> 1 H/Ph W/Pw (Ph*Pw*4*3)
        nt=T // Pt,
        nh=H // Ph,
        nw=W // Pw,
        pt=Pt,
        ph=Ph,
        pw=Pw,
    )
    feats = rearrange(feats, "t h w c -> (t h w) c")  

    # Generate coordinates
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(T // Pt, device=image.device),
            torch.arange(H // Ph, device=image.device),
            torch.arange(W // Pw, device=image.device),
            torch.arange(1, device=image.device),  # z always 0
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 4).int()

    return feats, coords


def images_to_concatenated(
    images: List[torch.Tensor],
    patch_size: Tuple[int, int, int],
    normalize: bool = True,
    padding_mode: str = 'zero',
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Convert list of images/videos with arbitrary resolutions to concatenated tensors.

    This is the ZERO-PADDING version that concatenates all patches without padding.
    Memory footprint: O(N) where N = sum of patches across all images.

    Args:
        images: List of tensors, each can be:
                - [H, W, C] - single image
                - [T, H, W, C] - video or image sequence
                - [C, H, W] - single image (channel-first)
                - [T, C, H, W] - video (channel-first)
                Different images can have different resolutions.
        patch_size: (Pt, Ph, Pw) patch dimensions
        normalize: Whether to normalize to [-1, 1] (default: True)

    Returns:
        feats: [N, patch_dim] concatenated patch features (NO PADDING)
        coords: [N, 5] concatenated coordinates [batch_idx, t, x, y, z] (NO PADDING)
        seqlens: List[int] - number of patches per image

    Example:
        >>> img1 = torch.randn(256, 256, 3)  # Small image -> 256 patches
        >>> img2 = torch.randn(512, 512, 3)  # Large image -> 1024 patches
        >>> feats, coords, seqlens = images_to_concatenated(
        ...     [img1, img2],
        ...     patch_size=(4, 16, 16)
        ... )
        >>> feats.shape  # [1280, 3072] - 256+1024 = 1280 (NO PADDING!)
        >>> seqlens  # [256, 1024] - patches per image
    """
    if not images:
        raise ValueError("images list cannot be empty")

    device = images[0].device
    dtype = images[0].dtype

    all_feats = []
    all_coords = []
    seqlens = []
    data_types = []
    orig_shapes = []

    # Process each image independently
    for batch_idx, image in enumerate(images):
        # Normalize shape to [T, H, W, C]
        if image.dim() == 3:
            if image.shape[0] == 3 or image.shape[0] == 1:
                # [C, H, W] -> [H, W, C]
                image = image.permute(1, 2, 0)
            # [H, W, C] -> [1, H, W, C]
            image = image.unsqueeze(0)
            
            data_types.append('image')
        elif image.dim() == 4:
            if image.shape[1] == 3 and image.shape[1] < min(image.shape[2:]):
                # [T, C, H, W] -> [T, H, W, C]
                image = image.permute(0, 2, 3, 1)
            elif image.shape[0] == 3 and image.shape[-1] != 3:
                # [C, T, H, W] -> [T, H, W, C]
                image = image.permute(1, 2, 3, 0)
            # else already [T, H, W, C]
            
            if image.shape[0] == 1:
                data_types.append('image')
            else:
                data_types.append('volume')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        orig_shapes.append((image.shape[0], image.shape[1], image.shape[2]))
        
        # Process to patches
        feats, coords = process_single_image(
            image,
            patch_size,
            normalize=normalize,
            padding_mode=padding_mode
        )

        # Add batch index to coords
        batch_indices = torch.full(
            (coords.shape[0], 1),
            batch_idx,
            dtype=torch.int32,
            device=device,
        )
        coords = torch.cat([batch_indices, coords], dim=1)  # [num_patches, 5]

        all_feats.append(feats)
        all_coords.append(coords)
        seqlens.append(len(feats))

    # Concatenate WITHOUT padding - this is the key difference!
    feats_cat = torch.cat(all_feats, dim=0)  # [N, patch_dim] where N = sum(seqlens)
    coords_cat = torch.cat(all_coords, dim=0)  # [N, 5]

    return feats_cat, coords_cat, seqlens, data_types, orig_shapes


def concatenated_to_img_list(
    feats: torch.Tensor,
    coords: torch.Tensor,
    seqlens: List[int],
    patch_size: Tuple[int, int, int],
) -> List[torch.Tensor]:
    """
    Convert concatenated tensors back to list of image tensors.

    Args:
        feats: [N, patch_dim] concatenated features (N = sum(seqlens))
        coords: [N, 5] concatenated coordinates
        seqlens: List[int] - patches per image
        patch_size: (Pt, Ph, Pw) patch dimensions

    Returns:
        List of tensors, each [T, C, H, W] or [C, H, W]
    """
    Pt, Ph, Pw = patch_size
    batch_size = len(seqlens)

    images = []

    # Precompute cumulative seqlens for O(1) slice calculation
    cum_seqlens = [0]
    for s in seqlens:
        cum_seqlens.append(cum_seqlens[-1] + s)

    for batch_idx in range(batch_size):
        # Extract patches for this batch using precomputed cumsum
        start = cum_seqlens[batch_idx]
        end = cum_seqlens[batch_idx + 1]
        batch_feats = feats[start:end]  # [seqlens[batch_idx], patch_dim]
        batch_coords = coords[start:end]  # [seqlens[batch_idx], 5]

        if batch_feats.shape[0] == 0:
            # Empty batch
            images.append(torch.zeros(1, 3, Ph, Pw, device=feats.device, dtype=feats.dtype))
            continue

        # Get spatial dimensions from coordinates (single GPU max operation)
        max_coords = batch_coords.max(dim=0)[0]
        # Use index_select to avoid .item() GPU->CPU sync where possible
        # But we need these values for tensor creation, so we batch the .item() calls
        nt = int(max_coords[1].item()) + 1  # t coordinate
        nh = int(max_coords[2].item()) + 1  # x coordinate
        nw = int(max_coords[3].item()) + 1  # y coordinate

        # Determine patch dimension and number of channels
        patch_dim = batch_feats.shape[1]
        C = patch_dim // (Pt * Ph * Pw)

        # Create feature grid
        feature_grid = torch.zeros(
            nt, nh, nw, patch_dim,
            device=feats.device,
            dtype=feats.dtype,
        )

        # VECTORIZED scatter: compute linear indices on GPU, no per-element .item() calls
        t_indices = batch_coords[:, 1].long()
        x_indices = batch_coords[:, 2].long()
        y_indices = batch_coords[:, 3].long()

        # Compute linear index: t * (nh * nw) + x * nw + y
        linear_indices = t_indices * (nh * nw) + x_indices * nw + y_indices

        # Reshape grid and scatter all patches in one operation
        feature_grid_flat = feature_grid.view(-1, patch_dim)
        feature_grid_flat[linear_indices] = batch_feats

        # Reverse patchify
        image = rearrange(
            feature_grid,
            "nt nh nw (pt ph pw c) -> (nt pt) c (nh ph) (nw pw)",
            pt=Pt,
            ph=Ph,
            pw=Pw,
            c=C,
        )

        images.append(image)

    return images



def concatenated_to_feature_maps(
    feats: torch.Tensor,
    coords: torch.Tensor,
    seqlens: List[int],
) -> List[torch.Tensor]:
    """
    Convert concatenated encoder features to spatial feature maps.

    Unlike concatenated_to_img_list which unpatchifies pixel data, this function
    places feature vectors directly at their spatial locations WITHOUT unpatchifying.

    Use this to reshape enc_full [N, model_channels] back to spatial format.

    Args:
        feats: [N, feat_dim] concatenated features (N = sum(seqlens))
               feat_dim can be any size (e.g., 1152 for enc_full, 96 for enc_feats)
        coords: [N, 5] concatenated coordinates [batch_idx, t, x, y, z]
        seqlens: List[int] - patches per image

    Returns:
        List of tensors, each [T, Nh, Nw, feat_dim] spatial feature maps
        where Nh = number of patches in height, Nw = number of patches in width

    Example:
        >>> enc_feats, enc_coords, enc_seqlens, image_feat, enc_full = model.encode(...)
        >>> # enc_full has shape [N, 1152]
        >>> feature_maps = concatenated_to_feature_maps(enc_full, enc_coords, enc_seqlens)
        >>> # feature_maps[0] has shape [T, Nh, Nw, 1152]
    """
    batch_size = len(seqlens)
    feat_dim = feats.shape[1]

    feature_maps = []

    # Precompute cumulative seqlens for O(1) slice calculation
    cum_seqlens = [0]
    for s in seqlens:
        cum_seqlens.append(cum_seqlens[-1] + s)

    for batch_idx in range(batch_size):
        # Extract features for this batch
        start = cum_seqlens[batch_idx]
        end = cum_seqlens[batch_idx + 1]
        batch_feats = feats[start:end]  # [seqlens[batch_idx], feat_dim]
        batch_coords = coords[start:end]  # [seqlens[batch_idx], 5]

        if batch_feats.shape[0] == 0:
            # Empty batch
            feature_maps.append(torch.zeros(1, 1, 1, feat_dim, device=feats.device, dtype=feats.dtype))
            continue

        # Get spatial dimensions from coordinates
        max_coords = batch_coords.max(dim=0)[0]
        nt = int(max_coords[1].item()) + 1  # t coordinate
        nh = int(max_coords[2].item()) + 1  # x/height coordinate
        nw = int(max_coords[3].item()) + 1  # y/width coordinate

        # Create feature grid [T, Nh, Nw, feat_dim]
        feature_grid = torch.zeros(
            nt, nh, nw, feat_dim,
            device=feats.device,
            dtype=feats.dtype,
        )

        # VECTORIZED scatter using linear indices
        t_indices = batch_coords[:, 1].long()
        x_indices = batch_coords[:, 2].long()
        y_indices = batch_coords[:, 3].long()

        # Compute linear index: t * (nh * nw) + x * nw + y
        linear_indices = t_indices * (nh * nw) + x_indices * nw + y_indices

        # Scatter all features in one operation
        feature_grid_flat = feature_grid.view(-1, feat_dim)
        feature_grid_flat[linear_indices] = batch_feats

        feature_maps.append(feature_grid)

    return feature_maps




def batch_tensor_to_dense(
    batch_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    normalize: bool = True,
    padding_mode: str = 'zero',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert batch of images/videos to dense padded tensors.

    Args:
        batch_tensor: [B, C, H, W] or [B, T, H, W, C] or [B, T, C, H, W]
        patch_size: (Pt, Ph, Pw) patch dimensions

    Returns:
        feats: [B, max_patches, patch_dim]
        coords: [B, max_patches, 5] with [batch_idx, t, x, y, z]
        mask: [B, max_patches] boolean mask
    """
    device = batch_tensor.device
    dtype = batch_tensor.dtype

    # Normalize input shape to [B, T, H, W, C]
    if batch_tensor.dim() == 4:
        # [B, C, H, W] -> [B, T, H, W, C]
        batch_tensor = rearrange(batch_tensor, "b c h w -> b h w c")
        batch_tensor = batch_tensor.unsqueeze(1)
    elif batch_tensor.dim() == 5:
        if batch_tensor.shape[2] == 3 and batch_tensor.shape[2] < min(batch_tensor.shape[3:]):
            # [B, T, C, H, W] -> [B, T, H, W, C]
            batch_tensor = rearrange(batch_tensor, "b t c h w -> b t h w c")
        elif batch_tensor.shape[1] == 3 and batch_tensor.shape[-1] != 3:
            # [B, C, T, H, W] -> [B, T, H, W, C]
            batch_tensor = rearrange(batch_tensor, "b c t h w -> b t h w c")
    else:
        raise ValueError(f"Unsupported input shape: {batch_tensor.shape}")

    # Process each item in batch
    all_feats = []
    all_coords = []
    max_patches = 0

    B = batch_tensor.shape[0]

    for batch_idx in range(B):
        feats, coords = process_single_image(
            batch_tensor[batch_idx],
            patch_size,
            normalize=normalize,
            padding_mode=padding_mode,
        )

        # Add batch index to coords
        batch_indices = torch.full(
            (coords.shape[0], 1),
            batch_idx,
            dtype=torch.int32,
            device=device,
        )
        coords = torch.cat([batch_indices, coords], dim=1)  # [num_patches, 5]

        all_feats.append(feats)
        all_coords.append(coords)
        max_patches = max(max_patches, feats.shape[0])

    # Pad all to max_patches
    padded_feats = []
    padded_coords = []
    masks = []

    patch_dim = all_feats[0].shape[1]

    for feats, coords in zip(all_feats, all_coords):
        num_patches = feats.shape[0]

        # Create mask
        mask = torch.zeros(max_patches, dtype=torch.bool, device=device)
        mask[:num_patches] = True
        masks.append(mask)

        # Pad features
        if num_patches < max_patches:
            padding = torch.zeros(
                max_patches - num_patches,
                patch_dim,
                dtype=dtype,
                device=device,
            )
            feats = torch.cat([feats, padding], dim=0)

        # Pad coords
        if num_patches < max_patches:
            padding = torch.zeros(
                max_patches - num_patches,
                5,
                dtype=torch.int32,
                device=device,
            )
            coords = torch.cat([coords, padding], dim=0)

        padded_feats.append(feats)
        padded_coords.append(coords)

    # Stack to create batch tensors
    feats_tensor = torch.stack(padded_feats, dim=0)  # [B, max_patches, patch_dim]
    coords_tensor = torch.stack(padded_coords, dim=0)  # [B, max_patches, 5]
    mask_tensor = torch.stack(masks, dim=0)  # [B, max_patches]

    return feats_tensor, coords_tensor, mask_tensor





def seqlens_to_dense(
    feats: torch.Tensor,
    coords: torch.Tensor,
    seqlens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert concatenated format to dense padded format for attention modules.

    Args:
        feats: [N, C] concatenated features
        coords: [N, 5] concatenated coordinates
        seqlens: List[int] patches per image

    Returns:
        feats_dense: [B, max_L, C] padded features
        coords_dense: [B, max_L, 5] padded coordinates
        mask: [B, max_L] boolean mask
    """
    B = len(seqlens)
    max_L = max(seqlens) if seqlens else 0
    C = feats.shape[1]
    device = feats.device
    dtype = feats.dtype

    if max_L == 0 or B == 0:
        return (
            torch.zeros(B, max_L, C, dtype=dtype, device=device),
            torch.zeros(B, max_L, 5, dtype=torch.int32, device=device),
            torch.zeros(B, max_L, dtype=torch.bool, device=device),
        )

    feats_dense = torch.zeros(B, max_L, C, dtype=dtype, device=device)
    coords_dense = torch.zeros(B, max_L, 5, dtype=torch.int32, device=device)
    mask = torch.zeros(B, max_L, dtype=torch.bool, device=device)

    # Vectorized scatter: compute indices on GPU
    seqlens_t = torch.tensor(seqlens, device=device, dtype=torch.long)
    batch_indices = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.long), seqlens_t
    )
    # Position within each batch: [0,1,...,s0-1, 0,1,...,s1-1, ...]
    position_indices = torch.cat([
        torch.arange(s, device=device, dtype=torch.long) for s in seqlens
    ])

    # Single vectorized scatter for all data
    feats_dense[batch_indices, position_indices] = feats
    coords_dense[batch_indices, position_indices] = coords
    mask[batch_indices, position_indices] = True

    return feats_dense, coords_dense, mask




def batch_tensor_to_concatenated(
    batch_tensor: torch.Tensor,
    patch_size: Tuple[int, int, int],
    normalize: bool = True,
    padding_mode: str = 'zero',
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Directly convert batch of images/videos to concatenated format (no dense padding).

    Args:
        batch_tensor: [B, C, H, W] or [B, T, H, W, C] or [B, T, C, H, W]
        patch_size: (Pt, Ph, Pw)

    Returns:
        feats:  [N, patch_dim] concatenated features
        coords: [N, 5] concatenated coords with [batch_idx, t, x, y, z]
        seqlens: List[int], number of patches per item in batch
    """
    device = batch_tensor.device
    feats_dtype = batch_tensor.dtype
    data_types = []
    orig_shapes = []
    batch_size = batch_tensor.shape[0]

    # Normalize input shape to [B, T, H, W, C]
    if batch_tensor.dim() == 4:
        # [B, C, H, W] -> [B, 1, H, W, C]
        batch_tensor = rearrange(batch_tensor, "b c h w -> b h w c").unsqueeze(1)
        data_types = ['image',] * batch_size
    elif batch_tensor.dim() == 5:
        if batch_tensor.shape[2] == 3 and batch_tensor.shape[2] < min(batch_tensor.shape[3:]):
            # [B, T, C, H, W] -> [B, T, H, W, C]
            batch_tensor = rearrange(batch_tensor, "b t c h w -> b t h w c")
        elif batch_tensor.shape[1] == 3 and batch_tensor.shape[-1] != 3:
            # [B, C, T, H, W] -> [B, T, H, W, C]
            batch_tensor = rearrange(batch_tensor, "b c t h w -> b t h w c")
        # else assume already [B, T, H, W, C]
        data_types = ['volume',] * batch_size
    else:
        raise ValueError(f"Unsupported input shape: {tuple(batch_tensor.shape)}")

    orig_shapes = [(batch_tensor.shape[1], batch_tensor.shape[2], batch_tensor.shape[3]),] * batch_size
    B = batch_tensor.shape[0]
    feats_list: List[torch.Tensor] = []
    coords_list: List[torch.Tensor] = []
    seqlens: List[int] = []

    for batch_idx in range(B):
        feats_i, coords_i = process_single_image(
            batch_tensor[batch_idx],
            patch_size,
            normalize=normalize,
            padding_mode=padding_mode,
        )  # feats_i: [Li, patch_dim], coords_i: [Li, 4] (t,x,y,z) or similar

        Li = int(feats_i.shape[0])
        seqlens.append(Li)

        if Li == 0:
            continue

        # Ensure dtypes/devices
        feats_i = feats_i.to(device=device, dtype=feats_dtype)

        # prepend batch index -> [Li, 5]
        batch_col = torch.full(
            (Li, 1),
            batch_idx,
            dtype=torch.int32,
            device=device,
        )
        coords_i = coords_i.to(device=device)
        if coords_i.dtype != torch.int32:
            coords_i = coords_i.to(torch.int32)
        coords_i = torch.cat([batch_col, coords_i], dim=1)

        feats_list.append(feats_i)
        coords_list.append(coords_i)

    if len(feats_list) == 0:
        patch_dim = 0
        feats = torch.zeros((0, patch_dim), dtype=feats_dtype, device=device)
        coords = torch.zeros((0, 5), dtype=torch.int32, device=device)
        return feats, coords, seqlens

    feats = torch.cat(feats_list, dim=0)   # [N, patch_dim]
    coords = torch.cat(coords_list, dim=0) # [N, 5]
    return feats, coords, seqlens, data_types, orig_shapes




def get_temporal_range(coords: torch.Tensor, seqlens: List[int]) -> Tuple[int, int]:
    """
    Get min and max temporal coordinates across all valid positions.

    Args:
        coords: [N, 5] coordinate tensor [batch_idx, t, x, y, z]
        seqlens: List[int] - patches per image

    Returns:
        (min_t, max_t) tuple
    """
    if coords.shape[0] == 0:
        return 0, 0

    temporal_coords = coords[:, 1]  # [N] - t dimension
    return temporal_coords.min().item(), temporal_coords.max().item()


def split_by_temporal_batches(
    feats: torch.Tensor,
    coords: torch.Tensor,
    seqlens: List[int],
    frame_batch_size: int,
    frame_batch_strides: Optional[int] = None,
    adjust_temporal: bool = True,
    offset: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor, List[int], int]]:
    """
    Split concatenated tensors into temporal windows with sliding support.

    Args:
        feats: [N, C] concatenated feature tensor
        coords: [N, 5] concatenated coordinate tensor
        seqlens: List[int] - patches per image
        frame_batch_size: Window size in frames
        frame_batch_strides: Stride between windows (default: frame_batch_size)
        adjust_temporal: Whether to adjust temporal coords to start from 0
        offset: Temporal offset to apply

    Returns:
        List of (feats, coords, seqlens, temporal_offset) tuples
    """
    if frame_batch_strides is None:
        frame_batch_strides = frame_batch_size

    min_frame, max_frame = get_temporal_range(coords, seqlens)

    slices = []
    current_start = min_frame

    while current_start <= max_frame:
        current_end = current_start + frame_batch_size

        # Extract temporal slice per batch
        slice_feats_list = []
        slice_coords_list = []
        slice_seqlens = []

        for batch_idx, seqlen in enumerate(seqlens):
            batch_slice = get_batch_slice(batch_idx, seqlens)
            batch_feats = feats[batch_slice]
            batch_coords = coords[batch_slice]

            # Filter by temporal range
            temporal_coords = batch_coords[:, 1]
            temporal_mask = (temporal_coords >= current_start) & (temporal_coords < current_end)

            if temporal_mask.any():
                filtered_feats = batch_feats[temporal_mask]
                filtered_coords = batch_coords[temporal_mask]

                # Adjust coordinates if requested
                if adjust_temporal:
                    filtered_coords = filtered_coords.clone()
                    filtered_coords[:, 1] = filtered_coords[:, 1] - current_start + offset

                slice_feats_list.append(filtered_feats)
                slice_coords_list.append(filtered_coords)
                slice_seqlens.append(len(filtered_feats))
            else:
                # Empty slice for this batch - keep empty placeholder
                slice_feats_list.append(torch.zeros(0, feats.shape[1], dtype=feats.dtype, device=feats.device))
                slice_coords_list.append(torch.zeros(0, 5, dtype=coords.dtype, device=coords.device))
                slice_seqlens.append(0)

        # Concatenate all batches
        if sum(slice_seqlens) > 0:
            slice_feats = torch.cat([f for f in slice_feats_list if f.shape[0] > 0], dim=0)
            slice_coords = torch.cat([c for c in slice_coords_list if c.shape[0] > 0], dim=0)
        else:
            slice_feats = torch.zeros(0, feats.shape[1], dtype=feats.dtype, device=feats.device)
            slice_coords = torch.zeros(0, 5, dtype=coords.dtype, device=coords.device)

        # Store temporal offset for reconstruction
        temporal_offset = current_start - offset

        slices.append((slice_feats, slice_coords, slice_seqlens, temporal_offset))
        current_start += frame_batch_strides

    return slices


def reconstruct_from_temporal_slices(
    slices: List[Tuple[torch.Tensor, torch.Tensor, List[int], int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Reconstruct full tensors from temporal slices, handling overlaps.

    Args:
        slices: List of (feats, coords, seqlens, temporal_offset) tuples

    Returns:
        (feats, coords, seqlens) tuple
    """
    if len(slices) == 0:
        raise ValueError("Cannot reconstruct from empty slice list")

    # Filter non-empty slices
    non_empty_slices = [(f, c, s, o) for f, c, s, o in slices if sum(s) > 0]

    if len(non_empty_slices) == 0:
        # Return first slice as-is
        return slices[0][:3]

    # Restore temporal offsets
    restored_slices = []
    for feats, coords, seqlens, offset in non_empty_slices:
        restored_coords = coords.clone()
        restored_coords[:, 1] = restored_coords[:, 1] + offset
        restored_slices.append((feats, restored_coords))

    # Concatenate all patches from all temporal slices
    all_feats = torch.cat([f for f, c in restored_slices], dim=0)
    all_coords = torch.cat([c for f, c in restored_slices], dim=0)

    # Re-group by batch_idx: gather each batch's patches, sort by temporal coord,
    # then concatenate in batch order so the result matches [all_batch0 | all_batch1 | ...].
    # This correctly handles both different temporal ranges (some batches absent in later
    # slices) and different spatial resolutions (different patch counts per batch).
    batch_size = len(slices[0][2])  # all slices share the same number of batches
    out_feats_list = []
    out_coords_list = []
    final_seqlens = []

    for batch_idx in range(batch_size):
        mask = all_coords[:, 0] == batch_idx
        if mask.any():
            bf = all_feats[mask]
            bc = all_coords[mask]
            order = torch.argsort(bc[:, 1])
            out_feats_list.append(bf[order])
            out_coords_list.append(bc[order])
            final_seqlens.append(int(mask.sum().item()))
        else:
            final_seqlens.append(0)

    final_feats = torch.cat(out_feats_list, dim=0) if out_feats_list else all_feats[:0]
    final_coords = torch.cat(out_coords_list, dim=0) if out_coords_list else all_coords[:0]

    return final_feats, final_coords, final_seqlens
