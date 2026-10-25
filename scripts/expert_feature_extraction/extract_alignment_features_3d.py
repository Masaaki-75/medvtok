"""
This script uses pretrained image encoders to extract and save CT3D/MR3D features.
MULTI-GPU VERSION: Supports DistributedDataParallel (DDP) for multi-GPU inference.

Usage:
    CT3D: python scripts/expert_feature_extraction/extract_alignment_features_3d.py --modality ct3d --meta_path /path/to/metadata.csv
    MR3D: python scripts/expert_feature_extraction/extract_alignment_features_3d.py --modality mr3d --meta_path /path/to/metadata.csv
    Multi-GPU: torchrun --nproc_per_node=4 scripts/expert_feature_extraction/extract_alignment_features_3d.py --modality ct3d --meta_path /path/to/metadata.csv
"""

import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"🪐 WORKSPACE={WORKSPACE}")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import argparse
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import torch.distributed as tdist
from tqdm import tqdm
from PIL import Image
from PIL import PngImagePlugin
from typing import Sequence


from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from encoders.hlip.builder import build_model as build_model_hlip
from encoders.spectre.builder import build_model as build_model_spectre
from encoders.medimageinsight.builder import build_model as build_model_mii, build_transform as build_transform_mii
from utilities.helpers import force_suffix, stable_linear_transform, read_json, write_json, linspace_sample_list
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    Cropd,
    CenterSpatialCrop,
    GridPatchd
)

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024 # Increase to 10 MB


SAVE_FEATURE_ROOT = os.path.join(WORKSPACE, 'outputs', 'alignment')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)




class LargestMultipleCenterCropd(Cropd):
    """
    Dictionary-based transform for channel-first arrays only.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: sequence of ints, e.g. (128, 128, 64). Number of components must match image spatial dims.
        allow_missing_keys: don't raise if key missing.
        lazy: whether the internal cropper should be lazy.
    """
    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        self.patch_size = patch_size
        cropper = CenterSpatialCrop(roi_size=patch_size, lazy=lazy)  # Placeholder, will be reset per image
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
    
    def compute_largest_multiple_crop(self, img: np.ndarray) -> np.ndarray:
        
        spatial_dims = img.shape[1:]  # Exclude channel dim
        patch_size = np.asarray(self.patch_size, dtype=int)
        multiples = np.array(spatial_dims) // patch_size
        crop_size = (multiples * patch_size).astype(int)
        # If axis smaller than patch (multiple == 0) keep original axis size
        crop_size = np.where(crop_size == 0, spatial_dims, crop_size)
        return tuple(int(x) for x in crop_size)

    def __call__(self, data: dict, lazy: bool | None = None) -> dict:
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy

        for key in self.key_iterator(d):
            # reset cropper based on current image shape
            self.cropper.roi_size = self.compute_largest_multiple_crop(d[key].cpu().numpy())
            d[key] = self.cropper(d[key], lazy=lazy_)  # type: ignore
            
        return d


class ConditionalMaxSizeCropd(MapTransform):
    """
    Conditionally center-crops a channel-first 3D tensor [C, H, W, D].

    - If H > max_hw OR W > max_hw: center-crop both H and W to max_hw.
    - If D > max_d: center-crop D to max_d.
    No cropping is applied to dimensions already within the limits.
    """
    def __init__(
        self,
        keys: KeysCollection,
        max_hw: int = 512,
        max_d: int = 512,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.max_hw = max_hw
        self.max_d = max_d

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            _, H, W, D = img.shape
            new_H = self.max_hw if (H > self.max_hw or W > self.max_hw) else H
            new_W = self.max_hw if (H > self.max_hw or W > self.max_hw) else W
            new_D = min(D, self.max_d)
            if (new_H, new_W, new_D) != (H, W, D):
                d[key] = CenterSpatialCrop(roi_size=(new_H, new_W, new_D))(img)
        return d



def get_read_transform(modality):
    if modality == 'mr3d':
        return Compose([
            LoadImaged(keys=("image",), reader="ITKReader", ensure_channel_first=True),
            EnsureTyped(keys=("image",), data_type="numpy"),
        ])

    return Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000,
            a_max=1000,
            b_min=-1000,
            b_max=1000,
            clip=True
        ),
        Orientationd(keys=("image",), axcodes="LPI"),
        EnsureTyped(keys=("image",), data_type="numpy"),
    ])


def get_specture_transform(patch_size=(128, 128, 64)):
    return Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",),
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
        ConditionalMaxSizeCropd(keys=("image",), max_hw=512, max_d=512),
        #Spacingd(keys=("image",), pixdim=(0.5, 0.5, 1.0), mode=("bilinear",)),
        # LargestMultipleCenterCropd(
        #     keys=("image",),
        #     patch_size=patch_size,
        # ),
        GridPatchd(
            keys=("image",),
            patch_size=patch_size,
            overlap=0.0,
        ),
    ])


def read_nii(nii_path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
    if len(arr.shape) == 2:
        arr = arr[None, ...]
    return arr


def get_feature_path(img_path, data_root, folder_name='default'):
    data_root = os.path.abspath(os.path.expanduser(data_root))
    img_path = os.path.abspath(os.path.expanduser(img_path))
    if os.path.commonpath((data_root, img_path)) != data_root:
        raise ValueError(f"Image path is outside data_root: {img_path} (data_root={data_root})")

    dst_root_ = os.path.join(SAVE_FEATURE_ROOT, folder_name)
    tmp_path = os.path.join(dst_root_, os.path.relpath(img_path, data_root))
    feat_path = force_suffix(tmp_path, suffix='.pt')
    return feat_path


class SimpleDataset(Dataset):
    def __init__(
        self,
        paths,
        transform,
        modality,
        encoder_type,
        ct_window_name='default',
        drop_slice_ratio=0.0,
        use_2d_encoder=False,
    ):
        self.paths = paths
        self.transform = transform
        self.modality = modality
        self.encoder_type = encoder_type
        assert modality in {'ct3d', 'mr3d'}, f"Unsupported modality: {modality}"
        if use_2d_encoder:
            assert encoder_type == 'mii', f"2D extraction requires MII, got {encoder_type}"
        else:
            expected_encoder = 'spectre' if modality == 'ct3d' else 'hlip'
            assert encoder_type == expected_encoder, (
                f"{modality} requires {expected_encoder}, got {encoder_type}"
            )
        self.ct_windows = {
            'default': (-1000, 1000),
            #'full': (-1000, 2000),
            #'soft': (-150, 250),
            #'lung': (-1400, 200),
        }
        assert ct_window_name in self.ct_windows, f"Invalid ct window name: {ct_window_name}"
        self.selected_window_name = ct_window_name
        self.selected_window = self.ct_windows[ct_window_name]
        self.drop_slice_ratio = drop_slice_ratio
        self.use_2d_encoder = use_2d_encoder
        self.read_transform = get_read_transform(modality)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        window_name = self.selected_window_name
        window = self.selected_window
        data = {'path': path, 'window_name': window_name, 'is_windowed_ct': False}

        if path.endswith('.nii.gz') or path.endswith('.nii'):

            if self.use_2d_encoder:
                image = self.read_transform({'image': path})['image'].astype(np.float32)
                image = image.squeeze().transpose(2, 1, 0) # [D, H, W]
                if len(image.shape) == 4:
                    raise ValueError(f"Invalid shape ({image.shape}) for {path}")

                if 0 < self.drop_slice_ratio < 1:
                    num_slices = image.shape[0]
                    num_slices_drop = int(num_slices * self.drop_slice_ratio)
                    s0 = max(num_slices_drop // 2, 0)
                    s1 = min(num_slices - s0, num_slices-1)
                    image = image[s0:s1+1]

                if self.modality == 'ct3d':
                    x = stable_linear_transform(image, x_min=min(window), x_max=max(window), y_min=0, y_max=1)
                    data['is_windowed_ct'] = True
                else:
                    input_min = np.percentile(image, 0.5)
                    input_max = np.percentile(image, 99.5)
                    x = stable_linear_transform(image, x_min=input_min, x_max=input_max, y_min=0, y_max=1)

                x = torch.from_numpy(x).unsqueeze(1).expand(-1, 3, -1, -1)
                x = self.transform(x)  # Process slices in parallel: [D, 3, H, W].
                x = x.permute(1, 0, 2, 3)  # [3, D, H, W]
                data['image'] = x
            elif self.encoder_type == 'spectre':
                data['image'] = self.transform({'image': path})['image']  # [num_crops, 128, 128, 64]
            else:
                image = self.read_transform({'image': path})['image'].astype(np.float32)
                image = image.squeeze().transpose(2, 1, 0)  # [D, H, W]
                if len(image.shape) != 3:
                    raise ValueError(f"Invalid shape ({image.shape}) for {path}")
                input_min = np.percentile(image, 0.5)
                input_max = np.percentile(image, 99.5)
                image = stable_linear_transform(
                    image, x_min=input_min, x_max=input_max, y_min=0, y_max=1
                )
                x = torch.from_numpy(image).unsqueeze(0)  # [1, D, H, W]
                x = F.interpolate(
                    x[None], size=(48, 224, 224), mode='trilinear', align_corners=False
                )[0]
                hlip_mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean()
                hlip_std = torch.as_tensor(IMAGENET_DEFAULT_STD).mean()
                data['image'] = (x - hlip_mean) / hlip_std

        else:
            raise ValueError(f"Only accept nii/nii.gz file. Got {path}")

        return data



def encode_chunk(imgs, encoder, chunk_size=128):
    N = imgs.shape[0]
    out = None
    if N > chunk_size:
        for start in range(0, N, chunk_size):
            x = imgs[start:start + chunk_size]  # [n, C, H, W]; the last chunk may be smaller.
            y = encoder(x, do_proj=False)["cls_token"]  # [n, D]

            if out is None:
                D = y.shape[1]
                out = torch.empty((N, D), device=y.device, dtype=y.dtype)

            out[start:start + y.shape[0]].copy_(y)  # Write in place to avoid concatenation.
    else:
        out = encoder(imgs, do_proj=False)["cls_token"]  # [N, D]

    return out


def get_grid_size_from_monai_tensor(tensor):
    # expected batched tensor.
    B, N, C, H, W, D = tensor.shape
    loc = tensor.data.meta["location"]
    #print(f">>>>>>>> loc shape: {loc.shape}")
    Hp, Wp, Dp = tuple(int(np.unique(loc[i, :]).size) for i in range(3))
    assert N == (Hp * Wp * Dp), f"Number of patches {N} does not match computed grid size {Hp}x{Wp}x{Dp}"
    return (Hp, Wp, Dp), tensor.as_tensor()



@torch.inference_mode()
def inference_encoder(
    encoder, 
    dataset, 
    device, 
    data_root,
    batch_size=64, 
    num_workers=16, 
    rank=0, 
    world_size=1, 
    local_rank=0, 
    chunk_size=64,
    force=False,
):
    """
    Multi-GPU aware feature extraction function.

    Args:
        encoder: The encoder model
        dataset: Dataset to process
        device: Device to use
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        rank: Current process rank (0 for single-GPU or master in multi-GPU)
        world_size: Total number of processes (1 for single-GPU)
        local_rank: Local rank within the node (for DDP device_ids)
    """

    # Move encoder to device and set to eval mode
    encoder = encoder.eval().to(device)
    use_2d_encoder = dataset.use_2d_encoder
    encoder_type = dataset.encoder_type
    print(f"⚠️⚠️⚠️ USING 2D ENCODERS? {use_2d_encoder}")

    # Wrap with DDP if in distributed mode
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ Wrapped encoder with DistributedDataParallel")

    # Create sampler and dataloader
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Keep deterministic ordering
            drop_last=False  # Process all samples
        )
        if rank == 0:
            print(f"✅ Created DistributedSampler with world_size={world_size}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # No shuffle when using sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Progress bar only on rank 0
    if rank == 0:
        print(f"🚀 Starting feature extraction on {world_size} GPU(s)")
        print(f"   Total samples: {len(dataset)}, Batches per GPU: {len(loader)}")

    pbar = tqdm(
        loader,
        total=len(loader),
        disable=(rank != 0),
        desc=f"[Rank {rank}/{world_size}] Extracting features"
    )

    for data in pbar:
        img_paths = data['path']
        flags = data['is_windowed_ct']
        imgs = data['image']
        if encoder_type == 'spectre':
            grid_size, imgs = get_grid_size_from_monai_tensor(imgs)
        else:
            grid_size = None

        windows = data['window_name']
        current_batch_size = imgs.shape[0]
        feat_paths = []

        # Check whether feat_path exists before loading data to avoid reading images unnecessarily.
        for i in range(current_batch_size):
            folder_name = 'ablation' if use_2d_encoder else 'default'
            feat_path = get_feature_path(img_paths[i], data_root, folder_name=folder_name)
            feat_paths.append(feat_path)

        # Skip if all features already exist
        if (not force) and all([os.path.exists(p) for p in feat_paths]):
            continue

        # Forward pass
        imgs = imgs.to(device, non_blocking=True)
        if use_2d_encoder:
            assert len(imgs.shape) == 5, f"Expected [B, C, D, H, W], got {imgs.shape}"            
            # [B, C, D, H, W]
            depth = imgs.shape[2]
            imgs = rearrange(imgs, 'b c d h w -> (b d) c h w', d=depth, b=current_batch_size)
            out = encode_chunk(imgs, encoder, chunk_size=chunk_size)
            out = rearrange(out, '(b d) c -> b d c', d=depth, b=current_batch_size)
            out = out.mean(dim=1)
        elif encoder_type == 'spectre':
            assert len(imgs.shape) == 6, f"Expected [B, num_crops, 1, H, W, D], got {imgs.shape}"
            #print(f">>>>> INPUT SHAPE: {imgs.shape}; INPUT RANGE: {imgs.min(), imgs.max()}")
            out = encoder(imgs, grid_size=grid_size)["cls_token"]  # [N, D]
            #print(f">>>>> OUTPUT SHAPE: {out.shape}; OUTPUT RANGE: {out.min(), out.max()}")
        else:
            assert encoder_type == 'hlip'
            assert len(imgs.shape) == 5, f"Expected [B, 1, D, H, W], got {imgs.shape}"
            out = encoder(imgs)["cls_token"]  # [B, D]
        cls_tokens = out.detach().cpu()  # [B, C]

        # Save features
        for i in range(current_batch_size):
            cls_token = cls_tokens[i].squeeze().detach().clone()
            save_path = feat_paths[i]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(cls_token, save_path)

    # Synchronize all processes
    if world_size > 1:
        tdist.barrier()
        if rank == 0:
            print(f"✅ All {world_size} GPUs finished processing!")
    else:
        print(f"✅ All done!")


def get_data_paths(
    meta_path,
    data_root,
    modality=None,
    num_samples=None,
    force=False,
    use_2d_encoder=False,
):
    print(f"✌️ Reading metadata from existing metadata: {meta_path}")
    data_root = os.path.abspath(os.path.expanduser(data_root))
    if meta_path.endswith('.json'):
        data_paths = read_json(meta_path)
        data_paths = [p if os.path.isabs(p) else os.path.join(data_root, p) for p in data_paths]
    elif meta_path.endswith('.csv'):
        metadata = pd.read_csv(meta_path, usecols=['identifier', 'submodality'])
        if modality is not None:
            metadata = metadata[metadata['submodality'] == modality]
        metadata = metadata.drop_duplicates(subset='identifier', keep='first')
        data_paths = metadata['identifier'].tolist()
        print(f"[get_data_paths] len(data_paths) == {len(data_paths)}")
        data_paths = [os.path.join(data_root, p) for p in data_paths]
    else:
        raise ValueError(f"Only support csv/json, got {meta_path}")

    if num_samples is not None:
        data_paths = linspace_sample_list(data_paths, num_samples)
    
    if (not force):
        data_paths_ = []
        folder_name = 'ablation' if use_2d_encoder else 'default'
        for p in data_paths:
            feat_path = get_feature_path(p, data_root, folder_name=folder_name)
            if not os.path.exists(feat_path):
                data_paths_.append(p)
        return data_paths_
    else:
        return data_paths


def get_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Feature Extraction')
    parser.add_argument('--modality', type=str, required=True,
                       choices=('ct3d', 'ct', 'mr3d', 'mr'),
                       help='3D modality to process')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=10,
                       help='Number of data loading workers')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for data paths')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Starting index for data paths')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by launcher)')
    parser.add_argument('--ct_window', type=str, default='default')
    parser.add_argument('--drop_slice_ratio', type=float, default=0.0)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--meta_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/inspire/qb-ilm/project/qproject-multimedicine/public/jinye/data/medical_preprocessed',
                       help='Root directory containing the source medical images')
    parser.add_argument('--feature_root', type=str, default=SAVE_FEATURE_ROOT,
                       help='Feature output root; features are saved under its default/ subfolder')
    parser.add_argument('--device', default=DEVICE, help='Torch device for single-process extraction')
    parser.add_argument('--use_2d_encoder', action='store_true', default=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = get_args()
    modality = {'ct': 'ct3d', 'mr': 'mr3d'}.get(args.modality, args.modality)
    start_idx = args.start_idx
    end_idx = args.end_idx
    batch_size = args.batch_size
    force = args.force
    num_workers = args.num_workers
    meta_path = args.meta_path
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    SAVE_FEATURE_ROOT = os.path.abspath(os.path.expanduser(args.feature_root))
    use_2d_encoder = args.use_2d_encoder
    
    if end_idx is not None:
        if end_idx > 0:
            assert end_idx > start_idx, f"start_idx={start_idx}, end_idx={end_idx}"

    # Initialize distributed mode if RANK environment variable is set
    is_distributed = 'RANK' in os.environ
    rank = 0
    world_size = 1
    local_rank = 0
    device = torch.device(args.device)

    if is_distributed:
        # Initialize distributed environment
        try:
            # Get distributed parameters from environment
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))

            # Set device
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')

            # Initialize process group
            if not tdist.is_initialized():
                tdist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

            if rank == 0:
                print(f"🌐 Distributed mode initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        except Exception as e:
            print(f"⚠️  Failed to initialize distributed mode: {e}")
            print(f"⚠️  Falling back to single-GPU mode")
            is_distributed = False
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device(args.device)
    else:
        print(f"🖥️  Running in single-GPU mode")

    if rank == 0:
        print(f"💡💡💡 Working with modality: {modality} (force={force}, starting from {start_idx} to {end_idx}). Now searching image paths...")

    drop_slice_ratio = 0
    
    data_paths = get_data_paths(
        meta_path,
        data_root=data_root,
        modality=modality,
        force=force,
        use_2d_encoder=use_2d_encoder,
    )
    end_idx = len(data_paths) if end_idx is None else end_idx
    data_paths = data_paths[start_idx:end_idx]
    if not data_paths:
        if rank == 0:
            print("No pending volumes; existing features are kept (use --force to regenerate).")
        if is_distributed and tdist.is_initialized():
            tdist.destroy_process_group()
        sys.exit(0)
    
    if modality in ['ct3d', 'ct']:
        ct_window_name = args.ct_window
        
        if use_2d_encoder:    
            encoder = build_model_mii(device=device)
            transform = build_transform_mii()
            encoder_type = 'mii'
            drop_slice_ratio = float(args.drop_slice_ratio)
        else:
            encoder = build_model_spectre(device=device)
            transform = get_specture_transform()
            encoder_type = 'spectre'

    elif modality in ['mr3d', 'mr']:
        ct_window_name = 'default'

        if use_2d_encoder:
            encoder = build_model_mii(device=device)
            transform = build_transform_mii()
            encoder_type = 'mii'
            drop_slice_ratio = float(args.drop_slice_ratio)
        else:
            encoder = build_model_hlip(device=device)
            transform = None
            encoder_type = 'hlip'

    else:
        raise NotImplementedError(f"For now the modality is not supported: {modality}")

    if rank == 0:
        print(f"#images for {modality}: {len(data_paths)}, with CT window {ct_window_name}")

    dataset = SimpleDataset(
        data_paths,
        transform,
        modality=modality,
        encoder_type=encoder_type,
        ct_window_name=ct_window_name,
        drop_slice_ratio=drop_slice_ratio,
        use_2d_encoder=use_2d_encoder,
    )
    inference_encoder(
        encoder, dataset, device,
        batch_size=batch_size,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        force=force,
        data_root=data_root,
    )

    # Finalize distributed mode
    if is_distributed and rank == 0:
        print(f"🎉 All processes completed successfully!")

    # Clean up distributed process group
    if is_distributed and tdist.is_initialized():
        tdist.destroy_process_group()
