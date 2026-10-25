"""
This script uses pretrained image encoders to extract and save the features for given medical image datasets.
MULTI-GPU VERSION: Supports DistributedDataParallel (DDP) for multi-GPU inference.

Usage:
    Single-GPU: python scripts/expert_feature_extraction/extract_alignment_features.py --modality dermoscopy --meta_path /path/to/metadata.csv
    Multi-GPU: torchrun --nproc_per_node=4 scripts/expert_feature_extraction/extract_alignment_features.py --modality dermoscopy --meta_path /path/to/metadata.csv
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
import torch
import SimpleITK as sitk
import torch.distributed as tdist
from tqdm import tqdm
from PIL import Image
from PIL import PngImagePlugin

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from encoders.dermlip.builder import build_model as build_model_dermlip, build_transform as build_transform_dermlip
from encoders.conch.builder import build_model as build_model_conch, build_transform as build_transform_conch
from encoders.retclip.builder import build_model as build_model_retclip, build_transform as build_transform_retclip
from encoders.medimageinsight.builder import build_model as build_model_mii, build_transform as build_transform_mii
from encoders.peskavlp.builder import build_model as build_model_peskavlp, build_transform as build_transform_peskavlp
from utilities.helpers import force_suffix, stable_linear_transform, read_json, linspace_sample_list

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024 # Increase to 10 MB


SAVE_FEATURE_ROOT = os.path.join(WORKSPACE, 'outputs', 'alignment')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, paths, transform, ct_window_name='default', drop_slice_ratio=0.0, modality=None):
        self.paths = paths
        self.transform = transform
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
        self.modality = modality

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        is_ct = self.modality in ('ct', 'ct2d', 'ct3d2d', 'ct3d') if self.modality is not None else '/ct' in path
        window_name = self.selected_window_name
        window = self.selected_window
        data = {'path': path, 'window_name': window_name, 'is_windowed_ct': False}

        if path.endswith('.npy'):
            # converted to float so they will not be scaled by ToTensor transform
            image = np.load(path).astype(np.float32)
            if image.ndim != 2:
                raise ValueError(f"Expected a 2D .npy array for {path}, got {image.shape}")
            if is_ct:  # HU values of CT slices are saved
                x = stable_linear_transform(image, x_min=min(window), x_max=max(window), y_min=0, y_max=1)
                data['is_windowed_ct'] = True
            else:  # MR slices
                x = stable_linear_transform(image, x_min=None, x_max=None, y_min=0, y_max=1)
            
            x = np.stack([x,]*3, axis=-1)  # make it [H, W, 3] within range [0, 1]
            x = self.transform(x)  # this contains a ToTensor transform in it
            data['image'] = x
        elif path.endswith('.nii.gz'):
            image = read_nii(path).astype(np.float32)
            if len(image.shape) == 4:
                raise ValueError(f"Invalid shape ({image.shape}) for {path}")

            if 0 < self.drop_slice_ratio < 1:
                num_slices = image.shape[0]
                num_slices_drop = int(num_slices * self.drop_slice_ratio)
                s0 = max(num_slices_drop // 2, 0)
                s1 = min(num_slices - s0, num_slices-1)
                image = image[s0:s1+1]

            if is_ct:
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

        else:
            try:
                image = Image.open(path).convert('RGB')
                x = self.transform(image)
                data['image'] = x
            except Exception as err:
                print(f"Error when dealing with {path}: {err}")
                raise err

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
    chunk_size=128,
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
        windows = data['window_name']

        current_batch_size = imgs.shape[0]
        feat_paths = []

        # Check whether feat_path exists before loading data to avoid reading images unnecessarily.
        for i in range(current_batch_size):
            folder_name = windows[i] if flags[i] else 'default'
            feat_path = get_feature_path(img_paths[i], data_root, folder_name=folder_name)
            feat_paths.append(feat_path)

        # Skip if all features already exist
        if (not force) and all([os.path.exists(p) for p in feat_paths]):
            continue

        # Forward pass
        imgs = imgs.to(device, non_blocking=True)
        if len(imgs.shape) == 4:  # [B, C, H, W]
            #out = encoder(imgs, do_proj=False)['cls_token']  # Returns a dictionary with 'cls_token'
            out = encode_chunk(imgs, encoder, chunk_size=chunk_size)
        elif len(imgs.shape) == 5: # [B, C, D, H, W]
            depth = imgs.shape[2]
            imgs = rearrange(imgs, 'b c d h w -> (b d) c h w', d=depth, b=current_batch_size)
            out = encode_chunk(imgs, encoder, chunk_size=chunk_size)
            out = rearrange(out, '(b d) c -> b d c', d=depth, b=current_batch_size)
            out = out.mean(dim=1)
        else:
            raise ValueError(f"Invalid input shape: {imgs.shape}")

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


def get_data_paths(meta_path, data_root, num_samples=None, force=False):
    print(f"✌️ Reading metadata from existing metadata: {meta_path}")
    data_root = os.path.abspath(os.path.expanduser(data_root))
    if meta_path.endswith('.json'):
        data_paths = read_json(meta_path)
        data_paths = [p if os.path.isabs(p) else os.path.join(data_root, p) for p in data_paths]
    elif meta_path.endswith('.csv'):
        metadata = pd.read_csv(meta_path, usecols=['identifier', 'submodality'])
        num_rows = len(metadata)
        metadata = metadata.drop_duplicates(subset='identifier', keep='first')
        print(
            f"✌️ Deduplicated CSV image identifiers: "
            f"{num_rows} rows -> {len(metadata)} unique images"
        )
        data_paths = metadata['identifier'].tolist()
        data_paths = [os.path.join(data_root, p) for p in data_paths]
    else:
        raise ValueError(f"❌ Unsupported data format: {meta_path}")

    if num_samples is not None:
        data_paths = linspace_sample_list(data_paths, num_samples)
    
    if (not force):
        data_paths_ = []
        for p in data_paths:
            feat_path = get_feature_path(p, data_root, folder_name='default')
            if not os.path.exists(feat_path):
                data_paths_.append(p)
        return data_paths_
    else:
        return data_paths


def get_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Feature Extraction')
    parser.add_argument('--modality', type=str, required=True,
                       help='Modality to process (dermoscopy, eye, etc.)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=16,
                       help='Number of data loading workers')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting index for data paths')
    parser.add_argument('--end', type=int, default=None,
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
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = get_args()
    modality = args.modality
    start_idx = args.start
    end_idx = args.end
    batch_size = args.batch_size
    force = args.force
    num_workers = args.num_workers
    meta_path = args.meta_path
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    SAVE_FEATURE_ROOT = os.path.abspath(os.path.expanduser(args.feature_root))
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
                tdist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )

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

    data_paths = get_data_paths(meta_path, data_root=data_root, force=force)
    end_idx = len(data_paths) if end_idx is None else end_idx
    data_paths = data_paths[start_idx:end_idx]
    if not data_paths:
        if rank == 0:
            print("No pending images; existing features are kept (use --force to regenerate).")
        if is_distributed and tdist.is_initialized():
            tdist.destroy_process_group()
        sys.exit(0)
    
    if modality == 'dermoscopy':
        encoder = build_model_dermlip(arch='dermlip-panderm', device=device)
        transform = build_transform_dermlip()
        ct_window_name = 'default'

        if rank == 0:
            print(f"#images for {modality}: {len(data_paths)}, with CT window {ct_window_name}")

    elif modality in ['cfp']:
        #encoder = build_model_eyeclip(device=device)
        #transform = build_transform_eyeclip()
        encoder = build_model_retclip(device=device)
        transform = build_transform_retclip()
        ct_window_name = 'default'

        if rank == 0:
            print(f"#images for {modality}: {len(data_paths)}, with CT window {ct_window_name}")
        
    elif modality in [
        'ct2d', 'mr2d', 'ultrasound', 'xray',
        'mammography', 'oct', 'pet', 'unknown'
    ]:
        encoder = build_model_mii(device=device)
        transform = build_transform_mii()
        ct_window_name = 'default'

    elif modality in ['endoscopy', 'ophthalmoscope', 'ophthalmoscopy', 'surgical']:
        encoder = build_model_peskavlp(device=device)
        transform = build_transform_peskavlp()
        ct_window_name = 'default'

    elif modality  == 'histopathology':
        encoder = build_model_conch(device=device)
        transform = build_transform_conch()
        ct_window_name = 'default'

    elif modality in ['ct3d2d', 'mr3d2d']:
        encoder = build_model_mii(device=device)
        transform = build_transform_mii()
        ct_window_name = args.ct_window if modality == 'ct3d2d' else 'default'

    elif modality in ['ct3d', 'mr3d']:
        encoder = build_model_mii(device=device)
        transform = build_transform_mii()
        drop_slice_ratio = float(args.drop_slice_ratio)
        ct_window_name = args.ct_window if modality == 'ct3d' else 'default'

    else:
        raise NotImplementedError(f"For now the modality is not supported: {modality}")

    if rank == 0:
        print(f"#images for {modality}: {len(data_paths)}, with CT window {ct_window_name}")

    dataset = SimpleDataset(data_paths, transform, ct_window_name=ct_window_name,
                            drop_slice_ratio=drop_slice_ratio, modality=modality)
    inference_encoder(
        encoder, dataset, device,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        force=force,
    )

    # Finalize distributed mode
    if is_distributed and rank == 0:
        print(f"🎉 All processes completed successfully!")

    # Clean up distributed process group
    if is_distributed and tdist.is_initialized():
        tdist.destroy_process_group()
