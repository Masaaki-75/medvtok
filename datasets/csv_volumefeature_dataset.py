import os
import gc
import random
import torch
import torch.nn.functional as F
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import List, Iterator, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler

from datasets.common import DataInfo
from datasets.csv_imagefeature_dataset import (
    get_modality_group,
    stratified_subsample_dataframe,
    CT3D_MODALITIES,
    MR3D_MODALITIES,
)
from utilities import dist
from utilities.helpers import force_suffix


VOLUME_MODALITY_INDEX_CT3D = 9
VOLUME_MODALITY_INDEX_MR3D = 10

# Encode a per-batch forced depth into the sample index so all DDP ranks
# produce volumes with the same temporal depth per training step.
# Supports datasets up to 10M samples.
DEPTH_ENCODE_FACTOR = 10_000_000

SHAPE_TABLE = {
    256: [16, 32],
    384: [16, 32],
    512: [16, 32],
}
CT_WINDOW_DICT = {
    'default': (-1000, 1000), 
    'soft': (-150, 250), 
    'lung': (-1400, 200), 
    'full': (-1000, 2000),
}


class VolumeTransform:
    def __init__(self, is_train: bool = True, dtype: str = "float32", volume_size=None, preset_resolutions=(256, 384, 512), preset_resolution_probs=(0.4, 0.2, 0.4)):
        assert dtype in ("float16", "float32")
        self.is_train = is_train
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.ct_window_dict = CT_WINDOW_DICT
        self.ct_window_names = ('default', 'soft', 'lung', 'full')
        self.ct_window_probs = (0.5, 0.3, 0.05, 0.15)
        
        self.volume_size = volume_size
        self.preset_resolutions = preset_resolutions
        self.preset_resolution_probs = preset_resolution_probs
        self.shape_table = SHAPE_TABLE
        if preset_resolutions is not None:
            for r in preset_resolutions:
                assert r in self.shape_table, f"Invalid resolution {r}. Not in shape table: {self.shape_table}"

    def __call__(self, path: str, modality: str, forced_depth: int = 0) -> torch.Tensor:
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            vol = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)  # [D, H, W]
        elif path.endswith('.npy'):
            vol = np.load(path).astype(np.float32)  # [D, H, W]
        else:
            raise ValueError(f"Unsupported file format: {path}")

        vol = self._normalize(vol, modality)

        if self.volume_size is not None:
            # Force volume_size
            vol = self._smart_resize(vol, dhw=self.volume_size)
        elif self.volume_size is None and self.preset_resolutions is not None:
            # Smart resize
            vol = self._smart_resize(vol, dhw=None, forced_depth=forced_depth)
        else:
            # No resize
            vol = self._smart_resize(vol, dhw=vol.shape)

        vol = torch.from_numpy(vol).to(self.dtype)    # [D, H, W]
        vol = vol.unsqueeze(0).expand(3, -1, -1, -1)  # [3, D, H, W]

        if self.is_train:
            for axis in range(3):
                if random.random() < 0.5:
                    vol = vol.flip(axis + 1)  # +1 because dim 0 is channel
            if random.random() < 0.2:
                vol = vol.permute(0, 1, 3, 2)

        return vol.contiguous()

    def _normalize(self, vol: np.ndarray, modality: str) -> np.ndarray:
        modality = modality.lower().strip()
        if modality in CT3D_MODALITIES:
            if self.is_train:
                window_name = np.random.choice(self.ct_window_names, p=self.ct_window_probs)
            else:
                window_name = 'default'
            lo, hi = self.ct_window_dict[window_name]
        elif modality in MR3D_MODALITIES:
            lo, hi = np.percentile(vol, [0.5, 99.5])
            if hi - lo < 1e-6:
                hi = lo + 1.0
        else:
            raise ValueError(f"Unsupported modality for volume normalization: {modality}")

        # [T, H, W]
        vol = np.clip(vol, lo, hi)
        vol = 2.0 * (vol - lo) / (hi - lo) - 1.0

        return vol

    def _smart_resize(self, vol: np.ndarray, dhw=None, forced_depth: int = 0) -> np.ndarray:
        D, H, W = vol.shape
        if dhw is not None:
            if D != dhw[0] or H != dhw[1] or W != dhw[2]:
                return F.interpolate(torch.from_numpy(vol).unsqueeze(0).unsqueeze(0), size=dhw, mode='trilinear', align_corners=False).squeeze(0).squeeze(0).numpy()
            else:
                return vol

        input_hw = max(H, W)
        target_hw = np.random.choice(self.preset_resolutions, p=self.preset_resolution_probs) #min(self.preset_resolutions, key=lambda x: abs(x - input_hw))

        if forced_depth > 0:
            target_d = forced_depth
            resize_depth = (0.5 * target_d <= D <= 2.0 * target_d)
        else:
            chooseable_depths = SHAPE_TABLE[target_hw]
            close_depths = [d for d in chooseable_depths if 0.5 * d <= D <= 1.5 * d]
            if close_depths:
                target_d = min(close_depths, key=lambda d: abs(d - D))
                resize_depth = True
            else:
                target_d = random.choice(chooseable_depths)
                resize_depth = False

        if resize_depth:
            # Resize all three dims via trilinear interpolation
            t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
            t = F.interpolate(t, size=(target_d, target_hw, target_hw), mode='trilinear', align_corners=False)
            vol = t.squeeze(0).squeeze(0).numpy()
        else:
            # Crop or pad depth, then resize H/W only
            vol = self._crop_or_pad_depth(vol, target_d)
            if H != target_hw or W != target_hw:
                t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                t = F.interpolate(t, size=(target_d, target_hw, target_hw), mode='trilinear', align_corners=False)
                vol = t.squeeze(0).squeeze(0).numpy()

        return vol

    @staticmethod
    def _crop_or_pad_depth(vol: np.ndarray, target_d: int) -> np.ndarray:
        D, H, W = vol.shape
        if D > target_d:
            lo = int(D * 0.1)
            hi = int(D * 0.9) - target_d
            if hi >= lo:
                start = random.randint(lo, hi)
            else:
                start = max(0, (D - target_d) // 2)
            vol = vol[start:start + target_d]
        elif D < target_d:
            pad_total = target_d - D
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            vol = np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=vol.min())
        return vol


class BalancedCTMRSampler(Sampler[int]):
    """Infinite sampler that yields balanced CT3D / MR3D batches.

    Each batch contains batch_size//2 CT3D samples and the rest MR3D samples.
    When a pool is exhausted it reshuffles and restarts, making the iteration
    infinite. Compatible with DataLoader(sampler=..., batch_size=N).

    Args:
        submodalities: List of submodality strings for every sample in the dataset.
        batch_size: Per-GPU batch size.
        num_replicas: World size for distributed training.
        rank: Current process rank.
        shuffle: Whether to shuffle within each pool.
        seed: Base random seed.
        num_batches_per_epoch: How many batches constitute one "epoch" for
            __len__ reporting and the outer training loop. If None, defaults
            to total_samples // batch_size.
    """

    def __init__(
        self,
        submodalities: List[str],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        num_batches_per_epoch: int = None,
    ):
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Split dataset indices into CT3D and MR3D pools
        self.ct_indices = []
        self.mr_indices = []
        for idx, submod in enumerate(submodalities):
            group = get_modality_group(submod)
            if group == VOLUME_MODALITY_INDEX_CT3D:
                self.ct_indices.append(idx)
            elif group == VOLUME_MODALITY_INDEX_MR3D:
                self.mr_indices.append(idx)
            else:
                raise ValueError(
                    f"CsvVolumeFeatureDataset only supports ct3d/mr3d, got submodality='{submod}' (group={group})"
                )

        self.ct_indices = np.array(self.ct_indices)
        self.mr_indices = np.array(self.mr_indices)
        total = len(self.ct_indices) + len(self.mr_indices)

        print(f"[BalancedCTMRSampler] CT3D={len(self.ct_indices)}, MR3D={len(self.mr_indices)}, total={total}")

        if len(self.ct_indices) == 0 or len(self.mr_indices) == 0:
            non_empty = "ct3d" if len(self.ct_indices) > 0 else "mr3d"
            empty = "mr3d" if non_empty == "ct3d" else "ct3d"
            print(f"[BalancedCTMRSampler] WARNING: {empty} pool is empty. "
                  f"All batch slots will be filled from {non_empty}.")

        # Per-batch allocation
        self.ct_per_batch = batch_size // 2
        self.mr_per_batch = batch_size - self.ct_per_batch

        # If one pool is empty, give all slots to the other
        if len(self.ct_indices) == 0:
            self.ct_per_batch = 0
            self.mr_per_batch = batch_size
        if len(self.mr_indices) == 0:
            self.mr_per_batch = 0
            self.ct_per_batch = batch_size

        if num_batches_per_epoch is not None:
            self.num_batches = num_batches_per_epoch
        else:
            self.num_batches = max(1, total // batch_size)

        self.total_size = self.num_batches * batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_rank_pool(self, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shuffle and shard indices for the current rank."""
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        pool = indices.copy()
        rng.shuffle(pool)
        seps = np.linspace(0, len(pool), self.num_replicas + 1, dtype=int)
        return pool[seps[self.rank]:seps[self.rank + 1]]

    def _infinite_pool(self, base_pool: np.ndarray, rng: np.random.Generator):
        """Yield indices from pool forever, reshuffling when exhausted."""
        if len(base_pool) == 0:
            return
        pool = base_pool.copy()
        ptr = 0
        while True:
            if ptr >= len(pool):
                rng.shuffle(pool)
                ptr = 0
            yield int(pool[ptr])
            ptr += 1

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)

        ct_pool = self._get_rank_pool(self.ct_indices, rng)
        mr_pool = self._get_rank_pool(self.mr_indices, rng)

        # Separate RNGs so reshuffles are independent
        ct_iter = self._infinite_pool(ct_pool, np.random.default_rng(self.seed + self.epoch + 1))
        mr_iter = self._infinite_pool(mr_pool, np.random.default_rng(self.seed + self.epoch + 2))

        # Rank-independent RNG for per-batch depth selection (same seed on all ranks
        # so all ranks pick the same depth for the same batch index).
        depth_rng = np.random.default_rng(self.seed + self.epoch + 9999)
        depth_choices = [16, 32]

        for _ in range(self.num_batches):
            target_depth = int(depth_rng.choice(depth_choices))
            batch = []
            for _ in range(self.ct_per_batch):
                batch.append(next(ct_iter) + target_depth * DEPTH_ENCODE_FACTOR)
            for _ in range(self.mr_per_batch):
                batch.append(next(mr_iter) + target_depth * DEPTH_ENCODE_FACTOR)
            # Shuffle within batch so CT and MR are interleaved
            rng.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self) -> int:
        return self.total_size


class CsvVolumeFeatureDataset(Dataset):
    """Dataset for 3D medical volumes (CT3D and MR3D) with precomputed features.

    Mirrors the interface of CsvImageFeatureDataset but uses MONAI transforms
    for 3D volume loading and random cropping. Each __getitem__ returns one
    randomly cropped sub-volume.
    """

    def __init__(
        self,
        input_filename,
        image_key='identifier',
        image_root=None,
        image_feature_root=None,
        tokenizer=None,
        is_train=True,
        volume_size=None,
        dtype="float32",
        num_samples=None,
        load_image_feature=True,
        preset_resolutions=(256, 384, 512),
        ablation_modality=None,
        subset_fraction=1.0,
        subset_seed=42,
    ):
        logging.debug(f'Loading csv data from {input_filename}.')
        usecols = ['identifier', 'modality', 'submodality', 'multiwindow']
        self.images = pd.read_csv(input_filename, usecols=usecols)
        self.images = stratified_subsample_dataframe(
            self.images,
            fraction=subset_fraction,
            seed=subset_seed,
            dataset_name='3D dataset',
        )

        logging.debug(f'[CsvVolFeatDataset] Successfully loaded csv data from {input_filename}.')
        self.image_key = image_key
        self.tokenize = tokenizer
        self.image_root = image_root
        self.image_feature_root = image_feature_root
        self.load_image_feature = load_image_feature
        self.ablation_modality = ablation_modality
        self.volume_size = volume_size
        self.preset_resolutions = preset_resolutions

        if num_samples is not None:
            num_samples = int(num_samples)
            self.images = self.images.iloc[:num_samples]
        self.num_samples = num_samples

        print(f"[CsvVolFeatDataset] Loaded data (n={len(self.images)}, is_train={is_train})")

        self.transforms = VolumeTransform(is_train=is_train, dtype=dtype, volume_size=volume_size, preset_resolutions=preset_resolutions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, encoded_idx):
        forced_depth = encoded_idx // DEPTH_ENCODE_FACTOR
        idx = encoded_idx % DEPTH_ENCODE_FACTOR
        row = self.images.iloc[idx]
        data = self.read_data(row, forced_depth=forced_depth)
        return data["volume"], data["image_feature"], data["modality_index"]

    def get_volume_path(self, row):
        image_identifier = row[self.image_key]
        if self.image_root is None:
            return image_identifier
        return os.path.join(self.image_root, image_identifier)

    def get_image_feature_path(self, row, window_name='default'):
        image_identifier = row[self.image_key]
        feature_identifier = force_suffix(image_identifier, suffix='.pt')
        feature_path = os.path.join(self.image_feature_root, window_name, feature_identifier)
        return feature_path

    def read_data(self, row, forced_depth=0):
        volume_path = self.get_volume_path(row)
        submodality = row.get('submodality', 'unknown')
        midx = get_modality_group(submodality)

        volume = self.transforms(volume_path, submodality, forced_depth=forced_depth)  # [3, D, H, W]

        # Load precomputed image feature
        if self.load_image_feature:
            if not self.ablation_modality:
                feature_window = 'default'
            elif self.ablation_modality in CT3D_MODALITIES:
                feature_window = 'ablation' if midx == VOLUME_MODALITY_INDEX_CT3D else 'default'
            elif self.ablation_modality in MR3D_MODALITIES:
                feature_window = 'ablation' if midx == VOLUME_MODALITY_INDEX_MR3D else 'default'
            else:
                feature_window = 'default'
                
            feature_path = self.get_image_feature_path(row, window_name=feature_window)
            try:
                image_feature = torch.load(feature_path, map_location='cpu').squeeze()
            except Exception as err:
                raise ValueError(f"❌ Error loading image feature from {feature_path}: \n{err}")
        else:
            image_feature = None

        return {"volume": volume, "image_feature": image_feature, "modality_index": midx}

    def get_submodalities(self) -> List[str]:
        return self.images['submodality'].tolist()

    def __del__(self):
        if hasattr(self, 'images'):
            del self.images
        gc.collect()


def collate_volume_batch(batch):
    """Collate function for 3D volume batches.

    Returns the same 4-tuple format as collate_mixed_batch so the training loop
    can consume 2D and 3D batches with the same unpacking logic:
        (images_or_volumes, image_features, modality_indices)
    """
    volumes, image_features, modality_indices = zip(*batch)

    modality_indices = torch.tensor(modality_indices, dtype=torch.long)

    # Volumes as list (variable sizes across configs / future flexibility)
    volumes = list(volumes)

    # Image features as list (same convention as 2D MEPA)
    image_features = list(image_features)

    return volumes, image_features, modality_indices


def get_csvvolfeat_dataset(args, is_train, epoch=0, iters=0, tokenizer=None):
    """Factory function that creates a CsvVolumeFeatureDataset with DataLoader.

    Returns DataInfo compatible with the existing training loop.
    """
    input_filename = args.train_data_3d if is_train else args.val_data_3d
    num_samples = getattr(args, 'train_num_samples_3d', None) if is_train else getattr(args, 'val_num_samples_3d', None)
    image_root = getattr(args, 'train_root_3d', None) if is_train else getattr(args, 'val_root_3d', None)
    image_feature_root = args.feature_root
    load_image_feature = getattr(args, 'lshare', 0) > 0

    # Volume-specific settings with defaults
    volume_dtype = getattr(args, 'volume_dtype', 'float32')
    num_batches_per_epoch = getattr(args, 'num_batches_per_epoch_3d', None)
    mixed_resolution_training = bool(args.mixed_resolution_training)
    preset_resolution = (256, 384, 512) if mixed_resolution_training else None
    ablation_modality = getattr(args, 'ablation_modality', None)
    subset_fraction = getattr(args, 'train_fraction_3d', 1.0) if is_train else 1.0
    subset_seed = getattr(args, 'subset_seed', 42)

    print(f"[get_csvvolfeat_dataset] Settings: local_bs_3d={args.local_bs_3d}, is_train={is_train}, num_batches_per_epoch={num_batches_per_epoch}")

    assert input_filename, "train_data_3d (or val_data_3d) must be specified for csvvolfeat dataset"

    dataset = CsvVolumeFeatureDataset(
        input_filename,
        image_key=args.csv_img_key,
        tokenizer=tokenizer,
        image_root=image_root,
        image_feature_root=image_feature_root,
        is_train=is_train,
        dtype=volume_dtype,
        volume_size=None,
        num_samples=num_samples,
        load_image_feature=load_image_feature,
        preset_resolutions=preset_resolution,
        ablation_modality=ablation_modality,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )

    if is_train:
        sampler = BalancedCTMRSampler(
            submodalities=dataset.get_submodalities(),
            batch_size=args.local_bs_3d,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=args.seed if args.seed is not None else 0,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_bs_3d,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.workers > 0,
            collate_fn=collate_volume_batch,
            prefetch_factor=4 if args.workers > 0 else None,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_bs_3d,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate_volume_batch,
        )

    num_samples = len(dataset)
    num_batches = len(dataloader)

    return DataInfo(dataloader, num_samples, num_batches, sampler=sampler)
