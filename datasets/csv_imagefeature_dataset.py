import os
import gc
import sys
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Iterator
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from datasets.common import DataInfo

Image.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 3)
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = True  # pathgen has images with large metadata

from utilities import dist
from utilities.helpers import force_suffix
from datasets.transforms import ReadMedicalImage, build_image_transform


# Modality group mapping for diversity sampling
NUM_MODALITY_GROUPS_2D = 9
NUM_MODALITY_GROUPS_ALL = NUM_MODALITY_GROUPS_2D + 2

DERM_MODALITIES = {'dermoscopy', 'dermatology', 'derm'}
CFP_MODALITIES = {'cfp', 'fundus', 'fundus-photography'}
OTHER_MODALITIES = {'oct', 'unknown'}
PATH_MODALITIES = {'histopathology', 'pathology', 'histo', 'patho', 'microscopy'}
SURG_MODALITIES = {'surgical', 'endoscopy', 'ophthalmoscope', 'ophthalmoscopy'}
CT2D_MODALITIES = {'ct2d', 'ct3d2d'}
MR2D_MODALITIES = {'mr2d', 'mr3d2d'}
US_MODALITIES = {'ultrasound', 'us'}
XRAY_MODALITIES = {'xray', 'x-ray', 'mammography', 'cxr'}
CT3D_MODALITIES = {'ct3d'}
MR3D_MODALITIES = {'mr3d'}

MODALITY_GROUP_NAMES = {
    0: 'dermoscopy',
    1: 'cfp',
    2: 'pathology',
    3: 'surgical',
    4: 'ct2d',
    5: 'mr2d',
    6: 'ultrasound',
    7: 'xray',
    8: 'other',
    9: 'ct3d',
    10: 'mr3d',
}


def get_modality_group(modality):
    # [NOTE] align with models/medvtok_wrapper.py
    modality = modality.lower().strip().replace('_', '-')
    if modality in DERM_MODALITIES:
        return 0
    elif modality in CFP_MODALITIES:
        return 1
    elif modality in PATH_MODALITIES:
        return 2
    elif modality in SURG_MODALITIES:
        return 3
    elif modality in CT2D_MODALITIES:
        return 4
    elif modality in MR2D_MODALITIES:
        return 5
    elif modality in US_MODALITIES:
        return 6
    elif modality in XRAY_MODALITIES:
        return 7
    elif modality in OTHER_MODALITIES:
        return 8
    elif modality in CT3D_MODALITIES:
        return 9
    elif modality in MR3D_MODALITIES:
        return 10
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def exclude_modalities_from_dataframe(
    dataframe: pd.DataFrame,
    exclude_modalities: List[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Remove requested modality groups or exact modality/submodality names."""
    requested = {
        str(name).lower().strip().replace("_", "-")
        for name in (exclude_modalities or [])
        if str(name).strip()
    }
    if not requested:
        return dataframe

    modality_values = dataframe["modality"].fillna("").astype(str).str.lower().str.strip().str.replace("_", "-", regex=False)
    submodality_values = dataframe["submodality"].fillna("").astype(str).str.lower().str.strip().str.replace("_", "-", regex=False)
    exact_match = modality_values.isin(requested) | submodality_values.isin(requested)

    requested_groups = {}
    for name in requested:
        try:
            requested_groups[name] = get_modality_group(name)
        except ValueError:
            pass
    sample_groups = submodality_values.map(get_modality_group)
    group_match = sample_groups.isin(set(requested_groups.values()))

    available_names = set(modality_values.unique()) | set(submodality_values.unique())
    available_groups = set(sample_groups.unique())
    matched = (requested & available_names) | {
        name for name, group in requested_groups.items() if group in available_groups
    }
    unmatched = requested - matched
    if unmatched:
        raise ValueError(
            f"{dataset_name}: exclude_modalities contains unknown values: {sorted(unmatched)}"
        )

    excluded_mask = exact_match | group_match
    filtered = dataframe.loc[~excluded_mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"{dataset_name}: exclude_modalities removed every sample")
    print(
        f"[{dataset_name}] Excluded modalities {sorted(requested)}: "
        f"removed {int(excluded_mask.sum())} samples, retained {len(filtered)}"
    )
    return filtered


def stratified_subsample_dataframe(
    dataframe: pd.DataFrame,
    fraction: float,
    seed: int,
    dataset_name: str,
) -> pd.DataFrame:
    """Deterministically subsample each modality group by the same fraction."""
    fraction = float(fraction)
    seed = int(seed)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"{dataset_name} fraction must satisfy 0 < fraction <= 1, got {fraction}."
        )
    if 'submodality' not in dataframe.columns:
        raise ValueError(
            f"{dataset_name} stratified subsampling requires a 'submodality' column."
        )

    if fraction == 1.0:
        print(
            f"[{dataset_name}] Stratified subsampling disabled "
            f"(fraction=1.0, n={len(dataframe)})."
        )
        return dataframe.reset_index(drop=True)

    modality_groups = dataframe['submodality'].map(get_modality_group)
    sampled_groups = []
    count_messages = []
    for group_id in sorted(modality_groups.unique()):
        group_data = dataframe.loc[modality_groups == group_id]
        num_original = len(group_data)
        num_keep = max(1, int(np.floor(num_original * fraction)))
        num_keep = min(num_keep, num_original)
        sampled_groups.append(
            group_data.sample(
                n=num_keep,
                replace=False,
                random_state=seed + int(group_id),
            )
        )
        group_name = MODALITY_GROUP_NAMES.get(int(group_id), str(group_id))
        count_messages.append(f"{group_name}={num_keep}/{num_original}")

    sampled = pd.concat(sampled_groups, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[{dataset_name}] Stratified subsampling: fraction={fraction}, seed={seed}, "
        f"total={len(sampled)}/{len(dataframe)}, per_group=[{', '.join(count_messages)}]"
    )
    return sampled



class DiversitySampler(Sampler[int]):
    """
    Sampler ensuring batch diversity when used with DataLoader.

    Yields individual indices in an order such that consecutive batch_size
    indices contain samples from all modality groups. When DataLoader batches
    these consecutive indices, each batch is guaranteed to be diverse.

    Compatible with standard DataLoader(sampler=..., batch_size=...).

    Args:
        submodalities: List of submodality strings for each sample in the dataset
        batch_size: Number of samples per batch (needed to arrange indices)
        num_replicas: Number of distributed processes (world_size)
        rank: Rank of the current process
        shuffle: Whether to shuffle indices each epoch
        seed: Random seed for reproducibility
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        submodalities: List[str],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        include_3d_data: bool = False,
        num_appear_each_modality: int = 1,
    ):
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_appear_each_modality = num_appear_each_modality

        # Build index pools for each modality group
        self.num_modality_groups = NUM_MODALITY_GROUPS_2D #NUM_MODALITY_GROUPS_ALL if include_3d_data else NUM_MODALITY_GROUPS_2D
        self.group_indices = [[] for _ in range(self.num_modality_groups)]
        print(f"[DiversitySampler] self.num_modality_groups: {self.num_modality_groups}")
        print(f"[DiversitySampler] submodalities ({len(submodalities)})")
        
        for idx, submod in enumerate(submodalities):
            group = get_modality_group(submod)
            self.group_indices[group].append(idx)

        # Convert to numpy arrays for efficient shuffling
        self.group_indices = [np.array(indices) for indices in self.group_indices]

        # Log group sizes
        group_sizes = [len(g) for g in self.group_indices]
        print(f"[DiversitySampler] Group sizes: derm={group_sizes[0]}, cfp={group_sizes[1]}, histopathology={group_sizes[2]}, "
              f"surgical={group_sizes[3]}, total={sum(group_sizes)}")

        # Check if all groups have samples
        self.non_empty_groups = [i for i, g in enumerate(self.group_indices) if len(g) > 0]
        empty_groups = [i for i, g in enumerate(self.group_indices) if len(g) == 0]
        if empty_groups:
            print(f"[DiversitySampler] WARNING: Empty modality groups: {empty_groups}. Will sample from available groups only.")

        # Calculate total samples for this rank
        total_samples = len(submodalities)
        samples_per_rank = total_samples // num_replicas
        self.num_batches = samples_per_rank // batch_size
        if drop_last:
            self.total_size = self.num_batches * batch_size
        else:
            self.total_size = samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling."""
        self.epoch = epoch

    def _get_shuffled_group_indices(self, rng: np.random.Generator) -> List[np.ndarray]:
        """Get shuffled copies of group indices."""
        shuffled = []
        for group in self.group_indices:
            if len(group) > 0:
                indices = group.copy()
                rng.shuffle(indices)
                shuffled.append(indices)
            else:
                shuffled.append(np.array([], dtype=np.int64))
        return shuffled

    def __iter__(self) -> Iterator[int]:
        """
        Yield individual indices in an order that ensures batch diversity.

        Indices are arranged so that every consecutive batch_size indices
        contain at least one sample from each non-empty modality group.
        """
        rng = np.random.default_rng(self.seed + self.epoch)

        # Get shuffled group indices
        shuffled_groups = self._get_shuffled_group_indices(rng)

        # Create a shuffled pool of all indices for filling remaining slots
        all_indices = np.concatenate([g for g in shuffled_groups if len(g) > 0])
        rng.shuffle(all_indices)

        # Split all indices by rank for distributed training
        total_samples = len(all_indices)
        samples_per_rank = total_samples // self.num_replicas
        rank_start = self.rank * samples_per_rank
        rank_end = rank_start + samples_per_rank
        rank_all_indices = all_indices[rank_start:rank_end]

        # Split group indices by rank
        rank_group_indices = []
        for group in shuffled_groups:
            if len(group) == 0:
                rank_group_indices.append(np.array([], dtype=np.int64))
            else:
                group_per_rank = max(1, len(group) // self.num_replicas)
                start = self.rank * group_per_rank
                end = min(start + group_per_rank, len(group))
                rank_group_indices.append(group[start:end])

        # Pointers for cycling through each group
        group_pointers = [0] * self.num_modality_groups
        fill_pointer = 0

        # Pre-build all batches, then flatten to yield individual indices
        for batch_idx in range(self.num_batches):
            batch = []
            batch_set = set()

            # Step 1: Sample num_appear_each_modality from each non-empty group
            for g in self.non_empty_groups:
                group = rank_group_indices[g]
                if len(group) > 0:
                    n_take = min(self.num_appear_each_modality, len(group))
                    for _ in range(n_take):
                        idx_in_group = group_pointers[g] % len(group)
                        sample_idx = int(group[idx_in_group])
                        group_pointers[g] += 1
                        if sample_idx not in batch_set:
                            batch.append(sample_idx)
                            batch_set.add(sample_idx)

            # Step 2: Fill remaining slots from the general pool, skipping duplicates
            remaining = self.batch_size - len(batch)
            for _ in range(remaining):
                if len(rank_all_indices) > 0:
                    attempts = 0
                    while attempts < len(rank_all_indices):
                        candidate = int(rank_all_indices[fill_pointer % len(rank_all_indices)])
                        fill_pointer += 1
                        if candidate not in batch_set:
                            batch.append(candidate)
                            batch_set.add(candidate)
                            break
                        attempts += 1

            # Step 3: Shuffle the batch order
            rng.shuffle(batch)

            # Yield individual indices (not the batch list)
            for idx in batch:
                yield idx

    def __len__(self) -> int:
        """Return the total number of samples (not batches)."""
        return self.total_size


def collate_mixed_batch(batch, image_list_as_batch=True, feature_list_as_batch=True):
    """
    Custom collate function for batches with potentially different-sized tensors.

    This handles the case where expert features from different encoders have different
    dimensions (e.g., DermLIP: 768, RetCLIP: 768, Conch: 1024, MII: 2048), and where
    images may have different resolutions (for future native-resolution support).

    Args:
        batch: List of (image, feature, modality_index) tuples from the dataset
        image_list_as_batch: If True, return images as list of tensors (for variable sizes);
                             if False, stack into single tensor [B, C, H, W]
        feature_list_as_batch: If True, return expert image features as list of tensors (for variable dims);
                               if False, attempt to stack into single tensor [B, D]

    Returns:
        images: Tensor [B, C, H, W] or List of tensors [C, H, W] (if image_list_as_batch=True)
        image_features: List of tensors [D_i,] or Tensor [B, D] (if feature_list_as_batch=False and dims match)
        modality_indices: Tensor [B,]
    """
    images, image_features, modality_indices = zip(*batch)

    # Always stack modality_indices into a tensor
    modality_indices = torch.tensor(modality_indices, dtype=torch.long)

    # Handle images
    if image_list_as_batch:
        # Keep as list for variable-size images
        images = list(images)
    else:
        # Try to stack; will raise error if shapes differ
        image_shapes = [tuple(f.shape) for f in images]
        try:
            images = torch.stack(images, dim=0)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                raise RuntimeError(
                    f"Images have different shapes ({image_shapes}) and cannot be stacked. "
                    f"Set image_list_as_batch=True to handle variable-size images. "
                    f"Original error: {e}"
                )
            raise

    # Handle features
    if all(feature is None for feature in image_features):
        image_features = None
    elif feature_list_as_batch:
        # Keep as list for variable-dimension features (default for MEPA)
        image_features = list(image_features)
    else:
        # Try to stack; will raise error if dims differ
        feature_shapes = [tuple(f.shape) for f in image_features]
        try:
            image_features = torch.stack(image_features, dim=0)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                raise RuntimeError(
                    f"Image features have different dimensions ({feature_shapes}) and cannot be stacked. "
                    f"Set feature_list_as_batch=True to handle variable-dimension features. "
                    f"Original error: {e}"
                )
            raise

    return images, image_features, modality_indices


class CsvImageFeatureDataset(Dataset):
    def __init__(
        self, 
        input_filename, 
        image_key='identifier', 
        caption_key='caption', 
        image_root=None,
        image_feature_root=None,
        tokenizer=None, 
        do_aug=False,
        prob_flip=0.5,
        prob_rot=0.1,
        prob_grayscale=0.1,
        image_size=(256, 256),
        output_min=-1,
        output_max=1,
        num_samples=None,
        load_image_feature=True,
        use_default_feature=True,
        max_size=None,
        keep_ratio=True,
        preset_resolutions=None,
        random_preset=False,
        ablation_modality=None,
        subset_fraction=1.0,
        subset_seed=42,
        exclude_modalities=None,
    ):
        logging.debug(f'Loading csv data from {input_filename}.')
        usecols = ['identifier', 'modality', 'submodality', 'multiwindow']
        self.images = pd.read_csv(input_filename, usecols=usecols)
        self.images = exclude_modalities_from_dataframe(
            self.images,
            exclude_modalities=exclude_modalities,
            dataset_name="2D dataset",
        )
        self.images = stratified_subsample_dataframe(
            self.images,
            fraction=subset_fraction,
            seed=subset_seed,
            dataset_name='2D dataset',
        )

        logging.debug(f'[CsvImageFeatDataset] Successfully loaded csv data from {input_filename}.')
        self.image_key = image_key
        self.caption_key = caption_key
        self.tokenize = tokenizer
        self.image_root = image_root
        self.image_feature_root = image_feature_root
        self.do_aug = do_aug
        self.prob_flip = prob_flip
        self.prob_rot = prob_rot
        self.prob_grayscale = prob_grayscale
        self.image_size = image_size
        self.output_min = output_min  # [NOTE] min value for normalization, model-specific
        self.output_max = output_max  # [NOTE] max value for normalization, model-specific
        self.load_image_feature = load_image_feature
        self.use_default_feature = use_default_feature
        self.ablation_modality = ablation_modality

        if num_samples is not None:
            num_samples = int(num_samples)
            self.images = self.images.iloc[:num_samples]
        self.num_samples = num_samples

        print(f"[CsvImageFeatDataset] Loaded data (n={len(self.images)}, image_size={image_size}, do_aug={do_aug}, output_range={(output_min, output_max)})")
        self.ct_hu_modalities = ('ct', 'ct3d', 'ct3d2d')
        self.read_ct_image = ReadMedicalImage(output_min, output_max, modality='ct', ct_modalities=self.ct_hu_modalities)
        self.read_common_image = ReadMedicalImage(output_min, output_max, modality=None, ct_modalities=self.ct_hu_modalities)
        self.transforms = build_image_transform(
            image_size,
            is_train=do_aug,
            flip_prob=prob_flip,
            rot_prob=prob_rot,
            grayscale_prob=prob_grayscale,
            max_size=max_size,
            keep_ratio=keep_ratio,
            preset_resolutions=preset_resolutions,
            random_preset=random_preset
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                row = self.images.iloc[idx]
                data = self.read_data(row)
                image = data["image"]
                image_feature = data["image_feature"]
                midx = data["modality_index"]
                return image, image_feature, midx
            except Exception as e:
                warnings.warn(f"[CsvImageFeatDataset] Skipping broken sample idx={idx}: {e}")
                idx = np.random.randint(0, len(self.images))
        raise RuntimeError(f"[CsvImageFeatDataset] Failed to load a valid sample after {max_retries} retries")
    
    def get_image_path(self, row):
        image_identifier = row[self.image_key]
        image_root = self.image_root
        image_path = image_identifier if image_root is None else os.path.join(image_root, image_identifier)
        return image_path
    
    def get_image_feature_path(self, row, window_name='default', modality_index=None):
        image_identifier = row[self.image_key]
        feature_identifier = force_suffix(image_identifier, suffix='.pt')
        feature_path = os.path.join(self.image_feature_root, window_name, feature_identifier)
        direct_feature_path = os.path.join(self.image_feature_root, feature_identifier)
        if not os.path.exists(feature_path) and os.path.exists(direct_feature_path):
            feature_path = direct_feature_path

        return feature_path
    
    def read_data(self, row):
        image_path = self.get_image_path(row)
        submodality = row.get('submodality', 'unknown')
        midx = get_modality_group(submodality)
        
        if submodality in self.ct_hu_modalities:
            image = np.load(image_path) if image_path.endswith('.npy') else Image.open(image_path)
            data = self.read_ct_image(image)
            image = data["image"]
            window_name = data["window_name"]
        else:
            image = np.load(image_path) if image_path.endswith('.npy') else Image.open(image_path)
            data = self.read_common_image(image)
            image = data["image"]
            window_name = 'default'
        
        image = self.transforms(image)
        image = image.contiguous().clone()  # Ensure tensor has its own memory for pin_memory

        if self.load_image_feature:
            if not self.ablation_modality:
                feature_window = 'default' if self.use_default_feature else window_name
            else:
                if self.ablation_modality in DERM_MODALITIES:
                    feature_window = 'ablation' if midx == 0 else 'default'
                elif self.ablation_modality in CFP_MODALITIES:
                    feature_window = 'ablation' if midx == 1 else 'default'
                elif self.ablation_modality in PATH_MODALITIES:
                    feature_window = 'ablation' if midx == 2 else 'default'
                elif self.ablation_modality in SURG_MODALITIES:
                    feature_window = 'ablation' if midx == 3 else 'default'
                elif self.ablation_modality == 'all':
                    feature_window_tmp = 'default' if self.use_default_feature else window_name
                    feature_window = 'ablation' if midx in [0, 1, 2, 3] else feature_window_tmp
                else:
                    feature_window = 'default' if self.use_default_feature else window_name

            feature_path = self.get_image_feature_path(row, window_name=feature_window, modality_index=midx)
            
            try:
                image_feature = torch.load(feature_path, map_location='cpu').squeeze()  # [dim,]
            except Exception as err:
                raise ValueError(
                    f"Failed to load image feature from {feature_path}: {err}"
                ) from err
            
        else:
            image_feature = None
        
        return {"image": image, "image_feature": image_feature, "modality_index": midx}
    
    def __del__(self):
        if hasattr(self, 'images'):
            del self.images
        gc.collect()

    def get_submodalities(self) -> List[str]:
        """Return list of submodalities for all samples (used by DiversityBatchSampler)."""
        return self.images['submodality'].tolist()


def get_csvimgfeat_dataset(args, is_train, epoch=0, iters=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    num_samples = args.train_num_samples if is_train else args.val_num_samples
    image_root = args.train_root if is_train else args.val_root
    image_feature_root = args.feature_root
    load_image_feature = args.lshare > 0
    do_aug = True if is_train else False
    
    mixed_resolution_training = bool(args.mixed_resolution_training)
    keep_ratio = bool(args.keep_ratio)
    preset_resolutions = (256, 384, 512) if mixed_resolution_training else None
    # This can be True or False during training; evaluation does not use SmartResize.
    # It selects random or nearest target resolutions during training. Enabled by default.
    random_preset = True if is_train else False
    ablation_modality = getattr(args, 'ablation_modality', None)
    include_3d_data = getattr(args, 'include_3d_data', False)
    num_modality_groups =  NUM_MODALITY_GROUPS_2D #NUM_MODALITY_GROUPS_ALL if include_3d_data else NUM_MODALITY_GROUPS_2D
    subset_fraction = getattr(args, 'train_fraction_2d', 1.0) if is_train else 1.0
    subset_seed = getattr(args, 'subset_seed', 42)
    exclude_modalities = getattr(args, 'exclude_modalities', None) if is_train else None

    print(f"[get_csvimgfeat_dataset] ⚠️ Settings: preset_resolutions={preset_resolutions}, ablation_modality={ablation_modality}") #, include_3d_data={include_3d_data}")

    assert input_filename
    dataset = CsvImageFeatureDataset(
        input_filename,
        image_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        image_root=image_root,
        image_feature_root=image_feature_root,
        load_image_feature=load_image_feature,
        do_aug=do_aug,
        prob_flip=args.prob_flip,
        prob_rot=args.prob_rot,
        prob_grayscale=args.prob_grayscale,
        image_size=args.img_size,
        num_samples=num_samples,
        max_size=512,
        keep_ratio=keep_ratio,
        preset_resolutions=preset_resolutions,
        random_preset=random_preset,
        ablation_modality=ablation_modality,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        exclude_modalities=exclude_modalities,
    )

    # Custom collate function for handling variable-dimension features and variable-size images
    image_list_as_batch = getattr(args, 'image_list_as_batch', True)
    feature_list_as_batch = getattr(args, 'feature_list_as_batch', True)
    collate_fn = partial(
        collate_mixed_batch,
        image_list_as_batch=image_list_as_batch,
        feature_list_as_batch=feature_list_as_batch
    )

    # Determine sampler/batch_sampler based on diversity_sampling setting
    diversity_sampling = getattr(args, 'diversity_sampling', True)
    num_appear_each_modality = getattr(args, 'num_appear_each_modality', 1)
    num_available_modality_groups = len({
        get_modality_group(submodality)
        for submodality in dataset.get_submodalities()
    })
    use_diversity_sampler = (
        is_train and
        diversity_sampling and
        args.local_bs >= num_available_modality_groups * num_appear_each_modality
    )

    if use_diversity_sampler:
        # Use DiversitySampler for diverse batches (standard Sampler interface)
        sampler = DiversitySampler(
            submodalities=dataset.get_submodalities(),
            batch_size=args.local_bs,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=args.seed if args.seed is not None else 0,
            drop_last=True,
            #include_3d_data=include_3d_data,
            num_appear_each_modality=num_appear_each_modality,
        )
        # Standard DataLoader interface - compatible with existing training code
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_bs,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            collate_fn=collate_fn
        )
    else:
        # Fall back to standard DistributedSampler
        if is_train and args.local_bs < num_available_modality_groups * num_appear_each_modality and diversity_sampling:
            print(f"[get_csvimgfeat_dataset] ⚠️❌⚠️ WARNING: batch_size ({args.local_bs}) < {num_available_modality_groups} * {num_appear_each_modality}, "
                  f"cannot guarantee diversity. Using standard sampler.")
        sampler = DistributedSampler(dataset) if is_train else None
        shuffle = is_train and sampler is None
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_bs,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
            persistent_workers=False,
            collate_fn=collate_fn
        )
    num_samples = len(dataset)
    num_batches = len(dataloader)

    return DataInfo(dataloader, num_samples, num_batches, sampler=sampler)
