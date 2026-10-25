import os
import PIL
import math
import torch
import random
import numpy as np

from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False

from typing import Optional, Sequence, Union, List, Tuple

from torchvision.transforms import v2 as transv2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F



OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


_Number = Union[int, float]
_SizeArg = Union[int, Sequence[int], None]


class SmartResize(transv2.Transform):
    """
    A Resize-like transform with extra features:
      1) keep_ratio (only effective when size is a scalar int)
      2) ensure_patch_size: make H and W divisible by a given patch size
      3) preset_resolutions: snap the LONG side to nearest preset value; short side depends on keep_ratio

    Notes:
      - If preset_resolutions is not None, then size must be None; otherwise raise.
      - If size is a (h, w) tuple/list, keep_ratio is ignored (exact resize).
      - If ensure_patch_size > 1 and preset_resolutions is provided, every preset must be divisible by ensure_patch_size.
    """
    _transformed_types = (torch.Tensor, Image.Image, np.ndarray)
    def __init__(
        self,
        size: _SizeArg = None,
        *,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = True,
        keep_ratio: bool = True,
        ensure_patch_size: Optional[int] = 16,
        preset_resolutions: Optional[Sequence[int]] = None,
        random_preset: bool = False,
    ):
        super().__init__()
        self.random_preset = bool(random_preset)
        
        # -------- validate preset vs size --------
        if preset_resolutions is not None:
            if size is not None:
                raise ValueError(f'Two arguments are conflicting: `preset_resolution={preset_resolutions}`, and `size={size}`. Try disabling one.')
            if not isinstance(preset_resolutions, (list, tuple)) or len(preset_resolutions) == 0:
                raise ValueError("preset_resolutions must be a non-empty list/tuple of ints.")
            for r in preset_resolutions:
                if not isinstance(r, int) or r <= 0:
                    raise ValueError(f"Invalid preset resolution: {r}. Must be positive int.")
                if max_size is not None and r > max_size:
                    raise ValueError(f"Invalid preset resolution: {r}. Must be smaller than max_size={max_size}")
            self.preset_resolutions = tuple(sorted(set(preset_resolutions)))
        else:
            self.preset_resolutions = None

        # -------- validate size / max_size --------
        if size is not None:
            if isinstance(size, (list, tuple)):
                if len(size) != 2:
                    raise ValueError("When size is a sequence, it must have length 2 (H, W).")
                if max_size is not None:
                    # aligns with torchvision Resize behavior
                    raise ValueError("max_size should be None when size is a sequence (H, W).")
                if not (isinstance(size[0], int) and isinstance(size[1], int) and size[0] > 0 and size[1] > 0):
                    raise ValueError("When size is a sequence, both H and W must be positive ints.")
                self.size = (int(size[0]), int(size[1]))
            else:
                if not isinstance(size, int) or size <= 0:
                    raise ValueError("When size is a scalar, it must be a positive int.")
                self.size = int(size)
        else:
            self.size = None

        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError("max_size must be a positive int or None.")

        # -------- validate ensure_patch_size --------
        if ensure_patch_size is not None:
            if not isinstance(ensure_patch_size, int) or ensure_patch_size < 1:
                raise ValueError("ensure_patch_size must be a positive int or None.")
            if ensure_patch_size == 1:
                ensure_patch_size = None  # treat as disabled
        self.ensure_patch_size = ensure_patch_size

        if self.preset_resolutions is not None and self.ensure_patch_size is not None and self.ensure_patch_size > 1:
            p = self.ensure_patch_size
            bad = [r for r in self.preset_resolutions if (r % p) != 0]
            if bad:
                raise ValueError(f"All preset_resolutions must be divisible by ensure_patch_size={p}. Bad: {bad}")

        self.keep_ratio = bool(keep_ratio)
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def make_params(self, flat_inputs):
        # deterministic by default
        if self.preset_resolutions is not None and self.random_preset:
            idx = int(torch.randint(0, len(self.preset_resolutions), (1,)).item())
            return {"forced_preset_short": int(self.preset_resolutions[idx])}
        return {}

    @staticmethod
    def _round_to_multiple(x: int, m: int) -> int:
        # avoid returning 0 when x < m
        if x <= 0:
            return m
        if x < m:
            return m

        lower = (x // m) * m
        upper = lower + m

        # Tie-breaker: if exactly halfway, choose lower.
        return lower if (x - lower <= upper - x) else upper

    @staticmethod
    def _round_int(x: float) -> int:
        # stable "round half up"
        return int(math.floor(x + 0.5))

    def _choose_nearest_preset(self, long_side: int) -> int:
        # nearest by abs diff; if tie, choose the smaller preset
        presets = self.preset_resolutions
        assert presets is not None
        return min(presets, key=lambda r: (abs(r - long_side), r))

    def _compute_target_hw(self, h: int, w: int, forced_preset_short=None) -> Tuple[int, int]:
        """
        Compute target (H, W) based on:
          - preset_resolutions OR size
          - keep_ratio when size is scalar OR preset_resolutions
          - max_size when size is scalar & keep_ratio=True (align with Resize)
          - ensure_patch_size post adjustment
        """
        # ---- case 1: preset_resolutions mode ----
        if self.preset_resolutions is not None:
            long_side = max(h, w)
            short_side = min(h, w)

            if forced_preset_short is not None:
                target_short = int(forced_preset_short)
            else:
                target_short = self._choose_nearest_preset(short_side)

            if self.keep_ratio:
                scale = target_short / float(short_side)
                new_short = target_short
                new_long = self._round_int(long_side * scale)
                # max_size takes priority over keep_ratio.
                if self.max_size is not None:
                    new_long = min(self.max_size, new_long)

                if h <= w:
                    new_h, new_w = new_short, new_long
                else:
                    new_h, new_w = new_long, new_short
            else:
                new_h = target_short
                new_w = target_short

        # ---- case 2: size mode ----
        else:
            if self.size is None:
                raise ValueError("Either size or preset_resolutions must be provided.")

            # size is (H, W)
            if isinstance(self.size, tuple):
                new_h, new_w = int(self.size[0]), int(self.size[1])

            # size is scalar int
            else:
                s = int(self.size)
                if self.keep_ratio:
                    short_side = min(h, w)
                    long_side = max(h, w)

                    scale = s / float(short_side)
                    new_short = s
                    new_long = self._round_int(long_side * scale)

                    # Apply max_size like torchvision Resize (only meaningful for keep_ratio=True, scalar size)
                    if self.max_size is not None and new_long > self.max_size:
                        scale2 = self.max_size / float(long_side)
                        new_long = int(self.max_size)
                        new_short = self._round_int(short_side * scale2)

                    if h <= w:
                        new_h, new_w = new_short, new_long
                    else:
                        new_h, new_w = new_long, new_short
                else:
                    new_h, new_w = s, s

        # ---- ensure_patch_size adjustment ----
        if self.ensure_patch_size is not None and self.ensure_patch_size > 1:
            p = self.ensure_patch_size
            if (new_h % p) != 0:
                new_h = self._round_to_multiple(new_h, p)
            if (new_w % p) != 0:
                new_w = self._round_to_multiple(new_w, p)

        # Avoid degenerate
        new_h = max(1, int(new_h))
        new_w = max(1, int(new_w))
        return new_h, new_w

    def _as_tensor_preserve_layout(self, x):
        """
        Convert numpy / tensor to a tensor that F.resize understands (channel-first when needed),
        and return (tensor, meta) so we can restore the original type/layout.
        """
        meta = {"kind": None, "layout": None, "dtype": None}

        if isinstance(x, np.ndarray):
            meta["kind"] = "numpy"
            meta["dtype"] = x.dtype
            arr = np.ascontiguousarray(x)

            if arr.ndim == 2:
                meta["layout"] = "HW"
                t = torch.from_numpy(arr).unsqueeze(0)
                return t, meta

            if arr.ndim == 3:
                # prefer HWC if last dim looks like channel count
                if arr.shape[-1] in (1, 3, 4):
                    meta["layout"] = "HWC"
                    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
                elif arr.shape[0] in (1, 3, 4):
                    meta["layout"] = "CHW"
                    t = torch.from_numpy(arr)  # already CHW
                else:
                    # fallback assume HWC
                    meta["layout"] = "HWC"
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                return t, meta

            if arr.ndim == 4:
                # prefer NHWC if last dim looks like channel count
                if arr.shape[-1] in (1, 3, 4):
                    meta["layout"] = "NHWC"
                    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # NCHW
                elif arr.shape[1] in (1, 3, 4):
                    meta["layout"] = "NCHW"
                    t = torch.from_numpy(arr)
                else:
                    meta["layout"] = "NHWC"
                    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
                return t, meta

            raise ValueError(f"Unsupported numpy ndim={arr.ndim}")

        if isinstance(x, torch.Tensor):
            meta["kind"] = "tensor"
            # We generally assume torch tensor images are CHW/NCHW, but try to fix obvious HWC/NHWC.
            t = x
            if t.ndim == 3 and (t.shape[-1] in (1, 3, 4)) and (t.shape[0] not in (1, 3, 4)):
                meta["layout"] = "HWC"
                t = t.permute(2, 0, 1)  # CHW
            elif t.ndim == 4 and (t.shape[-1] in (1, 3, 4)) and (t.shape[1] not in (1, 3, 4)):
                meta["layout"] = "NHWC"
                t = t.permute(0, 3, 1, 2)  # NCHW
            else:
                meta["layout"] = "as_is"
            return t, meta

        # PIL or others: keep as-is, F.resize can handle PIL
        meta["kind"] = "pil"
        meta["layout"] = "as_is"
        return x, meta

    def transform(self, inpt, params):
        x, meta = self._as_tensor_preserve_layout(inpt)

        # infer H,W robustly: for tensor, last two dims are H,W (since we normalized layout)
        if isinstance(x, torch.Tensor):
            h, w = int(x.shape[-2]), int(x.shape[-1])
        else:  # PIL: size is (W,H)
            w, h = x.size
            h, w = int(h), int(w)

        forced = params.get("forced_preset_short", None)
        new_h, new_w = self._compute_target_hw(h, w, forced_preset_short=forced)

        return F.resize(
            x,
            size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, preset_resolutions={self.preset_resolutions}, random_preset={self.random_preset}"
            f"keep_ratio={self.keep_ratio}, ensure_patch_size={self.ensure_patch_size}, "
            f"interpolation={self.interpolation}, max_size={self.max_size}, antialias={self.antialias}"
        )


class ReadMedicalImage(transv2.Transform):
    _transformed_types = (torch.Tensor, Image.Image, np.ndarray)
    def __init__(self, output_min=-1, output_max=1, modality=None, p=0.5, ct_modalities=('ct', 'ct3d', 'ct3d2d')):
        super().__init__()
        self.p = p
        self.output_min = output_min
        self.output_max = output_max
        self.modality = modality.lower() if modality else 'unknown'
        
        self.ct_modalities = ct_modalities
        self.ct_window_names = ('default', 'soft', 'lung', 'full')
        self.ct_window_probs = (0.4, 0.3, 0.1, 0.2)
        self.ct_window_dict = {
            'default': (-1000, 1000), 
            'soft': (-150, 250), 
            'lung': (-1400, 200), 
            'full': (-1000, 2000),
        }
    
    def make_params(self, flat_inputs):
        apply = random.random() < self.p

        if self.modality in self.ct_modalities:
            if apply:
                window_name = np.random.choice(self.ct_window_names, p=self.ct_window_probs)
            else:
                window_name = 'default'
            window = self.ct_window_dict[window_name]
        else:
            window_name = 'default'
            window = (0, 255)

        return {"window": window, "window_name": window_name}

    def transform(self, inpt, params):
        if isinstance(inpt, np.ndarray):
            img = np.atleast_3d(inpt)
        elif isinstance(inpt, Image.Image):
            if self.modality in self.ct_modalities:
                img = np.array(inpt, dtype=np.int16)
            else:
                img = np.array(inpt.convert('RGB'))
        else:
            img = inpt

        img = self.to_rgb_tensor(img)

        window = params["window"]
        window_name = params["window_name"]
        min_val, max_val = min(window), max(window)
        img = stable_linear_transform(img, y_min=self.output_min, y_max=self.output_max, x_min=min_val, x_max=max_val)
        
        return {"image": img, "window_name": window_name}
    
    def to_rgb_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32).squeeze()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and (x.shape[-1] <= 4):  # Some images have four channels.
            x = x.permute(2, 0, 1).contiguous()
        
        if x.shape[0] == 1:
            x = x.repeat_interleave(3, dim=0)
        elif x.shape[0] > 3:
            x = x[:3]
        
        if x.shape[0] != 3:
            raise ValueError(f"Invalid shape: {x.shape}")
        
        return x
        


class RandomFixedRotation(transv2.Transform):
    def __init__(self, angles=(90, -90, 180), p=0.5):
        super().__init__()
        assert 0. <= p <= 1.
        self.angles = angles
        self.p = p

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        angle = random.choice(self.angles) if apply else 0
        return {"apply": apply, "angle": angle}

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        
        angle = params['angle']
        k = angle // 90
        if k > 0:
            return torch.rot90(inpt, k=k, dims=[-2,-1])
        elif k < 0:
            return torch.rot90(inpt, k=k, dims=[-1,-2])
        else:
            return inpt


class RandomGrayScale(transv2.Transform):
    def __init__(self, p=0.2):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p
        self.transf = transv2.Grayscale(num_output_channels=3)

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        return {"apply": apply}

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        else:
            return self.transf(inpt)


class RandomColorJitter(transv2.Transform):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.1):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transf = transv2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        return {"apply": apply}
    
    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        else:
            return self.transf(inpt)



def build_image_transform(
    image_size: int,
    is_train: bool,
    scale_range=(0.9, 1.0),
    color_jitter_prob=0,
    color_jitter_kwargs=None,
    grayscale_prob=0,
    flip_prob=0,
    rot_kwargs=None,
    rot_prob=0,
    max_size=None,
    keep_ratio=True,
    preset_resolutions=None,
    random_preset=False,
):

    # if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
    #     # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
    #     image_size = image_size[0]
    
    color_jitter_kwargs = {} if color_jitter_kwargs is None else color_jitter_kwargs
    rot_kwargs = {} if rot_kwargs is None else rot_kwargs

    if is_train:
        image_size = None if preset_resolutions is not None else image_size
        transforms = [
            #transv2.RandomResizedCrop(scale=scale_range, size=image_size, antialias=True),
            #transv2.Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            SmartResize(
                image_size, 
                max_size=max_size,
                interpolation=InterpolationMode.BICUBIC, 
                preset_resolutions=preset_resolutions,
                keep_ratio=keep_ratio,
                ensure_patch_size=16,
                random_preset=random_preset,
            ),
            transv2.RandomHorizontalFlip(p=flip_prob),
            transv2.RandomVerticalFlip(p=flip_prob),
            RandomFixedRotation(p=rot_prob, **rot_kwargs),
        ]

        if color_jitter_prob:
            transforms.append(RandomColorJitter(p=color_jitter_prob, **color_jitter_kwargs))

        if grayscale_prob:
            transforms.append(RandomGrayScale(grayscale_prob))
        
    else:
        if not isinstance(image_size, (list, tuple)):
            image_size = (image_size, image_size)
        transforms = [transv2.Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC)]

    return transv2.Compose(transforms)
