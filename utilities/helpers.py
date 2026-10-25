import os
import re
import ast
import html
import json
import math
import random
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist

from PIL import Image
from io import BytesIO
from scipy import stats
from typing import Union
from matplotlib import cm
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter, deque
from torchvision.transforms.functional import rotate
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


################################################################################################
# NUMBER PROCESSING
################################################################################################

def true_round(value):
    """
    Perform rounding as taught in school, i.e., rounding to nearest integer 
    without considering it's odd or even.
    """
    return np.floor(value + 0.5).astype(int) if value >= 0 else np.ceil(value - 0.5).astype(int)


def nearest_divisible(num, n=4, output_upper=True):
    lower = num - (num % n)
    upper = lower + n
    if output_upper:
        return upper if lower != num else lower
    else:
        return lower


def nearest_power(n, p=2, output_type='upper'):
    assert n > 0, "Input number must be positive."

    lower_power = p ** (int(n).bit_length() - 1)
    upper_power = lower_power * p

    if output_type == 'upper':
        return upper_power if lower_power != n else lower_power
    elif output_type == 'lower':
        return lower_power
    else:
        return lower_power if abs(n - lower_power) <= abs(n - upper_power) else upper_power


def is_invalid_num(x):
    return math.isnan(x) or (x is None) or (x == '') or (not isinstance(x, (int, float)))


################################################################################################
# FILE READING, WRITING AND SEARCH
################################################################################################

def make_paths(*paths):
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)

def get_filename(path):
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return filename

def force_suffix(path, suffix):
    if suffix:  # if suffix is not empty, it should start with a dot
        suffix = ('.' + suffix) if not suffix.startswith('.') else suffix
    filename = get_filename(path)
    dirname = os.path.dirname(path)
    return os.path.join(dirname, filename + suffix)


def exist(path):
    return (path is not None) and os.path.exists(path)

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def write_txt(data, txt_path, encoding='utf-8'):
    with open(txt_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + '\n')


def read_json(json_path, encoding='utf-8'):
    with open(json_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def write_json(data, json_path, write_mode='w', encoding='utf-8', ensure_ascii=False):
    with open(json_path, write_mode, encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)


def read_jsonl(file_path, encoding='utf-8', skip_error=False):
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))  # Convert each JSONL line to a dictionary
            except Exception as err:
                print(f"Error when loading Line {idx} in {file_path}: {err}")
                if skip_error:
                    continue
                else:
                    raise err
    return data


def write_jsonl(dicts, jsonl_path, encoding='utf-8', write_mode='w', ensure_ascii=False):
    if not isinstance(dicts, (list, tuple)):
        dicts = [dicts,]
        
    with open(jsonl_path, write_mode, encoding=encoding) as f:
        for adict in dicts:
            f.write(json.dumps(adict, ensure_ascii=ensure_ascii) + '\n')
                    
        

def find_files_recursively(directory, ext='', inclusions='', exclusions=None):
    """
    Args:
    - ext: can be a string or a tuple (find multiple strings).
    - inclusions: can be a string or a list. Ensure the name of each retrieved file includes all the strings specified.
    - exclusions: can be a string or a list or None. Ensure the name of each retrieved file does not include any of the strings.
    """
    matched_paths = []
    
    # Ensure inclusions and exclusions are lists
    if not isinstance(inclusions, (list, tuple)):
        inclusions = [inclusions]
    if exclusions is not None and not isinstance(exclusions, (list, tuple)):
        exclusions = [exclusions]

    if not os.path.exists(directory):
        raise ValueError(f'Path does not exist: {directory}.')

    for root, dirs, files in os.walk(directory):
        filtered_dirs = []
        
        for d in dirs:
            full_path = os.path.join(root, d)
            # Only treat it as a file if it's actually a file
            if d.endswith('.nii.gz') and os.path.isfile(full_path):
                files.append(d)  # Treat it as a file
            else:
                filtered_dirs.append(d)  # Keep directories that are actually directories

        dirs[:] = filtered_dirs  # Modify dirs in place to control recursion

        for file in files:
            cond1 = all(s in file for s in inclusions)
            cond2 = file.endswith(ext)
            cond3 = all(s not in file for s in exclusions) if exclusions else True

            if cond1 and cond2 and cond3:
                matched_paths.append(os.path.join(root, file))

    return sorted(matched_paths)



def count_files(root: str, *, follow_symlinks: bool = False) -> int:
    """
    Count files under root, including all nested subdirectories.
    By default, skip directory symlinks to avoid loops, and skip entries without access permission.
    """
    count = 0
    stack = deque([root])
    # Use local references for a small speed improvement.
    scandir = os.scandir
    pop = stack.pop
    push = stack.append
    while stack:
        path = pop()
        try:
            with scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=follow_symlinks):
                            count += 1
                        elif entry.is_dir(follow_symlinks=follow_symlinks):
                            push(entry.path)
                    except OSError:
                        # Skip broken symlinks or files deleted during the scan.
                        continue
        except PermissionError:
            # Skip directories we do not have permission to read.
            continue
        except FileNotFoundError:
            # Skip directories deleted during the scan.
            continue
    return count


def find_files_recursively_fast(
    directory,
    ext = "",
    inclusions="",
    exclusions=None,
    sort=False,
    follow_symlinks=False,
):
    """
    Faster recursive file finder using os.scandir().

    Args:
        directory: root directory to scan.
        ext: '' (match all), a string (single suffix), or a tuple of suffixes.
        inclusions: string or iterable of strings; all must be substrings of filename.
        exclusions: None, string, or iterable; none may be substrings of filename.
        sort: whether to sort results (can be expensive for huge lists).
        follow_symlinks: whether to follow symlinks when deciding file/dir.

    Returns:
        List of matched file paths.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory or does not exist: {directory}")

    # Normalize ext
    if isinstance(ext, str):
        ext_tuple = () if ext == "" else (ext,)
    else:
        ext_tuple = tuple(e for e in ext if e)  # drop empty strings

    # Normalize inclusions/exclusions, drop empty strings to avoid wasted checks
    if isinstance(inclusions, (str, bytes)):
        inc_list = [inclusions]
    else:
        inc_list = list(inclusions)
    inc_list = [s for s in inc_list if s]

    if exclusions is None:
        exc_list = []
    elif isinstance(exclusions, (str, bytes)):
        exc_list = [exclusions]
    else:
        exc_list = list(exclusions)
    exc_list = [s for s in exc_list if s]

    matched: List[str] = []
    stack = [directory]

    while stack:
        root = stack.pop()
        try:
            with os.scandir(root) as it:
                for entry in it:
                    # Skip broken entries / permission issues per-entry
                    try:
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            stack.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=follow_symlinks):
                            continue
                    except OSError:
                        continue

                    name = entry.name

                    # Extension check
                    if ext_tuple and not name.endswith(ext_tuple):
                        continue

                    # Inclusion/exclusion substring checks (filename only, as your code does)
                    if inc_list and not all(s in name for s in inc_list):
                        continue
                    if exc_list and any(s in name for s in exc_list):
                        continue

                    matched.append(entry.path)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            # Directory vanished or inaccessible — ignore and keep going
            continue

    if sort:
        matched.sort()
    return matched


def find_folders(directory, inclusions=''):
    if not isinstance(inclusions, (list, tuple)):
        inclusions = [inclusions]  # make it a list by default
        
    names = os.listdir(directory)
    folders = []
    for n in names:
        folder = os.path.join(directory, n)
        if os.path.isdir(folder):
            if all(s in n for s in inclusions):
                folders.append(folder)
    
    return sorted(folders)


def check_multiple_exist(paths):
    if not isinstance(paths, (tuple, list)):
        paths = [paths]
    
    for p in paths:
        assert os.path.exists(p), f'Path does not exist: {p}'


################################################################################################
# STRING PROCESSING
################################################################################################

def normalize_spaces(s: str) -> str:
    """
    Replace all groups of multiple spaces with a single space.
    Keeps other whitespace (tabs, newlines) untouched.
    """
    return re.sub(r' {2,}', ' ', s)

def extract_text_from_html(raw_html: str) -> str:
    # Step 1: Use BeautifulSoup to parse and remove tags
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n")  # Preserve some structure with line breaks

    # Step 2: Decode HTML entities (e.g., &lt;, &gt;)
    clean_text = html.unescape(text)

    # Step 3: Strip leading/trailing whitespace
    clean_text = clean_text.strip()
    clean_text = clean_text.replace('\n', ' ')
    clean_text = normalize_spaces(clean_text)
    return clean_text


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')



################################################################################################
# LIST and DICT PROCESSING
################################################################################################


def all_tuples_equal(tuples_list):
    if not tuples_list:
        return True
    first = tuples_list[0]
    return all(t == first for t in tuples_list[1:])
    

def all_equal(iterable):
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        # Treat all elements of an empty iterable as equal.
        return True
    
    return all(elem == first for elem in iterator)


def interleave_lists(a, b):
    if len(a) != len(b):
        raise ValueError("The two lists must have the same length.")
    
    interleaved = [item for pair in zip(a, b) for item in pair]
    return interleaved


def concat_lists(list_list):
    return [item for alist in list_list for item in alist]


def count_elements(lst):
    return dict(Counter(lst))


def sort_dict_by_keys(d, reverse=False):
    return dict(sorted(d.items(), key=lambda item: item[0], reverse=reverse))


def concat_dicts(dict_list):
    merged_dict = {}
    for d in dict_list:
        overlapping_keys = merged_dict.keys() & d.keys()  # Find common keys
        if overlapping_keys:
            raise ValueError(f"Overlapping keys found: {overlapping_keys}")
        merged_dict.update(d)  # Merge the dictionary if no conflicts
    return merged_dict
    

def divide_list_with_overlap(a, group_len, overlap=0):
    if group_len <= 0 or overlap < 0 or group_len <= overlap:
        raise ValueError("group_len must be positive and greater than overlap.")

    groups = []
    start = 0

    while start < len(a):
        end = start + group_len
        groups.append(a[start:end])
        start += group_len - overlap
    return groups


def get_consecutive_pairs(lst):
    return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]


def get_moving_average(values, window_size=5, mode="shrink"):
    """
    For each index i, average a (near-)centered window around i.
    - Includes the element at i.
    - `mode='shrink'`: window shrinks near edges.
    - `mode='keep_size'`: try to keep exactly `window_size` elements
      by shifting the window inward when close to the edges.
    """
    assert window_size >= 1, f"window_size must be >= 1, got {window_size}"
    if window_size == 1:
        return values
        
    assert mode in ['shrink', 'keep_size'], f"mode must be 'shrink' or 'keep_size', got {mode}"
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
    if n == 0:
        return []

    half_left = window_size // 2                # how many to the left (target)
    half_right = window_size - 1 - half_left    # how many to the right (target)

    # prefix sums for O(1) range-sum queries
    prefix = np.zeros(n + 1, dtype=float)
    prefix[1:] = np.cumsum(x)

    out = np.empty(n, dtype=float)

    for i in range(n):
        # start/end of a centered window before clipping
        s = i - half_left
        e = i + half_right + 1  # exclusive

        if mode == "shrink":
            s = max(0, s)
            e = min(n, e)
        else:
            # shift right if left spills past start
            if s < 0:
                shift = -s
                s = 0
                e = min(n, e + shift)
            # shift left if right spills past end
            if e > n:
                shift = e - n
                e = n
                s = max(0, s - shift)

        # compute mean over [s, e)
        count = e - s
        # (count can be < window_size at the edges, especially in 'shrink' mode)
        total = prefix[e] - prefix[s]
        out[i] = total / count if count > 0 else np.nan

    return out.tolist()


def dynamic_counter():
    seen_elems = {}

    def count_elements(s):
        if s not in seen_elems:
            seen_elems[s] = 0
        seen_elems[s] += 1
        return seen_elems[s]
    
    return count_elements
    

class DotDict(dict):
    """
    dict with dot-attribute access:
      d.name <-> d['name']

    Safety:
      - If a key conflicts with existing dict attributes/methods (e.g., 'items', 'keys'),
        setting/accessing via dot will raise AttributeError.
    """
    __slots__ = ()

    @staticmethod
    def _conflicts_with_dict_attr(key: str) -> bool:
        # Any attribute/method name on dict or on this class is considered a conflict
        return hasattr(dict, key) or hasattr(DotDict, key)

    def __getattr__(self, name):
        # Called only if normal attribute lookup fails
        if name in self:
            if self._conflicts_with_dict_attr(name):
                raise AttributeError(
                    f"Key '{name}' conflicts with built-in dict attribute/method; "
                    f"use d['{name}'] instead."
                )
            return self[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute '{name}'")

    def __setattr__(self, name, value):
        # dot-setting writes into dict, unless conflict
        if self._conflicts_with_dict_attr(name):
            raise AttributeError(
                f"Cannot set attribute '{name}': conflicts with dict attribute/method. "
                f"Use d['{name}'] = value instead."
            )
        self[name] = value

    def __delattr__(self, name):
        if self._conflicts_with_dict_attr(name):
            raise AttributeError(
                f"Cannot delete attribute '{name}': conflicts with dict attribute/method. "
                f"Use del d['{name}'] instead."
            )
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"No such attribute/key: '{name}'") from e


def as_dotdict(d: dict) -> DotDict:
    """
    Convert a dict to DotDict, with a strict conflict check on existing keys.
    Raises if any existing key conflicts with dict's native attributes/methods.
    """
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict, got {type(d)}")

    bad = [k for k in d.keys() if isinstance(k, str) and DotDict._conflicts_with_dict_attr(k)]
    if bad:
        raise ValueError(
            "These keys conflict with dict built-ins and cannot be accessed via dot: "
            + ", ".join(repr(k) for k in sorted(set(bad)))
        )

    return DotDict(d)


def rename_dict(adict, mapping, drop=False):
    renamed_dict = {}
    
    for key, value in adict.items():
        if key in mapping:
            # If the key is in mapping, rename it
            renamed_dict[mapping[key]] = value
        elif not drop:
            # If key is not in mapping and drop is False, keep the original key
            renamed_dict[key] = value

    return renamed_dict


def ensure_rep(x, n=2):
    if n < 0:
        raise ValueError("n must be >= 0")
    if isinstance(x, (list, tuple)):
        m = len(x)
        if m == n:
            return x
        if m == 0:
            raise ValueError("Cannot repeat an empty sequence")
        k, r = divmod(n, m)
        return x * k + x[:r]   # preserves list/tuple type
    return [x] * n             # treat non-sequences (and strings) as scalars


def len_dict_of_lists(d):
    return sum(len(v) for v in d.values() if isinstance(v, list))


def update_dict_of_lists(d, key, value, only_unique_value=False):
    if key not in d.keys():
        d[key] = []
    
    do_append = value not in d[key] if only_unique_value else True

    if do_append:
        d[key].append(value)


def duplicate_list_to_length(lst, target_len):
    """
    duplicate a list until its length reaches or exceeds target_len.
    The duplicated list is NOT truncated.
    """
    assert target_len > 0, f"Length to be reached should be a positive integer, got {target_len}."
    if not lst:
        raise ValueError("Input list cannot be empty.")

    n = len(lst)
    if n >= target_len:
        return lst
    
    reps = -(-target_len // n)  # ceiling division
    return lst * reps


def pad_list(lst, target_len, mode="reflect", seed=0,):
    """
    Pad a list to a given length using different padding modes.

    Parameters
    ----------
    lst : list
        Input list to pad.
    target_len : int
        Desired output length.
    mode : {"random", "reflect", "edge", "cycle"}, default "reflect"
        - "reflect": mirror-like repetition, e.g. [1,2,3] -> [1,2,3,2,1,2,3,2]
        - "edge": repeat last element, e.g. [1,2,3] -> [1,2,3,3,3,...]
        - "cycle": repeat list, e.g. [1,2,3] -> [1,2,3,1,2,3,...]
        - "random": use random permutations of the list to pad,
                    minimizing repeats in the padded part
    seed : int or None
        Random seed for "random" mode.

    Returns
    -------
    list
        Padded list of length target_len.

    Notes
    -----
    - If len(lst) >= target_len, the list is truncated to target_len.
    - If lst is empty and target_len > 0, raises ValueError.
    """
    if target_len < 0:
        raise ValueError("target_len must be non-negative")

    n = len(lst)
    if target_len == 0:
        return []
    if n == 0:
        raise ValueError("Cannot expand an empty list when target_len > 0")

    # If already long enough, exist
    if n >= target_len:
        return lst

    extra = target_len - n

    if mode == "edge":
        # Repeat last element
        return lst + [lst[-1]] * extra

    elif mode == "cycle":
        # Repeat the whole list cyclically
        times = extra // n
        rem = extra % n
        return lst + lst * times + lst[:rem]

    elif mode == "random":
        rng = random.Random(seed)

        padded = []
        # Use as many full permutations of lst as needed
        full_blocks = extra // n
        remainder = extra % n

        for _ in range(full_blocks):
            # random.sample without replacement -> a permutation of lst
            block = rng.sample(lst, n)
            padded.extend(block)

        if remainder > 0:
            # Take a partial permutation (subset) of lst
            block = rng.sample(lst, remainder)
            padded.extend(block)

        return lst + padded

    elif mode == "reflect":
        if n == 1:
            # Reflection of a single element is just edge padding
            return lst + [lst[0]] * extra

        # Build a reflection pattern like [1,2,3] -> [1,2,3,2]
        pattern = lst + lst[-2:0:-1]

        # Repeat the pattern until we reach the desired length
        reps = math.ceil(target_len / len(pattern))
        extended = (pattern * reps)[:target_len]
        return extended

    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def random_sample_list(items, n: int, seed: int = None):
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n > len(items):
        raise ValueError("n cannot be larger than the number of items.")

    # Use a local Random instance so we don't affect global random state
    rng = random.Random(seed)
    return rng.sample(items, n)



def linspace_sample_list(lst, n):
    """
    Sample evenly spaced elements from a long list.

    Args:
        lst: list (or any sequence supporting __len__ and __getitem__)
        n:  - if n > 1: sample exactly int(n) elements (uniformly spaced)
            - if 0 < n <= 1: sample int(len(lst) * n) elements (uniformly spaced)

    Returns:
        A new list containing the sampled elements.
    """
    L = len(lst)
    if L == 0:
        return []

    if n <= 0:
        return []

    # Determine target count k
    k = int(n) if n > 1 else int(L * n)

    if k <= 0:
        return []
    if k >= L:
        return lst

    # Uniform indices across [0, L-1]
    # Use integer arithmetic to avoid floating-point accumulation errors.
    if k == 1:
        return [lst[0]]

    step_num = L - 1
    step_den = k - 1
    idxs = [(i * step_num) // step_den for i in range(k)]
    return [lst[i] for i in idxs]



################################################################################################
# ARRAY PROCESSING
################################################################################################

def print_array_info(x, name='', compute_stat=False):

    def is_array(inp):
        return hasattr(inp, 'min') and hasattr(inp, 'max') and hasattr(inp, 'mean') and hasattr(inp, 'std')

    if is_array(x):
        xmin = x.min().item()
        xmax = x.max().item()
        if compute_stat:
            mean_val = x.mean().item()
            std_val = x.std().item()
            print(f"--> {name}: shape={tuple(x.shape)}, range=({xmin:.4f}, {xmax:.4f}), mean={mean_val:.4f}, std={std_val:.4f}")
        else:
            print(f"--> {name}: shape={tuple(x.shape)}, range=({xmin:.4f}, {xmax:.4f})")
    else:
        s = f"--> {name}: Not an array "
        if isinstance(x, (list, tuple)):
            s += f"(type={type(x)}, len={len(x)}): "
            for ii, xx in enumerate(x):
                if is_array(xx):
                    xxmin = xx.min().item()
                    xxmax = xx.max().item()
                    if compute_stat:
                        mean_val = x.mean().item()
                        std_val = x.std().item()
                        s += f"\n-----> [{ii}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f}), mean={mean_val:.4f}, std={std_val:.4f}"
                    else:
                        s += f"\n-----> [{ii}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f})"
                elif isinstance(xx, (int, float)):
                    s += f"\n-----> [{ii}]: Not an array (type={type(xx)}): {xx}"
                else:
                    s += f"\n-----> [{ii}]: Not an array (type={type(xx)})"

        elif isinstance(x, dict):
            s += f"(type={type(x)}, len={len(x)}): "
            for key, xx in x.items():
                if is_array(xx):
                    xxmin = xx.min().item()
                    xxmax = xx.max().item()
                    if compute_stat:
                        mean_val = x.mean().item()
                        std_val = x.std().item()
                        s += f"\n-----> [{ii}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f}), mean={mean_val:.4f}, std={std_val:.4f}"
                    else:
                        s += f"\n-----> [{key}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f})"
                elif isinstance(xx, (int, float)):
                    s += f"\n-----> [{ii}]: Not an array (type={type(xx)}): {xx}"
                else:
                    s += f"\n-----> [{key}]: Not an array (type={type(xx)})"

        else:
            s += f"(type={type(x)})"

        print(s)


def unique(data, output_type='list'):
    out = set(data)
    if output_type == 'tuple':
        return tuple(out)
    elif output_type == 'numpy':
        return np.array(list(out))
    elif output_type == 'tensor':
        return torch.tensor(list(out))
    elif output_type == 'set':
        return out
    else:
        return list(out)
    

def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y



def unwrap_module(m):
    # works for DDP, DataParallel, torch.compile wrappers, etc.
    while hasattr(m, "module"):
        m = m.module
    return m


def is_integer_after_conv(n, num_convs, kernel_size=3, stride=2, padding=1, dilation=1):
    def get_conv_out_size(n):
        return ((n + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1

    for _ in range(num_convs):
        n = get_conv_out_size(n)
        if n != int(n):  # Check if it's not an integer during intermediate steps
            return False
    return True


def rotate_tensor(x: torch.Tensor, angle_deg, center=None, fill=None):
    # Counterclock-wise rotation
    fill = x.min().item() if fill is None else fill
    rotated_x = rotate(x, angle_deg, center=center, fill=fill)
    return rotated_x


def stable_rotate_tensor(x: torch.Tensor, k=1, dims=[-2,-1]):
    # Clock-wise rotation if dims=[-1,-2]
    # Counterclock-wise rotation if dims=[-2,-1]
    return torch.rot90(x, k=k, dims=dims)

################################################################################################
# IMAGE PROCESSING
################################################################################################


def read_and_resize_image(image_path, max_size=512):
    img = Image.open(image_path)
    width, height = img.size

    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        img = img.resize((new_width, new_height))

    return img

def arrange_corner_points(points):
    if len(points) != 4:
        raise ValueError("The input must contain exactly four points.")

    # Sort points by their y-coordinates first (top to bottom), and then by x-coordinates (left to right)
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

    # The first two points in the sorted list will be the upper points
    upper_points = sorted(sorted_points[:2], key=lambda p: p[0])  # Sort by x to get upper-left and upper-right
    # The last two points in the sorted list will be the lower points
    lower_points = sorted(sorted_points[2:], key=lambda p: p[0])  # Sort by x to get lower-left and lower-right

    # Combine the points in the desired order
    arranged_points = [upper_points[0], upper_points[1], lower_points[1], lower_points[0]]

    return arranged_points



def get_image_num_channels(img) -> int:
    if isinstance(img, Image.Image):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    elif isinstance(img, torch.Tensor):
        if img.ndim == 2:
            return 1
        elif img.ndim > 2:
            return img.shape[-3]
    raise TypeError(f"Unexpected type {type(img)}")


def pad_images_to_max_size(images, pad_value=0):
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    padded_images = []
    
    for img in images:
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]
        pad_value_ = pad_value if pad_value is not None else img.min()
        padded_img = np.pad(
            img,
            pad_width=((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=pad_value_
        )
        padded_images.append(padded_img)
    
    return padded_images


def pseudo_color_array(arr, cmap='seismic'):
    assert len(arr.shape) == 2, f'Input should be 2-dim (x,y), got {arr.shape}.'
    assert (arr.min() >= 0) and (arr.max() <= 1), f'Input should be normalized to [0, 1].'
    colormap = plt.get_cmap(cmap)
    pc_arr = colormap(arr)  # Returns an RGBA (4-channel) image
    pc_arr = (pc_arr[:, :, :3] * 255).astype(np.uint8)
    return pc_arr


def get_volume_preview(
    volume, do_pad=True, is_depth_first=True, 
    input_min=-1000, input_max=1000, 
    concat_output=True, concat_axis=1, to_uint8=False,
    cmap=None
):
    
    xval_dict = dict(x_min=input_min, x_max=input_max)
    
    if is_depth_first:
        d, h, w = volume.shape
        slices = [volume[d//2], volume[:,h//2], volume[:,:,w//2]]
    else:
        h, w, d = volume.shape
        slices = [volume[h//2], volume[:,w//2], volume[:,:,d//2]]
        
    if do_pad:
        slices = pad_images_to_max_size(slices, pad_value=input_min)
    
    if cmap is not None:
        slices = [pseudo_color_array(
            stable_linear_transform(
                v, y_min=0, y_max=1, **xval_dict
            ), cmap=cmap) for v in slices]  # [H,W,C], [0, 255]
    else:
        slices = [stable_linear_transform(
            v, y_min=0, y_max=255, **xval_dict
        ) for v in slices]
    
    if to_uint8:
        slices = [v.astype(np.uint8) for v in slices]
    
    if concat_output:
        slices = np.concatenate(slices, axis=concat_axis)
    
    return slices


def concat_pil_images(images, save_path=None, height_limit=400, force_mode='RGB'):
    """
    Concatenate a list of PIL images along the width dimension.
    """
    if len(images) == 1:
        return images[0]
    if len(images) == 0:
        return images

    # Ensure all images have the same height and mode
    heights = [img.size[1] for img in images]
    widths = [img.size[0] for img in images]
    modes = [img.mode for img in images]
    max_height = max(heights)

    if (len(set(heights)) > 1) or (max_height > height_limit):
        #raise ValueError("All images must have the same height.")
        max_height = min(max_height, height_limit)
        new_widths = [int(max_height / h * w) for (w, h) in zip(widths, heights)]
        images = [im.resize((w, max_height)) for (im, w) in zip(images, new_widths)]

    if len(set(modes)) > 1 or force_mode:
        #raise ValueError("All images must have the same mode (e.g., RGB, L).")
        images = [im.convert(force_mode) for im in images]
    
    # Compute the dimensions of the new image
    total_width = sum(img.size[0] for img in images)
    height = images[0].size[1]  # All images have the same height

    # Create a new blank image with the concatenated size
    new_image = Image.new(images[0].mode, (total_width, height))

    # Paste each image into the new image
    current_x = 0
    for img in images:
        new_image.paste(img, (current_x, 0))
        current_x += img.size[0]

    # Save the result if a path is provided
    if save_path is not None:
        new_image.save(save_path)

    return new_image


def show_multiple_images(
    images, 
    nrows=1, 
    ncols=None, 
    titles=None, 
    suptitle=None, 
    tight=True, 
    cmaps='gray', 
    figsize=None, 
    dpi=None, 
    masks=None,
    set_axis_off=True,
    return_pil=False,
):

    num_imgs = len(images)
    ncols = num_imgs // nrows if ncols is None else ncols
    if nrows * ncols < num_imgs:
        ncols += 1
    num_plots = int(nrows * ncols)
    cmaps = [cmaps] * num_imgs if not isinstance(cmaps, (tuple, list)) else cmaps
    masks = [masks] * num_imgs if not isinstance(masks, (tuple, list)) else masks
    titles = [titles] * num_imgs if not isinstance(titles, (tuple, list)) else titles
    assert num_imgs <= num_plots, f'num_imgs = {num_imgs}, nrows = {nrows}, ncols = {ncols}.'
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    for i in range(num_imgs):
        if isinstance(images[i], torch.Tensor):
            img = images[i].cpu().squeeze()
        elif isinstance(images[i], np.ndarray):
            img = images[i].squeeze()
        elif isinstance(images[i], Image.Image):
            img = np.array(images[i]).squeeze()
        else:
            img = images[i]
        
        mask = masks[i]
        if mask is not None:
            img[mask == 0] = img.min()
        axes[i].imshow(img, cmap=cmaps[i])
        if set_axis_off:
            axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i])
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    if num_imgs < num_plots:
        for i in range(num_imgs, num_plots):
            axes[i].axis('off')
            
    if tight:
        fig.tight_layout()
    
    if return_pil:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        pil_image = Image.open(buf).convert('RGB')
        buf.close()
        plt.close(fig)  # Close the figure to free memory
        return pil_image
    
    return None



def paint_masks_on_image(image, masks, alpha=0.5, color_cycle=None):
    """
    Overlay a list of 2D masks onto an image.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        Input image. If ndarray, can be:
          - H x W (grayscale)
          - H x W x C (C=1 or 3)
        Values can be [0,1] or [0,255].
    masks : list of np.ndarray
        List of 2D arrays of shape (H, W), values in [0,1].
    alpha : float
        Base opacity of the mask overlay.
    color_cycle : list of (r, g, b) tuples in [0,1], optional
        Colors used for each mask; if None, a default list is used.

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 3) with values in [0,1].
    """

    # --- Normalize image to float32 [0,1] and ensure 3 channels ---
    if isinstance(image, Image.Image):
        img = np.asarray(image.convert("RGB"), dtype=np.float32)
        img /= 255.0
    else:
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 2:  # H, W -> grayscale to RGB
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError("Unsupported image shape: {}".format(img.shape))

        # Normalize if not already in [0,1]
        if img.max() > 1.0:
            img /= 255.0

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    h, w, _ = img.shape

    # --- Default color cycle (red -> green -> blue -> ...) ---
    if color_cycle is None:
        color_cycle = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 0.5, 0.0),  # orange-ish
            (0.6, 0.2, 1.0),  # purple-ish
        ]

    # --- Overlay each mask ---
    out = img.copy()

    for i, mask in enumerate(masks):
        if mask is None:
            continue

        m = np.asarray(mask, dtype=np.float32)
        if m.shape != (h, w):
            raise ValueError(f"Mask {i} shape {m.shape} does not match image shape {(h, w)}")

        m = np.clip(m, 0.0, 1.0)
        m_exp = m[..., None]  # (H, W, 1) for broadcasting

        color = np.array(color_cycle[i % len(color_cycle)], dtype=np.float32)
        color = color.reshape(1, 1, 3)  # broadcast to (H, W, 3)

        # Alpha blending: stronger where mask is closer to 1
        eff_alpha = alpha * m_exp
        out = out * (1.0 - eff_alpha) + color * eff_alpha

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def grayscale_to_cmap(x, cmap_name='viridis'):
    assert (0 <= x.min() <= 1) and (0 <= x.max() <= 1)
    cmap = cm.get_cmap(cmap_name)
    orig_shape = list(x.shape)
    new_shape = orig_shape + [4,]
    colored_x = cmap(x.flatten()).reshape(*new_shape)[..., :3]  # [*orig_shape, 3]
    # The resulting array has shape (height, width, 4) where the last dimension is RGBA
    # We can drop the alpha channel to get an RGB image
    return colored_x




def mean_std_ci(data, confidence=0.95):
    """
    Calculate the mean, standard deviation, and confidence interval using the t-distribution.

    Parameters
    ----------
    data : list or np.ndarray
        One-dimensional numeric data.
    confidence : float
        Confidence level (default: 0.95).

    Returns
    -------
    mean : float
    std : float
    (ci_low, ci_high) : tuple
        Confidence interval.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    assert n >= 2, "At least 2 samples are required"
    #
    mean = np.nanmean(data).item()
    std = np.nanstd(data, ddof=1).item()
    sem = std / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2.0, df=n-1)
    #
    margin = t_val * sem
    ci = margin.item() if hasattr(margin, 'item') else margin
    #ci_min = (mean - margin).item()
    #ci_max = (mean + margin).item()
    #ci = (ci_min, ci_max)
    #
    return {'mean': mean, 'std': std, 'ci': ci, 'confidence': confidence}


################################################################################################
# DATAFRAME PROCESSING
################################################################################################

def add_stratified_split(
    df: pd.DataFrame,
    stratify_col: str,
    ratios=(0.6, 0.2, 0.2),
    random_state: int = 0,
    inplace: bool = False,
    val_as_test: bool = False,
    key_for_split: str = 'split',
) -> pd.DataFrame:
    """
    Add a 'split' column with values in {'train','val','test'} by
    stratifying within each class of `stratify_col` and using the
    given ratios.

    Returns the modified DataFrame (or the same object if inplace=True).
    """
    # Normalize ratios in case ints like (6,2,2) are passed
    ratios = np.array(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    r_train, r_val, r_test = ratios

    if not inplace:
        df = df.copy()

    rng = np.random.RandomState(random_state)
    split_col = pd.Series(index=df.index, dtype="object")

    # Treat NaN as its own group to avoid dropping rows
    for _, g in df.groupby(stratify_col, dropna=False, sort=False):
        idx = g.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(np.floor(r_train * n))
        n_val   = int(np.floor(r_val * n))
        n_test  = n - n_train - n_val  # ensures total == n

        split_col.loc[idx[:n_train]] = "train"
        split_col.loc[idx[n_train:n_train + n_val]] = "val"
        split_col.loc[idx[n_train + n_val: n_train + n_val + n_test]] = "val" if val_as_test else "test"

    df[key_for_split] = split_col
    return df


def add_simple_split(df, ratios=(0.8, 0.1, 0.1), tags=("train", "val", "test"), seed=0, exact=False):
    """
    Add a 'split' column to `df` by randomly assigning rows to the given tags
    using `ratios`. If `exact=True`, the number of rows per split matches the
    ratios as closely as possible (up to rounding).
    """
    if len(ratios) != len(tags):
        raise ValueError("ratios and tags must have the same length.")
    df2 = df.copy()
    n = len(df2)
    if n == 0:
        df2["split"] = []
        return df2

    rng = np.random.default_rng(seed)
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()  # normalize

    if exact:
        # Target counts via largest-remainder method
        ideal = ratios * n
        counts = np.floor(ideal).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            # Distribute remaining items to the largest fractional parts
            frac_order = np.argsort(-(ideal - counts))
            counts[frac_order[:remainder]] += 1

        # Build label array with exact counts and shuffle
        splits = np.repeat(np.array(tags, dtype=object), counts)
        rng.shuffle(splits)
    else:
        # Approximate multinomial sampling
        splits = rng.choice(tags, size=n, p=ratios)

    df2["split"] = splits
    return df2


def make_few_shot_subset(
    df,
    k: int,
    label_key = 'label',
    seed: int = 3407,
) -> pd.DataFrame:
    
    assert k > 0, f"Number of shots (k) must be a positive integer, got {k}."
    
    if isinstance(df, str):
        df = pd.read_csv(df)
    elif isinstance(df, list):  # a list of dictionary
        if isinstance(df[0], dict):
            df = pd.DataFrame(df)
        else:
            raise ValueError(f"For a list input, each element should be a dictionary, got {type(df[0])}.")
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise ValueError(f"Unsupported input table type: {type(df)}. Expected one of [string to a csv file, dataframe, list of dictionary].")
    
    rng = np.random.RandomState(seed)

    first_non_null = df[label_key].dropna().iloc[0]
    if isinstance(first_non_null, (int, np.integer, float)):
        # If it's float but all values are integer-like, treat as multiclass
        task_type = "multiclass"
    elif isinstance(first_non_null, (list, tuple, np.ndarray)):
        task_type = "multilabel"
    elif isinstance(first_non_null, str) and first_non_null.strip().startswith("["):
        task_type = "multilabel"
    else:
        raise ValueError(f"Cannot infer task type from label value: {first_non_null!r}")

    # --- Multiclass case ---
    if task_type == "multiclass":
        parts = []
        for label_value, group in df.groupby(label_key):
            n_select = min(k, len(group))
            # sample indices with our RNG
            if n_select == len(group):
                chosen = group
            else:
                chosen_idx = rng.choice(group.index, size=n_select, replace=False)
                chosen = df.loc[chosen_idx]
            parts.append(chosen)

        subset = pd.concat(parts, axis=0)
        # Optional: shuffle final subset to mix classes
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return subset

    # --- Multilabel case ---
    elif task_type == "multilabel":

        def parse_multilabel_cell(x: Union[str, list, tuple, np.ndarray]):
            """Parse label cell into a 1D numpy array of ints."""
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.array(x, dtype=int)
            elif isinstance(x, str):
                s = x.strip()
                if s.startswith("["):
                    # String representation of a Python list, e.g., "[0, 1, 0, 1]"
                    arr = np.array(ast.literal_eval(s), dtype=int)
                else:
                    # Fallback: space/comma separated "0 1 0 1" or "0,1,0,1"
                    tokens = s.replace(",", " ").split()
                    arr = np.array([int(t) for t in tokens], dtype=int)
            else:
                raise ValueError(f"Unsupported multilabel cell type: {type(x)}")
            return arr

        label_arrays = df[label_key].apply(parse_multilabel_cell)
        # Stack into an (N, C) matrix
        y = np.vstack(label_arrays.to_numpy())
        num_classes = y.shape[1]

        selected_indices = set()

        # For each label index, sample up to k rows where that label is 1
        for j in range(num_classes):
            idx_j = np.where(y[:, j] == 1)[0]
            if len(idx_j) == 0:
                continue  # no examples for this label
            if len(idx_j) <= k:
                chosen = idx_j
            else:
                chosen = rng.choice(idx_j, size=k, replace=False)
            selected_indices.update(chosen.tolist())

        # Build subset from all chosen indices (may be > k * num_classes due to overlaps)
        selected_indices = sorted(selected_indices)
        subset = df.iloc[selected_indices].copy()
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return subset

    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    

def count_by_column(df, column, include_na=False):
    """
    Return a dict mapping each unique value in `column` -> number of rows.
    Set `include_na=True` to count NaNs as well.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    return df[column].value_counts(dropna=not include_na, sort=False).to_dict()


def concat_dataframes(dataframes, fill_value="", ignore_index=True):
    """
    Concatenates multiple DataFrames, aligning columns and padding missing columns with empty values.
    """
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for concatenation.")
    
    all_columns = set().union(*[df.columns for df in dataframes])
    
    # Align all DataFrames to have the same columns
    aligned_dfs = [df.reindex(columns=all_columns, fill_value=fill_value) for df in dataframes]
    result = pd.concat(aligned_dfs, ignore_index=ignore_index)
    return result


################################################################################################
# DEEP LEARNING THINGS
################################################################################################

def fix_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False



################################################
# DISTRIBUTED DATA PARALLEL
################################################


def dprint(msg, local_rank=0):
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(msg)
    else:
        print(msg)


def only_on_rank0(func):
    '''wrapper for only log on the first rank'''
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)
    return wrapper


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


################################################################################################
# OTHERS
################################################################################################

def copy_geometry(sitk_image, sitk_ref):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image

def get_now(num=19):
    return str(datetime.now())[:num]

def print_dict(d, keys=None, indent=2):
    orig_keys = list(d.keys())
    d_ = d
    if keys is not None:
        keys = [keys] if isinstance(keys, str) else keys
        keys = [k for k in keys if k in orig_keys]
        d_ = {k:d[k] for k in keys}
    print(json.dumps(d_, indent=indent))


def print_and_log(message, log_type='info', logger=None):
    print(message)
    if log_type == 'error':
        if logger is None:
            logging.error(message)
        else:
            logger.error(message)
    elif log_type == 'warn':
        if logger is None:
            logging.warning(message)
        else:
            logger.warning(message)
    else:
        if logger is None:
            logging.info(message)
        else:
            logger.info(message)




    
