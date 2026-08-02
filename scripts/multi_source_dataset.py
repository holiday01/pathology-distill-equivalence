#!/usr/bin/env python3
"""
Multi-source RGB patch dataset for FM distillation.

Sources:
  - HDF5 (extract_patches.py output): C16, BRCA
  - Image folder (PNG/TIF/JPG): NCT-CRC, Kather-MSI, PanNuke images
  - Numpy packed (PanNuke fold_*.npy with shape [N,H,W,3])

Each sample: (tensor[3,H,W] normalized, source_tag:str, group_id:str).
group_id is used for group-aware split (slide for WSI, parent folder for
image-folder sources, image-index for PanNuke).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(arr: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """uint8 HWC → float CHW, normalized, resized to (target_size, target_size)."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    if (h, w) != (target_size, target_size):
        img = Image.fromarray(arr)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.asarray(img)
    t = torch.from_numpy(np.ascontiguousarray(arr).copy()).permute(2, 0, 1).float() / 255.0
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


class HDF5Source:
    """Source backed by an HDF5 produced by extract_patches.py."""

    def __init__(self, h5_path: str, source_tag: str, max_samples: int | None = None,
                 seed: int = 42):
        self.h5_path = h5_path
        self.source_tag = source_tag
        self._h5 = None
        with h5py.File(h5_path, "r") as h5:
            self.n_total = h5["patches"].shape[0]
            self.slide_ids = h5["slide_ids"][:].astype(np.int64)
        if max_samples is not None and max_samples < self.n_total:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(self.n_total, max_samples, replace=False))
        else:
            self.indices = np.arange(self.n_total)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self.indices)

    def get_group(self, i: int) -> str:
        real = int(self.indices[i])
        sid = int(self.slide_ids[real])
        return f"{self.source_tag}:slide{sid}"

    def get(self, i: int):
        self._ensure_open()
        real = int(self.indices[i])
        patch = self._h5["patches"][real]
        return patch, self.source_tag, self.get_group(i)


class ImageFolderSource:
    """Source backed by a folder of PNG/TIF/JPG images (each = one patch).

    group_strategy:
      - "parent": group by parent folder name (e.g. class label)
      - "random": every sample is its own group (effectively random split)
      - "bucket:<N>": hash filename into N buckets (coarse pseudo-groups)
    """

    def __init__(self, root: str, source_tag: str, patterns=("*.tif", "*.png", "*.jpg"),
                 max_samples: int | None = None, seed: int = 42,
                 group_strategy: str = "parent", use_parent_as_group: bool | None = None):
        self.source_tag = source_tag
        if use_parent_as_group is not None:
            group_strategy = "parent" if use_parent_as_group else "random"
        self.group_strategy = group_strategy
        paths: list[Path] = []
        root_p = Path(root).expanduser()
        for pat in patterns:
            paths.extend(root_p.rglob(pat))
        paths = sorted(paths)
        if max_samples is not None and max_samples < len(paths):
            rng = random.Random(seed)
            paths = rng.sample(paths, max_samples)
            paths.sort()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def get_group(self, i: int) -> str:
        p = self.paths[i]
        gs = self.group_strategy
        if gs == "parent":
            return f"{self.source_tag}:{p.parent.name}"
        if gs == "random":
            return f"{self.source_tag}:item{i}"
        if gs.startswith("bucket:"):
            n = int(gs.split(":", 1)[1])
            return f"{self.source_tag}:b{hash(p.name) % n:03d}"
        return f"{self.source_tag}:all"

    def get(self, i: int):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img)
        return arr, self.source_tag, self.get_group(i)


class NumpyArraySource:
    """Source backed by stacked numpy arrays (PanNuke fold_*.npy shape [N,H,W,3])."""

    def __init__(self, npy_paths: list[str], source_tag: str,
                 max_samples: int | None = None, seed: int = 42,
                 group_strategy: str = "bucket:20"):
        self.source_tag = source_tag
        self.group_strategy = group_strategy
        self.arrays = []
        self.offsets = []
        total = 0
        for p in npy_paths:
            a = np.load(p, mmap_mode="r")
            self.arrays.append((p, a))
            self.offsets.append((total, total + len(a), p))
            total += len(a)
        self.n_total = total
        if max_samples is not None and max_samples < total:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(total, max_samples, replace=False))
        else:
            self.indices = np.arange(total)

    def __len__(self):
        return len(self.indices)

    def _find(self, real: int):
        for lo, hi, p in self.offsets:
            if lo <= real < hi:
                for pp, a in self.arrays:
                    if pp == p:
                        return p, a, real - lo
        raise IndexError(real)

    def get_group(self, i: int) -> str:
        real = int(self.indices[i])
        p, _, _ = self._find(real)
        gs = self.group_strategy
        if gs == "random":
            return f"{self.source_tag}:item{real}"
        if gs.startswith("bucket:"):
            n = int(gs.split(":", 1)[1])
            return f"{self.source_tag}:{os.path.basename(p)}:b{real % n:03d}"
        return f"{self.source_tag}:{os.path.basename(p)}"

    def get(self, i: int):
        real = int(self.indices[i])
        p, a, local = self._find(real)
        patch = np.asarray(a[local])
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)
        return patch, self.source_tag, self.get_group(i)


class MultiSourceDataset(Dataset):
    def __init__(self, sources: list[Any], target_size: int = 224,
                 indices: np.ndarray | None = None):
        self.sources = sources
        self.target_size = target_size
        offsets = []
        total = 0
        for s in sources:
            offsets.append((total, total + len(s)))
            total += len(s)
        self.offsets = offsets
        self.n_total = total
        self.indices = np.arange(total) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def _locate(self, global_idx: int):
        for si, (lo, hi) in enumerate(self.offsets):
            if lo <= global_idx < hi:
                return si, global_idx - lo
        raise IndexError(global_idx)

    def __getitem__(self, idx):
        gi = int(self.indices[idx])
        si, li = self._locate(gi)
        patch, src_tag, group = self.sources[si].get(li)
        t = _to_tensor(patch, target_size=self.target_size)
        return t, src_tag, group

    def build_group_index(self) -> dict[str, list[int]]:
        """Map group_id → list of positions (within self.indices). Lightweight: uses get_group."""
        groups: dict[str, list[int]] = {}
        for pos, gi in enumerate(self.indices):
            si, li = self._locate(int(gi))
            g = self.sources[si].get_group(li)
            groups.setdefault(g, []).append(pos)
        return groups


def group_split(dataset: MultiSourceDataset, val_frac: float = 0.1,
                test_frac: float = 0.1, seed: int = 42):
    """Group-aware random split; returns (train_idx, val_idx, test_idx) as arrays
    indexing into dataset.indices. Groups are split whole to avoid leakage."""
    groups = dataset.build_group_index()
    rng = random.Random(seed)
    group_names = sorted(groups.keys())
    rng.shuffle(group_names)
    n = len(group_names)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_g = set(group_names[:n_test])
    val_g = set(group_names[n_test:n_test + n_val])
    train_g = set(group_names[n_test + n_val:])
    train_idx, val_idx, test_idx = [], [], []
    for g, pos_list in groups.items():
        if g in test_g:
            test_idx.extend(pos_list)
        elif g in val_g:
            val_idx.extend(pos_list)
        else:
            train_idx.extend(pos_list)
    return (np.asarray(sorted(train_idx)),
            np.asarray(sorted(val_idx)),
            np.asarray(sorted(test_idx)))
