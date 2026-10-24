"""Datasets.

Two on-disk formats, one per regime:

2D slices (memmap, zero-copy random access)
    data/slices.npy   uint8  (N, 4, H, W)   values 0..255 -> [-1, 1]
    data/masks.npy    uint8  (N, H, W)      labels 0..K
    data/meta.json    cases, splits, slice->case index, size, affines

3D volumes (one npz per case, uint8)
    data/cases/<name>.npz   image (4, X, Y, Z), mask (X, Y, Z), affine

Labels follow MSD Task01: 0 background, 1 edema, 2 non-enhancing tumour,
3 enhancing tumour.  Condition channels are the K = 3 foreground classes.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

N_CLASSES = 3


def _mask_to_channels(mask: np.ndarray) -> np.ndarray:
    return np.stack([(mask == k + 1).astype(np.float32) for k in range(N_CLASSES)])


def _uint8_to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 127.5 - 1.0


def load_meta(root: str) -> dict:
    with open(os.path.join(root, "meta.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


class SliceDataset(Dataset):
    """Random axial slices from the memmap, one sample per index.

    ``__len__`` is cases x slices_per_case, so an epoch covers every case once
    even though each access draws a random slice.
    """

    def __init__(self, root: str, split: str = "train", augment: bool | None = None):
        meta = load_meta(root)
        self.root = root
        self.meta = meta
        self.size = meta["size"]
        self.slices_per_case = meta["slices_per_case"]
        self.augment = meta["augment_default"] if augment is None else augment

        case_to_idx = {c: i for i, c in enumerate(meta["cases"])}
        wanted = {case_to_idx[c] for c in meta["splits"][split]}
        idx = np.array([i for i, c in enumerate(meta["slice_case"]) if c in wanted], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"split '{split}' selects no slices")
        self.index = idx

        self.images = np.load(os.path.join(root, "slices.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(root, "masks.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, i: int) -> dict:
        j = int(self.index[i])
        image = _uint8_to_float(np.asarray(self.images[j]))
        mask = _mask_to_channels(np.asarray(self.masks[j]))

        if self.augment and random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "cond": torch.from_numpy(np.ascontiguousarray(mask)),
            "case": self.meta["cases"][int(self.meta["slice_case"][j])],
        }


class VolumeDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool | None = None):
        meta = load_meta(root)
        self.root = root
        self.meta = meta
        self.size = meta["size"]
        self.augment = meta["augment_default"] if augment is None else augment

        case_to_idx = {c: i for i, c in enumerate(meta["cases"])}
        self.cases = [c for c in meta["splits"][split] if c in case_to_idx]
        if not self.cases:
            raise ValueError(f"split '{split}' selects no volumes")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> dict:
        case = self.cases[i]
        z = np.load(os.path.join(self.root, "cases", f"{case}.npz"))
        image = _uint8_to_float(z["image"])
        cond = _mask_to_channels(z["mask"])

        if self.augment and random.random() < 0.5:
            image = image[:, ::-1]
            cond = cond[:, ::-1]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "cond": torch.from_numpy(np.ascontiguousarray(cond)),
            "case": case,
        }


def make_dataset(root: str, split: str, mode: str, augment: bool | None = None):
    if mode == "2d":
        return SliceDataset(root, split, augment)
    if mode == "3d":
        return VolumeDataset(root, split, augment)
    raise ValueError(f"unknown mode '{mode}'")
