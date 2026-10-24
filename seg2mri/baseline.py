"""A segmentation-only baseline, so the diffusion numbers have something to beat.

For every class k, take the mean intensity of class-k voxels across the training
split, then paint those means into the test masks.  This is the best a model can
do using the mask alone and no spatial model, so SSIM above this baseline is the
part that the diffusion model genuinely contributes.

    python -m seg2mri.baseline --config configs/demo2d.yaml --split test --out samples/baseline
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np

from .config import load_config
from .data import N_CLASSES, load_meta

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def class_means(root: str, split: str, mode: str) -> dict:
    meta = load_meta(root)
    cases = [meta["cases"].index(c) for c in meta["splits"][split]]
    sums = np.zeros((4, N_CLASSES + 1), dtype=np.float64)
    counts = np.zeros(N_CLASSES + 1, dtype=np.float64)

    for case_idx in cases:
        if mode == "3d":
            z = np.load(os.path.join(root, "cases", f"{meta['cases'][case_idx]}.npz"))
            image = z["image"].astype(np.float32) / 127.5 - 1.0
            mask = z["mask"].astype(np.int64)
        else:
            ids = [i for i, c in enumerate(meta["slice_case"]) if c == case_idx]
            images = np.load(os.path.join(root, "slices.npy"), mmap_mode="r")
            masks = np.load(os.path.join(root, "masks.npy"), mmap_mode="r")
            image = np.stack([np.asarray(images[i]) for i in ids], axis=-1).astype(np.float32) / 127.5 - 1.0
            mask = np.stack([np.asarray(masks[i]) for i in ids], axis=-1).astype(np.int64)

        for cls in range(N_CLASSES + 1):
            sel = mask == cls
            counts[cls] += sel.sum()
            for ch in range(4):
                sums[ch, cls] += image[ch][sel].sum()

    return {"means": sums / np.maximum(counts, 1), "counts": counts}


def build_case(means: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((4,) + mask.shape, dtype=np.float32)
    for cls in range(N_CLASSES + 1):
        sel = mask == cls
        for ch in range(4):
            out[ch][sel] = means[ch, cls]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="samples/baseline")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root, mode = cfg["data"]["root"], cfg["data"]["mode"]
    meta = load_meta(root)

    stats = class_means(root, "train", mode)
    print("class means (T1, T1ce, T2, FLAIR):")
    for cls in range(N_CLASSES + 1):
        print(f"  class {cls}: " + "  ".join(f"{v:+.3f}" for v in stats["means"][:, cls]))

    os.makedirs(args.out, exist_ok=True)
    for case in meta["splits"][args.split]:
        if mode == "3d":
            z = np.load(os.path.join(root, "cases", f"{case}.npz"))
            vol = build_case(stats["means"], z["mask"].astype(np.int64))
        else:
            case_idx = meta["cases"].index(case)
            ids = [i for i, c in enumerate(meta["slice_case"]) if c == case_idx]
            masks = np.load(os.path.join(root, "masks.npy"), mmap_mode="r")
            mask = np.stack([np.asarray(masks[i]) for i in ids], axis=-1).astype(np.int64)
            vol = build_case(stats["means"], mask)

        affine = np.array(meta["affines"][case], dtype=np.float64)
        for ch, mod in enumerate(MODALITIES):
            arr = vol[ch].transpose(1, 2, 0)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            nib.save(nib.Nifti1Image(arr.astype(np.float32), affine),
                     os.path.join(args.out, f"{case}_{mod}.nii.gz"))
        print(f"  {case}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
