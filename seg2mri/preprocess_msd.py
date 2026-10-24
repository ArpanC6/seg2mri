"""Convert MSD Task01_BrainTumour into this repo's on-disk format.

Download first (public S3 mirror, ~6 GB):

    wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
    tar -xf Task01_BrainTumour.tar

Then:

    python -m seg2mri.preprocess_msd \
        --src Task01_BrainTumour --out data/msd_prepared --size 128 --mode both

The 2D format is a memmap of axial slices (fast random access for training);
the 3D format is one npz per case.  Both store uint8, which costs ~1/4 of float32
and is lossless enough at this resolution.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def normalise(volume: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    body = volume[volume != 0]
    if body.size == 0:
        body = volume.ravel()
    p_lo, p_hi = np.percentile(body, [lo, hi])
    out = np.clip((volume - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
    return out * 2.0 - 1.0


def resize_case(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = torch.from_numpy(image.astype(np.float32))[None]
    img = F.interpolate(img, size=(size, size, size), mode="trilinear", align_corners=False)
    msk = torch.from_numpy(mask.astype(np.float32))[None, None]
    msk = F.interpolate(msk, size=(size, size, size), mode="nearest")
    return img[0].numpy(), msk[0, 0].numpy().astype(np.uint8)


def scaled_affine(affine: np.ndarray, shape: tuple[int, int, int], size: int) -> np.ndarray:
    scale = np.array([shape[0] / size, shape[1] / size, shape[2] / size])
    out = np.array(affine, dtype=np.float64)
    out[:3, :3] = np.asarray(affine, dtype=np.float64)[:3, :3] @ np.diag(scale)
    return out


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round((image + 1.0) * 127.5), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing imagesTr/ and labelsTr/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N cases")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import nibabel as nib

    image_paths = sorted(glob.glob(os.path.join(args.src, "imagesTr", "*.nii.gz")))
    if not image_paths:
        raise SystemExit(f"no imagesTr/*.nii.gz under {args.src}")
    if args.limit:
        image_paths = image_paths[: args.limit]

    cases = [os.path.basename(p).replace(".nii.gz", "") for p in image_paths]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cases))
    n_train = int(0.8 * len(cases))
    n_val = int(0.1 * len(cases))
    splits = {
        "train": sorted(cases[i] for i in order[:n_train]),
        "val": sorted(cases[i] for i in order[n_train : n_train + n_val]),
        "test": sorted(cases[i] for i in order[n_train + n_val :]),
    }

    os.makedirs(args.out, exist_ok=True)
    if args.mode in ("3d", "both"):
        os.makedirs(os.path.join(args.out, "cases"), exist_ok=True)

    n = len(cases)
    size = args.size
    images_mm = masks_mm = None
    if args.mode in ("2d", "both"):
        images_mm = np.lib.format.open_memmap(
            os.path.join(args.out, "slices.npy"), mode="w+", dtype=np.uint8, shape=(n * size, 4, size, size)
        )
        masks_mm = np.lib.format.open_memmap(
            os.path.join(args.out, "masks.npy"), mode="w+", dtype=np.uint8, shape=(n * size, size, size)
        )

    slice_case: list[int] = []
    affines: dict[str, list] = {}

    for ci, (case, path) in enumerate(tqdm(list(zip(cases, image_paths)), desc="cases")):
        img = nib.load(path)
        raw = np.asanyarray(img.dataobj, dtype=np.float32)
        if raw.ndim != 4 or raw.shape[-1] != 4:
            raise ValueError(f"{case}: expected 4 modalities, got {raw.shape}")
        image = np.transpose(raw, (3, 0, 1, 2))
        for m in range(4):
            image[m] = normalise(image[m])

        label_path = os.path.join(args.src, "labelsTr", f"{case}.nii.gz")
        mask = np.asanyarray(nib.load(label_path).dataobj).astype(np.uint8) if os.path.exists(label_path) \
            else np.zeros(image.shape[1:], dtype=np.uint8)

        orig_shape = image.shape[1:]
        image, mask = resize_case(image, mask, size)
        affines[case] = scaled_affine(np.asarray(img.affine), orig_shape, size).tolist()

        packed = to_uint8(image)
        if images_mm is not None:
            for z in range(size):
                images_mm[ci * size + z] = packed[:, :, :, z]
                masks_mm[ci * size + z] = mask[:, :, z]
                slice_case.append(ci)

        if args.mode in ("3d", "both"):
            np.savez_compressed(
                os.path.join(args.out, "cases", f"{case}.npz"),
                image=packed,
                mask=mask,
                affine=affines[case],
            )

    if images_mm is not None:
        images_mm.flush()
        masks_mm.flush()

    meta = {
        "cases": cases,
        "splits": splits,
        "slice_case": slice_case,
        "slices_per_case": size if images_mm is not None else 0,
        "size": size,
        "augment_default": True,
        "affines": affines,
        "source": "MSD Task01_BrainTumour",
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumour", "3": "enhancing tumour"},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    print(f"wrote {n} cases to {args.out}  (train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
