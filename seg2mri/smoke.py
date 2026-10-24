"""Synthetic data in the repo's on-disk format.

Lets the whole pipeline (train -> sample -> evaluate) run on a laptop CPU in
under a minute, and gives CI something to test without a dataset download.

    python -m seg2mri.smoke --out data/smoke --size 32 --cases 6
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _synth_case(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    c = (size - 1) / 2
    brain = ((xx - c) ** 2 + (yy - c) ** 2 + ((zz - c) * 1.2) ** 2) < (size * 0.4) ** 2

    mask = np.zeros((size, size, size), dtype=np.uint8)
    for cls in (1, 2, 3):
        cx, cy, cz = rng.uniform(c - size * 0.2, c + size * 0.2, size=3)
        radius = rng.uniform(size * 0.05, size * 0.14)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < radius ** 2
        mask[blob & (mask == 0)] = cls

    volume = np.zeros((4, size, size, size), dtype=np.float32)
    base = rng.uniform(-0.4, 0.1, size=4)
    for m in range(4):
        vol = np.full(mask.shape, base[m], dtype=np.float32)
        vol += 0.3 * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * (size * 0.25) ** 2))
        for cls in (1, 2, 3):
            vol[mask == cls] += rng.uniform(0.1, 0.5)
        vol[~brain] = -1.0
        volume[m] = np.clip(vol, -1.0, 1.0)
    return volume, mask


def make_smoke(root: str = "data/smoke", n_cases: int = 6, size: int = 32, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "cases"), exist_ok=True)

    cases = [f"SYNTH_{i:03d}" for i in range(n_cases)]
    splits = {"train": cases[:4], "val": cases[4:5], "test": cases[5:]}

    images = np.lib.format.open_memmap(
        os.path.join(root, "slices.npy"), mode="w+", dtype=np.uint8, shape=(n_cases * size, 4, size, size)
    )
    masks = np.lib.format.open_memmap(
        os.path.join(root, "masks.npy"), mode="w+", dtype=np.uint8, shape=(n_cases * size, size, size)
    )

    slice_case, affines = [], {}
    for ci, case in enumerate(cases):
        volume, mask = _synth_case(rng, size)
        packed = np.clip(np.round((volume + 1.0) * 127.5), 0, 255).astype(np.uint8)
        for z in range(size):
            images[ci * size + z] = packed[:, :, :, z]
            masks[ci * size + z] = mask[:, :, z]
            slice_case.append(ci)
        affines[case] = np.eye(4).tolist()
        np.savez_compressed(os.path.join(root, "cases", f"{case}.npz"),
                            image=packed, mask=mask, affine=np.eye(4))

    images.flush()
    masks.flush()

    meta = {
        "cases": cases,
        "splits": splits,
        "slice_case": slice_case,
        "slices_per_case": size,
        "size": size,
        "augment_default": False,
        "affines": affines,
        "source": "synthetic",
    }
    with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/smoke")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    root = make_smoke(args.out, args.cases, args.size)
    print(f"wrote synthetic data to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
