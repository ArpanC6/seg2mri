"""Sample held-out cases and export NIfTI + a PNG montage.

Works for diffusion checkpoints (DDIM) and for regressor checkpoints (one
forward pass); the architecture is read from the checkpoint.

    python -m seg2mri.sample --ckpt runs/demo2d/ckpt-last.pt --split test \
        --n 3 --out samples/demo2d --export-real
"""

from __future__ import annotations

import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from .data import N_CLASSES, _mask_to_channels, load_meta
from .diffusion import GaussianDiffusion
from .train import build_model

MODALITIES = ["t1", "t1ce", "t2", "flair"]


class RegressorSampler(torch.nn.Module):
    """Single-step regressor exposed through the same sampling interface."""

    def __init__(self, model: torch.nn.Module, image_channels: int = 4):
        super().__init__()
        self.model = model
        self.image_channels = image_channels

    @torch.no_grad()
    def ddim_sample(self, cond: torch.Tensor, steps: int = 0, clip: bool = True, generator=None) -> torch.Tensor:
        zeros = cond.new_zeros((cond.shape[0], self.image_channels, *cond.shape[2:]))
        t0 = cond.new_zeros((cond.shape[0],), dtype=torch.long)
        out = self.model(zeros, t0, cond)
        return out.clamp(-1.0, 1.0) if clip else out


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    arch = ckpt.get("arch", "diffusion")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if arch == "regressor":
        sampler = RegressorSampler(model).to(device).eval()
    else:
        sampler = GaussianDiffusion(
            model,
            image_channels=4,
            timesteps=cfg["diffusion"]["timesteps"],
            schedule=cfg["diffusion"]["schedule"],
            prediction=cfg["diffusion"]["prediction"],
        ).to(device).eval()
    return sampler, cfg, ckpt["step"], arch


def _tile(sl: np.ndarray) -> np.ndarray:
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    return (sl * 255).astype(np.uint8)


def save_montage(gen: np.ndarray, real: np.ndarray | None, path: str) -> None:
    """One middle axial slice, four modalities per row: generated on top, real below."""
    idx = gen.shape[2] // 2
    rows = []
    for source in ([gen, real] if real is not None else [gen]):
        rows.append(np.concatenate([_tile(source[c, :, :, idx]) for c in range(source.shape[0])], axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(path)


def sample_2d(diffusion, root: str, case: str, batch: int, steps: int, seed: int) -> np.ndarray:
    meta = load_meta(root)
    case_idx = meta["cases"].index(case)
    slice_ids = [i for i, c in enumerate(meta["slice_case"]) if c == case_idx]
    masks = np.load(os.path.join(root, "masks.npy"), mmap_mode="r")
    size = meta["size"]

    gen = np.zeros((4, size, size, len(slice_ids)), dtype=np.float32)
    torch.manual_seed(seed)
    for start in range(0, len(slice_ids), batch):
        chunk = slice_ids[start : start + batch]
        cond = torch.stack([torch.from_numpy(_mask_to_channels(np.asarray(masks[j]))) for j in chunk])
        out = diffusion.ddim_sample(cond.to(next(diffusion.parameters()).device), steps=steps)
        gen[:, :, :, start : start + len(chunk)] = out.cpu().numpy().transpose(1, 2, 3, 0)
    return gen


def sample_3d(diffusion, root: str, case: str, steps: int, seed: int) -> np.ndarray:
    z = np.load(os.path.join(root, "cases", f"{case}.npz"))
    cond = torch.from_numpy(_mask_to_channels(z["mask"]))[None]
    torch.manual_seed(seed)
    out = diffusion.ddim_sample(cond.to(next(diffusion.parameters()).device), steps=steps)
    return out[0].cpu().numpy()


def load_real(root: str, case: str, mode: str) -> np.ndarray:
    if mode == "3d":
        z = np.load(os.path.join(root, "cases", f"{case}.npz"))
        return z["image"].astype(np.float32) / 127.5 - 1.0
    meta = load_meta(root)
    case_idx = meta["cases"].index(case)
    ids = [i for i, c in enumerate(meta["slice_case"]) if c == case_idx]
    images = np.load(os.path.join(root, "slices.npy"), mmap_mode="r")
    vol = np.stack([np.asarray(images[i]) for i in ids], axis=-1).astype(np.float32)
    return vol / 127.5 - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3, help="number of test cases; 0 means all")
    ap.add_argument("--cases", default="")
    ap.add_argument("--out", default="samples/run")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-real", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sampler, cfg, trained_steps, arch = load_model(args.ckpt, args.device)
    root = cfg["data"]["root"]
    mode = cfg["data"]["mode"]
    meta = load_meta(root)
    steps = 0 if arch == "regressor" else (args.steps or cfg["diffusion"]["ddim_steps"])

    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or meta["splits"][args.split]
    cases = cases[: args.n] if args.n > 0 else cases

    gen_dir = os.path.join(args.out, "gen")
    real_dir = os.path.join(args.out, "real")
    os.makedirs(gen_dir, exist_ok=True)
    if args.export_real:
        os.makedirs(real_dir, exist_ok=True)

    print(f"checkpoint step {trained_steps} | arch {arch} | mode {mode} | ddim steps {steps} | cases {cases}")
    for case in cases:
        gen = sample_3d(sampler, root, case, steps, args.seed) if mode == "3d" else \
            sample_2d(sampler, root, case, args.batch, steps, args.seed)
        affine = np.array(meta["affines"][case], dtype=np.float64)

        for ch, mod in enumerate(MODALITIES):
            vol = gen[ch].transpose(1, 2, 0)  # (H, W, D)
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            nib.save(nib.Nifti1Image(vol.astype(np.float32), affine),
                     os.path.join(gen_dir, f"{case}_{mod}.nii.gz"))

        real = load_real(root, case, mode)
        save_montage(gen, real, os.path.join(args.out, f"{case}_montage.png"))

        if args.export_real:
            for ch, mod in enumerate(MODALITIES):
                vol = real[ch].transpose(1, 2, 0)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                nib.save(nib.Nifti1Image(vol.astype(np.float32), affine),
                         os.path.join(real_dir, f"{case}_{mod}.nii.gz"))

        print(f"  {case}: wrote 4 volumes")

    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"ckpt": args.ckpt, "arch": arch, "step": trained_steps, "ddim_steps": steps,
                   "cases": cases, "mode": mode}, fh, indent=2)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
