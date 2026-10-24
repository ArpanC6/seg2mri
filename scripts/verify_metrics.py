#!/usr/bin/env python3
"""Recompute SSIM / PSNR / NRMSE / Tenengrad directly from NIfTI files.

The point of this script is independent verification: anyone who receives the
NIfTI trio can re-derive every number reported in the note.

    python verify_metrics.py --gen gen/ --real real/ --out results.csv
    python verify_metrics.py --selftest

Conventions (deliberately explicit, because SSIM is implementation-sensitive):
  * Each volume is min-max normalised to [0, 1] independently.
  * SSIM is 3D (skimage, win_size=7, data_range=1.0) on the full volume.
  * Axial slice-wise SSIM is reported separately (mean over slices).
  * If --mask is given, both volumes are zeroed outside the mask before scoring.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np


def load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    return np.asanyarray(img.dataobj, dtype=np.float32), np.asarray(img.affine)


def to_unit(vol):
    lo, hi = float(vol.min()), float(vol.max())
    return (vol - lo) / (hi - lo + 1e-8)


def ssim_3d(gen, real):
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(real, gen, data_range=1.0, win_size=7, channel_axis=None))


def ssim_slices(gen, real):
    from skimage.metrics import structural_similarity as ssim
    scores = []
    for z in range(real.shape[2]):
        a, b = real[:, :, z], gen[:, :, z]
        if a.std() < 1e-6 and b.std() < 1e-6:
            continue
        scores.append(ssim(a, b, data_range=1.0, win_size=7))
    return float(np.mean(scores)) if scores else float("nan")


def psnr(gen, real, data_range=1.0):
    mse = float(np.mean((gen - real) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def nrmse(gen, real):
    rmse = float(np.sqrt(np.mean((gen - real) ** 2)))
    return rmse / (float(real.max() - real.min()) + 1e-8)


def tenengrad(vol):
    from scipy.ndimage import sobel
    gx = sobel(vol, axis=0)
    gy = sobel(vol, axis=1)
    return float(np.mean(gx ** 2 + gy ** 2))


def compare(gen_path, real_path, mask_path=None):
    gen, _ = load_nifti(gen_path)
    real, _ = load_nifti(real_path)
    if gen.shape != real.shape:
        raise ValueError(f"shape mismatch {gen_path} {gen.shape} vs {real.shape}")

    gen, real = to_unit(gen), to_unit(real)

    if mask_path:
        mask, _ = load_nifti(mask_path)
        if mask.shape != real.shape:
            raise ValueError(f"mask shape mismatch {mask.shape} vs {real.shape}")
        m = (mask > 0).astype(np.float32)
        gen, real = gen * m, real * m

    return {
        "ssim3d": ssim_3d(gen, real),
        "ssim_slice_mean": ssim_slices(gen, real),
        "psnr": psnr(gen, real),
        "nrmse": nrmse(gen, real),
        "tenengrad_gen": tenengrad(gen),
        "tenengrad_real": tenengrad(real),
    }


def selftest():
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(np.arange(64), np.arange(64), np.arange(48), indexing="ij")
    bubble = ((zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 24) ** 2) < 18 ** 2
    real = bubble.astype(np.float32) + 0.05 * rng.standard_normal(bubble.shape).astype(np.float32)
    gen = real + 0.02 * rng.standard_normal(real.shape).astype(np.float32)

    import tempfile
    import nibabel as nib
    tmp = tempfile.mkdtemp()
    paths = {}
    for name, vol in (("real", real), ("gen", gen)):
        p = os.path.join(tmp, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), p)
        paths[name] = p

    identical = compare(paths["gen"], paths["gen"])
    perturbed = compare(paths["gen"], paths["real"])

    assert abs(identical["ssim3d"] - 1.0) < 1e-6, identical
    assert identical["psnr"] == float("inf"), identical
    assert perturbed["ssim3d"] < 0.999, perturbed
    assert perturbed["psnr"] < identical["psnr"]

    print("selftest PASS")
    print(f"  identical : SSIM={identical['ssim3d']:.6f} PSNR={identical['psnr']}")
    print(f"  perturbed : SSIM={perturbed['ssim3d']:.6f} PSNR={perturbed['psnr']:.2f} dB")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", help="directory with generated .nii.gz")
    ap.add_argument("--real", help="directory with ground-truth .nii.gz")
    ap.add_argument("--mask", help="optional directory with brain masks (same filenames)")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.gen and args.real):
        ap.error("--gen and --real are required (or pass --selftest)")

    gen_files = sorted(glob.glob(os.path.join(args.gen, "*.nii.gz")))
    if not gen_files:
        ap.error(f"no .nii.gz found in {args.gen}")

    rows = []
    for gen_path in gen_files:
        name = os.path.basename(gen_path)
        real_path = os.path.join(args.real, name)
        if not os.path.exists(real_path):
            print(f"[skip] no ground truth for {name}", file=sys.stderr)
            continue
        mask_path = os.path.join(args.mask, name) if args.mask else None
        if mask_path and not os.path.exists(mask_path):
            mask_path = None
        m = compare(gen_path, real_path, mask_path)
        m["case"] = name.replace(".nii.gz", "")
        rows.append(m)
        print(f"{m['case']:<28} SSIM={m['ssim3d']:.4f} PSNR={m['psnr']:.2f} "
              f"NRMSE={m['nrmse']:.4f} TENG={m['tenengrad_gen']:.4f}")

    if not rows:
        ap.error("no cases compared")

    fields = ["case", "ssim3d", "ssim_slice_mean", "psnr", "nrmse",
              "tenengrad_gen", "tenengrad_real"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def stat(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()), float(vals.std())

    s_m, s_s = stat("ssim3d")
    p_m, p_s = stat("psnr")
    n_m, n_s = stat("nrmse")
    print()
    print(f"| metric | mean ± std (n={len(rows)}) |")
    print("|---|---|")
    print(f"| SSIM (3D) | {s_m:.4f} ± {s_s:.4f} |")
    print(f"| PSNR (dB) | {p_m:.2f} ± {p_s:.2f} |")
    print(f"| NRMSE | {n_m:.4f} ± {n_s:.4f} |")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
