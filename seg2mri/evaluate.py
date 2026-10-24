"""Compare generated volumes against ground truth and write a results table.

    python -m seg2mri.evaluate --gen samples/demo2d/gen --real samples/demo2d/real \
        --out samples/demo2d/results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

from .metrics import evaluate_pair


def load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(path).dataobj, dtype=np.float32)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    gen_files = sorted(glob.glob(os.path.join(args.gen, "*.nii.gz")))
    if not gen_files:
        ap.error(f"no .nii.gz in {args.gen}")

    rows = []
    for gen_path in gen_files:
        name = os.path.basename(gen_path)
        real_path = os.path.join(args.real, name)
        if not os.path.exists(real_path):
            print(f"[skip] no ground truth for {name}", file=sys.stderr)
            continue
        row = {"case": name.replace(".nii.gz", "")}
        row.update(evaluate_pair(load_volume(gen_path), load_volume(real_path)))
        rows.append(row)
        print(f"{row['case']:<24} SSIM={row['ssim3d']:.4f} MS-SSIM={row['msssim_slice_mean']:.4f} "
              f"PSNR={row['psnr']:.2f} NRMSE={row['nrmse']:.4f}")

    if not rows:
        ap.error("no cases compared — check that file names match")

    fields = ["case", "ssim3d", "msssim_slice_mean", "psnr", "nrmse"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def mean_std(key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()), float(vals.std())

    s_m, s_s = mean_std("ssim3d")
    ms_m, ms_s = mean_std("msssim_slice_mean")
    p_m, p_s = mean_std("psnr")
    n_m, n_s = mean_std("nrmse")

    table = (
        f"| metric | mean ± std (n={len(rows)}) |\n"
        "|---|---|\n"
        f"| SSIM (3D) | {s_m:.4f} ± {s_s:.4f} |\n"
        f"| MS-SSIM (2D axial, mean) | {ms_m:.4f} ± {ms_s:.4f} |\n"
        f"| PSNR (dB) | {p_m:.2f} ± {p_s:.2f} |\n"
        f"| NRMSE | {n_m:.4f} ± {n_s:.4f} |\n"
    )
    print()
    print(table)
    print(f"wrote {args.out}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
