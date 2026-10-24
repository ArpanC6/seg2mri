"""Merge per-method result CSVs into one Markdown table.

    python -m seg2mri.compare \
        --run "class-mean=samples/class_mean/results.csv" \
        --run "regressor=samples/regressor/results.csv" \
        --run "diffusion=samples/diffusion/results.csv" \
        --reference regressor --out docs/RESULTS.md

The table is the deliverable: same test cases, same metrics, methods differ only
in how the image is produced from the mask.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

METRICS = [
    ("ssim3d", "SSIM (3D)", "{:.4f}"),
    ("msssim_slice_mean", "MS-SSIM (axial)", "{:.4f}"),
    ("psnr", "PSNR (dB)", "{:.2f}"),
    ("nrmse", "NRMSE", "{:.4f}"),
]


def summarise(path: str) -> dict:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")

    stats = {"n": len(rows)}
    for key, _, _ in METRICS:
        vals = np.array([float(r[key]) for r in rows if r.get(key) not in (None, "")], dtype=float)
        vals = vals[np.isfinite(vals)]
        stats[key] = (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), float("nan"))
    return stats


def format_table(results: dict[str, dict], reference: str = "") -> str:
    names = list(results)
    header = "| method | n | " + " | ".join(label for _, label, _ in METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    lines = [header, sep]

    for name in names:
        stats = results[name]
        cells = []
        for key, _, fmt in METRICS:
            mean, std = stats[key]
            cells.append(fmt.format(mean) + " ± " + fmt.format(std))
        lines.append(f"| {name} | {stats['n']} | " + " | ".join(cells) + " |")

    if reference and reference in results and len(names) > 1:
        ref = results[reference]
        lines += ["", f"Δ vs {reference} (positive = better, except NRMSE):", "",
                  "| method | ΔSSIM | ΔPSNR (dB) |", "|---|---|---|"]
        for name in names:
            if name == reference:
                continue
            d_ssim = results[name]["ssim3d"][0] - ref["ssim3d"][0]
            d_psnr = results[name]["psnr"][0] - ref["psnr"][0]
            lines.append(f"| {name} | {d_ssim:+.4f} | {d_psnr:+.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=CSV",
                    help="method name and its evaluate.py CSV; repeat per method")
    ap.add_argument("--reference", default="", help="method the deltas are computed against")
    ap.add_argument("--out", default="", help="write the Markdown here")
    args = ap.parse_args()

    results = {}
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run expects NAME=CSV, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"missing CSV for {name}: {path}")
        results[name] = summarise(path)

    table = format_table(results, args.reference)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
