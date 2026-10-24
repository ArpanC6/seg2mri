"""One command from public dataset to a finished results package.

Designed for a fresh Kaggle notebook (GPU on, internet on):

    !git clone https://github.com/<you>/seg2mri.git
    %cd seg2mri
    !python scripts/run_all.py --work /kaggle/working/seg2mri_run

Stages, each skipped if its output already exists (use --force to redo):

    data       download MSD Task01, extract the cases we need, preprocess
    diffusion  train the conditional diffusion model
    regressor  train the single-step U-Net on the identical data and split
    sample     draw held-out cases from both checkpoints, export NIfTI + PNG
    baseline   paint training class means into the test masks
    evaluate   per-case CSV for all three methods
    compare    one Markdown table with deltas against the regressor
    package    zip the numbers, volumes, montages, logs, and manifest

Everything written to --work survives the session; --tmp is for the raw
download and prepared arrays, which can be regenerated and are large.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- helpers

def run(cmd: list[str], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=REPO)
    start = time.time()
    subprocess.run(cmd, cwd=REPO, env=env, check=True)
    print(f"--- {title} done in {time.time() - start:.0f}s", flush=True)


def have(path: str) -> bool:
    return os.path.exists(path)


def read_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_yaml(cfg: dict, path: str) -> None:
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def count_cases(images_dir: str) -> int:
    if not os.path.isdir(images_dir):
        return 0
    return len([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])


# ----------------------------------------------------------------------------- stages

def stage_data(args) -> str:
    raw = args.raw_dir or os.path.join(args.tmp, "Task01_BrainTumour")
    prepared = os.path.join(args.tmp, f"msd_prepared_{args.mode}_{args.size}")
    if have(os.path.join(prepared, "meta.json")):
        print(f"[data] reusing {prepared}")
        return prepared

    if not args.raw_dir and count_cases(os.path.join(raw, "imagesTr")) < (args.limit or 1):
        tar_path = os.path.join(args.tmp, "Task01_BrainTumour.tar")
        os.makedirs(args.tmp, exist_ok=True)
        if not have(tar_path) or os.path.getsize(tar_path) < 5_000_000_000:
            run(["curl", "-L", "-C", "-", "--retry", "3", "-o", tar_path, MSD_URL],
                f"download MSD Task01 (~6.4 GB) -> {tar_path}")
        extract_cases(tar_path, args.tmp, args.limit)

    if count_cases(os.path.join(raw, "imagesTr")) == 0:
        raise SystemExit(f"no imagesTr/*.nii.gz under {raw} — pass --raw-dir")

    run([sys.executable, "-m", "seg2mri.preprocess_msd", "--src", raw, "--out", prepared,
         "--size", str(args.size), "--mode", args.mode] + (["--limit", str(args.limit)] if args.limit else []),
        "preprocess to memmap slices")
    return prepared


def extract_cases(tar_path: str, dest: str, limit: int) -> None:
    print(f"\n=== extract {'first ' + str(limit) if limit else 'all'} cases from {tar_path} ===", flush=True)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".nii.gz")]
        if limit:
            images = sorted((m for m in members if "/imagesTr/" in m.name), key=lambda m: m.name)
            keep = {os.path.basename(m.name)[:-7] for m in images[:limit]}
            members = [m for m in members if os.path.basename(m.name)[:-7] in keep]
        for m in members:
            try:
                tf.extract(m, dest, filter="data")
            except TypeError:  # Python < 3.11.4
                tf.extract(m, dest)
    print(f"extracted {len(members)} files", flush=True)


def make_configs(args, prepared: str) -> tuple[str, str]:
    cfg_dir = os.path.join(args.work, "configs")
    diff = read_yaml(os.path.join(REPO, "configs", "demo2d.yaml"))
    diff["data"].update({"root": prepared, "mode": args.mode, "num_workers": args.workers})
    diff["train"].update({"out_dir": os.path.join(args.work, "runs", "diffusion"),
                          "steps": args.steps, "batch_size": args.batch, "ckpt_every": 0})
    reg = read_yaml(os.path.join(REPO, "configs", "reg2d.yaml"))
    reg["data"].update({"root": prepared, "mode": args.mode, "num_workers": args.workers})
    reg["train"].update({"out_dir": os.path.join(args.work, "runs", "regressor"),
                         "steps": args.reg_steps, "batch_size": args.batch, "ckpt_every": 0})

    diff_path, reg_path = os.path.join(cfg_dir, "diffusion.yaml"), os.path.join(cfg_dir, "regressor.yaml")
    write_yaml(diff, diff_path)
    write_yaml(reg, reg_path)
    return diff_path, reg_path


def stage_train(args, cfg_path: str, out_dir: str, module: str, label: str) -> None:
    if have(os.path.join(out_dir, "ckpt-last.pt")) and not args.force:
        print(f"[{label}] reusing {out_dir}/ckpt-last.pt")
        return
    run([sys.executable, "-m", module, "--config", cfg_path], f"train {label}")


def stage_sample(args, ckpt: str, out: str, export_real: bool, steps: int) -> None:
    if have(os.path.join(out, "run.json")) and not args.force:
        print(f"[sample] reusing {out}")
        return
    cmd = [sys.executable, "-m", "seg2mri.sample", "--ckpt", ckpt, "--split", "test",
           "--n", "0" if args.sample_cases == 0 else str(args.sample_cases),
           "--out", out, "--steps", str(steps), "--batch", str(args.sample_batch)]
    if export_real:
        cmd.append("--export-real")
    run(cmd, f"sample -> {out}")


def stage_baseline(args, cfg_path: str, out: str) -> None:
    if os.path.isdir(out) and any(f.endswith(".nii.gz") for f in os.listdir(out)) and not args.force:
        print(f"[baseline] reusing {out}")
        return
    run([sys.executable, "-m", "seg2mri.baseline", "--config", cfg_path, "--split", "test", "--out", out],
        "class-mean baseline")


def stage_evaluate(gen: str, real: str, out_csv: str, label: str) -> None:
    run([sys.executable, "-m", "seg2mri.evaluate", "--gen", gen, "--real", real,
         "--out", out_csv, "--md", out_csv.replace(".csv", ".md")], f"evaluate {label}")


def stage_compare(args, results_dir: str) -> str:
    out_md = os.path.join(results_dir, "RESULTS.md")
    cmd = [sys.executable, "-m", "seg2mri.compare",
           "--run", f"class-mean={os.path.join(results_dir, 'class_mean.csv')}",
           "--run", f"regressor={os.path.join(results_dir, 'regressor.csv')}",
           "--run", f"diffusion={os.path.join(results_dir, 'diffusion.csv')}",
           "--reference", "regressor", "--out", out_md]
    run(cmd, "merge results table")
    return out_md


def stage_package(args) -> str:
    zip_path = os.path.join(args.work, "seg2mri_results.zip")
    manifest = os.path.join(args.work, "MANIFEST.txt")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write(f"seg2mri run — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"python {platform.python_version()} | torch {torch_version()} | cuda {cuda_name()}\n")
        fh.write(f"steps diffusion={args.steps} regressor={args.reg_steps} | "
                 f"data limit={args.limit or 'all'} size={args.size} mode={args.mode}\n")
        fh.write(f"ddim={args.ddim}\n\n")
        for root, _, files in os.walk(args.work):
            for name in sorted(files):
                if name in ("seg2mri_results.zip", "MANIFEST.txt"):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, args.work)
                fh.write(f"{os.path.getsize(path):>12,}  {rel}\n")

    include = [os.path.join(args.work, d) for d in ("results", "samples", "runs", "configs")]
    include.append(manifest)
    print(f"\n=== package -> {zip_path} ===", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for item in include:
            if os.path.isfile(item):
                zf.write(item, os.path.relpath(item, args.work))
                continue
            for root, _, files in os.walk(item):
                for name in sorted(files):
                    if name.endswith(".pt"):
                        continue
                    path = os.path.join(root, name)
                    zf.write(path, os.path.relpath(path, args.work))
    print(f"zip size {os.path.getsize(zip_path) / 1e6:.1f} MB", flush=True)
    return zip_path


def torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "missing"


def cuda_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    except Exception:
        return "unknown"


def print_loss_summary(work: str) -> None:
    for label in ("diffusion", "regressor"):
        path = os.path.join(work, "runs", label, "log.csv")
        if not have(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            last = rows[-1]
            print(f"[{label}] step {last['step']} loss {float(last['loss']):.4f}")


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/seg2mri_run" if os.path.isdir("/kaggle/working")
                    else os.path.join(REPO, "kaggle_run"))
    ap.add_argument("--tmp", default="/kaggle/temp" if os.path.isdir("/kaggle/temp")
                    else os.path.join(REPO, "kaggle_tmp"))
    ap.add_argument("--raw-dir", default="", help="skip download; folder with imagesTr/ and labelsTr/")
    ap.add_argument("--limit", type=int, default=300, help="number of MSD cases (0 = all 484)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--reg-steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--sample-cases", type=int, default=0, help="0 = every test case")
    ap.add_argument("--sample-batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to check the wiring: 60 cases, 128px, 600/300 steps, 4 samples")
    ap.add_argument("--force", action="store_true", help="redo stages whose outputs exist")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.limit = min(args.limit, 60)
        args.steps, args.reg_steps = 600, 300
        args.sample_cases = 4
        args.ddim = 20

    results_dir = os.path.join(args.work, "results")
    samples_dir = os.path.join(args.work, "samples")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    print(f"torch {torch_version()} | cuda {cuda_name()} | python {platform.python_version()}", flush=True)

    prepared = stage_data(args)
    diff_cfg, reg_cfg = make_configs(args, prepared)

    stage_train(args, diff_cfg, os.path.join(args.work, "runs", "diffusion"), "seg2mri.train", "diffusion")
    stage_train(args, reg_cfg, os.path.join(args.work, "runs", "regressor"), "seg2mri.regress", "regressor")

    stage_sample(args, os.path.join(args.work, "runs", "diffusion", "ckpt-last.pt"),
                 os.path.join(samples_dir, "diffusion"), export_real=True, steps=args.ddim)
    stage_sample(args, os.path.join(args.work, "runs", "regressor", "ckpt-last.pt"),
                 os.path.join(samples_dir, "regressor"), export_real=False, steps=0)
    stage_baseline(args, diff_cfg, os.path.join(samples_dir, "class_mean"))

    real = os.path.join(samples_dir, "diffusion", "real")
    stage_evaluate(os.path.join(samples_dir, "diffusion", "gen"), real,
                   os.path.join(results_dir, "diffusion.csv"), "diffusion")
    stage_evaluate(os.path.join(samples_dir, "regressor", "gen"), real,
                   os.path.join(results_dir, "regressor.csv"), "regressor")
    stage_evaluate(os.path.join(samples_dir, "class_mean"), real,
                   os.path.join(results_dir, "class_mean.csv"), "class-mean")

    print_loss_summary(args.work)
    stage_compare(args, results_dir)

    if not args.no_zip:
        stage_package(args)
    print("\nDONE. results/RESULTS.md is the table; samples/ has the NIfTI volumes and montages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
