"""Training entry point.

    python -m seg2mri.train --config configs/smoke.yaml
    python -m seg2mri.train --config configs/demo2d.yaml --steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import make_dataset
from .diffusion import GaussianDiffusion
from .models import UNet, count_parameters


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_model(cfg: dict) -> UNet:
    mode = cfg["data"]["mode"]
    model_cfg = cfg["model"]
    return UNet(
        dim=2 if mode == "2d" else 3,
        in_channels=4 + 3,
        out_channels=4,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        attn_levels=tuple(model_cfg["attn_levels"]),
        dropout=model_cfg["dropout"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0, help="override config batch size (use 8 if CUDA OOM)")
    ap.add_argument("--lr", type=float, default=0.0, help="override config learning rate")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.out:
        cfg["train"]["out_dir"] = args.out

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = make_dataset(cfg["data"]["root"], "train", cfg["data"]["mode"], cfg["data"]["augment"])
    model = build_model(cfg)

    if args.dry_run:
        sample = dataset[0]
        print(f"mode          {cfg['data']['mode']}")
        print(f"dataset       {cfg['data']['root']}  ({len(dataset)} samples)")
        print(f"image         {tuple(sample['image'].shape)}  range [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
        print(f"cond          {tuple(sample['cond'].shape)}  classes {int(sample['cond'].sum())} voxels")
        print(f"parameters    {count_parameters(model):,}")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    diffusion = GaussianDiffusion(
        model,
        image_channels=4,
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        prediction=cfg["diffusion"]["prediction"],
    ).to(device)

    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        pin_memory=device == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = EMA(model, cfg["train"]["ema_decay"])

    steps = cfg["train"]["steps"]
    log_every = cfg["train"]["log_every"]
    log_path = os.path.join(out_dir, "log.csv")
    log_fh = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    writer.writerow(["step", "loss", "lr", "sec_per_step"])

    print(f"device {device} | params {count_parameters(model):,} | steps {steps} | batch {cfg['train']['batch_size']}")
    it = iter(loader)
    start = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x0 = batch["image"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)

        loss, _ = diffusion.training_loss(x0, cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["train"]["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        lr = lr_at(step, cfg["train"]["lr"], cfg["train"]["warmup_steps"], steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        ema.update(model)

        running += float(loss.detach())
        if step % log_every == 0:
            elapsed = time.time() - start
            writer.writerow([step, running / log_every, lr, elapsed / log_every])
            log_fh.flush()
            print(f"step {step:>6}/{steps}  loss {running / log_every:.4f}  lr {lr:.2e}  {elapsed / log_every:.2f}s/it")
            running = 0.0
            start = time.time()

        if cfg["train"]["ckpt_every"] and step % cfg["train"]["ckpt_every"] == 0:
            torch.save(
                {"model": ema.shadow, "raw": model.state_dict(), "cfg": cfg, "step": step},
                os.path.join(out_dir, f"ckpt-{step:06d}.pt"),
            )

    torch.save(
        {"model": ema.shadow, "raw": model.state_dict(), "cfg": cfg, "step": steps},
        os.path.join(out_dir, "ckpt-last.pt"),
    )
    log_fh.close()
    print(f"saved {os.path.join(out_dir, 'ckpt-last.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
