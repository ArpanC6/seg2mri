"""Single-step regressor — the "coarse prior" without diffusion.

Predicts the four modalities directly from the mask with L1 + gradient loss.
Training this alongside the diffusion model is the point: on the same split, with
the same U-Net, the difference between them is the diffusion process and nothing
else.

    python -m seg2mri.regress --config configs/reg2d.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data import make_dataset
from .models import count_parameters
from .train import EMA, build_model, lr_at


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dim = pred.dim() - 2
    total = pred.new_zeros(())
    for axis in range(2, 2 + dim):
        i1 = [slice(None)] * pred.dim()
        i2 = [slice(None)] * pred.dim()
        i1[axis] = slice(1, None)
        i2[axis] = slice(None, -1)
        total = total + ((pred[tuple(i1)] - pred[tuple(i2)]) - (target[tuple(i1)] - target[tuple(i2)])).abs().mean()
    return total / dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.out:
        cfg["train"]["out_dir"] = args.out
    grad_weight = float(cfg["train"].get("grad_loss_weight", 0.1))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    dataset = make_dataset(cfg["data"]["root"], "train", cfg["data"]["mode"], cfg["data"]["augment"])
    model = build_model(cfg)

    if args.dry_run:
        sample = dataset[0]
        print(f"regressor | {len(dataset)} samples | {count_parameters(model):,} params "
              f"| image {tuple(sample['image'].shape)}")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"], drop_last=True, pin_memory=device == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = EMA(model, cfg["train"]["ema_decay"])

    steps = cfg["train"]["steps"]
    log_every = cfg["train"]["log_every"]
    log_fh = open(os.path.join(out_dir, "log.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    writer.writerow(["step", "loss", "l1", "grad", "lr", "sec_per_step"])

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
        zeros = torch.zeros_like(x0)
        t0 = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

        pred = model(zeros, t0, cond)
        l1 = F.l1_loss(pred, x0)
        grad = gradient_loss(pred, x0)
        loss = l1 + grad_weight * grad

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
            writer.writerow([step, running / log_every, float(l1.detach()), float(grad.detach()), lr, elapsed / log_every])
            log_fh.flush()
            print(f"step {step:>6}/{steps}  loss {running / log_every:.4f}  l1 {float(l1.detach()):.4f}  "
                  f"grad {float(grad.detach()):.4f}  {elapsed / log_every:.2f}s/it")
            running = 0.0
            start = time.time()

        if cfg["train"]["ckpt_every"] and step % cfg["train"]["ckpt_every"] == 0:
            torch.save({"arch": "regressor", "model": ema.shadow, "raw": model.state_dict(), "cfg": cfg, "step": step},
                       os.path.join(out_dir, f"ckpt-{step:06d}.pt"))

    torch.save({"arch": "regressor", "model": ema.shadow, "raw": model.state_dict(), "cfg": cfg, "step": steps},
               os.path.join(out_dir, "ckpt-last.pt"))
    log_fh.close()
    print(f"saved {os.path.join(out_dir, 'ckpt-last.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
