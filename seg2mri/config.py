"""YAML config loading with defaults, so every run is reproducible from one file."""

from __future__ import annotations

import copy

import yaml

DEFAULTS = {
    "seed": 42,
    "data": {
        "root": "data/msd_prepared",
        "mode": "2d",
        "augment": True,
        "num_workers": 0,
    },
    "model": {
        "base_channels": 64,
        "channel_mults": [1, 2, 4],
        "attn_levels": [2],
        "dropout": 0.0,
    },
    "diffusion": {
        "timesteps": 1000,
        "schedule": "cosine",
        "prediction": "v",
        "ddim_steps": 50,
    },
    "train": {
        "steps": 30000,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "ema_decay": 0.999,
        "warmup_steps": 500,
        "grad_clip": 1.0,
        "log_every": 50,
        "ckpt_every": 2000,
        "out_dir": "runs/smoke",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(copy.deepcopy(DEFAULTS), user)
    cfg["_config_path"] = path
    return cfg
