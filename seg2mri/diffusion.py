"""Gaussian diffusion with v-prediction, plus a deterministic DDIM sampler.

The model is trained exactly as described in the technical note: v-prediction,
mean-squared error on v, cosine schedule.  Sampling uses DDIM (eta = 0), so a
seed reproduces a volume bit-for-bit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    out = values.gather(0, t)
    return out.reshape((-1,) + (1,) * (ndim - 1))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_channels: int = 4,
        timesteps: int = 1000,
        schedule: str = "cosine",
        prediction: str = "v",
    ):
        super().__init__()
        if prediction not in ("v", "eps", "x0"):
            raise ValueError(f"unsupported prediction '{prediction}'")
        self.model = model
        self.image_channels = image_channels
        self.timesteps = timesteps
        self.prediction = prediction

        betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa = _extract(self.sqrt_alphas_cumprod, t, x0.dim())
        sb = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.dim())
        return sa * x0 + sb * noise

    def _target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.prediction == "eps":
            return noise
        if self.prediction == "x0":
            return x0
        sa = _extract(self.sqrt_alphas_cumprod, t, x0.dim())
        sb = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.dim())
        return sa * noise - sb * x0

    def training_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t, cond)
        target = self._target(x0, t, noise)
        loss = F.mse_loss(pred, target)
        return loss, {"loss": loss.detach(), "t_mean": t.float().mean().detach()}

    def _split_prediction(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
        sa = _extract(self.sqrt_alphas_cumprod, t, x_t.dim())
        sb = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.dim())
        if self.prediction == "eps":
            eps = pred
            x0 = (x_t - sb * eps) / torch.clamp(sa, min=1e-6)
        elif self.prediction == "x0":
            x0 = pred
            eps = (x_t - sa * x0) / torch.clamp(sb, min=1e-6)
        else:
            x0 = sa * x_t - sb * pred
            eps = sb * x_t + sa * pred
        return x0, eps

    @torch.no_grad()
    def ddim_sample(
        self,
        cond: torch.Tensor,
        steps: int = 50,
        clip: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = cond.device
        shape = (cond.shape[0], self.image_channels) + tuple(cond.shape[2:])
        x = torch.randn(shape, device=device, generator=generator)

        seq = torch.linspace(0, self.timesteps - 1, steps, device=device)
        seq = seq.round().long().unique_consecutive().flip(0)

        for i, t_scalar in enumerate(seq):
            t = torch.full((cond.shape[0],), int(t_scalar), device=device, dtype=torch.long)
            pred = self.model(x, t, cond)
            x0, eps = self._split_prediction(x, t, pred)
            if clip:
                x0 = x0.clamp(-1.0, 1.0)

            prev = int(seq[i + 1]) if i + 1 < len(seq) else -1
            abar_prev = 1.0 if prev < 0 else float(self.alphas_cumprod[prev])
            x = math.sqrt(abar_prev) * x0 + math.sqrt(max(0.0, 1.0 - abar_prev)) * eps

        return x
