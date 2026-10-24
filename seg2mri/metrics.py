"""Metrics, implemented once and used by both the CLI and the tests.

Conventions are explicit because SSIM and MS-SSIM move a lot with them:

* every volume is min-max normalised to [0, 1] independently before scoring;
* 3D SSIM is ``skimage`` with ``win_size=7`` and ``data_range=1``;
* MS-SSIM is the standard 5-scale version evaluated slice-by-slice in the
  axial plane, then averaged (it is a 2D metric; there is no accepted 3D form).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_MSSSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


def to_unit(vol: np.ndarray) -> np.ndarray:
    lo, hi = float(vol.min()), float(vol.max())
    return (vol - lo) / (hi - lo + 1e-8)


def ssim3d(gen: np.ndarray, real: np.ndarray, win_size: int = 7) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(real, gen, data_range=1.0, win_size=win_size, channel_axis=None))


def psnr(gen: np.ndarray, real: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((gen - real) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def nrmse(gen: np.ndarray, real: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((gen - real) ** 2)))
    return rmse / (float(real.max() - real.min()) + 1e-8)


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def _ssim_components(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, data_range: float = 1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, window, padding=window.shape[-1] // 2)
    mu_y = F.conv2d(y, window, padding=window.shape[-1] // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    s_x = F.conv2d(x * x, window, padding=window.shape[-1] // 2) - mu_x2
    s_y = F.conv2d(y * y, window, padding=window.shape[-1] // 2) - mu_y2
    s_xy = F.conv2d(x * y, window, padding=window.shape[-1] // 2) - mu_xy
    luminance = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    cs = (2 * s_xy + c2) / (s_x + s_y + c2)
    return luminance, cs


def msssim2d(gen: np.ndarray, real: np.ndarray, data_range: float = 1.0) -> float:
    """MS-SSIM for one 2D slice, inputs in [0, 1].

    Contrast-structure is combined over the first four scales, luminance is
    taken at the coarsest scale, as in Wang et al. (2003).
    """
    x = torch.from_numpy(np.ascontiguousarray(gen, dtype=np.float32))[None, None]
    y = torch.from_numpy(np.ascontiguousarray(real, dtype=np.float32))[None, None]
    window = _gaussian_window(device=x.device)

    cs_scores = []
    luminance = None
    for scale in range(len(_MSSSIM_WEIGHTS)):
        luminance, cs = _ssim_components(x, y, window, data_range)
        cs_scores.append(float(cs.clamp(min=0.0).mean()))
        if scale < len(_MSSSIM_WEIGHTS) - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)

    cs_term = float(np.prod([c ** w for c, w in zip(cs_scores[:-1], _MSSSIM_WEIGHTS[:-1])]))
    lum_term = float(luminance.clamp(min=0.0).mean()) ** _MSSSIM_WEIGHTS[-1]
    return cs_term * lum_term


def msssim_slices(gen: np.ndarray, real: np.ndarray) -> float:
    scores = []
    for z in range(real.shape[2]):
        a, b = real[:, :, z], gen[:, :, z]
        if a.std() < 1e-6 and b.std() < 1e-6:
            continue
        scores.append(msssim2d(a, b))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_pair(gen: np.ndarray, real: np.ndarray) -> dict:
    if gen.shape != real.shape:
        raise ValueError(f"shape mismatch {gen.shape} vs {real.shape}")
    gen, real = to_unit(gen), to_unit(real)
    return {
        "ssim3d": ssim3d(gen, real),
        "msssim_slice_mean": msssim_slices(gen, real),
        "psnr": psnr(gen, real),
        "nrmse": nrmse(gen, real),
    }
