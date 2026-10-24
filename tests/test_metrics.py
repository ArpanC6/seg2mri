import numpy as np

from seg2mri.metrics import evaluate_pair, msssim2d, nrmse, psnr, ssim3d, to_unit


def _structured(size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    c = size / 2
    vol = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2 < (size * 0.3) ** 2).astype(np.float32)
    vol += 0.05 * rng.standard_normal(vol.shape).astype(np.float32)
    return vol


def test_identical_volumes_score_perfectly():
    vol = to_unit(_structured())
    assert abs(ssim3d(vol, vol) - 1.0) < 1e-5
    assert psnr(vol, vol) == float("inf")
    assert nrmse(vol, vol) < 1e-6


def test_perturbation_lowers_every_metric():
    rng = np.random.default_rng(1)
    real = to_unit(_structured())
    gen = to_unit(real + 0.1 * rng.standard_normal(real.shape).astype(np.float32))
    assert ssim3d(gen, real) < 0.99
    assert np.isfinite(psnr(gen, real))
    assert psnr(gen, real) < 40
    assert nrmse(gen, real) > 0


def test_msssim_is_one_for_identical_slices():
    sl = to_unit(_structured()[4])
    assert abs(msssim2d(sl, sl) - 1.0) < 1e-3


def test_msssim_drops_with_noise():
    sl = to_unit(_structured()[4])
    rng = np.random.default_rng(2)
    noisy = np.clip(sl + 0.15 * rng.standard_normal(sl.shape), 0, 1).astype(np.float32)
    assert msssim2d(noisy, sl) < msssim2d(sl, sl)


def test_evaluate_pair_returns_all_keys():
    rng = np.random.default_rng(3)
    real = _structured()
    gen = real + 0.1 * rng.standard_normal(real.shape).astype(np.float32)
    out = evaluate_pair(gen, real)
    assert set(out) == {"ssim3d", "msssim_slice_mean", "psnr", "nrmse"}
    assert all(np.isfinite(v) for v in out.values())
