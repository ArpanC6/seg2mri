import torch
import torch.nn as nn

from seg2mri.diffusion import GaussianDiffusion


class ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, cond):
        return torch.zeros_like(x)


def test_v_prediction_roundtrip_is_exact():
    diffusion = GaussianDiffusion(ZeroModel(), image_channels=4, timesteps=100, prediction="v")
    x0 = torch.randn(3, 4, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 100, (3,))

    v = diffusion._target(x0, t, noise)
    x_t = diffusion.q_sample(x0, t, noise)
    x0_hat, eps_hat = diffusion._split_prediction(x_t, t, v)

    assert torch.allclose(x0_hat, x0, atol=1e-5)
    assert torch.allclose(eps_hat, noise, atol=1e-5)


def test_q_sample_at_final_step_is_noise_like():
    diffusion = GaussianDiffusion(ZeroModel(), image_channels=4, timesteps=1000)
    x0 = torch.zeros(256, 4, 4, 4)
    t = torch.full((256,), 999, dtype=torch.long)
    out = diffusion.q_sample(x0, t, torch.randn_like(x0))
    assert abs(float(out.mean())) < 0.2
    assert abs(float(out.std()) - 1.0) < 0.2


def test_ddim_sample_shape_and_finiteness():
    diffusion = GaussianDiffusion(ZeroModel(), image_channels=4, timesteps=100)
    cond = torch.randn(2, 3, 8, 8)
    out = diffusion.ddim_sample(cond, steps=5)
    assert out.shape == (2, 4, 8, 8)
    assert torch.isfinite(out).all()


def test_training_loss_is_finite():
    diffusion = GaussianDiffusion(ZeroModel(), image_channels=4, timesteps=100)
    loss, logs = diffusion.training_loss(torch.randn(2, 4, 8, 8), torch.randn(2, 3, 8, 8))
    assert torch.isfinite(loss)
    assert "loss" in logs
