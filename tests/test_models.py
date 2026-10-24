import torch

from seg2mri.models import UNet, count_parameters


def test_unet_shapes_2d():
    model = UNet(dim=2, in_channels=7, out_channels=4, base_channels=8, channel_mults=(1, 2), attn_levels=())
    x = torch.randn(2, 4, 32, 32)
    cond = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    assert model(x, t, cond).shape == (2, 4, 32, 32)


def test_unet_shapes_3d():
    model = UNet(dim=3, in_channels=7, out_channels=4, base_channels=8, channel_mults=(1, 2), attn_levels=())
    x = torch.randn(1, 4, 16, 16, 16)
    cond = torch.randn(1, 3, 16, 16, 16)
    t = torch.randint(0, 10, (1,))
    assert model(x, t, cond).shape == (1, 4, 16, 16, 16)


def test_attention_level_keeps_shape():
    model = UNet(dim=2, in_channels=7, out_channels=4, base_channels=8, channel_mults=(1, 2), attn_levels=(1,))
    x = torch.randn(1, 4, 32, 32)
    cond = torch.randn(1, 3, 32, 32)
    out = model(x, torch.zeros(1, dtype=torch.long), cond)
    assert out.shape == (1, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_parameter_count_scales_with_channels():
    small = UNet(dim=2, in_channels=7, out_channels=4, base_channels=8, channel_mults=(1, 2), attn_levels=())
    large = UNet(dim=2, in_channels=7, out_channels=4, base_channels=16, channel_mults=(1, 2), attn_levels=())
    assert count_parameters(large) > count_parameters(small)
