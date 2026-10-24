"""A small 2D/3D U-Net noise estimator with time conditioning.

One class covers both regimes; ``dim=2`` and ``dim=3`` differ only in the
convolution type.  Parameter count is printed by ``train --dry-run``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim: int, cin: int, cout: int, k: int = 3, p: int = 1) -> nn.Module:
    cls = nn.Conv2d if dim == 2 else nn.Conv3d
    return cls(cin, cout, k, padding=p)


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(min(8, ch), ch)


def _broadcast(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t[(slice(None), slice(None)) + (None,) * dim]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = dim * 4
        self.mlp = nn.Sequential(nn.Linear(dim, self.out_dim), nn.SiLU(), nn.Linear(self.out_dim, self.out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, dim: int, cin: int, cout: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = _norm(cin)
        self.conv1 = _conv(dim, cin, cout)
        self.norm2 = _norm(cout)
        self.conv2 = _conv(dim, cout, cout)
        self.temb = nn.Linear(temb_dim, cout * 2)
        self.skip = _conv(dim, cin, cout, k=1, p=0) if cin != cout else nn.Identity()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.temb(F.silu(temb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + _broadcast(scale, self.dim)) + _broadcast(shift, self.dim)
        h = self.conv2(F.silu(h))
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = _norm(ch)
        self.qkv = _conv(dim, ch, ch * 3, k=1, p=0)
        self.proj = _conv(dim, ch, ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        out = (attn @ v).reshape(b, c, *spatial)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, dim: int, ch: int):
        super().__init__()
        self.dim = dim
        self.conv = _conv(dim, ch, ch, k=3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 2:
            return self.conv(F.avg_pool2d(x, 2))
        return self.conv(F.avg_pool3d(x, 2))


class Upsample(nn.Module):
    def __init__(self, dim: int, ch: int):
        super().__init__()
        self.conv = _conv(dim, ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        attn_levels: tuple[int, ...] = (2,),
        dropout: float = 0.0,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = len(channel_mults)
        self.num_res_blocks = num_res_blocks

        temb = TimeEmbedding(base_channels)
        self.time = temb
        self.stem = _conv(dim, in_channels, base_channels)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(dim, ch, base_channels * mult, temb.out_dim, dropout))
                ch = base_channels * mult
                if level in attn_levels:
                    blocks.append(SelfAttention(dim, ch))
            self.down_blocks.append(blocks)
            if level != self.num_levels - 1:
                self.downsamples.append(Downsample(dim, ch))

        self.mid = nn.ModuleList([
            ResBlock(dim, ch, ch, temb.out_dim, dropout),
            SelfAttention(dim, ch),
            ResBlock(dim, ch, ch, temb.out_dim, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            if level != self.num_levels - 1:
                self.upsamples.append(Upsample(dim, ch))
            cout = base_channels * channel_mults[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(dim, ch + cout, cout, temb.out_dim, dropout))
                ch = cout
                if level in attn_levels:
                    blocks.append(SelfAttention(dim, ch))
            self.up_blocks.append(blocks)

        self.out_norm = _norm(ch)
        self.out = _conv(dim, ch, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        temb = self.time(t)
        h = self.stem(torch.cat([x, cond], dim=1))

        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, temb)
                    skips.append(h)
                else:
                    h = block(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        for block in self.mid:
            h = block(h, temb) if isinstance(block, ResBlock) else block(h)

        for i, blocks in enumerate(self.up_blocks):
            if i > 0:
                h = self.upsamples[i - 1](h)
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(torch.cat([h, skips.pop()], dim=1), temb)
                else:
                    h = block(h)

        return self.out(self.out_norm(F.silu(h)))
