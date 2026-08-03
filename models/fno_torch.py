"""FNO family models for CFO experiments (PyTorch version)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Create sinusoidal time embeddings for FNO conditioning."""
    half_dim = embed_dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(10000.0), half_dim, device=t.device, dtype=t.dtype)
    )
    angles = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in Fourier space for 1D tensors."""
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in Fourier space for 2D tensors."""
    return torch.einsum("bixy,ioxy->boxy", a, b)




class SpectralConv2d(nn.Module):
    """2D spectral convolution over truncated Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.wr1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.wi1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.wr2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.wi2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C_in) -- channels-last, matching Flax
        x = x.permute(0, 3, 1, 2)  # (B, C_in, H, W)
        w1 = torch.complex(self.wr1, self.wi1)
        w2 = torch.complex(self.wr2, self.wi2)

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-2], x_ft.shape[-1], dtype=torch.complex64, device=x.device
        )
        m1 = min(self.modes1, x.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = compl_mul2d(x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = compl_mul2d(x_ft[:, :, -m1:, :m2], w2[:, :, :m1, :m2])
        x_out = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), dim=(-2, -1))
        return x_out.permute(0, 2, 3, 1)  # back to (B, H, W, C_out)


class FNO2d(nn.Module):
    """2D Fourier Neural Operator backbone.

    Same note as `FNO1d`: `in_channels` (channel count of `x`) and, if
    `use_condition=True`, `condition_channels` (channel count of `c`) must be
    passed explicitly since `nn.Linear` needs a fixed `in_features`.
    """

    def __init__(
        self,
        num_channels: int = 3,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        num_blocks: int = 4,
        use_condition: bool = True,
        use_time: bool = False,
        in_channels: int = 3,
        condition_channels: int = 3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_blocks = num_blocks
        self.use_condition = use_condition
        self.use_time = use_time

        fc0_in = in_channels + 2  # +2 for the (y, x) grid coordinates
        if use_condition:
            fc0_in += condition_channels

        self.fc0 = nn.Linear(fc0_in, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_blocks)]
        )
        self.ws = nn.ModuleList([nn.Linear(width, width) for _ in range(num_blocks)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        N, H, W, _ = x.shape

        if grid is None:
            gy = torch.linspace(0.0, 1.0, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(0.0, 1.0, W, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            grid = torch.stack([yy, xx], dim=-1)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1)

        x = torch.cat([x, grid], dim=-1)
        if self.use_condition and c is not None:
            if c.dim() == 3:
                c = c.unsqueeze(-1)
            x = torch.cat([x, c], dim=-1)

        x = self.fc0(x)
        if self.use_time and t is not None:
            x = x + _sinusoidal_time_embedding(t, self.width)[:, None, None, :]

        for conv, w in zip(self.convs[:-1], self.ws[:-1]):
            x = F.gelu(conv(x) + w(x))
        x = self.convs[-1](x) + self.ws[-1](x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


__all__ = [
    "SpectralConv2d",
    "FNO2d",
]