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


class SpectralConv1d(nn.Module):
    """1D spectral convolution over truncated Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Flax's nn.initializers.normal(stddev=scale) draws from N(0, scale^2),
        # i.e. `scale * standard_normal`, so this reproduces it exactly.
        scale = 1.0 / (in_channels * out_channels)
        self.wr = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.wi = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C_in) -- kept channels-last to match the Flax convention
        x = x.permute(0, 2, 1)  # (B, C_in, S)
        weights = torch.complex(self.wr, self.wi)

        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x_ft.shape[-1], dtype=torch.complex64, device=x.device
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = compl_mul1d(x_ft[:, :, :m], weights[:, :, :m])
        x_out = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return x_out.permute(0, 2, 1)  # back to (B, S, C_out)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator backbone.

    NOTE: unlike Flax's `nn.Dense`, `nn.Linear` isn't shape-polymorphic — it
    needs `in_features` fixed at construction time. The original module could
    lazily infer the input width of `fc0` from whatever `x`/`grid`/`c`
    happened to be at first call. Here you must pass `in_channels` (the
    channel count of `x` itself, e.g. 1 if `x` is a plain scalar field) and,
    if `use_condition=True`, `condition_channels` (channel count of `c`).
    """

    def __init__(
        self,
        num_channels: int,
        modes: int = 16,
        width: int = 64,
        num_blocks: int = 4,
        use_condition: bool = False,
        use_time: bool = True,
        in_channels: int = 1,
        condition_channels: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.modes = modes
        self.width = width
        self.num_blocks = num_blocks
        self.use_condition = use_condition
        self.use_time = use_time

        fc0_in = in_channels + 1  # +1 for the grid coordinate
        if use_condition:
            fc0_in += condition_channels

        self.fc0 = nn.Linear(fc0_in, width)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_blocks)])
        # A kernel_size=1 conv over a channels-last tensor is exactly a
        # per-position Linear over the channel dim, so nn.Linear replaces
        # Flax's `nn.Conv(width, kernel_size=(1,))` with no permute needed.
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
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, S, _ = x.shape
        if grid is None:
            g = torch.linspace(0.0, 1.0, S, device=x.device, dtype=x.dtype)
            grid = g.view(1, S, 1).expand(B, S, 1)
        x = torch.cat([x, grid], dim=-1)
        if self.use_condition and c is not None:
            if c.dim() == 2:
                c = c.unsqueeze(-1)
            x = torch.cat([x, c], dim=-1)

        x = self.fc0(x)
        if self.use_time and t is not None:
            x = x + _sinusoidal_time_embedding(t, self.width)[:, None, :]

        for conv, w in zip(self.convs[:-1], self.ws[:-1]):
            x = F.gelu(conv(x) + w(x))
        x = self.convs[-1](x) + self.ws[-1](x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


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
        num_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        num_blocks: int = 4,
        use_condition: bool = False,
        use_time: bool = False,
        in_channels: int = 1,
        condition_channels: int = 1,
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
    "SpectralConv1d",
    "SpectralConv2d",
    "FNO1d",
    "FNO2d",
]