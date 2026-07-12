"""Example script showing how to build a PyTorch DataLoader using OceanDataset.

This script:
- tries to open a merged xarray Dataset at ./merged_dataset.nc (expected variable names: ssh,u,v,wind_u,wind_v,slp,bathymetry)
- if not found, creates a small synthetic dataset for demonstration
- computes per-variable mean/std (over time) and constructs `OceanDataset`
- creates a `DataLoader` and prints a single batch shapes

Run from the project root:
python scripts/dataloader_example.py
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader

from utils.utils import OceanDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = PROJECT_ROOT / 'merged_dataset.nc'

VAR_ORDER = ['ssh', 'u', 'v', 'wind_u', 'wind_v', 'slp']


def make_synthetic_ds(nt=12, ny=32, nx=32):
    times = np.arange(nt)
    y = np.arange(ny)
    x = np.arange(nx)
    data = {}
    for v in VAR_ORDER:
        data[v] = (('time', 'y', 'x'), np.random.randn(nt, ny, nx).astype(np.float32))
    # bathymetry: static [y,x]
    data['bathymetry'] = (('y', 'x'), np.abs(np.random.randn(ny, nx).astype(np.float32)))
    ds = xr.Dataset(
        data_vars=data,
        coords={'time': times, 'y': y, 'x': x}
    )
    return ds


def compute_mean_std(ds):
    """Compute mean and std per-variable (order follows VAR_ORDER).
    Returns two 1D numpy arrays: mean (6,) and std (6,).
    """
    means = []
    stds = []
    for v in VAR_ORDER:
        if v not in ds:
            raise KeyError(f"Variable {v} not found in dataset")
        arr = ds[v]
        # mean/std over time and spatial dims, but keep per-variable single value
        m = float(arr.mean(dim=('time', 'y', 'x')).values)
        s = float(arr.std(dim=('time', 'y', 'x')).values)
        means.append(m)
        stds.append(s)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def main():
    if MERGED_PATH.exists():
        print(f"Loading merged dataset from {MERGED_PATH}")
        ds = xr.open_dataset(MERGED_PATH)
    else:
        print("Merged dataset not found — creating synthetic dataset for demonstration")
        ds = make_synthetic_ds(nt=20, ny=64, nx=64)

    # Ensure bathymetry exists
    if 'bathymetry' not in ds:
        # create a simple bathy if missing
        ny, nx = ds['y'].size, ds['x'].size
        ds['bathymetry'] = (('y', 'x'), np.zeros((ny, nx), dtype=np.float32))

    mean, std = compute_mean_std(ds)
    print('Per-variable mean:', mean)
    print('Per-variable std:', std)

    # Build dataset: OceanDataset accepts either a merged xarray.Dataset or individual DataArrays.
    # Here we pass the full merged dataset and bathymetry.
    od = OceanDataset(ds, bathymetry=ds['bathymetry'], horizon=1, mean=mean, std=std)

    loader = DataLoader(od, batch_size=4, shuffle=True, num_workers=0)

    # Iterate a single batch to show shapes
    for xb, yb in loader:
        print('Input batch shape:', xb.shape)   # [B, 7, H, W]
        print('Target batch shape:', yb.shape)  # [B, 3, H, W]
        break


if __name__ == '__main__':
    main()
