"""
Name: Esther Zijerveld
Training a FNO on the North sea data

1. loading and preprocessing the data
2. creating FNO model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
5. Evaluating predictions
"""
#Import dependencies
import os
from utils.utils import OceanDataset, regrid_xy, train_one_epoch, eval_epoch
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import LpLoss

# -------------------------
# 0. Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# 1. Load data
# -------------------------
ssh = xr.open_mfdataset(
    "data/*ssh.nc",
    combine="by_coords",
    chunks={"time_counter": 10},
    engine="netcdf4",
    lock=False
)["ssh"].rename("ssh")

u = xr.open_mfdataset(
    "data/*ubar.nc",
    combine="by_coords",
    chunks={"time_counter": 10},
    engine="netcdf4",
    lock=False
)["ubar"].rename("u")

v = xr.open_mfdataset(
    "data/*vbar.nc",
    combine="by_coords",
    chunks={"time_counter": 10},
    engine="netcdf4",
    lock=False
)["vbar"].rename("v")

forcing = xr.open_mfdataset(
    "data/forcing/*.nc",
    combine="by_coords",
    chunks={"time_counter": 10},
    engine="netcdf4",
    lock=False
)

bath = xr.open_dataset(
    "data/nordic_seas_domain_cfg.nc",
    engine="netcdf4",
    lock=False
)

# Regrid forcing to NEMO grid
forcing = forcing.interp(time=ssh["time_counter"], method="nearest")
forcing = forcing.rename({"time": "time_counter"})

src_lon = forcing["lon"].values
src_lat = forcing["lat"].values
src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
source_points = np.column_stack((src_lat2d.ravel(), src_lon2d.ravel()))

nemo_lat = ssh["nav_lat"].values
nemo_lon = ssh["nav_lon"].values
target_points = np.column_stack((nemo_lat.ravel(), nemo_lon.ravel()))



wind_u = regrid_xy(forcing["u10"]).rename("wind_u")
wind_v = regrid_xy(forcing["v10"]).rename("wind_v")
slp    = regrid_xy(forcing["msl"]).rename("slp")

ssh, u, v, wind_u, wind_v, slp = xr.align(
    ssh, u, v, wind_u, wind_v, slp,
    join="inner"
)

# -------------------------
# 2. Train/val/test split
# -------------------------
train = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice("1980", "2010"))
val   = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice("2011", "2018"))
test  = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice("2019", "2024"))

bathymetry = bath["bathy_metry"]  # [y, x]

# -------------------------
# 3. Dataset with shared normalization
# -------------------------
# Compute normalization from train only
train_ssh = torch.from_numpy(train["ssh"].values).float()
train_u   = torch.from_numpy(train["u"].values).float()
train_v   = torch.from_numpy(train["v"].values).float()
train_wu  = torch.from_numpy(train["wind_u"].values).float()
train_wv  = torch.from_numpy(train["wind_v"].values).float()
train_slp = torch.from_numpy(train["slp"].values).float()

train_stack = torch.stack(
    [train_ssh, train_u, train_v, train_wu, train_wv, train_slp],
    dim=1
)  # [T, 6, H, W]

mean = train_stack.mean(dim=(0, 2, 3), keepdim=True)  # [1, 6, 1, 1]
std  = train_stack.std(dim=(0, 2, 3), keepdim=True) + 1e-6

train_dataset = OceanDataset(train, bathymetry, mean=mean, std=std)
val_dataset   = OceanDataset(val,   bathymetry, mean=mean, std=std)
test_dataset  = OceanDataset(test,  bathymetry, mean=mean, std=std)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=4)

# Quick sanity check
x_batch, y_batch = next(iter(train_loader))
print("Input batch:", x_batch.shape)   # [B, 7, H, W]
print("Target batch:", y_batch.shape)  # [B, 3, H, W]

# -------------------------
# 4. FNO model
# -------------------------
# Choose n_modes smaller than H/2, W/2
H, W = x_batch.shape[-2], x_batch.shape[-1]
n_modes = (min(16, H // 2), min(16, W // 2))

model = FNO(
    n_modes=n_modes,
    in_channels=7,
    out_channels=3,
    hidden_channels=64,
    n_layers=4,
    padding=8,          # important for non-periodic domain
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = LpLoss(d=2, p=2)  # L2 loss over spatial domain

# -------------------------
# 5. Training loop
# -------------------------
n_epochs = 20
for epoch in range(1, n_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss   = eval_epoch(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch:03d} | train loss: {train_loss:.4e} | val loss: {val_loss:.4e}")

# -------------------------
# 6. Simple test evaluation
# -------------------------
test_loss = eval_epoch(model, test_loader, loss_fn, device)
print("Test loss:", test_loss)
