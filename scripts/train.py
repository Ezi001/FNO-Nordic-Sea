"""
Name: Esther Zijerveld
Training a FNO on the North sea data for my Master's Thesis

1. loading and preprocessing the data
2. creating FNO model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
5. Evaluating predictions
"""
#Import dependencies
import os
from utils.utils import regrid_xy
from utils.DataLoader import NordicSeaCurrentDataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr


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
)["ssh"]

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
    "data/fno_ERA5forcing_y1980m01.nc",
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
if "time" in forcing.coords and "time_counter" in forcing.coords:
    forcing = forcing.drop_vars("time")

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
bathymetry = bath["bathy_metry"].rename("bathymetry")

train = xr.merge([ssh, u, v, wind_u, wind_v, slp, bathymetry]).sel(time_counter=slice("1980", "2010"))
val   = xr.merge([ssh, u, v, wind_u, wind_v, slp, bathymetry]).sel(time_counter=slice("2011", "2018"))
test  = xr.merge([ssh, u, v, wind_u, wind_v, slp, bathymetry]).sel(time_counter=slice("2019", "2024"))

# -------------------------
# 3. Dataset with shared normalization
# -------------------------
input_vars = ("ssh", "u", "v", "wind_u", "wind_v", "slp")

mean = np.array([train[var].mean().item() for var in input_vars], dtype=np.float32)
std  = np.array([train[var].std().item() for var in input_vars], dtype=np.float32)
std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

train_dataset = NordicSeaCurrentDataset(
    train,
    input_vars=input_vars,
    target_vars=("ssh", "u", "v"),
    bathymetry_var="bathymetry",
    horizon=1,
    temporal_window=4,
    return_temporal=True,
    mean=mean,
    std=std,
)
val_dataset = NordicSeaCurrentDataset(
    val,
    input_vars=input_vars,
    target_vars=("ssh", "u", "v"),
    bathymetry_var="bathymetry",
    horizon=1,
    temporal_window=4,
    return_temporal=True,
    mean=mean,
    std=std,
)
test_dataset = NordicSeaCurrentDataset(
    test,
    input_vars=input_vars,
    target_vars=("ssh", "u", "v"),
    bathymetry_var="bathymetry",
    horizon=1,
    temporal_window=4,
    return_temporal=True,
    mean=mean,
    std=std,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=0)

# Quick sanity check
x_batch, y_batch = next(iter(train_loader))
print("Input batch:", x_batch.shape)   # [B, 7, T, H, W]
print("Target batch:", y_batch.shape)  # [B, 3, H, W]


class FNOtDWrapper(nn.Module):
    def __init__(self, in_channels, out_channels=3, hidden_channels=64, n_layers=4, n_modes=(16, 16), padding=8):
        super().__init__()
        self.spatial_model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            padding=padding,
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out_flat = self.spatial_model(x_flat)
        out = out_flat.reshape(b, t, out_flat.shape[1], h, w)
        return out[:, -1]


# -------------------------
# 4. FNO model
# -------------------------
# Choose n_modes smaller than H/2, W/2
H, W = x_batch.shape[-2], x_batch.shape[-1]
n_modes = (min(16, H // 2), min(16, W // 2))

model = FNOtDWrapper(
    in_channels=len(input_vars) + 1,
    out_channels=3,
    hidden_channels=64,
    n_layers=4,
    n_modes=n_modes,
    padding=8,
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = LpLoss(d=2, p=2)  # L2 loss over spatial domain


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / n_samples


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / n_samples


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
