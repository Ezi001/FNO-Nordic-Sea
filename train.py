"""
Name: Esther Maud Zijerveld
Date: 14.07.2026
Training a FNO on the North sea data for my Master's Thesis to be completed in 2027.

This training pipeline includes:
1. loading and preprocessing the data
2. creating FNO model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
5. Evaluating predictions
"""
#Import dependencies
import os

from utils.DataLoader import NordicSeaDataset
from models.fno import FNOtDWrapper
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import xarray as xr
import sklearn


from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import LpLoss

# -------------------------
# 0. Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

kwargs = {'num_workers': 0, 'pin_memory': False if device=="cpu" else True}
root_dir = "../data"

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -------------------------
# Hyperparameters
# -------------------------


# Remove top-level dataset creation to avoid multiprocessing at import time.

# -------------------------
#Load data
# -------------------------
train_dataset = NordicSeaDataset(
    root_dir=root_dir,
    start_time="1980-01-01",
    end_time="2009-12-31 23:00",
    temporal_window=4,

)

val_dataset = NordicSeaDataset(
    root_dir=root_dir,
    start_time="2010-01-01",
    end_time="2019-12-31 23:00",
    temporal_window=4,

)

test_dataset = NordicSeaDataset(
    root_dir=root_dir,
    start_time="2020-01-01",
    end_time="2024-12-31 23:00",
    temporal_window=4,
)



# # Regrid forcing to NEMO grid
# forcing = forcing.interp(time=ssh["time_counter"], method="nearest")
# if "time" in forcing.coords and "time_counter" in forcing.coords:
#     forcing = forcing.drop_vars("time")

# src_lon = forcing["lon"].values
# src_lat = forcing["lat"].values
# src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
# source_points = np.column_stack((src_lat2d.ravel(), src_lon2d.ravel()))

# nemo_lat = ssh["nav_lat"].values
# nemo_lon = ssh["nav_lon"].values
# target_points = np.column_stack((nemo_lat.ravel(), nemo_lon.ravel()))



# wind_u = regrid_xy(forcing["u10"]).rename("wind_u")
# wind_v = regrid_xy(forcing["v10"]).rename("wind_v")
# slp    = regrid_xy(forcing["msl"]).rename("slp")

# ssh, u, v, wind_u, wind_v, slp = xr.align(
#     ssh, u, v, wind_u, wind_v, slp,
#     join="inner"
# )



# -------------------------
# 3. Dataset with shared normalization
# -------------------------
input_vars = ("ssh", "u", "v", "wind_u", "wind_v", "slp")
output_vars = ("ssh", "u", "v")

# mean = np.array([train[var].mean().item() for var in input_vars], dtype=np.float32)
# std  = np.array([train[var].std().item() for var in input_vars], dtype=np.float32)
# std = np.where(std < 1e-6, 1.0, std).astype(np.float32)


# -----------------------
# Creating dataloader for train, validation and test.
# -----------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

# Quick sanity check
x_batch, y_batch = next(iter(train_loader))
print("Input batch:", x_batch.shape)   # [B, 7, T, H, W]
print("Target batch:", y_batch.shape)  # [B, 3, H, W]





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

# defining optimiser and loss function.
optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = LpLoss(d=2, p=2)  # L2 loss over spatial domain


from utils.utils import train_one_epoch, eval_epoch



if __name__ == '__main__':
    # -------------------------
    # 5. Training loop
    # -------------------------
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4e} | val loss: {val_loss:.4e}")

    # -------------------------
    # 6. Simple test evaluation
    # -------------------------
    test_loss = eval_epoch(model, test_loader, loss_fn, device)
    print("Test loss:", test_loss)
