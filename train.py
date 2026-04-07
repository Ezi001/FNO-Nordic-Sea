# Required libraries
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
import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from neuralop.models import FNO
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
import sys
from functools import wraps
import xarray as xr

from neuralop import Trainer
# Each sample shape: [channels, lat, lon]
# Channels:
# 0: SSH
# 1: U velocity
# 2: V velocity
# 3: Wind U
# 4: Wind V
# 5: Sea level pressure
# 6: Bathymetry (static)

# Input shape:  [batch, 7, lat, lon]
# Output shape: [batch, 3, lat, lon]  (SSH, U, V at t+1)

# Using CPU
device = "cpu"
#Loading the dataset
# Create a data processor for the Nordic sea dataset
from data_processing import OceanDataset


from scipy.interpolate import griddata

# Load variables from your NetCDF files
ssh = xr.open_mfdataset("data/*ssh.nc", combine="by_coords", chunks={"time_counter": 10})["ssh"].rename("ssh")
u = xr.open_mfdataset("data/*ubar.nc", combine="by_coords", chunks={"time_counter": 10})["ubar"].rename("u")
v = xr.open_mfdataset("data/*vbar.nc", combine="by_coords", chunks={"time_counter": 10})["vbar"].rename("v")
forcing = xr.open_mfdataset("data/forcing/*.nc", combine="by_coords", chunks={"time": 10})
bath = xr.open_dataset("data/nordic_seas_domain_cfg.nc")

# Regrid forcing from regular lat/lon to the NEMO nav_lat/nav_lon grid.
forcing = forcing.interp(time=ssh["time_counter"], method="nearest")
forcing = forcing.rename({"time": "time_counter"})

src_lon = forcing["lon"].values
src_lat = forcing["lat"].values
src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
source_points = np.column_stack((src_lat2d.ravel(), src_lon2d.ravel()))

nemo_lat = ssh["nav_lat"].values
nemo_lon = ssh["nav_lon"].values
target_points = np.column_stack((nemo_lat.ravel(), nemo_lon.ravel()))


def regrid_xy(da: xr.DataArray) -> xr.DataArray:
    values = da.values
    result = np.empty((values.shape[0],) + nemo_lat.shape, dtype=np.float32)
    for t in range(values.shape[0]):
        linear = griddata(
            source_points,
            values[t].ravel(),
            target_points,
            method="linear",
            fill_value=np.nan,
        )
        if np.isnan(linear).any():
            nearest = griddata(
                source_points,
                values[t].ravel(),
                target_points,
                method="nearest",
            )
            linear[np.isnan(linear)] = nearest[np.isnan(linear)]
        result[t] = linear.reshape(nemo_lat.shape)
    return xr.DataArray(
        result,
        coords={"time_counter": ssh["time_counter"], "y": ssh["y"], "x": ssh["x"]},
        dims=("time_counter", "y", "x"),
    )

wind_u = regrid_xy(forcing["u10"]).rename("wind_u")
wind_v = regrid_xy(forcing["v10"]).rename("wind_v")
slp = regrid_xy(forcing["msl"]).rename("slp")

ssh, u, v, wind_u, wind_v, slp = xr.align(
    ssh, u, v, wind_u, wind_v, slp,
    join="inner"
)

# Split by time
train = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice('1980', '2010'))
val = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice('2011', '2018'))
test = xr.merge([ssh, u, v, wind_u, wind_v, slp]).sel(time_counter=slice('2019', '2024'))

# Create datasets
train_dataset = OceanDataset(
    ssh=train['ssh'],
    u=train['u'],
    v=train['v'],
    wind_u=train['wind_u'],
    wind_v=train['wind_v'],
    slp=train['slp'],
    bathymetry=bath['bathy_metry']
)

val_dataset = OceanDataset(
    ssh=val['ssh'],
    u=val['u'],
    v=val['v'],
    wind_u=val['wind_u'],
    wind_v=val['wind_v'],
    slp=val['slp'],
    bathymetry=bath['bathy_metry']
)

test_dataset = OceanDataset(
    ssh=test['ssh'],
    u=test['u'],
    v=test['v'],
    wind_u=test['wind_u'],
    wind_v=test['wind_v'],
    slp=test['slp'],
    bathymetry=bath['bathy_metry']
)
# For each timestep t:
# input_t = torch.stack([
#     ssh[t], u[t], v[t],        # current state
#     wind_u[t], wind_v[t],      # forcings
#     slp[t],                     # forcing
#     bathymetry                  # static
# ], dim=0)

# target_t = torch.stack([
#     ssh[t+1], u[t+1], v[t+1]  # next state to predict
# ], dim=0)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,        # Adjust based on GPU memory
    shuffle=True,        # Shuffle training data
    num_workers=4,       # Parallel data loading
    pin_memory=True      # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,       # Don't shuffle validation
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# Check a single batch
input_batch, target_batch = next(iter(train_loader))
print(f"Input shape:  {input_batch.shape}")   # [8, 7, lat, lon]
print(f"Target shape: {target_batch.shape}")  # [8, 3, lat, lon]
# normalise everything

# 6 hourly data

#train test split

# Dataloader creation

# Model creation
# Create FNO model with specified parameters
"""model = FNO(
    n_modes=(16, 16),           # Fourier modes for each dimension
    in_channels=1,              # Input channels
    out_channels=1,             # Output channels
    hidden_channels=32,         # Hidden layer width
    projection_channel_ratio=2  # Channel expansion ratio
)
model = model.to(device)

print(f"Model parameters: {count_model_params(model)}")

# Setup optimizer and loss function
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

# Training step - works exactly as before
for batch_idx, (input_data, target_data) in enumerate(train_loader):
    # Move data to device
    input_data = input_data.to(device)    # Shape: (batch, channels, height, width)
    target_data = target_data.to(device)  # Shape: (batch, channels, height, width)

    # Forward pass - activations automatically offloaded to CPU
    output = model(input_data)

    # Compute loss
    loss = l2loss(output, target_data)

    # Backward pass - gradients computed with CPU-stored activations
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Count and display the number of parameters
n_params = count_model_params(operator)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


optimizer = AdamW(operator.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)



l2loss = LpLoss(d=2, p=2)  # L2 loss for function values
h1loss = H1Loss(d=2)  # H1 loss includes gradient information

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

#Training the model
#Display training configuration
print("\n### MODEL ###\n", operator)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


trainer = Trainer(
    model=operator,
    n_epochs=15,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Disable Weights & Biases logging for this tutorial
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# Train the model on our Nordic sea dataset. The trainer will:
# 1. Run the forward pass through the FNO
# 2. Compute the H1 loss
# 3. Backpropagate and update weights
# 4. Evaluate on test data every 3 epochs
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)"""