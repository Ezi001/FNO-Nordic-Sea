# Import necessary libraries
import xarray as xr
import numpy as np
import zarr
import pandas as pd
import numcodecs

print("xarray:", xr.__version__)
print("zarr:", zarr.__version__)
print("numcodecs:", numcodecs.__version__)

path = r"C:\Users\esthe\Documents\UiO\FNO-Nordic-Sea\data\nemo_ocean.zarr"

z = zarr.open(path, mode="r")

print(z)
print(z.metadata)

zarr_path_ocean = "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/nemo_ocean.zarr"
zarr_path_forcing = "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/nemo_forcing.zarr"
#C:\Users\esthe\Documents\UiO\FNO-Nordic-Sea\data

from pathlib import Path

print("zarr path:", zarr_path_ocean)
print("exists:", Path(zarr_path_ocean).exists())
print("is directory:", Path(zarr_path_ocean).is_dir())

print("project data directory:")
print(list(Path(r"C:\Users\esthe\Documents\UiO\FNO-Nordic-Sea\data").iterdir()))

ds_ocean = xr.open_zarr(zarr_path_ocean)
ds_forcing = xr.open_zarr(zarr_path_forcing)

import matplotlib.pyplot as plt

ssh = ds_ocean["ssh"].isel(time=0)

plt.figure(figsize=(10, 7))
ssh.plot()

plt.title("SSH at first timestep")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

msl = ds_forcing["msl"].isel(time=0)

plt.figure(figsize=(10, 7))
msl.plot()

plt.title("MSL at first timestep")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

bathy = np.load("./data/nemo_bathymetry.npz")
print(bathy.files)

bathy = bathy["bathy_metry"]

plt.figure(figsize=(10, 7))
plt.imshow(bathy)

plt.title("Bathymetry")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# print("Ocean points:", ocean_mask.sum().item())
# print("Land points:", (~ocean_mask).sum().item())

print(ds_forcing.data_vars)
print(ds_forcing["time"].values)

# ssh = ds_ocean["ssh"]
# land_mask = ~ocean_mask



print("SSH:")
print(ssh.dims)
print(ssh.shape)

# print("\nOcean mask:")
# print(ocean_mask.dims)
# print(ocean_mask.shape)

#start_date = np.datetime64("2020-06-01")
#end_date = pd.Timestamp("2024-12-31") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


#forcing = ds_forcing.sel(time=slice(start_date, end_date))
#ocean = ds_ocean.sel(time=slice(start_date, end_date)) # Selecting the correct time period

# From 2020-06 to 2024
ssh = ssh.sel(time=slice("2020-01-01", "2023-12-31"))
trajectory_window=48
stride=48

# Select a trajectory
trajectory = ssh.isel(time=slice(0, trajectory_window))

print(trajectory.shape)


import torch

trajectory = torch.from_numpy(
    trajectory.values.astype("float32")
)

# Calculate spline coefficients
coefficients = (trajectory)

# Save coefficients
np.savez(
    "splines_2020_2024.npz",
    coefficients=coefficients,
)


data = np.load("splines_2020_2024.npz")

coefficients = data["coefficients"]

# nan_count = ssh.isnull().sum().compute().item()
# print("Number of NaNs:", nan_count)

#land_not_nan = ssh.where(land_mask).compute().notnull()

#print("Land values that are NOT NaN:",
#      land_not_nan.sum().item())