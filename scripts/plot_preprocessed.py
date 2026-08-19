# Import necessary libraries
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import time

print(time.time)
os.add_dll_directory("C:\\Users\\esthe\\anaconda3\\DLLs")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

files = {
    "state": PROCESSED / "state_1980-01-01_1980-01-07.nc",
    "forcing": PROCESSED / "forcing_1980-01-01_1980-01-07.nc", 
    "mask": PROCESSED / "mask_1980-01-01_1980-01-07.nc",
}

for key, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")

state_ds = xr.open_dataset(files["state"])
forcing_ds = xr.open_dataset(files["forcing"])
mask_ds = xr.open_dataset(files["mask"])

print(forcing_ds)
print("COORDINATES:")
print(forcing_ds.coords)
print("####")
print("vars")
print(forcing_ds.variables)
print(forcing_ds.dims)
print("STATE DATASET")
print(state_ds)
print("FORCING DATASET")
print(forcing_ds)
print("MASK DATASET")
print(mask_ds)



def get_data_array(ds):
    if len(ds.data_vars) != 1:
        raise ValueError(f"Expected one data variable, found {list(ds.data_vars)}")
    return ds[list(ds.data_vars)[0]]


def first_timestep(da):
    if "time" in da.dims:
        return da.isel(time=0)
    return da

def select_spatial_coords(arr):
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field, got dims {arr.dims}")

    if "nav_lon" in arr.coords and "nav_lat" in arr.coords:
        return arr.coords["nav_lon"].values, arr.coords["nav_lat"].values
    if "lon" in arr.coords and "lat" in arr.coords:
        return arr.coords["lon"].values, arr.coords["lat"].values
    if "x" in arr.coords and "y" in arr.coords:
        return arr.coords["x"].values, arr.coords["y"].values

    y_name, x_name = arr.dims
    if y_name in arr.coords and x_name in arr.coords:
        return arr.coords[x_name].values, arr.coords[y_name].values

    return np.arange(arr.shape[1]), np.arange(arr.shape[0])


def plot_2d(arr, ax, title):

    arr = arr.squeeze(drop=True)

    if "nav_lon" in arr.coords and "nav_lat" in arr.coords:

        mesh = ax.pcolormesh(
            arr.nav_lon.values,
            arr.nav_lat.values,
            arr.values,
            shading="auto"
        )

    else:

        mesh = ax.pcolormesh(
            np.arange(arr.shape[1]),
            np.arange(arr.shape[0]),
            arr.values,
            shading="auto"
        )

    ax.set_title(title)
    plt.colorbar(mesh, ax=ax)


state_da = get_data_array(state_ds)
forcing_da = get_data_array(forcing_ds)
mask_da = get_data_array(mask_ds)

print("STATE spatial coordinates:")
print(state_da.coords)

print("FORCING spatial coordinates:")
print(forcing_da.coords)

print("STATE shape:", state_da.shape)
print("FORCING shape:", forcing_da.shape)

print("State nav_lon shape:", state_ds.nav_lon.shape)
print("State nav_lat shape:", state_ds.nav_lat.shape)

state_lon = state_da.coords["nav_lon"]
state_lat = state_da.coords["nav_lat"]

# forcing_lon = forcing_da.coords["nav_lon"]
# forcing_lat = forcing_da.coords["nav_lat"]

# print("Same longitude grid:",
#       np.allclose(state_lon.values, forcing_lon.values))

# print("Same latitude grid:",
#       np.allclose(state_lat.values, forcing_lat.values))

state_t0 = first_timestep(state_da)
forcing_t0 = first_timestep(forcing_da)
mask_t0 = first_timestep(mask_da)

print("STATE first timestep dims:", state_t0.dims)
print("FORCING first timestep dims:", forcing_t0.dims)
print("MASK first timestep dims:", mask_t0.dims)

# Plot state and forcing channels in separate panels
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

state_channels = []
if "channel" in state_t0.dims:
    state_channels = list(state_t0.coords["channel"].values) if "channel" in state_t0.coords else [str(i) for i in range(state_t0.sizes["channel"])]
    for idx, name in enumerate(state_channels[:3]):
        field = forcing_t0.sel(channel="u10")

        print(field.shape)
        print(field.nav_lon.shape)
        print(field.nav_lat.shape)

        plot_2d(field, axes[idx], f"State: {name}")
else:
    plot_2d(state_t0, axes[0], "State")

forcing_channels = []
if "channel" in forcing_t0.dims:
    forcing_channels = list(forcing_t0.coords["channel"].values) if "channel" in forcing_t0.coords else [str(i) for i in range(forcing_t0.sizes["channel"])]
    for idx, name in enumerate(forcing_channels[:3]):
        field = forcing_t0.sel(channel="u10")

        print(field.shape)
        print(field.nav_lon.shape)
        print(field.nav_lat.shape)

        plot_2d(field, axes[idx + 3], f"Forcing: {name}")
else:
    plot_2d(forcing_t0, axes[3], "Forcing")

for ax in axes[len(state_channels) + len(forcing_channels):]:
    ax.axis("off")

fig.suptitle("Preprocessed fields from the first timestep")
fig.tight_layout(rect=[0, 0, 1, 0.98])
out_path = OUT / "preprocessed_overview.png"
fig.savefig(out_path, dpi=200)
print(f"Saved overview plot to {out_path}")



fig, axes = plt.subplots(1, 1, figsize=(18, 10))
mask = mask_ds.top_level
msl = forcing_t0.sel(channel="msl")
msl = msl.where(mask)
plot_2d(msl, axes, f"Forcing: msl")
fig.tight_layout(rect=[0, 0, 1, 0.98])
out_path = OUT / "preprocessed_msl.png"
fig.savefig(out_path, dpi=200)
print(f"Saved msl plot to {out_path}")

print(mask.dims)
print(msl.dims)

print(mask.shape)
print(msl.shape)

# A second plot for the mask
fig2, ax2 = plt.subplots(figsize=(6, 5))
mask_plot = mask_t0.squeeze(drop=True)
if mask_plot.ndim != 2:
    mask_plot = mask_plot.isel(channel=0) if "channel" in mask_plot.dims else mask_plot
if mask_plot.ndim != 2:
    raise ValueError(f"Cannot plot mask with dims {mask_plot.dims}")
x, y = select_spatial_coords(mask_plot)
mesh = ax2.pcolormesh(x, y, mask_plot.values.astype(float), shading="auto", cmap="gray")
ax2.set_title("Ocean mask")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.04)
fig2.tight_layout()
out_path2 = OUT / "ocean_mask.png"
fig2.savefig(out_path2, dpi=200)
print(f"Saved mask plot to {out_path2}")
