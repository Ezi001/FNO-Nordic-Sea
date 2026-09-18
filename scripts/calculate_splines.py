





# Create 2 day windows
# train/val/test split
# Calculate spline coefficients
# Save to Zarr
# CFO training dataloader


import numpy as np
import xarray as xr

from pathlib import Path

from splines_torch import quintic_spline_batch

ds = xr.open_zarr(
    "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/nemo_ocean.zarr"
)

ssh = ds["ssh"].sel(time=slice("2020-01-01", "2024-12-31"))

# First 9 time points = 2 days
X = ssh.isel(time=slice(0, 9)).values

print(X.shape)

X = X[None, ...]

print(X.shape)

coeffs, starts = quintic_spline_batch(
    X,
    time=None,
    batch_size=1,
    num_point=3,
    device="cpu"
)

print(coeffs.shape)
print(starts.shape)


trajectory_length = 49
stride = 48


for start in range(0, len(ssh) - trajectory_length + 1, stride):
    trajectory = ssh.isel(
        time=slice(start, start + trajectory_length)
    )


output_path = Path(
    "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/splines_2020_2024.npz"
)

np.savez(
    output_path,
    coefficients=coeffs,
    starts=starts,
)

print("Saved:", output_path)