# Esther Zijerveld
# Create 2 day windows
# train/val/test split
# Calculate spline coefficients
# Save to Zarr
# CFO training dataloader
"""
This script should calculate splines for 2 day trajectories within the training period of 2020-06-01 to 2022-12-31

"""
import numpy as np
import xarray as xr
from pathlib import Path
import zarr

from .utils.splines_torch import quintic_spline_batch

ds = xr.open_zarr(
    "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/nemo_ocean.zarr"
)

ssh_train = ds["ssh"].sel(time=slice("2020-06-01", "2022-12-31"))
ssh_test = ds["ssh"].sel(time=slice("2023-01-01", "2024-12-31"))

print("Train:", ssh_train.shape)
print("Test: ", ssh_test.shape)

trajectory_length = 49 # define trajectory length 49 is 2 days
stride = 48 

# Calculate splines for one dataset

def calculate_splines(ssh, trajectory_length, stride, output_path):

    n_windows = len(range(0, len(ssh) - trajectory_length + 1, stride)
    )

    # Calculate first trajectory to determine coefficient shape
    trajectory = ssh.isel(
        time=slice(0, trajectory_length)
    ).values

    trajectory = trajectory[None, ...]

    coeffs, starts = quintic_spline_batch(
        trajectory,
        time=None,
        batch_size=1,
        num_point=3,
        device="cpu"
    )

    print("One coefficient shape:", coeffs.shape)
    print("One starts shape:", starts.shape)

    # Create Zarr store
    root = zarr.open_group(
        output_path,
        mode="w"
    )

    coefficients = root.create_array(
        "coefficients",
        shape=(n_windows,) + coeffs.shape[1:],
        dtype=coeffs.dtype,
        chunks=(1,) + coeffs.shape[1:]
    )

    spline_starts = root.create_array(
        "starts",
        shape=(n_windows,) + starts.shape[1:],
        dtype=starts.dtype,
        chunks=(1,) + starts.shape[1:]
    )

    # Save first trajectory
    coefficients[0] = coeffs[0]
    spline_starts[0] = starts[0]

    # Remaining trajectories
    for i, start in enumerate(
        range(
            stride,
            len(ssh) - trajectory_length + 1,
            stride
        ),
        start=1
    ):

        trajectory = ssh.isel(
            time=slice(start, start + trajectory_length)
        ).values

        trajectory = trajectory[None, ...]

        coeffs, starts = quintic_spline_batch(
            trajectory,
            time=None,
            batch_size=1,
            num_point=3,
            device="cpu"
        )

        coefficients[i] = coeffs[0]
        spline_starts[i] = starts[0]

        if i % 10 == 0:
            print(f"Saved {i + 1}/{n_windows} trajectories")

    print("Finished:", output_path)

# Training splines

# train_coeffs, train_starts = calculate_splines(
#     ssh_train,
#     trajectory_length,
#     stride
# )

# # Test splines


# test_coeffs, test_starts = calculate_splines(
#     ssh_test,
#     trajectory_length,
#     stride
# )

# print("Test coefficients:", test_coeffs.shape)
# print("Test starts:", test_starts.shape)

# Saving 

calculate_splines(
    ssh_train,
    trajectory_length=49,
    stride=48,
    output_path="C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/splines_train.zarr"
)

calculate_splines(
    ssh_test,
    trajectory_length=49,
    stride=48,
    output_path="C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data/splines_test.zarr"
)

# First 49 time points = 2 days
X_train = ssh_train.isel(time=slice(0, 49)).values

print(X_train.shape)

X_train = X_train[None, ...]

print(X_train.shape)

coeffs, starts = quintic_spline_batch(
    X_train,
    time=None,
    batch_size=1,
    num_point=3,
    device="cpu"
)

print(coeffs.shape)
print(starts.shape)


for start in range(0, len(ssh_train) - trajectory_length + 1, stride):
    trajectory = ssh_train.isel(
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
