"""
Create raw Nordic Sea SSH trajectories for CFO training.

Input:
    nemo_ocean.zarr

Output:
    nordic_trajectories.npy

Shape:
    (N, T, Y, X, 1)

N = 1000 trajectories
T = 48 hourly snapshots
stride = 49
"""

from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    "C:/Users/esthe/Documents/UiO/FNO-Nordic-Sea/data"
)

INPUT_PATH = DATA_DIR / "nemo_ocean.zarr"

OUTPUT_PATH = DATA_DIR / "nordic_trajectories.npy"


# ============================================================
# Parameters
# ============================================================

N_TRAJECTORIES = 1000

TRAJECTORY_LENGTH = 48

STRIDE = 49

START_DATE = "2020-06-01"

END_DATE = "2024-12-31"


# ============================================================
# Open original dataset
# ============================================================

print("Opening input dataset...")

ds = xr.open_zarr(INPUT_PATH)

print(ds)

ssh = ds["ssh"].sel(
    time=slice(START_DATE, END_DATE)
)

print("\nSSH:")
print(ssh)

print("\nSSH shape:")
print(ssh.shape)


# ============================================================
# Determine dimensions
# ============================================================

n_time = ssh.sizes["time"]
Y = ssh.sizes["y"]
X = ssh.sizes["x"]


# ============================================================
# Calculate number of trajectories
# ============================================================

n_possible = 1 + (
    (n_time - TRAJECTORY_LENGTH) // STRIDE
)

n_trajectories = min(
    N_TRAJECTORIES,
    n_possible,
)

print("\nNumber of time points:", n_time)
print("Possible trajectories:", n_possible)
print("Trajectories requested:", N_TRAJECTORIES)
print("Trajectories created:", n_trajectories)

if n_trajectories < N_TRAJECTORIES:
    print(
        "\nWARNING: Fewer than 1000 trajectories "
        "are available."
    )


# ============================================================
# Create memory-mapped output
# ============================================================

print("\nCreating output file...")

trajectories = np.lib.format.open_memmap(
    OUTPUT_PATH,
    mode="w+",
    dtype=np.float32,
    shape=(
        n_trajectories,
        TRAJECTORY_LENGTH,
        Y,
        X,
        1,
    ),
)

print("Output shape:")
print(trajectories.shape)


# ============================================================
# Create trajectories
# ============================================================

for i in range(n_trajectories):

    start = i * STRIDE

    end = start + TRAJECTORY_LENGTH

    print(
        f"Trajectory {i + 1}/{n_trajectories}: "
        f"{start}:{end}"
    )

    # Load only this trajectory
    data = ssh.isel(
        time=slice(start, end)
    ).values.astype(np.float32)

    # Expected:
    # (48, Y, X)

    if data.shape != (
        TRAJECTORY_LENGTH,
        Y,
        X,
    ):
        raise ValueError(
            f"Unexpected shape: {data.shape}"
        )

    # Add channel dimension
    #
    # (48, Y, X)
    #       ↓
    # (48, Y, X, 1)

    trajectories[i, :, :, :, 0] = data


# ============================================================
# Flush data to disk
# ============================================================

print("\nSaving...")

trajectories.flush()

del trajectories


# ============================================================
# Verify
# ============================================================

print("\nVerifying output...")

loaded = np.load(
    OUTPUT_PATH,
    mmap_mode="r",
)

print("Output:")
print(OUTPUT_PATH)

print("Shape:")
print(loaded.shape)

print("Dtype:")
print(loaded.dtype)

print("\nFirst trajectory:")
print(loaded[0].shape)

print("\nDone!")
