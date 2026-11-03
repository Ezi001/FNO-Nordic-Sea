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

# Import necessary libraries
from pathlib import Path
import argparse
import sys

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Paths
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-dir", default="./data/nordic")
    
    parser.add_argument("--traj-length", type=int, default=49)
    parser.add_argument("--stride", type=int, default=49)

    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2022-12-31")

    parser.add_argument("--split", type=str, choices= ["eval", "test"], default="eval")

    return parser.parse_args()

args=parse_args()

if args.split=="eval":
    START_DATE = "2022-01-01"

    END_DATE = "2022-12-31"
else:
    START_DATE = "2023-01-01"

    END_DATE = "2023-12-31"

DATA_DIR = Path(args.data_dir)

INPUT_PATH = DATA_DIR / "nemo_ocean.zarr"

OUTPUT_PATH = DATA_DIR / "ssh_test_traj.npy"


# ============================================================
# Parameters
# ============================================================

N_TRAJECTORIES = 1000

TRAJECTORY_LENGTH = args.traj_length

STRIDE = args.stride


# ============================================================
# Open original dataset
# ============================================================

print("Opening input dataset...")

ds = xr.open_zarr(INPUT_PATH)

print("time dims:", ds.time.dims)
print("time shape:", ds.time.shape)
print("time values:", ds.time.values)

print(ds)

ssh = ds["ssh"].sel(
    time=slice(START_DATE, END_DATE)
)

print("\nSSH:")
print(ssh)

print("\nSSH shape:")
print(ssh.shape)

print("\ntimes: ")
print(ssh.time.values)


# ============================================================
# Determine dimensions
# ============================================================

n_time = ssh.sizes["time"]
Y = ssh.sizes["y"]
X = ssh.sizes["x"]

# Normalising
def compute_ssh_stats(ssh, chunk_size=100):
    """
    Compute mean and standard deviation of SSH over the
    training period, ignoring NaNs.
    """

    if ssh.ndim == 5 and ssh.shape[-1] == 1:
        ssh = ssh.isel(channel=0)

    if ssh.ndim != 3:
        raise ValueError(
            f"Expected SSH shape (time, Y, X), got {ssh.shape}"
        )

    n = ssh.shape[0]

    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        chunk = ssh.isel(time=slice(start, end)).values

        valid = np.isfinite(chunk)
        values = chunk[valid]

        sum_ += np.sum(values, dtype=np.float64)
        sum_sq += np.sum(values ** 2, dtype=np.float64)
        count += values.size

    mean = sum_ / count
    variance = sum_sq / count - mean**2
    std = np.sqrt(max(variance, 0.0))

    return np.float32(mean), np.float32(std)

ssh_train = ds["ssh"].sel(
    time=slice("2020-06-01", "2021-12-31")
)

ssh_mean, ssh_std = compute_ssh_stats(ssh_train)

ssh_std = (ssh - ssh_mean)/ ssh_std

print("mean, std" np.nanmean(ssh_std), np.nanstd(ssh_std))


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
    data = ssh_std.isel(
        time=slice(start, end)
    )

    # Converting nans to 0.
    data = np.nan_to_num(data, nan=0.0)

    #print(data.time.values)

    #data = data.values.astype(np.float32)

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
