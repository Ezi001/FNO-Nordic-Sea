"""
This file is supposed to calculate the splines for the trajectories. 
I am splitting the data into trajectories of 2 days with a stride of 2 days and 1 hour.
"""
print(__doc__)

__author__ = "Esther Zijerveld"
print(__author__)

__version__="0.1"
print(__version__)

# Neccessary imports
import time
import sys

import numpy as np
import xarray as xr
import zarr

from splines_torch import quintic_spline_batch
from pathlib import Path

# Add the parent folder to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Where the data is located
DATA_DIR = "./data"
OUTPUT_DIR = "./processed"

# trajectory length is 2 days
# Number of trajectories will be 48.
trajectory_length = 49
stride = 49

#ds_ocean_preprocessed = xr.open_zarr(f"{DATA_DIR}/nemo_ocean.zarr")

ds_ocean = xr.open_zarr(f"./{DATA_DIR}/nemo_ocean.zarr")
#ds_forcing = xr.open_zarr(f"{DATA_DIR}/nemo_forcing.zarr")

print(ds_ocean["time"].values)

# Training from 2020/06/01 to 2021/12/31
# Eval from 
# Test from 2023/01/01 to 2024/12/31
# Selecting the correct time period

ssh_train = ds_ocean["ssh"].sel(
    time=slice(
        "2020-06-01",
        "2022-12-31",
    )
)
print(ssh_train["time"].values)

print(ssh_train.ndim)

# forcing_train = ds_forcing.sel(
#     time=slice(
#         "2020-06-01",
#         "2021-12-31",
#     )
# )

# print(forcing_train.shape)

# forcing_train = (
#     forcing_train[["u10", "v10", "msl"]]
#     .to_array(dim="channel")
#     .transpose("time", "y", "x", "channel")
# )

# print("Forcing type:", type(forcing_train))
# print("Forcing dims:", forcing_train.dims)
# print("Forcing shape:", forcing_train.shape)

ssh_eval = ds_ocean["ssh"].sel(
    time=slice(
        "2023-01-01",
        "2023-12-31",
    )
)

# forcing_eval = ds_forcing.sel(
#     time=slice(
#         "2022-01-01",
#         "2023-12-31",
#     )
# )

ssh_test = ds_ocean["ssh"].sel(
    time=slice(
        "2024-01-01",
        "2024-12-31",
    )
)

# forcing_test = ds_forcing.sel(
#     time=slice(
#         "2023-01-01",
#         "2024-12-31",
#     )
# )

# forcing_eval = (
#     forcing_eval[["u10", "v10", "msl"]]
#     .to_array(dim="channel")
#     .transpose("time", "y", "x", "channel")
# )


# forcing_test = (
#     forcing_test[["u10", "v10", "msl"]]
#     .to_array(dim="channel")
#     .transpose("time", "y", "x", "channel")
# )

# print("Forcing type:", type(forcing_test))
# print("Forcing dims:", forcing_test.dims)
# print("Forcing shape:", forcing_test.shape)



print("Train:", ssh_train.shape)
print("eval:", ssh_eval.shape)
print("test:", ssh_test.shape)

print(ssh_eval["time"].values)
print(ssh_test["time"].values)


print("Normalising the data!")

def compute_forcing_stats(forcing, chunk_size=100):
    """
    For computing the normalisation statistics
    """
    n = forcing.shape[0]

    sum_ = np.zeros(forcing.shape[-1], dtype=np.float64)
    sum_sq = np.zeros(forcing.shape[-1], dtype=np.float64)
    count = np.zeros(forcing.shape[-1], dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Convert xarray chunk to NumPy
        chunk = forcing.isel(time=slice(start, end)).values

        for c in range(forcing.shape[-1]):
            x = chunk[..., c]
            valid = np.isfinite(x)

            sum_[c] += np.sum(x[valid], dtype=np.float64)
            sum_sq[c] += np.sum(x[valid] ** 2, dtype=np.float64)
            count[c] += np.sum(valid)

    mean = sum_ / count
    variance = sum_sq / count - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))

    return mean.astype(np.float32), std.astype(np.float32)

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

#print(help(compute_ssh_stats))
#print(compute_forcing_stats.__doc__)

#Normalising on training statistics
#mean_forcing, std_forcing = compute_forcing_stats(forcing_train)

#print("Forcing mean:", mean_forcing)
#print("Forcing std:", std_forcing)

# Norm on training stats
mean_state, std_state = compute_ssh_stats(ssh_train)

print("SSH mean:", mean_state)
print("SSH std:", std_state)

ssh_train_std = (ssh_train - mean_state)/std_state
ssh_eval_std = (ssh_eval - mean_state)/std_state
ssh_test_std = (ssh_test - mean_state)/std_state

print("Train mean:", ssh_train_std.mean(skipna=True).compute().item())
print("Train std:", ssh_train_std.std(skipna=True).compute().item())

print("Eval mean:", ssh_eval_std.mean(skipna=True).compute().item())
print("Eval std:", ssh_eval_std.std(skipna=True).compute().item())

print("Test mean:", ssh_test_std.mean(skipna=True).compute().item())
print("Test std:", ssh_test_std.std(skipna=True).compute().item())

# Mean: -5.601083685746744e-08
# Std:  0.9999998807907104
# Mean: 0.19018904864788055
# Std:  1.0267821550369263
# Mean: 0.14704376459121704
# Std:  1.0457528829574585

n_windows_train = len(
        range(
            0,
            len(ssh_train) - trajectory_length + 1,
            stride,
        )
    )

n_windows_eval = len(
        range(
            0,
            len(ssh_eval) - trajectory_length + 1,
            stride,
        )
    )

n_windows_test = len(
        range(
            0,
            len(ssh_test) - trajectory_length + 1,
            stride,
        )
    )

print("Number of trajectories train: ", n_windows_train)
print("Number of trajectories eval: ", n_windows_eval)
print("Number of trajectories test: ", n_windows_test)


def calculate_splines(
    data,
    trajectory_length,
    stride,
    output_path,
    #mean_state,
    #std_state
):
    """
    For calculating spline coefficients and forcing at start of trajectories.
    Inputs:
        data (Xarray of shape (time, y, x)): 
        trajectory length: the desired
        stride: Defining the overlap
        output_path: Where to save the trajectory data
    Does:
        Saves the trajectory data
    """
    # The number of trajectories (13896 - 49 + 1/49) is 283 for train
    n_windows = len(
        range(
            0,
            len(data) - trajectory_length + 1,
            stride,
        )
    )


    # --------------------------------------------------
    # Determine spline dimensions using first trajectory
    # --------------------------------------------------

    trajectory = data.isel(
        time=slice(0, trajectory_length)
    )

    mean_state, std_state = compute_ssh_stats(trajectory)

    
    x = trajectory.values.astype(np.float32)


    #Normalize
    x = (x - mean_state) / std_state


    print(np.isnan(x).sum())

    #Replace NaNs after normalization
    x = np.nan_to_num(x, nan=0.0)


    times = trajectory["time"].values
    print(times[0])

    times_hours = (
        (times - times[0])
        / np.timedelta64(1, "h")
    ).astype(np.float32)

    print(times_hours[0])

    coeffs, starts, ends = quintic_spline_batch(
        x[None, ...],
        time=times_hours[None, :],
        batch_size=1,
        num_point=3,
        device="cpu",
    )


    n_segments, n_coefficients, ny, nx = coeffs.shape

    print("Spline coefficient shape:", coeffs.shape)

    # --------------------------------------------------
    # Create Zarr store
    # --------------------------------------------------

    root = zarr.open_group(
        output_path,
        mode="w",
    )
    
    z_coeffs = root.create_array(
        "spline_coef",
        shape=(
            n_windows,
            n_segments,
            n_coefficients,
            ny,
            nx,
        ),
        chunks=(
            1,
            n_segments,
            n_coefficients,
            ny,
            nx,
        ),
        dtype=np.float32,
    )

    z_starts = root.create_array(
        "spline_start",
        shape=(
            n_windows,
            n_segments,
        ),
        chunks=(
            1,
            n_segments,
        ),
        dtype=np.float32,
    )

    z_ends = root.create_array(
        "spline_end",
        shape=(
            n_windows,
            n_segments,
        ),
        chunks=(
            1,
            n_segments,
        ),
        dtype=np.float32,
    )

    z_t_start = root.create_array(
        "t_start",
        shape=(n_windows,),
        chunks=(1,),
        dtype="datetime64[ns]",
    )

    z_t_end = root.create_array(
        "t_end",
        shape=(n_windows,),
        chunks=(1,),
        dtype="datetime64[ns]",
    )

    
    # --------------------------------------------------
    # Loop over trajectories
    # --------------------------------------------------

    for i, start in enumerate(
        range(
            0,
            len(data) - trajectory_length + 1,
            stride,
        )
    ):

        print(
            f"Trajectory {i + 1}/{n_windows}"
        )

        # ----------------------------------------------
        # SSH trajectory
        # ----------------------------------------------

        trajectory = data.isel(
            time=slice(
                start,
                start + trajectory_length,
            )
        )

        x = trajectory.values.astype(
            np.float32
        )

        # Normalize using TRAIN statistics
        x = (
            x - mean_state
        ) / std_state

        # Land → zero
        x = np.nan_to_num(
            x,
            nan=0.0,
        )

        

        times = trajectory["time"].values

        times_hours = (
            (times - times[0])
            / np.timedelta64(1, "h")
        ).astype(np.float32)

        

        # ----------------------------------------------
        # Calculate spline
        # ----------------------------------------------

        coeffs, starts, ends = quintic_spline_batch(
            x[None, ...],
            time=times_hours[None, :],
            batch_size=1,
            num_point=3,
            device="cpu",
        )

        # ----------------------------------------------
        # Save spline
        # ----------------------------------------------

        duration_hours = times_hours[-1] - times_hours[0]
        
        if duration_hours <= 0:
            raise ValueError(f"Invalid trajectory duration: {duration_hours}")

        z_coeffs[i] = coeffs
        z_starts[i] = starts / (duration_hours)
        z_ends[i] = ends / (duration_hours)

        
        # ----------------------------------------------
        # Absolute trajectory times
        # ----------------------------------------------

        z_t_start[i] = (
            data.time.isel(
                time=start
            ).values
        )

        z_t_end[i] = (
            data.time.isel(
                time=start + trajectory_length - 1
            ).values
        )

    

    print(
        f"Saved splines to {output_path}"
    )




# ---------------------------------------------------------
# Calculate splines
# ---------------------------------------------------------
start_time = time.time()

calculate_splines(
    ssh_train,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"./{OUTPUT_DIR}/splines_train_state.zarr",
)

calculate_splines(
    ssh_eval,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"./{OUTPUT_DIR}/splines_eval_state.zarr",
)

calculate_splines(
    ssh_test,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"./{OUTPUT_DIR}/splines_test_state.zarr",
)

end_time = time.time()

print(f"Time taken to calculate splines: {(end_time - start_time)/60} minutes")
