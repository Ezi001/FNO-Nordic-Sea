"""
This file is supposed to calculate the splines for the trajectories. 
I am splitting the data into trajectories of
2 days with a stride of 2 days and 1 hour.

Workflow:
1. Load the HDF5 trajectories and split them into training and test sets.

2. Normalize using statistics computed from the training set only.

3. Replace invalid values with zero after normalization, while preserving a consistent land/mask convention.

4. Calculate spline coefficients separately for each trajectory.

5. Save the coefficients, interval boundaries, and trajectory time metadata in Zarr, with the same key names your CFO ZarrSplineDataset expects.
"""
# Import necessary libraries
import time
from pathlib import Path

import h5py
import numpy as np
import zarr

from splines_torch import quintic_spline_batch

__author__ = "Esther Zijerveld"
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/shallow_water/2D_rdb_NA_NA.h5"
OUTPUT_DIR = PROJECT_ROOT / "processed/shallow_water/"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_RATIO = 0.9
RANDOM_SEED = 42

# Match your existing dataloader's time selection.
START_IND = 1
END_IND = 101


# ---------------------------------------------------------
# Load SWE trajectories
# ---------------------------------------------------------

def load_shallow_water_pdebench(
    file_path,
    training_ratio=0.9,
    random_seed=42,
    start_ind=1,
    end_ind=101,
):
    """Load and split PDEBench trajectories."""

    dataset = []

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data = np.asarray(f[key]["data"], dtype=np.float32)
            dataset.append(data)

    dataset = np.stack(dataset, axis=0)

    print("Original dataset shape:", dataset.shape)

    # Select time indices consistently for all trajectories.
    dataset = dataset[:, start_ind:end_ind]

    print("After time selection:", dataset.shape)

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(dataset))

    train_size = int(training_ratio * len(dataset))

    train_data = dataset[indices[:train_size]]
    test_data = dataset[indices[train_size:]]

    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)

    return train_data, test_data


# ---------------------------------------------------------
# Calculate normalization statistics from training data
# ---------------------------------------------------------

def compute_stats(data):
    """
    Calculate global mean and standard deviation over
    finite training values only.
    """

    values = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(values)

    if not np.any(valid):
        raise ValueError("Training data contains no finite values.")

    mean = np.mean(values[valid], dtype=np.float64)
    std = np.std(values[valid], dtype=np.float64)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid training standard deviation: {std}")

    return np.float32(mean), np.float32(std)


# ---------------------------------------------------------
# Calculate and save spline coefficients
# ---------------------------------------------------------

def calculate_splines(
    data,
    output_path,
    mean,
    std,
):
    """
    Parameters
    ----------
    data : ndarray
        Shape (n_trajectories, time, y, x, channels)
        or (n_trajectories, time, y, x).

    output_path : str or Path
        Output Zarr directory.

    mean, std : float
        Training-set normalization statistics.
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim not in (4, 5):
        raise ValueError(
            "Expected shape (N,T,Y,X) or (N,T,Y,X,C), "
            f"got {data.shape}"
        )

    n_trajectories = data.shape[0]
    trajectory_length = data.shape[1]

    # This implementation is for a single SWE state channel.
    if data.ndim == 5:
        if data.shape[-1] != 1:
            raise ValueError(
                "Expected one state channel, "
                f"got {data.shape[-1]} channels."
            )
        data = data[..., 0]

    n_time, ny, nx = data.shape[1:]

    # Time is measured in sample indices because the physical
    # time spacing has not been specified.
    time_points = np.arange(n_time, dtype=np.float32)

    # Determine coefficient dimensions from the first trajectory.
    x = (data[0] - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    coeffs, starts, ends = quintic_spline_batch(
        x[None, ...],
        time=time_points[None, :],
        batch_size=1,
        num_point=3,
        device="cpu",
    )

    coeffs = np.asarray(coeffs, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.float32).reshape(-1)
    ends = np.asarray(ends, dtype=np.float32).reshape(-1)

    if coeffs.ndim != 4:
        raise ValueError(
            "Expected spline coefficients with shape "
            f"(segments, coefficients, y, x), got {coeffs.shape}"
        )

    n_segments, n_coefficients, out_y, out_x = coeffs.shape

    if (out_y, out_x) != (ny, nx):
        raise ValueError("Spline output spatial dimensions changed.")

    print("Trajectory length:", trajectory_length)
    print("Spline coefficient shape:", coeffs.shape)
    print("Number of segments:", n_segments)
    print("Number of coefficients:", n_coefficients)

    # -----------------------------------------------------
    # Create output Zarr store
    # -----------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output_path), mode="w")

    z_coeffs = root.create_array(
        "spline_coef",
        shape=(
            n_trajectories,
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
        dtype="float32",
    )

    z_starts = root.create_array(
        "spline_start",
        shape=(n_trajectories, n_segments),
        chunks=(1, n_segments),
        dtype="float32",
    )

    z_ends = root.create_array(
        "spline_end",
        shape=(n_trajectories, n_segments),
        chunks=(1, n_segments),
        dtype="float32",
    )

    # Sample-index metadata rather than datetime metadata.
    z_t_start = root.create_array(
        "t_start",
        shape=(n_trajectories,),
        chunks=(1,),
        dtype="int32",
    )

    z_t_end = root.create_array(
        "t_end",
        shape=(n_trajectories,),
        chunks=(1,),
        dtype="int32",
    )

    # -----------------------------------------------------
    # Process every trajectory
    # -----------------------------------------------------

    for i in range(n_trajectories):
        print(f"Trajectory {i + 1}/{n_trajectories}")

        x = (data[i] - mean) / std

        # Convert invalid/masked values to zero after normalization.
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        coeffs, starts, ends = quintic_spline_batch(
            x[None, ...],
            time=time_points[None, :],
            batch_size=1,
            num_point=3,
            device="cpu",
        )

        coeffs = np.asarray(coeffs, dtype=np.float32)
        starts = np.asarray(starts, dtype=np.float32).reshape(-1)
        ends = np.asarray(ends, dtype=np.float32).reshape(-1)

        if coeffs.shape != (
            n_segments,
            n_coefficients,
            ny,
            nx,
        ):
            raise ValueError(
                f"Unexpected coefficient shape for trajectory {i}: "
                f"{coeffs.shape}"
            )

        duration = time_points[-1] - time_points[0]
        if duration <= 0:
            raise ValueError(f"Invalid trajectory duration: {duration}")

        z_coeffs[i] = coeffs
        z_starts[i] = starts / duration
        z_ends[i] = ends / duration

        z_t_start[i] = 0
        z_t_end[i] = n_time - 1

    # Basic metadata for later inspection.
    root.attrs["mean"] = float(mean)
    root.attrs["std"] = float(std)
    root.attrs["trajectory_length"] = int(trajectory_length)
    root.attrs["n_trajectories"] = int(n_trajectories)
    root.attrs["time_unit"] = "sample_index"
    root.attrs["spline_time_unit"] = "normalized_0_1"

    print(f"Saved splines to {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    train_data, test_data = load_shallow_water_pdebench(
        file_path=DATA_PATH,
        training_ratio=TRAINING_RATIO,
        random_seed=RANDOM_SEED,
        start_ind=START_IND,
        end_ind=END_IND,
    )

    mean_state, std_state = compute_stats(train_data)

    print("Training mean:", mean_state)
    print("Training std:", std_state)

    calculate_splines(
        train_data,
        output_path=OUTPUT_DIR / "splines_train_state.zarr",
        mean=mean_state,
        std=std_state,
    )

    test_data = (test_data - mean_state)/std_state
    # Save test trajectories in their original, raw form.
    test_path = OUTPUT_DIR / "ssh_test_traj.npy"
    np.save(test_path, test_data)

    print("Raw test trajectories shape:", test_data.shape)
    print(f"Saved raw test trajectories to {test_path}")

    print(
        f"Total time: {(time.time() - start_time) / 60:.2f} minutes"
    )
