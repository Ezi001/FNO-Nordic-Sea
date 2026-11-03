"""
Precompute piecewise-linear spline coefficients for the PDEBench 2D
shallow-water dataset with CFO's `linear_spline`, and store them in the Zarr
layout expected by ZarrSplineDataset.

Every pair of consecutive snapshots forms one spline interval.

Output Zarr store
-----------------
spline_coef  : (n_trajectories, n_intervals, 2, H, W), float32
spline_start : (n_trajectories, n_intervals),          float32
spline_end   : (n_trajectories, n_intervals),          float32

Coefficient convention (CFO):
    u(tau) = a0 + a1 * tau,   tau = (t - t_start) / (t_end - t_start) in [0, 1]
with
    a0 = spline_coef[..., 0, :, :] = snapshot at the start of the interval
    a1 = spline_coef[..., 1, :, :] = snapshot at the end minus snapshot at the start

Knots sit at t = 0, 1, ..., N-1, so t_end - t_start = 1 and tau = t - t_start.

Train/test split indices are saved in the root attributes:
    root.attrs["train_indices"], root.attrs["test_indices"]
and reproduce the split of load_shallow_water_pdebench (same seed and shuffle).

Usage
-----
python compute_splines_shallow_water_linear.py \
    --h5-path data/shallow_water/2D_rdb_NA_NA.h5 \
    --zarr-path data/shallow_water/splines_linear.zarr
"""

import argparse
import logging

import h5py
import numpy as np
import zarr

# Adjust this import to wherever the CFO spline file lives, e.g. save it next
# to this script as spline_utils.py (it needs torch installed because it
# imports torch at module level).
from utils.splines_torch import linear_spline

logger = logging.getLogger(__name__)

N_COEFFICIENTS = 2  # a0, a1


def load_trajectory(h5_file, key, start_ind, end_ind):
    """Read snapshots [start_ind:end_ind] of one trajectory as (T, H, W)."""
    data = np.asarray(h5_file[key]["data"][start_ind:end_ind], dtype=np.float64)

    # PDEBench shallow water stores (T, H, W, 1): a single channel (water depth)
    if data.ndim == 4:
        if data.shape[-1] != 1:
            raise ValueError(
                f"Expected a single channel, got data of shape {data.shape}."
            )
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"Expected (T, H, W) after squeezing, got {data.shape}.")
    return data


def compute_spline_coefficients(snapshots):
    """
    Compute piecewise-linear coefficients for (T, H, W) snapshots with CFO's
    `linear_spline`.

    Returns
    -------
    coef : (T - 1, 2, H, W)
    start_ind : (T - 1,) integer start index of each interval
    """
    # linear_spline expects a batch dimension: (B, T, ...) with B = 1 here.
    coef, start_ind = linear_spline(snapshots[None])
    # coef: (B * (T - 1), 2, H, W) -> already ordered by interval for B = 1
    return coef, start_ind


def make_array(root, name, shape, chunks, dtype):
    """Create a Zarr array on both zarr v2 and v3."""
    if hasattr(root, "create_array"):
        return root.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", default="data/shallow_water/2D_rdb_NA_NA.h5")
    parser.add_argument("--zarr-path", default="processed/shallow_water/splines_linear.zarr")
    parser.add_argument("--training-ratio", type=float, default=9 / 10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--start-ind", type=int, default=1)
    parser.add_argument("--end-ind", type=int, default=101)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with h5py.File(args.h5_path, "r") as h5_file:
        keys = list(h5_file.keys())
        n_trajectories = len(keys)

        # Inspect the first trajectory to get the shapes.
        first = load_trajectory(h5_file, keys[0], args.start_ind, args.end_ind)
        n_snapshots, height, width = first.shape
        n_intervals = n_snapshots - 1
        logger.info(
            "Found %d trajectories with %d snapshots on a %d x %d grid "
            "-> %d spline intervals each.",
            n_trajectories,
            n_snapshots,
            height,
            width,
            n_intervals,
        )

        # --------------------------------------------------
        # Create the Zarr store
        # --------------------------------------------------
        root = zarr.open_group(args.zarr_path, mode="w")

        # One sample = one [trajectory, interval] slice, so chunk accordingly.
        coef_array = make_array(
            root,
            "spline_coef",
            shape=(n_trajectories, n_intervals, N_COEFFICIENTS, height, width),
            chunks=(1, 1, N_COEFFICIENTS, height, width),
            dtype="float32",
        )

        # Knot times are identical for all trajectories.
        expected_start = np.arange(n_intervals)
        t_start = np.broadcast_to(
            expected_start.astype(np.float32),
            (n_trajectories, n_intervals),
        )
        t_end = t_start + 1.0

        for name, values in (("spline_start", t_start), ("spline_end", t_end)):
            array = make_array(
                root,
                name,
                shape=(n_trajectories, n_intervals),
                chunks=(n_trajectories, n_intervals),
                dtype="float32",
            )
            array[:] = values

        # --------------------------------------------------
        # Compute splines trajectory by trajectory (low memory)
        # --------------------------------------------------
        for i, key in enumerate(keys):
            snapshots = first if i == 0 else load_trajectory(
                h5_file, key, args.start_ind, args.end_ind
            )
            coef, start_ind = compute_spline_coefficients(snapshots)

            if i == 0:
                # linear_spline's start indices must match the knot times
                # written to spline_start above.
                if not np.array_equal(start_ind, expected_start):
                    raise RuntimeError(
                        "linear_spline start indices do not match the "
                        "knot times stored in spline_start."
                    )

                # At tau = 1 the spline must reproduce the next snapshot:
                # u(1) = a0 + a1.
                reconstructed = coef.sum(axis=1)  # (n_intervals, H, W)
                error = np.abs(reconstructed - snapshots[1:]).max()
                logger.info("Max knot reconstruction error (traj 0): %.3e", error)

            coef_array[i] = coef.astype(np.float32)

            if (i + 1) % 50 == 0 or i + 1 == n_trajectories:
                logger.info("  processed %d / %d", i + 1, n_trajectories)

    # --------------------------------------------------
    # Train/test split (same logic as load_shallow_water_pdebench)
    # --------------------------------------------------
    np.random.seed(args.random_seed)
    indices = np.arange(n_trajectories)
    np.random.shuffle(indices)
    train_size = int(args.training_ratio * n_trajectories)

    root.attrs["train_indices"] = indices[:train_size].tolist()
    root.attrs["test_indices"] = indices[train_size:].tolist()
    root.attrs["spline_type"] = "piecewise linear (CFO linear_spline)"
    root.attrs["coefficient_order"] = "a0 + a1 * tau, tau in [0, 1]"

    logger.info(
        "Wrote %s (train: %d, test: %d trajectories)",
        args.zarr_path,
        train_size,
        n_trajectories - train_size,
    )


if __name__ == "__main__":
    main()
