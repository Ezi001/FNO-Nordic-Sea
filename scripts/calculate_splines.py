"""
Precompute quintic-spline coefficients for the PDEBench 2D shallow-water dataset
and store them in the Zarr layout expected by ZarrSplineDataset.

For every trajectory, each pixel's time series (one value per snapshot) is
interpolated with a quintic spline whose knots sit at t = 0, 1, ..., N-1.
Every pair of consecutive snapshots therefore forms one spline interval.

Output Zarr store
-----------------
spline_coef  : (n_trajectories, n_intervals, 6, H, W), float32
spline_start : (n_trajectories, n_intervals),          float32
spline_end   : (n_trajectories, n_intervals),          float32

Coefficient convention:
    u(t) = sum_k spline_coef[..., k, :, :] * (t - t_start) ** k,
    for t in [t_start, t_end], with t_end - t_start = 1.
So k = 0 is the value at the start of the interval and k = 5 is the quintic term.

Train/test split indices are saved in the root attributes:
    root.attrs["train_indices"], root.attrs["test_indices"]
and reproduce the split of load_shallow_water_pdebench (same seed and shuffle).

Usage
-----
python compute_splines_shallow_water_quintic.py \
    --h5-path data/shallow_water/2D_rdb_NA_NA.h5 \
    --zarr-path processed/shallow_water/splines_quintic.zarr
"""

import argparse
import logging

import h5py
import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.splines_torch import quintic_spline_batch

logger = logging.getLogger(__name__)

N_COEFFICIENTS = 6  # quintic spline


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
    Fit a quintic spline through the time axis of (T, H, W) snapshots.

    Returns
    -------
    coef : (T - 1, 6, H, W)
    starts : (T - 1,)
    ends : (T - 1,)
    """
    n_snapshots = snapshots.shape[0]
    if n_snapshots < 2:
        raise ValueError("Need at least 2 snapshots to build a spline.")

    t = np.arange(n_snapshots, dtype=np.float32)

    # quintic_spline_batch expects (B, T, H, W) and returns
    # coeffs shaped like (n_segments, 6, H, W) for one trajectory.
    coeffs, starts, ends = quintic_spline_batch(
        snapshots[None, ...].astype(np.float32),
        time=t[None, :],
        batch_size=1,
        num_point=3,
        device="cpu",
    )

    # flatten the leading batch dimension
    coef = coeffs.reshape(n_snapshots - 1, N_COEFFICIENTS, *snapshots.shape[1:])
    return coef.astype(np.float32), starts.astype(np.float32), ends.astype(np.float32)


def make_array(root, name, shape, chunks, dtype):
    """Create a Zarr array on both zarr v2 and v3."""
    if hasattr(root, "create_array"):
        return root.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", default="data/shallow_water/2D_rdb_NA_NA.h5")
    parser.add_argument("--zarr-path", default="processed/shallow_water/splines_quintic.zarr")
    parser.add_argument("--training-ratio", type=float, default=9 / 10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--start-ind", type=int, default=1)
    parser.add_argument("--end-ind", type=int, default=101)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with h5py.File(args.h5_path, "r") as h5_file:
        keys = list(h5_file.keys())
        n_trajectories = len(keys)

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

        root = zarr.open_group(args.zarr_path, mode="w")

        coef_array = make_array(
            root,
            "spline_coef",
            shape=(n_trajectories, n_intervals, N_COEFFICIENTS, height, width),
            chunks=(1, 1, N_COEFFICIENTS, height, width),
            dtype="float32",
        )

        # Knot times are identical for all trajectories.
        # Keep the same convention as the rest of the project: local times
        # run 0, 1, ..., n_intervals-1 at the interval starts.
        t_start = np.broadcast_to(
            np.arange(n_intervals, dtype=np.float32),
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

        for i, key in enumerate(keys):
            snapshots = first if i == 0 else load_trajectory(
                h5_file, key, args.start_ind, args.end_ind
            )

            coef, starts, ends = compute_spline_coefficients(snapshots)

            if i == 0:
                # At the end of each interval, the spline should reproduce the next snapshot.
                reconstructed = coef.sum(axis=1)  # (n_intervals, H, W)
                error = np.abs(reconstructed - snapshots[1:]).max()
                logger.info("Max knot reconstruction error (traj 0): %.3e", error)

            coef_array[i] = coef

            if (i + 1) % 50 == 0 or i + 1 == n_trajectories:
                logger.info("  processed %d / %d", i + 1, n_trajectories)

    np.random.seed(args.random_seed)
    indices = np.arange(n_trajectories)
    np.random.shuffle(indices)
    train_size = int(args.training_ratio * n_trajectories)

    root.attrs["train_indices"] = indices[:train_size].tolist()
    root.attrs["test_indices"] = indices[train_size:].tolist()
    root.attrs["bc_type"] = "quintic-project-spline"
    root.attrs["coefficient_order"] = "ascending powers of (t - t_start); quintic"

    logger.info(
        "Wrote %s (train: %d, test: %d trajectories)",
        args.zarr_path,
        train_size,
        n_trajectories - train_size,
    )


if __name__ == "__main__":
    main()
