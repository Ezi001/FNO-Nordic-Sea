"""Dataset loading helpers for supported CFO datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

NordicSplit = tuple[np.ndarray, np.ndarray]
NordicSplits = dict[str, NordicSplit | dict[str, dict[str, np.ndarray]]]


def _open_processed_array(path: Path, chunks: dict[str, int] | None) -> xr.DataArray:
    try:
        return xr.open_dataarray(path, chunks=chunks)
    except ValueError:
        dataset = xr.open_dataset(path, chunks=chunks)
        data_vars = list(dataset.data_vars)
        if len(data_vars) != 1:
            raise ValueError(
                f"Expected one data variable in {path}, found {data_vars}."
            )
        return dataset[data_vars[0]]


def _require_dims(array: xr.DataArray, name: str) -> xr.DataArray:
    expected = ("time", "y", "x", "channel")
    missing = [dim for dim in expected if dim not in array.dims]
    if missing:
        raise ValueError(
            f"{name} must contain dims {expected}; got {array.dims}."
        )
    return array.transpose(*expected)


def _make_trajectory_windows(
    array: np.ndarray,
    trajectory_window: int,
    stride: int,
) -> np.ndarray:
    if trajectory_window < 2:
        raise ValueError("`trajectory_window` must be at least 2.")
    if stride < 1:
        raise ValueError("`stride` must be at least 1.")
    if array.shape[0] < trajectory_window:
        raise ValueError(
            f"Split has {array.shape[0]} time steps, which is shorter than "
            f"`trajectory_window={trajectory_window}`."
        )

    starts = range(0, array.shape[0] - trajectory_window + 1, stride)
    return np.stack(
        [array[start : start + trajectory_window] for start in starts],
        axis=0,
    )


def _normalize_with_train_stats(
    train: np.ndarray,
    eval_data: np.ndarray,
    test: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = train.mean(axis=(0, 1, 2), keepdims=True)
    std = train.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, eps)

    stats = {
        "mean": mean.reshape(-1),
        "std": std.reshape(-1),
    }
    return (
        (train - mean) / std,
        (eval_data - mean) / std,
        (test - mean) / std,
        stats,
    )


def load_nordic_seas(
    processed_dir: str | Path = "processed",
    *,
    trajectory_window: int,
    stride: int = 1,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.15,
    normalize: bool = False,
    chunks: dict[str, int] | None = None,
) -> NordicSplits:
    """Load preprocessed Nordic Seas data and return trajectory-window splits.

    Expected processed files:
        processed/state.nc   with dims (time, y, x, channel)
        processed/forcing.nc with dims (time, y, x, channel)

    Returns:
        {
            "train": (state_windows, forcing_windows),
            "eval":  (state_windows, forcing_windows),
            "test":  (state_windows, forcing_windows),
        }

    Each window array has shape:
        (num_windows, trajectory_window, y, x, channel)
    """
    processed_dir = Path(processed_dir)
    chunks = chunks or {"time": 100}

    state_da = _open_processed_array(processed_dir / "state.nc", chunks)
    forcing_da = _open_processed_array(processed_dir / "forcing.nc", chunks)

    state_da = _require_dims(state_da, "state")
    forcing_da = _require_dims(forcing_da, "forcing")

    if state_da.sizes["time"] != forcing_da.sizes["time"]:
        raise ValueError(
            "State and forcing have different time lengths: "
            f"{state_da.sizes['time']} vs {forcing_da.sizes['time']}."
        )
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("`train_ratio` must be between 0 and 1.")
    if not (0.0 < eval_ratio < 1.0):
        raise ValueError("`eval_ratio` must be between 0 and 1.")
    if train_ratio + eval_ratio >= 1.0:
        raise ValueError("`train_ratio + eval_ratio` must be less than 1.")

    state = state_da.to_numpy().astype(np.float32)
    forcing = forcing_da.to_numpy().astype(np.float32)

    n_time = state.shape[0]
    train_end = int(train_ratio * n_time)
    eval_end = int((train_ratio + eval_ratio) * n_time)

    state_train, state_eval, state_test = (
        state[:train_end],
        state[train_end:eval_end],
        state[eval_end:],
    )
    forcing_train, forcing_eval, forcing_test = (
        forcing[:train_end],
        forcing[train_end:eval_end],
        forcing[eval_end:],
    )

    stats = None
    if normalize:
        state_train, state_eval, state_test, state_stats = _normalize_with_train_stats(
            state_train, state_eval, state_test
        )
        (
            forcing_train,
            forcing_eval,
            forcing_test,
            forcing_stats,
        ) = _normalize_with_train_stats(forcing_train, forcing_eval, forcing_test)
        stats = {
            "state": state_stats,
            "forcing": forcing_stats,
        }

    splits = {
        "train": (
            _make_trajectory_windows(state_train, trajectory_window, stride),
            _make_trajectory_windows(forcing_train, trajectory_window, stride),
        ),
        "eval": (
            _make_trajectory_windows(state_eval, trajectory_window, stride),
            _make_trajectory_windows(forcing_eval, trajectory_window, stride),
        ),
        "test": (
            _make_trajectory_windows(state_test, trajectory_window, stride),
            _make_trajectory_windows(forcing_test, trajectory_window, stride),
        ),
    }
    if stats is not None:
        splits["normalization"] = stats
    return splits


def load_dataset_splits(
    dataset: str,
    *,
    processed_dir: str | Path = "processed",
    trajectory_window: int,
    stride: int = 1,
    normalize: bool = False,
) -> NordicSplits:
    """Load a dataset and return a standardized split dictionary."""
    key = dataset.lower()
    loaders: dict[str, Callable[..., NordicSplits]] = {
        "nordic": load_nordic_seas,
        "nordic_seas": load_nordic_seas,
    }
    if key not in loaders:
        raise ValueError(f"Unsupported dataset='{dataset}'. Supported: {sorted(loaders)}")

    return loaders[key](
        processed_dir=processed_dir,
        trajectory_window=trajectory_window,
        stride=stride,
        normalize=normalize,
    )


__all__ = [
    "load_nordic_seas",
    "load_dataset_splits",
]
