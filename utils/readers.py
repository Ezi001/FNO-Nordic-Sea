"""Dataset loading helpers for supported CFO datasets."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def load_nordic_sea(
    processed_dir: str | Path = "processed",
    *,
    chunks: dict | None = None,
):
    processed_dir = Path(processed_dir)
    chunks = chunks or {"time": 100}

    ocean = xr.open_dataarray(processed_dir / "state.nc", chunks=chunks)
    forcing = xr.open_dataarray(processed_dir / "forcing.nc", chunks=chunks)

    return ocean, forcing


def _time_dim(array: xr.DataArray) -> str:
    for dim in ("time", "time_counter"):
        if dim in array.dims:
            return dim
    raise ValueError(f"Could not find a time dimension in {array.dims}")


def load_nordic_sea_splits(ocean: xr.DataArray, forcing: xr.DataArray):
    ocean_time_dim = _time_dim(ocean)
    forcing_time_dim = _time_dim(forcing)

    n = ocean.sizes[ocean_time_dim]
    if forcing.sizes[forcing_time_dim] != n:
        raise ValueError(
            f"Ocean and forcing have different time lengths: "
            f"{n} vs {forcing.sizes[forcing_time_dim]}"
        )

    train_end = int(0.7 * n)
    eval_end = int(0.85 * n)

    return {
        "train": (
            ocean.isel({ocean_time_dim: slice(0, train_end)}),
            forcing.isel({forcing_time_dim: slice(0, train_end)}),
        ),
        "eval": (
            ocean.isel({ocean_time_dim: slice(train_end, eval_end)}),
            forcing.isel({forcing_time_dim: slice(train_end, eval_end)}),
        ),
        "test": (
            ocean.isel({ocean_time_dim: slice(eval_end, None)}),
            forcing.isel({forcing_time_dim: slice(eval_end, None)}),
        ),
    }


loaders = {
    "nordic": load_nordic_sea,
}