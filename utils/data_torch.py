"""PyTorch dataset for precomputed Nordic Seas spline trajectories."""

from __future__ import annotations

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class ZarrSplineDataset(Dataset):
    def __init__(self, zarr_path, forcing_path=None):
        self.root = zarr.open(zarr_path, mode="r")

        self.spline_coef = self._get_array(
            "spline_coef", "coefficients"
        )
        self.spline_start = self._get_array(
            "spline_start", "starts"
        )
        self.spline_end = self._get_array(
            "spline_end", "t_end"
        )

        self.n_trajectories = self.spline_coef.shape[0]
        self.n_intervals = self.spline_coef.shape[1]

        self.forcing = None

        if forcing_path is not None:
            self.forcing = np.load(
                forcing_path,
                mmap_mode="r",
            )

            if self.forcing.ndim != 4:
                raise ValueError(
                    "Forcing must have shape "
                    "(trajectories, height, width, channels), "
                    f"got {self.forcing.shape}."
                )

            if self.forcing.shape[0] != self.n_trajectories:
                raise ValueError(
                    "Forcing and spline stores must contain the "
                    "same number of trajectories: "
                    f"{self.forcing.shape[0]} vs "
                    f"{self.n_trajectories}."
                )

    def _get_array(self, *names):
        for name in names:
            if name in self.root:
                return self.root[name]

        raise KeyError(
            f"None of {names!r} found in spline store. "
            f"Available arrays: {list(self.root.array_keys())}."
        )

    def __len__(self):
        return self.n_trajectories * self.n_intervals

    def __getitem__(self, idx):
        trajectory_idx = idx // self.n_intervals
        interval_idx = idx % self.n_intervals

        coef = self.spline_coef[
            trajectory_idx,
            interval_idx,
        ]

        t_start = self.spline_start[
            trajectory_idx,
            interval_idx,
        ]

        t_end = self.spline_end[
            trajectory_idx,
            interval_idx,
        ]

        item = {
            "spline_coef": torch.from_numpy(
                np.asarray(coef)
            ).float(),

            "t_start": torch.tensor(
                t_start / 47,
                dtype=torch.float32,
            ),

            "t_end": torch.tensor(
                t_end / 47,
                dtype=torch.float32,
            ),
        }

        if self.forcing is not None:
            item["condition"] = torch.from_numpy(
                np.asarray(self.forcing[trajectory_idx])
            ).float()

        return item
