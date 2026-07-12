import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class NordicSeaCurrentDataset(Dataset):
    """Simple PyTorch dataset for Nordic Sea fields stored in xarray datasets (NetCDF (*.nc)).

    This version can return either:
    - spatial-only tensors shaped [C, H, W] for standard FNO-style models, or
    - temporal tensors shaped [C, T, H, W] for FNOtD-style models.
    """

    def __init__(
        self,
        ds,
        input_vars=("ssh", "u", "v", "wind_u", "wind_v", "slp"),
        target_vars=("ssh", "u", "v"),
        bathymetry_var="bathymetry",
        horizon=1,
        time_dim=None,
        temporal_window=1,
        return_temporal=False,
        mean=None,
        std=None,
    ):
        if isinstance(ds, (str, bytes)):
            ds = xr.open_dataset(ds)
        if not isinstance(ds, xr.Dataset):
            raise TypeError("ds must be an xarray.Dataset or a path to a NetCDF file")

        self.ds = ds
        self.input_vars = list(input_vars)
        self.target_vars = list(target_vars)
        self.bathymetry_var = bathymetry_var
        self.horizon = int(horizon)
        self.temporal_window = int(temporal_window)
        self.return_temporal = bool(return_temporal)
        self.time_dim = self._find_time_dim(ds, time_dim)
        self.mean = self._prepare_stats(mean, len(self.input_vars))
        self.std = self._prepare_stats(std, len(self.input_vars))

        self._validate_vars()

        bathy_da = ds[bathymetry_var]
        if bathy_da.ndim != 2:
            raise ValueError("bathymetry must be a 2D field with dimensions [y, x]")
        self.bathy = torch.from_numpy(np.asarray(bathy_da.values, dtype=np.float32)).unsqueeze(0)

        self.n_samples = ds.sizes[self.time_dim] - self.horizon - self.temporal_window + 1
        if self.n_samples < 1:
            raise ValueError("Not enough time steps for the requested horizon/window")

    def _find_time_dim(self, ds, time_dim):
        if time_dim is not None:
            return time_dim
        for candidate in ("time", "time_counter", "t"):
            if candidate in ds.dims:
                return candidate
        raise ValueError("Could not infer the time dimension in the dataset")

    def _validate_vars(self):
        for var in self.input_vars + self.target_vars + [self.bathymetry_var]:
            if var not in self.ds:
                raise KeyError(f"Variable {var!r} not found in dataset")

    def _prepare_stats(self, values, n_channels):
        if values is None:
            return None
        stats = torch.as_tensor(np.asarray(values, dtype=np.float32))
        if stats.ndim == 0:
            stats = stats.view(1)
        if stats.ndim != 1:
            raise ValueError("mean/std must be a 1D array")
        if stats.numel() < n_channels:
            raise ValueError(f"Expected at least {n_channels} mean/std values, got {stats.numel()}")
        return stats[:n_channels]

    def _to_tensor(self, da):
        return torch.from_numpy(np.asarray(da.values, dtype=np.float32))

    def _normalize(self, tensor, mean, std, n_channels):
        if mean is None or std is None:
            return tensor
        if tensor.dim() == 3:
            mean_view = mean.view(n_channels, 1, 1)
            std_view = std.view(n_channels, 1, 1)
        elif tensor.dim() == 4:
            mean_view = mean.view(n_channels, 1, 1, 1)
            std_view = std.view(n_channels, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimensionality: {tensor.dim()}")
        std_view = torch.where(std_view != 0, std_view, torch.ones_like(std_view))
        return (tensor - mean_view) / std_view

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idx = int(idx)

        if self.return_temporal:
            window = []
            for t in range(self.temporal_window):
                x_fields = []
                for var in self.input_vars:
                    da = self.ds[var].isel({self.time_dim: idx + t})
                    x_fields.append(self._to_tensor(da))
                x_t = torch.stack(x_fields, dim=0)
                window.append(x_t)
            x_t = torch.stack(window, dim=1)  # [C, T, H, W]
        else:
            x_fields = []
            for var in self.input_vars:
                da = self.ds[var].isel({self.time_dim: idx})
                x_fields.append(self._to_tensor(da))
            x_t = torch.stack(x_fields, dim=0)  # [C, H, W]

        target_fields = []
        for var in self.target_vars:
            da = self.ds[var].isel({self.time_dim: idx + self.horizon + self.temporal_window - 1})
            target_fields.append(self._to_tensor(da))

        y_t = torch.stack(target_fields, dim=0)

        if self.return_temporal:
            x_t = self._normalize(x_t, self.mean, self.std, len(self.input_vars))
            y_t = self._normalize(y_t, self.mean[: len(self.target_vars)], self.std[: len(self.target_vars)], len(self.target_vars))
            input_t = torch.cat([x_t, self.bathy.unsqueeze(0).expand(-1, self.temporal_window, -1, -1)], dim=0)
            return input_t, y_t

        x_t = self._normalize(x_t, self.mean, self.std, len(self.input_vars))
        y_t = self._normalize(y_t, self.mean[: len(self.target_vars)], self.std[: len(self.target_vars)], len(self.target_vars))
        input_t = torch.cat([x_t, self.bathy], dim=0)
        return input_t, y_t