# Name: Esther Maud Zijerveld
# Date: 13.07.2026


# Necessary imports
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from scipy.interpolate import griddata
from torch.utils.data import Dataset


class NordicSeaDataset(Dataset):
    """Simple PyTorch dataset for Nordic Sea fields stored in xarray datasets (NetCDF (*.nc) files).
    Dataset provides an interface for accessing data.
    Inputs: Ocean variables: u,v,ssh. Forcing: u, v, slp. Static: bathymetry.
    This version can return either:
    - spatial-only tensors shaped [C, H, W] for standard FNO-style models, or
    - temporal tensors shaped [C, T, H, W] for FNOtD-style models.
    Parameters:
        root_dir
        input_vars
        target_vars
        input_lag
        temporal_window
        return_temporal
        transform
        start_time
        end_time
    Output:
        x, y
    """

    def __init__(self, 
                 root_dir,
                 input_vars=("ssh", "u", "v", "wind_u", "wind_v", "slp"), 
                 target_vars=("ssh", "u", "v"), 
                temporal_window=1,
                return_temporal=False,
                transform=None,
                start_time=None,
                end_time=None,
                coarse_step=1,
                lat_trim=0,
                lon_trim=0,
                coarse_slice=None,
        ):
        """
        Initialises dataset with necessary attributes like file paths
        or data processing steps.
        Arguements:
        ds: xarray.Dataset or path to NetCDF file containing the data.

        """
        # if isinstance(ds, (str, bytes)):
        #     ds = xr.open_dataset(ds)
        # if not isinstance(ds, xr.Dataset):
        #     raise TypeError("ds must be an xarray.Dataset or a path to a NetCDF file")

        self.root = Path(root_dir).expanduser()
        if not self.root.is_absolute():
            self.root = (Path.cwd() / self.root).resolve()
        if not self.root.exists():
            fallback_root = (Path.cwd() / "data").resolve()
            self.root = fallback_root

        self.transform = transform
        self.input_vars = list(input_vars)
        self.target_vars = list(target_vars)
        self.temporal_window = int(temporal_window)
        self.return_temporal = bool(return_temporal)
        self.start_time = start_time
        self.end_time = end_time
        self.coarse_step = int(coarse_step)
        self.lat_trim = int(lat_trim)
        self.lon_trim = int(lon_trim)
        self.coarse_slice = coarse_slice

        self.ssh_ds = xr.open_mfdataset(
            str(self.root / "*ssh.nc"),
            combine="by_coords",
            chunks={"time_counter": 10},
            engine="netcdf4",
            lock=False,
        )["ssh"].astype("float32")

        self.u_ds = xr.open_mfdataset(
            str(self.root / "*ubar.nc"),
            combine="by_coords",
            chunks={"time_counter": 10},
            engine="netcdf4",
            lock=False,
        )["ubar"].astype("float32")

        self.v_ds = xr.open_mfdataset(
            str(self.root / "*vbar.nc"),
            combine="by_coords",
            chunks={"time_counter": 10},
            engine="netcdf4",
            lock=False,
        )["vbar"].astype("float32")

        self.wind_u_ds = xr.open_mfdataset(
            str(self.root / "fno_ERA5forcing*.nc"),
            combine="by_coords",
            chunks={"time": 10},
            engine="netcdf4",
            lock=False,
        )["u10"].astype("float32")

        self.wind_v_ds = xr.open_mfdataset(
            str(self.root / "fno_ERA5forcing*.nc"),
            combine="by_coords",
            chunks={"time": 10},
            engine="netcdf4",
            lock=False,
        )["v10"].astype("float32")

        self.slp_ds = xr.open_mfdataset(
            str(self.root / "fno_ERA5forcing*.nc"),
            combine="by_coords",
            chunks={"time": 10},
            engine="netcdf4",
            lock=False,
        )["msl"].astype("float32")

        self.ssh_ds = self.ssh_ds.rename({"time_counter": "time"})
        self.u_ds = self.u_ds.rename({"time_counter": "time"})
        self.v_ds = self.v_ds.rename({"time_counter": "time"})


        self.bathy_ds = xr.open_dataset(str(self.root / "nordic_seas_domain_cfg.nc"), engine="netcdf4")["bathy_metry"]
        self.bathy_array = np.asarray(self.bathy_ds.values, dtype=np.float32)

        self.land_mask = None
        if (self.root / "land_mask.nc").exists():
            self.land_mask = xr.open_dataset(str(self.root / "land_mask.nc"), engine="netcdf4").get("land_mask")
            if self.land_mask is not None:
                self.land_mask = np.asarray(self.land_mask.values, dtype=np.float32)

        self.var_map = {
            "ssh": self.ssh_ds,
            "u": self.u_ds,
            "v": self.v_ds,
            "wind_u": self.wind_u_ds,
            "wind_v": self.wind_v_ds,
            "slp": self.slp_ds,
        }

        if self.start_time is not None or self.end_time is not None:
            self.ssh_ds = self.ssh_ds.sel(time=slice(self.start_time, self.end_time))
            self.u_ds = self.u_ds.sel(time=slice(self.start_time, self.end_time))
            self.v_ds = self.v_ds.sel(time=slice(self.start_time, self.end_time))
            self.wind_u_ds = self.wind_u_ds.sel(time=slice(self.start_time, self.end_time))
            self.wind_v_ds = self.wind_v_ds.sel(time=slice(self.start_time, self.end_time))
            self.slp_ds = self.slp_ds.sel(time=slice(self.start_time, self.end_time))
            self.var_map = {
                "ssh": self.ssh_ds,
                "u": self.u_ds,
                "v": self.v_ds,
                "wind_u": self.wind_u_ds,
                "wind_v": self.wind_v_ds,
                "slp": self.slp_ds,
            }

        self.times = self.ssh_ds.time.values
        self.n_samples = max(len(self.times) - self.temporal_window, 0)

        self._source_points = None
        self._target_points = None

        # self._validate_vars()

    #     bathy_da = ds[bathymetry_var]
    #     if bathy_da.ndim != 2:
    #         raise ValueError("bathymetry must be a 2D field with dimensions [y, x]")
    #     self.bathy = torch.from_numpy(np.asarray(bathy_da.values, dtype=np.float32)).unsqueeze(0)

    #     self.n_samples = ds.sizes[self.time_dim] - self.horizon - self.temporal_window + 1
    #     if self.n_samples < 1:
    #         raise ValueError("Not enough time steps for the requested horizon/window")

    # def _find_time_dim(self, ds, time_dim):
    #     if time_dim is not None:
    #         return time_dim
    #     for candidate in ("time", "time_counter"):
    #         if candidate in ds.dims:
    #             return candidate
    #     raise ValueError("Could not infer the time dimension in the dataset")

    # def _validate_vars(self):
    #     for var in self.input_vars + self.target_vars + [self.bathymetry_var]:
    #         if var not in self.ds:
    #             raise KeyError(f"Variable {var!r} not found in dataset")

    # def _prepare_stats(self, values, n_channels):
    #     if values is None:
    #         return None
    #     stats = torch.as_tensor(np.asarray(values, dtype=np.float32))
    #     if stats.ndim == 0:
    #         stats = stats.view(1)
    #     if stats.ndim != 1:
    #         raise ValueError("mean/std must be a 1D array")
    #     if stats.numel() < n_channels:
    #         raise ValueError(f"Expected at least {n_channels} mean/std values, got {stats.numel()}")
    #     return stats[:n_channels]

    # def _to_tensor(self, da):
    #     return torch.from_numpy(np.asarray(da.values, dtype=np.float32))

    # def _normalize(self, tensor, mean, std, n_channels):
    #     if mean is None or std is None:
    #         return tensor
    #     if tensor.dim() == 3:
    #         mean_view = mean.view(n_channels, 1, 1)
    #         std_view = std.view(n_channels, 1, 1)
    #     elif tensor.dim() == 4:
    #         mean_view = mean.view(n_channels, 1, 1, 1)
    #         std_view = std.view(n_channels, 1, 1, 1)
    #     else:
    #         raise ValueError(f"Unsupported tensor dimensionality: {tensor.dim()}")
    #     std_view = torch.where(std_view != 0, std_view, torch.ones_like(std_view))
    #     return (tensor - mean_view) / std_view

    def _apply_coarse_sampling(self, array):
        if self.coarse_step <= 1 and self.coarse_slice is None:
            return array

        if self.coarse_slice is not None:
            y_slice, x_slice = self.coarse_slice
            return array[y_slice, x_slice]

        y_size, x_size = array.shape[-2], array.shape[-1]
        y_start = self.lat_trim
        x_start = self.lon_trim
        y_stop = y_size
        x_stop = x_size
        return array[y_start:y_stop:self.coarse_step, x_start:x_stop:self.coarse_step]

    def _regrid_to_ocean(self, da):
        if self._source_points is None:
            forcing_lat = self.wind_u_ds.lat.values
            forcing_lon = self.wind_u_ds.lon.values
            forcing_lon2d, forcing_lat2d = np.meshgrid(forcing_lon, forcing_lat)
            self._source_points = np.column_stack((forcing_lat2d.ravel(), forcing_lon2d.ravel()))

        if self._target_points is None:
            ocean_lat = self.ssh_ds.nav_lat.values
            ocean_lon = self.ssh_ds.nav_lon.values
            self._target_points = np.column_stack((ocean_lat.ravel(), ocean_lon.ravel()))

        values = np.asarray(da.values, dtype=np.float32).ravel()
        regridded = griddata(
            self._source_points,
            values,
            self._target_points,
            method="linear",
            fill_value=np.nan,
        )
        if np.isnan(regridded).any():
            nearest = griddata(
                self._source_points,
                values,
                self._target_points,
                method="nearest",
            )
            regridded[np.isnan(regridded)] = nearest[np.isnan(regridded)]

        target_shape = (self.ssh_ds.sizes.get("y"), self.ssh_ds.sizes.get("x"))
        return regridded.reshape(target_shape).astype(np.float32)

    def __len__(self):
        return self.n_samples #

    def __getitem__(self, idx):
        input_times = self.times[idx : idx + self.temporal_window]
        target_time = self.times[idx + self.temporal_window]

        input_frames = []

        for t in input_times:
            channels = []
            for var in self.input_vars:
                da = self.var_map.get(var)
                if da is None:
                    raise KeyError(f"Input variable {var!r} not recognised")
                if var in {"wind_u", "wind_v", "slp"}:
                    field = da.sel(time=t, method="nearest")
                    field = self._regrid_to_ocean(field)
                else:
                    field = da.sel(time=t, method="nearest").astype(np.float32).values

                field = self._apply_coarse_sampling(field)
                if self.land_mask is not None and var in {"ssh", "u", "v"}:
                    mask = self._apply_coarse_sampling(self.land_mask)
                    field = np.where(mask > 0, field, 0.0).astype(np.float32)
                channels.append(field)
            channels.append(self._apply_coarse_sampling(self.bathy_array))
            frame = np.stack(channels, axis=0)
            input_frames.append(frame)

        x = np.stack(input_frames, axis=0)  # [temporal_window, channels, lat, lon]

        # prepare targets
        target_channels = []
        for var in self.target_vars:
            da = self.var_map.get(var)
            if da is None:
                raise KeyError(f"Target variable {var!r} not recognised")
            target_channels.append(da.sel(time=target_time, method="nearest").astype(np.float32).values)

        y = np.stack(target_channels, axis=0)  # [n_target_vars, lat, lon]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Applying the transform
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y




