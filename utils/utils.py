
# Importing neccessary libraries
import torch
from torch.utils.data import Dataset
import xarray as xr
from scipy.interpolate import griddata
import numpy as np


def _open_dataarray(path_or_paths, var_name, chunks=None, concat_dim='time', use_cftime=False):
    if isinstance(path_or_paths, (list, tuple)):
        ds = xr.open_mfdataset(
            path_or_paths,
            concat_dim=concat_dim,
            combine='by_coords',
            decode_times=True,
            use_cftime=use_cftime,
            chunks=chunks,
        )
    else:
        ds = xr.open_dataset(
            path_or_paths,
            decode_times=True,
            use_cftime=use_cftime,
            chunks=chunks,
        )
    return ds[var_name]


def open_ocean_dataset(
    ssh_files,
    u_files,
    v_files,
    wind_u_files,
    wind_v_files,
    slp_files,
    bathymetry_file,
    var_names=None,
    chunks=None,
    concat_dim='time',
    join='inner',
    use_cftime=False,
):
    """Open separate ocean variable files and return a merged xarray Dataset.

    Parameters:
        ssh_files, u_files, v_files, wind_u_files, wind_v_files, slp_files:
            file path or list of file paths for each variable.
        bathymetry_file: file path or list of files for the bathymetry field.
        var_names: optional dict mapping canonical names to variable names in files.
        chunks: optional dask chunking dict for lazy loading.
        concat_dim: time dimension to concatenate along for multi-file inputs.
        join: join method for xr.align (default 'inner').
        use_cftime: whether to decode times with cftime.

    Returns:
        xarray.Dataset with variables ssh, u, v, wind_u, wind_v, slp, bathymetry.
    """
    if var_names is None:
        var_names = {
            'ssh': 'ssh',
            'u': 'u',
            'v': 'v',
            'wind_u': 'wind_u',
            'wind_v': 'wind_v',
            'slp': 'slp',
            'bathymetry': 'bathymetry',
        }

    ssh_da = _open_dataarray(ssh_files, var_names['ssh'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    u_da = _open_dataarray(u_files, var_names['u'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    v_da = _open_dataarray(v_files, var_names['v'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    wind_u_da = _open_dataarray(wind_u_files, var_names['wind_u'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    wind_v_da = _open_dataarray(wind_v_files, var_names['wind_v'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    slp_da = _open_dataarray(slp_files, var_names['slp'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)
    bathy_da = _open_dataarray(bathymetry_file, var_names['bathymetry'], chunks=chunks, concat_dim=concat_dim, use_cftime=use_cftime)

    ssh_da, u_da, v_da, wind_u_da, wind_v_da, slp_da = xr.align(
        ssh_da, u_da, v_da, wind_u_da, wind_v_da, slp_da,
        join=join,
    )

    if bathy_da.ndim == 2:
        try:
            bathy_da, ssh_da = xr.align(bathy_da, ssh_da.isel({ssh_da.dims[0]: 0}), join=join)
        except Exception:
            pass

    ds = xr.Dataset(
        {
            'ssh': ssh_da,
            'u': u_da,
            'v': v_da,
            'wind_u': wind_u_da,
            'wind_v': wind_v_da,
            'slp': slp_da,
            'bathymetry': bathy_da,
        }
    )
    return ds


class OceanDataset(Dataset):
    """
    Dataset for loading ocean state, forcing variables and a static bathymetry field.
    """
    def __init__(self, ssh, u=None, v=None, wind_u=None, wind_v=None, slp=None,
                 bathymetry=None, horizon=1, mean=None, std=None):
        if isinstance(ssh, xr.Dataset):
            ds = ssh
            ssh_da = ds["ssh"]
            u_da = ds["u"]
            v_da = ds["v"]
            wind_u_da = ds["wind_u"]
            wind_v_da = ds["wind_v"]
            slp_da = ds["slp"]
        else:
            if any(arg is None for arg in [u, v, wind_u, wind_v, slp, bathymetry]):
                raise ValueError("Provide either a merged xarray Dataset or all six variables plus bathymetry.")
            ssh_da = ssh
            u_da = u
            v_da = v
            wind_u_da = wind_u
            wind_v_da = wind_v
            slp_da = slp

        # converting all the variables to float32
        ssh_t = torch.from_numpy(ssh_da.values).float()
        u_t   = torch.from_numpy(u_da.values).float()
        v_t   = torch.from_numpy(v_da.values).float()
        wu_t  = torch.from_numpy(wind_u_da.values).float()
        wv_t  = torch.from_numpy(wind_v_da.values).float()
        slp_t = torch.from_numpy(slp_da.values).float()
        bathy_t = torch.from_numpy(bathymetry.values).float()  # [y, x]

        # Stack time-dependent fields into [T, C, H, W]
        data = torch.stack([ssh_t, u_t, v_t, wu_t, wv_t, slp_t], dim=1)  # [T, 6, H, W]

        # Standardise each variable independently.
        if mean is not None and std is not None:
            mean_t = torch.as_tensor(mean, dtype=data.dtype)
            std_t = torch.as_tensor(std, dtype=data.dtype)

            if mean_t.ndim == 1:
                mean_t = mean_t.view(1, -1, 1, 1)
                std_t = std_t.view(1, -1, 1, 1)
            elif mean_t.ndim == 0:
                mean_t = mean_t.view(1, 1, 1, 1)
                std_t = std_t.view(1, 1, 1, 1)

            std_t = torch.where(std_t != 0, std_t, torch.ones_like(std_t))
            data = (data - mean_t) / std_t

        self.fields = data                      # [T, 6, H, W]
        self.bathy = bathy_t.unsqueeze(0)       # [1, H, W]
        self.horizon = horizon
        self.n_samples = self.fields.shape[0] - horizon

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_t = self.fields[idx]          # [6, H, W]

        # Target: SSH, U, V at the next time step
        target_t = self.fields[idx + 1, 0:3]  # [3, H, W]

        # Input: 6 dynamic + 1 static bathy
        input_t = torch.cat([x_t, self.bathy], dim=0)  # [7, H, W]

        return input_t, target_t


class LazyOceanDataset(Dataset):
    """Lazy-loading Ocean dataset that reads xarray slices on demand."""

    def __init__(
        self,
        ssh,
        u=None,
        v=None,
        wind_u=None,
        wind_v=None,
        slp=None,
        bathymetry=None,
        horizon=1,
        mean=None,
        std=None,
    ):
        if isinstance(ssh, xr.Dataset):
            ds = ssh
            ssh_da = ds['ssh']
            u_da = ds['u']
            v_da = ds['v']
            wind_u_da = ds['wind_u']
            wind_v_da = ds['wind_v']
            slp_da = ds['slp']
            bathymetry_da = ds['bathymetry']
        else:
            if any(arg is None for arg in [u, v, wind_u, wind_v, slp, bathymetry]):
                raise ValueError(
                    'Provide either a merged xarray Dataset or all six variables plus bathymetry.'
                )
            ssh_da = ssh
            u_da = u
            v_da = v
            wind_u_da = wind_u
            wind_v_da = wind_v
            slp_da = slp
            bathymetry_da = bathymetry

        self.time_dim = ssh_da.dims[0]
        self.ssh = ssh_da
        self.u = u_da
        self.v = v_da
        self.wind_u = wind_u_da
        self.wind_v = wind_v_da
        self.slp = slp_da

        if bathymetry_da.ndim != 2:
            raise ValueError('bathymetry must be a 2D field with dimensions [y, x]')
        self.bathy = torch.from_numpy(np.asarray(bathymetry_da.values, dtype=np.float32)).unsqueeze(0)

        self.horizon = horizon
        self.n_samples = self.ssh.sizes[self.time_dim] - horizon
        if self.n_samples < 1:
            raise ValueError('Not enough time steps for the requested horizon.')

        if mean is not None and std is not None:
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
            std_t = torch.as_tensor(std, dtype=torch.float32)

            if mean_t.ndim == 1:
                mean_t = mean_t.view(-1, 1, 1)
                std_t = std_t.view(-1, 1, 1)
            elif mean_t.ndim == 0:
                mean_t = mean_t.view(1, 1, 1)
                std_t = std_t.view(1, 1, 1)

            std_t = torch.where(std_t != 0, std_t, torch.ones_like(std_t))
            self.mean = mean_t
            self.std = std_t
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return self.n_samples

    def _to_tensor(self, da):
        arr = np.asarray(da.values, dtype=np.float32)
        return torch.from_numpy(arr).float()

    def _normalize(self, tensor):
        if self.mean is None or self.std is None:
            return tensor
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx):
        idx = int(idx)
        x_t = torch.stack(
            [
                self._to_tensor(self.ssh.isel({self.time_dim: idx})),
                self._to_tensor(self.u.isel({self.time_dim: idx})),
                self._to_tensor(self.v.isel({self.time_dim: idx})),
                self._to_tensor(self.wind_u.isel({self.time_dim: idx})),
                self._to_tensor(self.wind_v.isel({self.time_dim: idx})),
                self._to_tensor(self.slp.isel({self.time_dim: idx})),
            ],
            dim=0,
        )

        target_idx = idx + self.horizon
        target_t = torch.stack(
            [
                self._to_tensor(self.ssh.isel({self.time_dim: target_idx})),
                self._to_tensor(self.u.isel({self.time_dim: target_idx})),
                self._to_tensor(self.v.isel({self.time_dim: target_idx})),
            ],
            dim=0,
        )

        if self.mean is not None:
            x_t = self._normalize(x_t)
            target_t = (target_t - self.mean[:3]) / self.std[:3]

        input_t = torch.cat([x_t, self.bathy], dim=0)
        return input_t, target_t



def regrid_xy(da: xr.DataArray) -> xr.DataArray:
    values = da.values  # [T, lat, lon]
    result = np.empty((values.shape[0],) + nemo_lat.shape, dtype=np.float32)
    for t in range(values.shape[0]):
        linear = griddata(
            source_points,
            values[t].ravel(),
            target_points,
            method="linear",
            fill_value=np.nan,
        )
        if np.isnan(linear).any():
            nearest = griddata(
                source_points,
                values[t].ravel(),
                target_points,
                method="nearest",
            )
            linear[np.isnan(linear)] = nearest[np.isnan(linear)]
        result[t] = linear.reshape(nemo_lat.shape)
    return xr.DataArray(
        result,
        coords={"time_counter": ssh["time_counter"], "y": ssh["y"], "x": ssh["x"]},
        dims=("time_counter", "y", "x"),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)  # [B, 7, H, W]
        y = y.to(device)  # [B, 3, H, W]

        optimizer.zero_grad()
        pred = model(x)   # [B, 3, H, W]
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)