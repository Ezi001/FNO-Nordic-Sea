import torch
from torch.utils.data import Dataset

class OceanDataset(Dataset):
    def __init__(self, ssh, u, v, wind_u, wind_v, slp, bathymetry,
                 mean=None, std=None):
        # ssh, u, v, wind_u, wind_v, slp: xarray DataArray [time, y, x]
        # bathymetry: xarray DataArray [y, x]

        ssh_t = torch.from_numpy(ssh.values).float()
        u_t   = torch.from_numpy(u.values).float()
        v_t   = torch.from_numpy(v.values).float()
        wu_t  = torch.from_numpy(wind_u.values).float()
        wv_t  = torch.from_numpy(wind_v.values).float()
        slp_t = torch.from_numpy(slp.values).float()
        bathy_t = torch.from_numpy(bathymetry.values).float()  # [y, x]

        # Stack time-dependent fields into [T, C, H, W]
        data = torch.stack([ssh_t, u_t, v_t, wu_t, wv_t, slp_t], dim=1)  # [T, 6, H, W]

        if mean is not None and std is not None:
            # mean, std: [C] or [1, C, 1, 1]
            data = (data - mean) / std

        self.fields = data                      # [T, 6, H, W]
        self.bathy = bathy_t.unsqueeze(0)       # [1, H, W]
        self.n_samples = self.fields.shape[0] - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_t = self.fields[idx]          # [6, H, W]
        x_tp1 = self.fields[idx + 1]    # [6, H, W]

        # Input: 6 dynamic + 1 static bathy
        input_t = torch.cat([x_t, self.bathy], dim=0)  # [7, H, W]

        # Target: SSH, U, V at t+1 → channels 0,1,2
        target_t = x_tp1[0:3, ...]      # [3, H, W]

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