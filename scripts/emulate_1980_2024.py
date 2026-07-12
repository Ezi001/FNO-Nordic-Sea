"""Autoregressive emulation script for 1980-2024 using a TorchScript FNOtD model.

Notes:
- Expects a TorchScript model (recommended) — use `torch.jit.trace` or `torch.jit.script` when saving.
- Expects input dataset (xarray) with variables: `ssh,u,v,wind_u,wind_v,slp,bathymetry` and a time coordinate named `time` (or `time_counter`).
- Expects a mean/std file (npz or npy): if `.npz` it should contain arrays `mean` and `std` (shape (6,)). If a `.npy` file, it should be an array of shape (2,6) or try to load as .npz.
- The model must accept input tensor of shape [B, 7, H, W] and return [B, 3, H, W] in normalized units (same normalization used for training).

Usage example:
python scripts/emulate_1980_2024.py \
  --model scripts/fnotd_scripted.pt \
  --meanstd data/meanstd.npz \
  --input data/merged_dataset.nc \
  --output results/emulation_1980_2024.nc \
  --start 1980-01-01 --end 2024-12-31

"""

import argparse
from pathlib import Path
import sys
import numpy as np
import xarray as xr
import torch

VAR_ORDER = ['ssh', 'u', 'v', 'wind_u', 'wind_v', 'slp']


def load_mean_std(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mean/std file not found: {path}")
    if path.suffix == '.npz':
        data = np.load(path)
        mean = data['mean']
        std = data['std']
    else:
        arr = np.load(path)
        # try to interpret formats: (6,), or (2,6)
        if arr.ndim == 1 and arr.size == 6:
            raise ValueError("Single array loaded; need both mean and std. Provide .npz with 'mean' and 'std'.")
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == 6:
            mean = arr[0]
            std = arr[1]
        else:
            raise ValueError("Unrecognized mean/std array shape. Provide an .npz with 'mean' and 'std' arrays of shape (6,).")
    return mean.astype(np.float32), std.astype(np.float32)


def choose_time_coord(ds):
    for name in ['time', 'time_counter', 'times']:
        if name in ds.coords:
            return name
    # fallback: first coordinate with dtype datetime-like or 1D
    for k in ds.coords:
        if ds.coords[k].ndim == 1:
            return k
    raise RuntimeError('No time coordinate found in dataset')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to TorchScript model (.pt)')
    p.add_argument('--meanstd', required=True, help='Path to mean/std .npz produced during training')
    p.add_argument('--input', required=True, help='Input xarray dataset with required variables')
    p.add_argument('--output', required=True, help='Output NetCDF path to write emulation results')
    p.add_argument('--start', required=True, help='Start time (inclusive), e.g. 1980-01-01')
    p.add_argument('--end', required=True, help='End time (inclusive), e.g. 2024-12-31')
    p.add_argument('--device', default='cpu')
    p.add_argument('--use_cftime', action='store_true', help='Open dataset with use_cftime=True')
    args = p.parse_args()

    device = torch.device(args.device)

    # Load model (TorchScript recommended)
    model_path = Path(args.model)
    if not model_path.exists():
        print('Model file not found:', model_path, file=sys.stderr)
        sys.exit(1)
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
    except Exception as e:
        print('Failed to load TorchScript model:', e, file=sys.stderr)
        sys.exit(1)

    # Load mean/std
    mean, std = load_mean_std(args.meanstd)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError('mean and std must have shape (6,) for variables: ' + ','.join(VAR_ORDER))

    # Load dataset
    ds = xr.open_dataset(args.input, decode_times=True, use_cftime=args.use_cftime)
    tname = choose_time_coord(ds)
    ds_time = ds.sel({tname: slice(args.start, args.end)})
    times = ds_time.coords[tname].values
    nt = len(times)
    if nt < 2:
        raise ValueError('Need at least two time steps in selected range')

    # Check variables
    for v in VAR_ORDER + ['bathymetry']:
        if v not in ds_time:
            raise KeyError(f"Variable '{v}' not found in dataset")

    # Prepare spatial grid
    # assume dims are (time, y, x)
    y = ds_time[VAR_ORDER[0]].coords[ds_time[VAR_ORDER[0]].dims[-2]]
    x = ds_time[VAR_ORDER[0]].coords[ds_time[VAR_ORDER[0]].dims[-1]]

    ny = ds_time[VAR_ORDER[0]].shape[-2]
    nx = ds_time[VAR_ORDER[0]].shape[-1]

    # Allocate outputs
    ssh_out = np.empty((nt, ny, nx), dtype=np.float32)
    u_out = np.empty((nt, ny, nx), dtype=np.float32)
    v_out = np.empty((nt, ny, nx), dtype=np.float32)

    # Initialize current dynamic fields from dataset at first time
    cur_ssh = ds_time['ssh'].isel({tname: 0}).values.astype(np.float32)
    cur_u = ds_time['u'].isel({tname: 0}).values.astype(np.float32)
    cur_v = ds_time['v'].isel({tname: 0}).values.astype(np.float32)

    bathy = ds_time['bathymetry'].values.astype(np.float32)  # [y,x]
    bathy_t = torch.from_numpy(bathy).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,y,x]

    # store initial state
    ssh_out[0] = cur_ssh
    u_out[0] = cur_u
    v_out[0] = cur_v

    # Precompute normalization tensors
    mean_dyn = torch.from_numpy(mean).to(device).view(1, -1, 1, 1)  # [1,6,1,1]
    std_dyn = torch.from_numpy(std).to(device).view(1, -1, 1, 1)
    std_dyn = torch.where(std_dyn != 0, std_dyn, torch.ones_like(std_dyn))

    # Loop forward in time (autoregressive)
    for i in range(nt - 1):
        # time index i corresponds to times[i]
        # extract forcings at time i (we assume known forcings at current time)
        wu = ds_time['wind_u'].isel({tname: i}).values.astype(np.float32)
        wv = ds_time['wind_v'].isel({tname: i}).values.astype(np.float32)
        slp = ds_time['slp'].isel({tname: i}).values.astype(np.float32)

        # build dynamic array [6, y, x]
        dyn = np.stack([cur_ssh, cur_u, cur_v, wu, wv, slp], axis=0)

        # convert to tensor and construct model input [1,7,y,x]
        dyn_t = torch.from_numpy(dyn).unsqueeze(0).to(device)  # [1,6,y,x]

        # normalize dynamic channels
        dyn_norm = (dyn_t - mean_dyn) / std_dyn

        inp = torch.cat([dyn_norm, bathy_t], dim=1)  # [1,7,y,x]

        # model forward
        with torch.no_grad():
            pred_norm = model(inp)  # expect [1,3,y,x]

        # denormalize first three channels (ssh,u,v)
        pred = pred_norm.clone()
        pred[:, 0:3, :, :] = pred[:, 0:3, :, :] * std_dyn[:, 0:3, :, :] + mean_dyn[:, 0:3, :, :]

        pred_np = pred.squeeze(0).cpu().numpy()  # [3,y,x]

        # store into outputs at i+1
        ssh_out[i + 1] = pred_np[0]
        u_out[i + 1] = pred_np[1]
        v_out[i + 1] = pred_np[2]

        # update current state for next step
        cur_ssh = pred_np[0]
        cur_u = pred_np[1]
        cur_v = pred_np[2]

        if (i + 1) % 50 == 0:
            print(f'Processed step {i+1}/{nt-1}')

    # Save outputs to NetCDF
    out_ds = xr.Dataset(
        {
            'ssh': (('time', y.name, x.name), ssh_out),
            'u': (('time', y.name, x.name), u_out),
            'v': (('time', y.name, x.name), v_out),
        },
        coords={
            'time': ds_time.coords[tname].values,
            y.name: ds_time[VAR_ORDER[0]].coords[y.name].values,
            x.name: ds_time[VAR_ORDER[0]].coords[x.name].values,
        }
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(str(out_path))
    print('Wrote emulation to', out_path)


if __name__ == '__main__':
    main()
