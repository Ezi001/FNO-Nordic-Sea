# Import necessary libraries
import numpy as np
import torch
from pathlib import Path

from utils.utils import LazyOceanDataset, open_ocean_dataset

BASE = Path(__file__).resolve().parents[1] / 'data'

input_dataset = open_ocean_dataset(
    ssh_files=BASE / 'NAA10KM_1h_19800101_19801231_ssh.nc',
    u_files=BASE / 'NAA10KM_1h_19800101_19801231_ubar.nc',
    v_files=BASE / 'NAA10KM_1h_19800101_19801231_vbar.nc',
    wind_u_files=BASE / 'fno_ERA5forcing_y1980m01.nc',
    wind_v_files=BASE / 'fno_ERA5forcing_y1980m01.nc',
    slp_files=BASE / 'fno_ERA5forcing_y1980m01.nc',
    bathymetry_file=BASE / 'nordic_seas_domain_cfg.nc',
    var_names={
        'ssh': 'ssh',
        'u': 'ubar',
        'v': 'vbar',
        'wind_u': 'wind_u',
        'wind_v': 'wind_v',
        'slp': 'slp',
        'bathymetry': 'bathymetry',
    },
    chunks={'time': 1},
    join='inner',
)

print('Dataset variables:', list(input_dataset.data_vars))
print('Time steps:', input_dataset['ssh'].sizes[input_dataset['ssh'].dims[0]])
print('Chunks:', input_dataset['ssh'].chunks)

mean = np.zeros(6, dtype=np.float32)
std = np.ones(6, dtype=np.float32)

lazy_dataset = LazyOceanDataset(
    input_dataset,
    bathymetry=input_dataset['bathymetry'],
    horizon=1,
    mean=mean,
    std=std,
)

print('Lazy dataset length:', len(lazy_dataset))
input_t, target_t = lazy_dataset[0]
print('input shape:', input_t.shape)
print('target shape:', target_t.shape)
