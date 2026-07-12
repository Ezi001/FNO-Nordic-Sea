import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from pathlib import Path

Path('temp_test').mkdir(exist_ok=True)

nt, ny, nx = 6, 16, 16
times = np.array(['1980-01-01','1980-01-02','1980-01-03','1980-01-04','1980-01-05','1980-01-06'], dtype='datetime64[D]')
y = np.arange(ny)
x = np.arange(nx)
coords = {'time': times, 'y': y, 'x': x}
vars = {}
for name in ['ssh','u','v','wind_u','wind_v','slp']:
    vars[name] = (('time','y','x'), np.random.randn(nt, ny, nx).astype(np.float32))
vars['bathymetry'] = (('y','x'), np.abs(np.random.randn(ny,nx).astype(np.float32)))
ds = xr.Dataset(data_vars=vars, coords=coords)
ds.to_netcdf('temp_test/synthetic_input.nc')
mean = np.zeros(6, dtype=np.float32)
std = np.ones(6, dtype=np.float32)
np.savez('temp_test/meanstd.npz', mean=mean, std=std)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    def forward(self, x):
        return self.net(x)

model = SimpleModel()
scripted = torch.jit.script(model)
scripted.save('temp_test/fnotd_synthetic.pt')
print('created synthetic files')
