import torch
from torch.utils import Dataset

class OceanDataset(Dataset):
    def __init__(self, ssh, u, v, wind_u, wind_v, slp, bathymetry):
        self.ssh = self.normalize(torch.FloatTensor(ssh.values))
        self.u = self.normalize(torch.FloatTensor(u.values))
        self.v = self.normalize(torch.FloatTensor(v.values))
        self.wind_u = self.normalize(torch.FloatTensor(wind_u.values))
        self.wind_v = self.normalize(torch.FloatTensor(wind_v.values))
        self.slp = self.normalize(torch.FloatTensor(slp.values))
        self.bathymetry = torch.FloatTensor(bathymetry.values)
        self.n_samples = self.ssh.shape[0] - 1

    def __getitem__(self, idx):
        static_bathy = self.bathymetry.unsqueeze(0).expand(1, *self.bathymetry.shape)
        input_t = torch.stack([
            self.ssh[idx],
            self.u[idx],
            self.v[idx],
            self.wind_u[idx],
            self.wind_v[idx],
            self.slp[idx],
            static_bathy[0],
        ], dim=0)
        target_t = torch.stack([
            self.ssh[idx + 1],
            self.u[idx + 1],
            self.v[idx + 1],
        ], dim=0)
        return input_t, target_t
    


