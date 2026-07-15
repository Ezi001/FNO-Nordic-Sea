# Necessary imports
import torch.nn as nn
from neuralop.models import FNO

class FNOtDWrapper(nn.Module):
    def __init__(self, in_channels, out_channels=3, hidden_channels=64, n_layers=4, n_modes=(16, 16), padding=8):
        super().__init__()
        self.spatial_model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            padding=padding,
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out_flat = self.spatial_model(x_flat)
        out = out_flat.reshape(b, t, out_flat.shape[1], h, w)
        return out[:, -1]