import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        output_channels=1,
        kernel_size=3
    ):
        super().__init__()

        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

        self.output_layer = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=1
        )

    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        batch_size, time_steps, channels, height, width = x.shape
        device = x.device

        h, c = self.cell.init_hidden(batch_size, height, width, device)

        for t in range(time_steps):
            h, c = self.cell(x[:, t], (h, c))

        y = self.output_layer(h)

        return y