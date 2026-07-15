from models.convLSTM import ConvLSTM
import torch


model = ConvLSTM(
    input_channels=5,
    hidden_channels=64,
    output_channels=1
)

x = torch.randn(8, 12, 5, 64, 64)
y = model(x)

print(y.shape)