import torch
from torch import nn

class AuthDecoder(nn.Module):
  def __init__(self):
    super().__init__()
    # decoder layer 1, decoder layer 2
    self.decoder_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=512,
                  out_channels=256,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2.0, mode='nearest')
    )

    self.decoder_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=512,
                  out_channels=256,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=128,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        nn.Conv2d(in_channels=128,
                  out_channels=128,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,
                  out_channels=64,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2.0,
                    mode='nearest'),
        nn.Conv2d(in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,
                    out_channels=3,
                    kernel_size=3,
                    padding=1)
    )

  def forward(self, F_cs45, F_cs3):
    dec1_result = self.decoder_layer1(F_cs45)

    I_cs = self.decoder_layer2(torch.cat((dec1_result, F_cs3), dim=1))
    return I_cs
