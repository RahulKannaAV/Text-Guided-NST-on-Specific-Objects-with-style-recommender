import torch
from torch import nn

class OneOneConv(nn.Module):
  def __init__(self, in_feat, out_feat):
    super().__init__()
    self.conv_layer = nn.Conv2d(in_channels=in_feat,
                                out_channels=out_feat,
                                kernel_size=1)

  def forward(self, x):
    result = self.conv_layer(x)

    return result