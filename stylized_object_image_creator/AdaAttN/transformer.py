import torch
from torch import nn
from .adaptive_attention_normalization import AuthorAttN

class AuthorTransformer(nn.Module):
  # ada_attn 4, ada_attn 5
  def __init__(self, embed_channels, input_channels):
    super().__init__()
    self.ada_attn_4_1 = AuthorAttN(embed_channels=embed_channels,
                                   input_channels=input_channels)
    self.ada_attn_5_1 = AuthorAttN(embed_channels=embed_channels+512,
                                   input_channels=input_channels)
    self.upsampler = nn.Upsample(scale_factor=2.0,
                                 mode="nearest")
    self.merge_conv_block = nn.Conv2d(in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1)

  def forward(self, fc_4, fs_4, fc_5, fs_5,
              fc_embed4, fs_embed4,
              fc_embed5, fs_embed5):
    fcs_4 = self.ada_attn_4_1(fc_4, fs_4, fc_embed4, fs_embed4)
    fcs_5 = self.ada_attn_5_1(fc_5, fs_5, fc_embed5, fs_embed5)
    upsampled_fcs5 = self.upsampler(fcs_5)

    cat_fcs4 = upsampled_fcs5 + fcs_4
    trans_result = self.merge_conv_block(cat_fcs4)

    return trans_result
