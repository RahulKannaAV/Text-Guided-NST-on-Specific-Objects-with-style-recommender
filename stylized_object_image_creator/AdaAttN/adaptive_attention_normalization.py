import torch
from torch import nn
from .one_conv import OneOneConv
from .util_functions import MeanVarianceNormalization

norm = MeanVarianceNormalization()

def AttnWeightedMean(V, A):
  N, C, Hs, Ws = V.shape
  V_reshaped = torch.reshape(input=V,
                             shape=(N, C, Hs * Ws))
  A_transpose = torch.transpose(A, 1, 2)
  M = V_reshaped @ A_transpose # V @ A^T

  return M

def AttnWeightedSD(V, A, M):
  N, C, Hs, Ws = V.shape
  V_reshaped = torch.reshape(input=V,
                             shape=(N, C, Hs * Ws))
  V_dot_prod = V_reshaped * V_reshaped # V * V

  A_transpose = torch.transpose(A, 1, 2) # A^T

  term1 = V_dot_prod @ A_transpose # (V * V) @ A^T
  term2 = M * M # M * M

  S = torch.sqrt(torch.clamp(term1 - term2, min=0.0)) # sqrt(((V * V) @ A^T) - (M * M))

  return S

import torch

def AdaptiveNormalization(content_feature, S, M):

    N, C, H, W = content_feature.shape

    normalized_content_feature = norm.normalize(content_feature)
    normalized_content_feature = normalized_content_feature.reshape(N, C, H * W)

    # S and M should have shape [N, C, H*W]
    assert S.shape == normalized_content_feature.shape, (
        f"S shape {S.shape} != {normalized_content_feature.shape}"
    )
    assert M.shape == normalized_content_feature.shape, (
        f"M shape {M.shape} != {normalized_content_feature.shape}"
    )

    fcs_x = S * normalized_content_feature + M

    return fcs_x.reshape(N, C, H, W)

class AuthorAttN(nn.Module):
  def __init__(self, embed_channels, input_channels):
    super().__init__()
    # f, g, h -> weights and biases
    self.f = OneOneConv(in_feat=embed_channels,
                             out_feat=embed_channels)
    self.g = OneOneConv(in_feat=embed_channels,
                             out_feat=embed_channels)
    self.h = OneOneConv(in_feat=input_channels,
                             out_feat=input_channels)

    norm = MeanVarianceNormalization()
    self.sm = nn.Softmax(dim=-1)

  def forward(self, content_feature, style_feature,
              content_embed_feature, style_embed_feature):
    Q_norm = norm.normalize(content_embed_feature)
    Q = self.f(Q_norm)
    K_norm = norm.normalize(style_embed_feature)
    K = self.g(K_norm).flatten(2)

    V = self.h(style_feature)
    q_transpose = Q.flatten(2).transpose(1, 2)

    # attention map
    A = self.sm(q_transpose @ K)

    M = AttnWeightedMean(V, A)
    S = AttnWeightedSD(V, A, M)

    fcs_x = AdaptiveNormalization(content_feature, S, M)

    return fcs_x
