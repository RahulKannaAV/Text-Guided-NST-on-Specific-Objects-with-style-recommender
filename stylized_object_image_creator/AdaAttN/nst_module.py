import torch
import os
from pathlib import Path
from torch import nn
from .image_encoder import get_relu_layers
from .util_functions import MeanVarianceNormalization, get_embedded_features, get_x_layer_feature
from .weight_loader import load_encoder_weights, load_adaattn_weights, load_transformer_weights, load_decoder_weights


def stats(name, x):
  print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.4f}, "
        f"max={x.max().item():.4f}, "
        f"mean={x.mean().item():.4f}, "
        f"std={x.std().item():.4f}"
  )

class NST(nn.Module):
  def __init__(self):
    super().__init__()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    ENCODER_WEIGHT_NAME = "encoder_weights.pth"
    ADAATTN_WEIGHT_NAME = "ada_attn_weights.pth"
    TRANSFORMER_WEIGHT_NAME = "transformer_weights.pth"
    DECODER_WEIGHT_NAME = "decoder_weights.pth"

    self.encoder = load_encoder_weights(Path(f"{BASE_DIR}\\{MODEL_DIR}\\{ENCODER_WEIGHT_NAME}"))
    self.encoder_layers = get_relu_layers(self.encoder)

    self._init_features(torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224), self.encoder_layers)

    if hasattr(self, 'fc_3') and hasattr(self, 'fc_embed3'):
      self.ada_attn3 = load_adaattn_weights(weight_path=Path(f"{BASE_DIR}\\{MODEL_DIR}\\{ADAATTN_WEIGHT_NAME}"),
                                            embed_channels=self.fc_embed3.shape[1],
                                            input_channels=self.fc_3.shape[1])

    if hasattr(self, 'fc_4') and hasattr(self, 'fc_embed4'):
      self.transformer = load_transformer_weights(weight_path=Path(f"{BASE_DIR}\\{MODEL_DIR}\\{TRANSFORMER_WEIGHT_NAME}"),
                                                  embed_channels=self.fc_embed4.shape[1],
                                                  input_channels=self.fc_4.shape[1])

    self.decoder = load_decoder_weights(Path(f"{BASE_DIR}\\{MODEL_DIR}\\{DECODER_WEIGHT_NAME}"))

  def _init_features(self, content_img, style_img, encoder_layers):
    for idx in range(1, 6):
      setattr(self,f"fc_{idx}", get_x_layer_feature(content_img, self.encoder_layers, idx))
      setattr(self,f"fs_{idx}", get_x_layer_feature(style_img, self.encoder_layers, idx))
      setattr(self,f"fc_embed{idx}", get_embedded_features(content_img, self.encoder_layers, idx))
      setattr(self,f"fs_embed{idx}", get_embedded_features(style_img, self.encoder_layers, idx))


  def forward(self, content_img, style_img):
    self._init_features(content_img, style_img, self.encoder_layers)

    for i in range(5):
      for name, param in self.encoder_layers[i].named_parameters():
        print(
          name,
          "min:", param.min().item(),
          "max:", param.max().item(),
          "mean:", param.mean().item(),
          "std:", param.std().item())


    stats("fc_3", self.fc_3)
    stats("fs_3", self.fs_3)
    stats("fc_embed3", self.fc_embed3)
    stats("fs_embed3", self.fs_embed3)

    stats("fc_4", self.fc_4)
    stats("fs_4", self.fs_4)
    stats("fc_5", self.fc_5)
    stats("fs_5", self.fs_5)

    norm = MeanVarianceNormalization()

    x = norm.normalize(self.fc_embed3)

    print("Normalized mean:", x.mean(dim=(0, 2, 3)).abs().max().item())
    print("Normalized std:", x.std(dim=(0, 2, 3), correction=0).mean().item())

    fcs_3 = self.ada_attn3(self.fc_3, self.fs_3,
                           self.fc_embed3, self.fs_embed3)
    stats("Fcs_3", fcs_3)
    fcs_45_merged = self.transformer(self.fc_4, self.fs_4,
                                     self.fc_5, self.fs_5,
                                     self.fc_embed4, self.fs_embed4,
                                     self.fc_embed5, self.fs_embed5)
    print("Fcs45 done")
    stats("Fcs45", fcs_45_merged)

    I_cs = self.decoder(fcs_45_merged, fcs_3)
    print("Ics done")
    stats("I_cs", I_cs)

    return I_cs
