from torchvision import transforms
import torch
from torch import nn
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image


def downsampler(input, x_input):
  interpolated_result = interpolate(input=input,
                                    size=x_input.shape[2:])

  return interpolated_result

def get_embedded_features(data, relu_layers, layer_number):
  norm = MeanVarianceNormalization()
  fx_features = norm.normalize(get_x_layer_feature(data, relu_layers, layer_number))

  f1_features = relu_layers[0](data)
  f1_x = downsampler(f1_features, fx_features)
  for i in range(2, layer_number):
    fi_features = get_x_layer_feature(data, relu_layers, i)

    fi_downsampled = norm.normalize(downsampler(fi_features, fx_features))

    f1_x = torch.cat([f1_x, fi_downsampled], dim=1)

  embedded_feature = torch.cat([f1_x, fx_features], dim=1)

  return embedded_feature

def get_x_layer_feature(data, relu_layers, layer_number):
  fx_features = data
  for i in range(layer_number):
    fx_layer = relu_layers[i]
    fx_features = fx_layer(fx_features)

  return fx_features

def save_nst_image(result_tensor:torch.Tensor,
                  image_name: str):
  image_numpy = tensor2im(result_tensor)
  image_pil = Image.fromarray(image_numpy)
  h, w, _ = image_numpy.shape
  image_pil.save(image_name)
  print(f"{image_name} saved successfully.")

# [N, C, H, W] => Channel-wise Mean Variance Normalization
class MeanVarianceNormalization:
    def normalize(self, data, eps=1e-5):
        N, C, H, W = data.shape

        mean = data.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        var = data.view(N, C, -1).var(dim=2, correction=0).view(N, C, 1, 1)

        std = (var + eps).sqrt()

        return (data - mean) / std

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0)), 0, 1) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)