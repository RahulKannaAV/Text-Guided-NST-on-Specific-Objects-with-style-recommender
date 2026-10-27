from torchvision.models import vgg19, VGG19_Weights
import torch
from torch import nn


def get_encoder_instance():
  pretrained_weights = VGG19_Weights.DEFAULT
  model_transform = pretrained_weights.transforms()

  model = vgg19(weights=pretrained_weights)

  for param in model.parameters():
    param.requires_grad = False

  return [model, model_transform]

def get_relu_layers(model):

  relu1_1_layer = model.features[:3]
  relu2_1_layer = model.features[3:8]
  relu3_1_layer = model.features[8:13]
  relu4_1_layer = model.features[13:22]
  relu5_1_layer = model.features[22:31]

  return [relu1_1_layer, relu2_1_layer, relu3_1_layer, relu4_1_layer, relu5_1_layer]
