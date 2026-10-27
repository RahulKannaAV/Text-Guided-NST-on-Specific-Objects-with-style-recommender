from pathlib import Path
import torch
from torch import nn
from .image_encoder import get_encoder_instance
from .adaptive_attention_normalization import AuthorAttN
from .transformer import AuthorTransformer
from .decoder import AuthDecoder

def load_encoder_weights(weight_path):
  # enc_weights['fc.weight']
  enc_weights = torch.load(weight_path, weights_only=True)

  # enc_weights['51.bias'] = torch.nn.Parameter(torch.tensor([79.264]))
  cnt = 0
  weights_idx = []
  for idx, (layer, weights) in enumerate(enc_weights.items()):
    cnt += 1
    weights_idx.append(int(layer.split('.')[0]))
  e, t = get_encoder_instance()

  name_list = []
  param_list = []

  d_cnt = 0

  e.features.insert(0, nn.Conv2d(
      in_channels=3,
      out_channels=3,
      stride=1,
      padding=1,
      kernel_size=1
  ))
  e.features[0].requires_grad = False

  our_encoder_idx = []
  for name, params in e.features.named_parameters():
    our_encoder_idx.append(int(name.split(".")[0]))

  # Loading weights to our encoder instance
  for idx, (layer, weights) in enumerate(enc_weights.items()):
    if(idx%2 == 0):
      e.features[our_encoder_idx[idx]].weight = nn.Parameter(enc_weights[layer])

    else:
      e.features[our_encoder_idx[idx]].bias = nn.Parameter(enc_weights[layer])

  print("Encoder successfully loaded with pretrained weights")
  return e


def load_adaattn_weights(weight_path, embed_channels, input_channels):
  ada_weights = torch.load(weight_path, weights_only=True)

  for layer, weights in ada_weights.items():
    print(f"Layer name: {layer} & Weight shape: {weights.shape}")

  our_adaattn = AuthorAttN(embed_channels=embed_channels,
                           input_channels=input_channels)
  print()
  for our_layer, our_params in our_adaattn.named_parameters():
    print(f"Our Layer: {our_layer} & Our param shape: {our_params.shape}")

  # print(getattr(our_adaattn, "f").conv_layer.weight.shape)

  cnt = 0
  for layer_name, weights in ada_weights.items():
    param_char = layer_name.split('.')[0]
    if(cnt%2 == 0): # Weight
      getattr(our_adaattn, param_char).conv_layer.weight = nn.Parameter(weights)
    else: # Bias
      getattr(our_adaattn, param_char).conv_layer.bias = nn.Parameter(weights)
    cnt += 1

  print("AdaAttN pretrained weights loaded successfully")
  return our_adaattn


def load_transformer_weights(weight_path, embed_channels, input_channels):
  trans_weights = torch.load(weight_path, weights_only=True)

  for layer, weights in trans_weights.items():
    print(f"Layer name: {layer} & Weights shape: {weights.shape}")

  our_trans = AuthorTransformer(embed_channels=embed_channels,
                                input_channels=input_channels)

  print()
  for our_layer, our_weights in our_trans.named_parameters():
    print(f"Our Layer: {our_layer} & Our Weights shape: {our_weights.shape}")

  print(our_trans.ada_attn_4_1.f.conv_layer.weight.shape)
  print(getattr(getattr(our_trans, "ada_attn_4_1"), "f").conv_layer.weight.shape)

  cnt = 0
  for layer, weights in trans_weights.items():
    layer_splits = layer.split('.')
    if(layer_splits[0] != "merge_conv"):
      param_num = int(layer_splits[0].split('_')[2])
      block_name = layer_splits[1]

      if(cnt%2 == 0):
        getattr(getattr(our_trans, f"ada_attn_{param_num}_1"), block_name).conv_layer.weight = nn.Parameter(weights)
      else:
        getattr(getattr(our_trans, f"ada_attn_{param_num}_1"), block_name).conv_layer.bias = nn.Parameter(weights)

    else:
      if(cnt%2 == 0):
        getattr(our_trans, "merge_conv_block").weight = nn.Parameter(weights)
      else:
        getattr(our_trans, "merge_conv_block").bias = nn.Parameter(weights)
    cnt += 1

  print("Transformer pretrained weights loaded successfully")
  return our_trans


def load_decoder_weights(weight_path):
  dec_weights = torch.load(weight_path, weights_only=True)

  a = 0
  for layer, weights in dec_weights.items():
    a += 1
    print(f"Layer {layer} & Weight shape: {weights.shape}")

  our_decoder = AuthDecoder()

  print()
  layer_idx = []
  b = 0
  for layer, param in our_decoder.named_parameters():
    b += 1
    layer_splits = layer.split('.')
    layer_idx.append(int(layer_splits[1]))

    print(f"Our layer {layer} & Parameter shape: {param.shape}")
#  print(getattr(our_decoder, "decoder_layer1")[0])

  for idx, (layer, weights) in enumerate(dec_weights.items()):
    dec_layer_splits = layer.split(".")
    which_dec_layer = dec_layer_splits[0].split("_")[-1]
    if(idx%2 == 0):
      # Weights
      getattr(our_decoder, f"decoder_layer{which_dec_layer}")[layer_idx[idx]].weight = nn.Parameter(weights)
    else:
      # Bias
      getattr(our_decoder, f"decoder_layer{which_dec_layer}")[layer_idx[idx]].bias = nn.Parameter(weights)

  print("Decoder weights loaded successfully")
  return our_decoder