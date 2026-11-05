from PIL import Image
from object_detection_recog import extract_object_from_image
import numpy as np
from torchvision.transforms import v2
import torch

#extracted_object_img = extract_object_from_image("input_images\man_dog_input.jpg", "dog")
totensor = v2.ToImage()

pil_image = Image.open("input_images/warli5_style.jpg").convert('RGB')
img_tensor = totensor(pil_image)

print(img_tensor.min(), img_tensor.max())

print(img_tensor, img_tensor.shape, torch.bincount(img_tensor.flatten(start_dim=0, end_dim=-1)))