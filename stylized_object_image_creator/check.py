from PIL import Image
from object_detection_recog import extract_object_from_image
import numpy as np


extracted_object_img = extract_object_from_image("input_images\man_dog_input.jpg", "dog")
