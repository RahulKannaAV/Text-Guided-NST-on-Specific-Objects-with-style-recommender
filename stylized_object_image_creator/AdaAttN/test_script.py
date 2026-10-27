from nst_module import NST
from pathlib import Path
import torch
from dataset.nst_dataset import NST_Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image
from util_functions import save_nst_image
from image_encoder import get_encoder_instance
from constants import DEVICE, CONTENT_IMG_PATH, STYLE_IMG_PATH, BATCH_SIZE, RESULT_PATH


torch.manual_seed(58)
nst_instance = NST().to(DEVICE)

cs_img_dataset = NST_Dataset(CONTENT_IMG_PATH, STYLE_IMG_PATH)

cs_img_dataloader = DataLoader(cs_img_dataset,
                                batch_size=BATCH_SIZE,
                                pin_memory=True)

for batch_idx, (result_name, content_batch, style_batch) in enumerate(iter(cs_img_dataloader)):
    print(f"BATCH {batch_idx + 1} is getting processed...")
    nst_instance.eval()
    with torch.no_grad():
            content_batch = content_batch.to(DEVICE)
            style_batch = style_batch.to(DEVICE)
            result_tensor = nst_instance(content_batch, style_batch).to(DEVICE)
            for idx in range(BATCH_SIZE):
                tensor = result_tensor[idx].unsqueeze(dim=0)
                save_nst_image(tensor,RESULT_PATH / f"{result_name[idx]}.jpg")


