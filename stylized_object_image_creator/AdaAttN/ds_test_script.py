from dataset.nst_dataset import NST_Dataset
from torch.utils.data import DataLoader
import torch
import os
from constants import CONTENT_IMG_PATH, STYLE_IMG_PATH, BATCH_SIZE, NUM_WORKERS

img_dataset = NST_Dataset(CONTENT_IMG_PATH, STYLE_IMG_PATH)

img_dataloader = DataLoader(img_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

for batch_idx, (result_name, content_batch, style_batch) in enumerate(iter(img_dataloader)):
    print(f"Batch: {batch_idx+1} Result Name: {result_name} Content Batch dim: {content_batch.shape} Style Batch dim: {style_batch.shape}")
    