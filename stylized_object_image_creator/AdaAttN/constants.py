import torch
from pathlib import Path

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" 
CONTENT_IMG_PATH = Path("data/contents")
STYLE_IMG_PATH = Path("data/styles")
RESULT_PATH = Path("nst_results")
BATCH_SIZE = 2
NUM_WORKERS = 0