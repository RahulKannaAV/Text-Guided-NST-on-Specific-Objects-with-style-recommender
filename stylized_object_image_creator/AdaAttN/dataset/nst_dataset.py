import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


CUSTOM_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((512, 512)),
])

class NST_Dataset(Dataset):
    def __init__(self, content_image_dir="", style_image_dir="", mode="normal", data=[], transform=CUSTOM_TRANSFORM):
        self.data_transform = transform
        if(mode != "TG-NST"):
            self.CONTENT_IMAGES_PATH = content_image_dir
            self.STYLE_IMAGES_PATH = style_image_dir
            self.accepted_img_formats = ["png", "jpg", "jpeg"]
            self.image_list = []
            self.content_list = []
            self.style_list = []


            for fmt in self.accepted_img_formats:
                if(content_image_dir != ""):
                    content_image_name_fmt = list(content_image_dir.glob(f"*.{fmt}"))
                    self.content_list.extend(content_image_name_fmt)
        
                if(style_image_dir != ""):
                    style_image_name_fmt = list(style_image_dir.glob(f"*\\*.{fmt}"))
                    self.style_list.extend(style_image_name_fmt)

            for content in self.content_list:
                self.image_list.extend([[content, style] for style in self.style_list])
        else:
            self.image_list = data

    
    def __len__(self):
        print(self.image_list)
        return len(self.image_list)

    def __getitem__(self, idx):
        if(idx < len(self.image_list)):
            print(self.image_list)
            content_image = self.data_transform(Image.open(self.image_list[idx][0]).convert('RGB'))
            style_image = self.data_transform(Image.open(self.image_list[idx][1]).convert('RGB'))

            result_image_name = f"{str(self.image_list[idx][0]).split('\\')[-1].split('.')[0]}_{str(self.image_list[idx][1]).split('\\')[-1].split('.')[0]}"

            return [result_image_name, content_image, style_image]

        else:
            return []    

