# Get the input from the user (both prompt and image)
# Using named_entity_recog.py identify entities and pass on to sentence transformer to find closest COCO_LABELS
# Pass the result along with input image to find out the object window
# Pass that window alongside style image to style the object window portion
# Using Image merge module (give the coordinate statistics, stylized object window, original window and input image) to merge the stylized object with the input image

from pathlib import Path
from AdaAttN.constants import BATCH_SIZE, NUM_WORKERS, DEVICE
from AdaAttN.dataset.nst_dataset import NST_Dataset
from AdaAttN.nst_module import NST
from torch.utils.data import DataLoader
import torch
from named_entity_recog import extract_object_details_from_input, get_similar_coco_labels
from object_detection_recog import extract_object_from_image
from AdaAttN.util_functions import save_nst_image


input_prompt = "style this dog"
entity_list = extract_object_details_from_input(input_prompt)
print(f"Extracted entities: {entity_list}")

recog_list = get_similar_coco_labels(entity_list)
print(f"Objects to be recognized: {recog_list}")

INPUT_PATH = Path("input_images")
OUTPUT_PATH = Path("output_images")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

accepted_formats = ['png', 'jpg', 'jpeg']
content_input = []
style_input = []

for fm in accepted_formats:
    content_input.extend(list(INPUT_PATH.glob(f"*_input.{fm}")))
    style_input.extend(list(INPUT_PATH.glob(f"*_style.{fm}")))

for obj_name in recog_list:
    extracted_object_img = extract_object_from_image(str(content_input[0]), obj_name)

    extracted_object_img["cropped_img"].show()
    cropped_path = f"{str(content_input[0]).split(".")[0]}_cropped.jpg"
    extracted_object_img["cropped_img"].save(cropped_path)
    content_input[0] = cropped_path

print(content_input, style_input)

img_dataset = NST_Dataset(mode="TG-NST",
                           data=[[str(content_input[0]), str(style_input[0])]])

img_dataloader = DataLoader(img_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=True)
nst_model = NST().to(DEVICE)

for idx, (name, cropped_content, style_batch) in enumerate(iter(img_dataloader)):
    cropped_content = cropped_content.to(DEVICE)
    style_batch = style_batch.to(DEVICE)

    nst_model.eval()
    with torch.no_grad():
        stylized_crop_results = nst_model(cropped_content, style_batch)
        for idx in range(stylized_crop_results.shape[0]):
            tensor = stylized_crop_results[idx].unsqueeze(dim=0)
            save_nst_image(tensor, OUTPUT_PATH / f"{name[idx]}_nst_result.jpg" )
print("Stylized object created successfully")



