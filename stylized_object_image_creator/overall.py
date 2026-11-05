# Get the input from the user (both prompt and image)
# Using named_entity_recog.py identify entities and pass on to sentence transformer to find closest COCO_LABELS
# Pass the result along with input image to find out the object window
# Pass that window alongside style image to style the object window portion
# Using Image merge module (give the coordinate statistics, stylized object window, original window and input image) to merge the stylized object with the input image

from pathlib import Path
import streamlit as st
from write_dataset import write_new_data
from AdaAttN.constants import BATCH_SIZE, NUM_WORKERS, DEVICE
from AdaAttN.dataset.nst_dataset import NST_Dataset
from AdaAttN.nst_module import NST
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from named_entity_recog import extract_object_details_from_input, get_similar_coco_labels
from object_detection_recog import extract_object_from_image
from AdaAttN.util_functions import save_nst_image
from image_merge import merge_image_to_original
from evaluation_metrics import calc_final_score
from PIL import Image


def all_results(input_prompt, content_pil_data, style_pil_data):
    print(content_pil_data, style_pil_data, input_prompt)

    entity_list = extract_object_details_from_input(input_prompt)

    recog_list = get_similar_coco_labels(entity_list)


    for obj_name in recog_list:
        extracted_object_img = extract_object_from_image(content_pil_data["img"], obj_name, streamlit_use=True)
    
    img_dataset = NST_Dataset(mode="streamlit",
                              s_content=content_pil_data["name"],
                              s_style=style_pil_data["name"],
                              data=[[extracted_object_img["cropped_img"], style_pil_data["img"]]])

    img_dataloader = DataLoader(img_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

    print(img_dataset, len(img_dataset), img_dataset[0])
    nst_model = NST().to(DEVICE)

    for idx, (name, cropped_content, style_batch) in enumerate(iter(img_dataloader)):
        cropped_content = cropped_content.to(DEVICE)
        style_batch = style_batch.to(DEVICE)

        nst_model.eval()
        with torch.no_grad():
            stylized_crop_results = nst_model(cropped_content, style_batch)
            tensor = stylized_crop_results[0].unsqueeze(dim=0)

            final_result = merge_image_to_original(detections=extracted_object_img, 
                                content_img=content_pil_data["img"],
                                nst_image_path=tensor,
                                name=recog_list[0],
                                mode="streamlit")
    
            st.session_state.results = {
                "cropped_portion": cropped_content,
                "stylized_portion": tensor,
                "merged": final_result
            }
            print(cropped_content.shape, tensor.shape, final_result)
    

