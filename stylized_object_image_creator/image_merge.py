from PIL import Image
import numpy as np
import cv2
from object_detection_recog import save_coords_draw_bounding_box

def merge_image_to_original(detections, content_img, nst_image_path, name):
    detection_supervision_coords = detections["xyxy_bounding"]
    cropped_img_np = np.asarray(detections["cropped_img"])
    print(cropped_img_np.shape)

    nst_img_pil = Image.open(nst_image_path)
    nst_img_np = np.asarray(nst_img_pil)

    nst_img_resized = cv2.resize(nst_img_np, [cropped_img_np.shape[1],cropped_img_np.shape[0]], interpolation=cv2.INTER_LINEAR)
    Image.fromarray(nst_img_resized).show()
    Image.open(nst_image_path).show()
    img = Image.fromarray(save_coords_draw_bounding_box(np.asarray(content_img), detections, nst_img_resized, name))

    img.save('test.jpg')