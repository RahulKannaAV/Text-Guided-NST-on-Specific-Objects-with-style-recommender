import supervision as sv
from rfdetr import RFDETRMedium, RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = RFDETRMedium()
seg_model = RFDETRSegMedium()

def extract_object_from_image(image_url: str,
                              object_name: str):
  content_img = Image.open(image_url).convert('RGB')
  cont_numpy = np.asarray(content_img)
  detections = model.predict(image_url, threshold=0.8)
  seg_detections = seg_model.predict(np.array(content_img), threshold=0.8)
  masks = seg_detections.mask

  labels = [f"{COCO_CLASSES[c_id]}" for c_id in detections.class_id]

  box_annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
  label_annotated_image = sv.LabelAnnotator().annotate(box_annotated_image, detections, labels)

  if masks is not None and len(masks) > 0:
    # Combine all instance masks into a single foreground mask (logical OR)
    combined_mask = np.any(masks, axis=0)  # Shape: (H, W), boolean

    alpha = (combined_mask * 255).astype(np.uint8)
    transparent_fg_alpha_channel = Image.fromarray(alpha, mode="L")
    transparent_fg = content_img.copy()
    transparent_fg.putalpha(transparent_fg_alpha_channel)
    transparent_fg.save("foreground_transparent.png")

  detect_dict = dict()
  for idx in range(len(detections.xyxy)):
    label_name = detections.data["class_name"][idx]
    detect_dict[label_name] = dict()

    detect_dict[label_name]["xyxy_bounding"] = [round(val) for val in detections.xyxy[idx]]

  for obj in detect_dict:
    obj_xyxy = detect_dict[obj]["xyxy_bounding"]
    crop_img = sv.crop_image(Image.open(image_url), obj_xyxy)

    detect_dict[obj]["cropped_img"] = crop_img
    

  print(f"Supervision coordinates for {object_name} : {detect_dict[object_name]["xyxy_bounding"]}")
  return detect_dict[object_name] if object_name in detect_dict else None

def show_all_detection(image_url):
  detections = model.predict(image_url, threshold=0.8)

  labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

  box_annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
  label_annotated_image = sv.LabelAnnotator().annotate(box_annotated_image, detections, labels)


  return (label_annotated_image, detections)

## Function which takes the coordinates, saves it and draws the bounding box for desired object
def save_coords_draw_bounding_box(image, detection_data, nst_data, obj_name):
  img_copy = image.copy()
  det_coords = detection_data["xyxy_bounding"]
  det_objects = [obj_name]

  coord_dict = dict()
  for idx in range(1):
    coord_dict[det_objects[idx]] = dict()

    obj_coords = det_coords
    coord_dict[det_objects[idx]]["top_left"] = (round(obj_coords[0]), round(obj_coords[1]))
    coord_dict[det_objects[idx]]["bottom_right"] = (round(obj_coords[2]), round(obj_coords[3]))

  print(coord_dict)

  # Draw bounding box for particular object
  if obj_name in coord_dict:
    req_obj_coords = coord_dict[obj_name]
    print(f"Custom Function bounding box coordinates: {req_obj_coords}")
    y1, x1 = req_obj_coords['top_left']
    y2, x2 = req_obj_coords['bottom_right']
    #cv2.rectangle(img_copy, req_obj_coords['top_left'], req_obj_coords["bottom_right"], (255, 0, 0), 1)

#   cropped_img = img_copy[y1+1: y2, x1+1: x2]
    print(f"NST data shape: {nst_data.shape}")
    print(f"Shape: {img_copy[x1+3: x2-3, y1+3: y2-3,  :].shape}")
    print(f"Width: {abs(x1 - x2)} and {x1}, {x2}")
    print(f"Height: {abs(y1 - y2)} and {y1}, {y2}")
    img_copy[x1+2: x2-2, y1+2: y2-2, :] = nst_data[2:-2, 2:-2, :]
    plt.imshow(img_copy)
    plt.axis("off")
    return img_copy

  else:
    print(f"Object {obj_name} undetected")


## USAGE
"""
import matplotlib.pyplot as plt

obj_data = extract_object_from_image("man_dog.jpg","dog")
print(obj_data)
if obj_data is not None:
  obj_data["cropped_img"].show()
  plt.axis("off")
else:
  print("Not found")

dog_img = save_coords_draw_bounding_box(img_rgb, det_data, "sports ball")

plt.imshow(dog_img)
"""