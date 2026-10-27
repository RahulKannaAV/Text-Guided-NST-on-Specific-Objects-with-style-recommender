import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
from PIL import Image

model = RFDETRMedium()

def extract_object_from_image(image_url: str,
                              object_name: str):
  detections = model.predict(image_url, threshold=0.8)

  labels = [f"{COCO_CLASSES[c_id]}" for c_id in detections.class_id]

  box_annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
  label_annotated_image = sv.LabelAnnotator().annotate(box_annotated_image, detections, labels)

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

import cv2

## Function which takes the coordinates, saves it and draws the bounding box for desired object
def save_coords_draw_bounding_box(image, detection_data, obj_name):
  img_copy = image.copy()
  det_coords = detection_data.xyxy
  det_objects = detection_data.data["class_name"]

  coord_dict = dict()
  for idx in range(len(det_objects)):
    coord_dict[det_objects[idx]] = dict()

    obj_coords = det_coords[idx]
    coord_dict[det_objects[idx]]["top_left"] = (round(obj_coords[0]), round(obj_coords[1]))
    coord_dict[det_objects[idx]]["bottom_right"] = (round(obj_coords[2]), round(obj_coords[3]))

  print(coord_dict)

  # Draw bounding box for particular object
  if obj_name in coord_dict:
    req_obj_coords = coord_dict[obj_name]
    print(f"Custom Function bounding box coordinates: {req_obj_coords}")
    x1, y1 = req_obj_coords['top_left']
    x2, y2 = req_obj_coords['bottom_right']
    cv2.rectangle(img_copy, req_obj_coords['top_left'], req_obj_coords["bottom_right"], (255, 0, 0), 1)

    cropped_img = img_copy[y1+1: y2, x1+1: x2]
    plt.imshow(img_copy)
    plt.axis("off")
    return cropped_img

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