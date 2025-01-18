import cv2
from ultralytics import YOLO
import gradio as gr

COLORS = {
    "checked": (47, 86, 233),
    "unchecked": (110, 255, 255),
}

DETECTION_MODEL = YOLO("runs/detect/train/weights/best.pt")

def run_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    results = DETECTION_MODEL.predict(source=image)
    # Since is a single image we only have one element
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("No elements of interest were found in the image")
        return image

    for box in boxes:
        cls = list(COLORS)[int(box.cls)]

        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

        # Draw bounding box of object
        image = cv2.rectangle(img=image,
                              pt1=start_box,
                              pt2=end_box,
                              color=COLORS[cls],
                              thickness=3)

    return image


iface = gr.Interface(fn=run_detection,
                     inputs=gr.Image(label="Upload image here", type="filepath"),
                     outputs="image")
iface.launch()
