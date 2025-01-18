import cv2
import numpy as np
import gradio as gr

COLORS = {
    "checked": (47, 86, 233),
    "unchecked": (110, 255, 255),
}

def convert_to_binary(img):
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, 127, 255, cv2.THRESH_BINARY)
    return ~img_bin

def get_merged_lines(img_bin):
    line_min_width = 15

    horizontal_kernel = np.ones((1, line_min_width), np.uint8)
    vertical_kernel = np.ones((line_min_width, 1), np.uint8)

    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)

    return horizontal_lines | vertical_lines

def dilate_lines(lines, shape):
    kernel = np.ones(shape, np.uint8)
    return cv2.dilate(lines, kernel, iterations=1)

def run_detection(image_path):
    image = cv2.imread(image_path)
    img_bin = convert_to_binary(image)
    merged_lines = get_merged_lines(img_bin)
    merged_lines = dilate_lines(merged_lines, (5, 5))

    stats = cv2.connectedComponentsWithStats(~merged_lines, connectivity=4, ltype=cv2.CV_32S)[2]

    # Ignored background and small noise components (dropped 0 and 1 in stats)
    for x, y, w, h, area in stats[2:]:

        # This is an area boundary calculated for check-boxes in the provided document (residential appraisal report)
        if 200 < area <= 500:

            # Crop the region of interest from the binary image
            roi = img_bin[y:y + h, x:x + w]

            # Count the number of non-zero (white) pixels inside the bounding box (filled)
            filled_pixels = cv2.countNonZero(roi)

            # Calculate the ratio of filled pixels to the total pixels in the bounding box
            filled_ratio = filled_pixels / (w * h)

            # DEBUG: print components found so far with stats
            # print(f"Component at ({x},{y}), Area: {area}, Filled Pixels: {filled_pixels}, Filled Ratio: {filled_ratio:.2f}")

            # Categorize if checked or unchecked based on the filled_ratio assigning a color to it
            color = COLORS["unchecked"]

            # Based on several tests conducted on documents in this format, the most reliable ratio found was 0.2
            if filled_ratio > 0.2:
                color = COLORS["checked"]

            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    return image

if __name__ == "__main__":
    iface = gr.Interface(fn=run_detection,
                         inputs=gr.Image(label="Upload image here", type="filepath"),
                         outputs="image")
    iface.launch()