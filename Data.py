# preprocess_data.py

import os
import cv2
import numpy as np

def preprocess_data(input_dir, output_dir):
    # Iterate through images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image = cv2.imread(os.path.join(input_dir, filename))
            # Resize to 256x256 pixels
            image_resized = cv2.resize(image, (256, 256))
            # Normalize to range [0, 1]
            image_normalized = image_resized / 255.0
            # Save the preprocessed image
            cv2.imwrite(os.path.join(output_dir, filename), image_normalized)

# Example usage
input_dir = "data/raw_images"
output_dir = "data/preprocessed_images"
preprocess_data(input_dir, output_dir)
