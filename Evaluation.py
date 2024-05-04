# evaluate_model.ipynb
# (Jupyter notebook for evaluating the model's performance)
# Code can be written in a Jupyter notebook

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load segmentation results and ground truth images
seg_result = cv2.imread("evaluation/segmentation_result.png")
ground_truth = cv2.imread("evaluation/ground_truth.png")

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(seg_result, cv2.COLOR_BGR2RGB))
plt.title("Segmentation Result")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth")
plt.axis("off")
plt.show()


