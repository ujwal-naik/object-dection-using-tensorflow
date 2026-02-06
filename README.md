## _Object Detection Visualization Utilities_
This project provides Python utilities for visualizing object detection results, including classification and localization [].

**Features**
Bounding Box Drawing: Functions to overlay bounding boxes on image arrays using PIL [].

Coordinate Handling: Supports both normalized and absolute coordinates [].

Dataset Integration: Includes helpers to convert TensorFlow datasets to NumPy arrays for easier processing [].

## 🏗️ Installation

Ensure you have the following libraries installed []:

`numpy`

`matplotlib`

`Pillow`

`tensorflow`

`tensorflow-datasets`

##Usage
The core function draw_bounding_boxes_on_image_array takes an image and box coordinates to generate a visual representation of detections [].


## 🚀 Setup
```python
import numpy as np
from visualisation_utilies_definition import draw_bounding_boxes_on_image_array

# Create a sample image and bounding box
sample_image = np.zeros((75, 75, 3), dtype=np.uint8)
boxes = np.array([[0.1, 0.1, 0.5, 0.5]]) # [ymin, xmin, ymax, xmax]

# Visualize
result_img = draw_bounding_boxes_on_image_array(sample_image, boxes, color=['red'])
```
