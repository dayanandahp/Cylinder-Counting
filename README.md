
# 🚛 Cylinder Counting from Moving Truck (Top-View)

## Problem Statement
In industrial logistics, accurately counting gas cylinders loaded on moving trucks is critical
for inventory management and safety compliance. This project detects and counts cylinders
from a top-view video using a deep learning–based computer vision pipeline.

## Approach
- **YOLOv8** for cylinder detection
- **SORT tracking** to avoid double counting
- **Virtual counting line** to ensure accurate counts
- Robust against:
  - Motion blur
  - Partial occlusion
  - Lighting variations

## Assumptions
- Cylinders are visible from the top view
- Cylinders move consistently with the truck
- Camera is stationary

## Tech Stack
- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy

## Installation
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python src/main.py --video data/Task.mp4
```

## Output
- Annotated video with bounding boxes & IDs
- Final cylinder count printed in terminal

## Resume Highlights
✔ End-to-end CV pipeline  
✔ YOLO + Object Tracking  
✔ Industry-grade folder structure  
✔ Real-world robustness  

---
**Author:** Dayananda H P  
