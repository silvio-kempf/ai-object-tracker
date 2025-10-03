# AI Object Tracker — YOLOv8 + IOU

Real-time **multi-object tracking** in Python.  
Combines a **state-of-the-art detector (YOLOv8)** with a **custom IOU-based tracker** for persistent IDs across frames.

![demo-gif](assets/demo.gif)  
*(Webcam demo: persons tracked with stable IDs)*

---

## Features
- **Deep Learning Detector** — Pretrained YOLOv8n (COCO, 80 classes)  
- **Custom Tracker** — Lightweight IOU-based multi-object tracking  
- **Real-Time** — Runs live on webcam or video stream  
- **Configurable** — `--conf`, `--iou_thr`, `--max_age`, `--class_filter`  
- **Demo Ready** — CLI, overlay with FPS + track IDs  

---

## Quickstart

```bash
# Clone & Setup
git clone https://github.com/YOURNAME/ai-object-tracker.git
cd ai-object-tracker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on webcam
python track_realtime.py --source 0 --device mps   # Apple Silicon
python track_realtime.py --source 0 --device cuda  # NVIDIA GPU

# Run on video
python track_realtime.py --source data/demo.mp4
```
## Example

```bash
# Only track persons and cars
python track_realtime.py --source 0 --class_filter person,car
```
Output overlay:
* Green bounding boxes
* Stable Track IDs
* FPS + parameter values

## How it works
### 1. YOLOv8 Detection:
* Outputs bounding boxes + class + confidence per frame
### 2. IOU-Tracking:
* Matches detections with existing tracks using Intersection over Union
* If IoU ≥ threshold → same ID
* If no match → new track
* If a track is not updated for max_age frames → removed
### Intersection over Union (IoU):
$$
IoU = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

## Limitations
* IDs reset when objects leave & re-enter the frame
* No ReID (appearance features)
* No motion model (Kalman)


​	
 
