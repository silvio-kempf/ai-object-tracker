# AI Object Tracker — YOLOv8 + IoU

A real-time **multi-object tracking** system in Python that combines state-of-the-art **YOLOv8 detection** with a **custom IoU-based tracker** for persistent object identification across video frames.

<img src="assets/demo.gif" alt="Demo" width="600"/>

---

## Features

- **State-of-the-art Detection**: Pre-trained YOLOv8n model (COCO dataset, 80 classes)
- **Smart Tracking**: Custom IoU-based multi-object tracking algorithm
- **Real-time Performance**: Optimized for live webcam and video stream processing
- **Highly Configurable**: Adjustable confidence, IoU threshold, track age, and class filtering
- **Rich Visualization**: Real-time overlay with FPS, track IDs, and parameter values
- **Cross-platform**: Supports macOS (MPS), NVIDIA GPU (CUDA), and CPU

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Camera/webcam (for real-time tracking)
- Recommended: GPU for better performance

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/silvio-kempf/ai-object-tracker.git
cd ai-object-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

```bash
# For macOS with Apple Silicon
# PyTorch MPS support is included by default

# For NVIDIA GPUs (Linux/Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Basic Usage

```bash
# Live tracking with webcam
python track_realtime.py --source 0

# Process video file
python track_realtime.py --source data/demo.mp4

# Detection only (no tracking)
python detect_only.py --source 0
```

### Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Video source (0 for webcam, path for video) |
| `--conf` | Confidence threshold for detections | `0.25` |
| `--iou_thr` | IoU threshold for track matching | `0.5` |
| `--max_age` | Maximum frames before deleting tracker | `30` |
| `--class_filter` | Comma-separated classes to track | `all` |
| `--device` | Device (cpu, cuda, mps) | `auto` |
| `--save` | Save output video to file |
| `--show` | Display video window | `True` |

### Examples

```bash
# Track only persons and cars with higher confidence
python track_realtime.py --source 0 --class_filter person,car --conf 0.4

# GPU-accelerated processing
python track_realtime.py --source data/demo.mp4 --device cuda

# Save tracked video to file
python track_realtime.py --source data/demo.mp4 --save output_tracked.mp4

# Conservative tracking (higher IoU threshold)
python track_realtime.py --source 0 --iou_thr 0.7
```

### Output Visualization

The system displays:
- **Green bounding boxes** around tracked objects
- **Stable track IDs** that persist across frames
- **FPS counter** showing real-time performance
- **Parameter values** in the corner

---

## 🔬 How It Works

### 1. **YOLOv8 Detection**
- Analyzes each frame using a pre-trained neural network
- Outputs bounding boxes, class labels, and confidence scores
- Supports all 80 COCO classes (person, car, dog, etc.)

### 2. **IoU-based Tracking**
- **Track Matching**: Compares new detections with existing tracks using Intersection over Union
- **Track Creation**: If IoU < threshold → creates new track with unique ID
- **Track Update**: If IoU ≥ threshold → updates existing track with same ID
- **Track Deletion**: Removes tracks that haven't been updated for `max_age` frames

### 3. **Intersection over Union (IoU)**
The IoU metric measures the overlap between two bounding boxes:

$$IoU = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

Where:
- `|A ∩ B|` = Intersection area between boxes A and B
- `|A|` = Area of box A
- `|B|` = Area of box B

---

## Limitations

- **ID Reset**: Track IDs reset when objects completely leave and re-enter the frame
- **No Appearance Matching**: Doesn't use visual features for re-identification
- **No Motion Prediction**: No Kalman filtering or motion model for track prediction
- **Performance**: Real-time performance depends on hardware (CPU/GPU)

---

## 🛠️ Architecture

```
Frame Input → YOLOv8 Detection → IoU Matching → Track Update → Visualization
     ↓              ↓                ↓            ↓            ↓
Webcam/Video → Bounding Boxes → Track IDs → Persistent IDs → Live Display || Saved Video
```

---

## 📁 Project Structure

```
ai-object-tracker/
├── track_realtime.py    # Main tracking script
├── detect_only.py       # Detection without tracking
├── iou_tracker.py       # Custom IoU tracking algorithm
├── requirements.txt     # Python dependencies
├── yolov8n.pt          # Pre-trained YOLOv8 model
├── data/
│   └── demo.mp4        # Demo video file
└── README.md           # This file
```



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
