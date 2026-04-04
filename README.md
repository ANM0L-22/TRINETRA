# 🔱 Trinetra — Intelligent Traffic Surveillance System

AI-powered traffic monitoring: vehicle detection, ANPR, violation detection, and density analytics.

---

## Project Structure

```
trinetra/
├── configs/
│   ├── config.yaml          ← Master config (edit this)
│   └── dataset.yaml         ← YOLOv8 dataset config
├── data/
│   ├── raw/                 ← Downloaded raw datasets
│   └── processed/
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── models/
│   ├── weights/             ← Trained .pt files go here
│   └── exports/             ← ONNX / TensorRT exports
├── src/
│   ├── detection/           ← Vehicle detector (YOLOv8)
│   ├── ocr/                 ← Number plate recognition
│   ├── violation/           ← Helmet, seatbelt, phone detection
│   ├── tracking/            ← DeepSORT vehicle tracking
│   ├── analytics/           ← Density, congestion, reporting
│   └── utils/               ← Shared helpers
├── scripts/
│   ├── prepare_dataset.py   ← Step 1: Get data
│   ├── train.py             ← Step 2: Train model (on Colab)
│   └── evaluate.py          ← Step 3: Test & run inference
├── notebooks/               ← Colab training notebooks
├── results/                 ← Output images, videos, reports
└── tests/                   ← Unit tests
```

---

## Quickstart (Windows + VS Code)

### 1. Setup environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare dataset

```bash
# Option A: Synthetic sample (no API key, pipeline test only)
python scripts/prepare_dataset.py --source sample --n-samples 300

# Option B: Real data from Roboflow (recommended)
python scripts/prepare_dataset.py --source roboflow --api-key YOUR_KEY
```

### 3. Train the model

**On Colab (recommended — free GPU):**
- Upload the project folder to Google Drive
- Open `notebooks/trinetra_training.ipynb` in Colab
- Set runtime to GPU (T4)
- Run all cells

**Local quick test (CPU, slow):**
```bash
python scripts/train.py --epochs 2 --batch 4 --device cpu
```

### 4. Run inference locally

```bash
# On an image
python scripts/evaluate.py --mode image --input path/to/image.jpg

# On a video
python scripts/evaluate.py --mode video --input path/to/video.mp4 --show

# Evaluate metrics on val set
python scripts/evaluate.py --mode eval
```

---

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0  | car | Passenger cars |
| 1  | bus | Buses, minibuses |
| 2  | bike | Motorcycles, scooters, 2-wheelers |
| 3  | truck | Trucks, lorries |
| 4  | auto_rickshaw | Indian auto-rickshaws |
| 5  | pedestrian | People on foot |

---

## Development Roadmap

- [x] **Phase 1** — Vehicle detection & classification (YOLOv8)
- [ ] **Phase 2** — Number plate recognition (OCR/ANPR)
- [ ] **Phase 3** — Violation detection (helmet, seatbelt, phone)
- [ ] **Phase 4** — Vehicle tracking (DeepSORT)
- [ ] **Phase 5** — Traffic density & congestion analytics
- [ ] **Phase 6** — Dashboard & reporting (Streamlit)

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| ultralytics | YOLOv8 training & inference |
| opencv-python | Video/image processing |
| paddleocr | Number plate OCR |
| deep-sort-realtime | Multi-object tracking |
| streamlit | Live dashboard |
| roboflow | Dataset management |

---

## Getting a Roboflow API Key

1. Sign up free at [roboflow.com](https://roboflow.com)
2. Go to Settings → API Keys
3. Copy your key and use with `--api-key YOUR_KEY`

Recommended public datasets:
- `vehicles-q0x2v` — General vehicle detection
- Search "Indian traffic" on [universe.roboflow.com](https://universe.roboflow.com)

Problem Description
TRINETRA is an AI-powered intelligent traffic monitoring system designed to improve road safety and traffic management using computer vision and deep learning. The system analyzes real-time CCTV/video feed to detect and classify vehicles such as cars, buses, bikes, and pedestrians, and performs automated vehicle counting for traffic analytics.

It also identifies traffic rule violations like helmet absence, seatbelt non-compliance, wrong-side driving, mobile phone usage while driving, and detects blackened/tampered number plates using OCR-based number plate recognition. TRINETRA generates time-based reports such as vehicle flow in one hour, violation logs, and traffic density status for congestion analysis.

By automating surveillance and reporting, the system reduces manual dependency, increases rule enforcement efficiency, and supports smarter traffic control for smart-city environments.
Objectives
To automatically detect and classify vehicles such as cars, buses, bikes, and pedestrians using AI.

To perform real-time OCR-based number plate recognition and track vehicle movement over time.

To detect traffic violations including:

Helmet violation

Seatbelt violation

Wrong-side driving

Mobile phone usage while driving

Blackened / tampered number plates

To analyze traffic density and identify congestion in real time.

To provide accurate traffic analytics for law enforcement and smart city planning.

To reduce dependency on manual traffic surveillance and improve road safety.
Proposed Solution
The proposed system, TRINETRA, is an AI-driven intelligent traffic monitoring platform that uses computer vision, deep learning, and OCR to analyze real-time traffic footage from CCTV cameras.

The system performs vehicle detection and classification, reads number plates using OCR, and monitors traffic violations such as helmet absence, seatbelt non-usage, wrong-side driving, phone usage, and blackened number plates. It also tracks vehicle movement over time to analyze traffic flow and congestion.

This unified approach enables real-time enforcement, congestion management, and emergency response, making roads safer and traffic systems smarter.
Technology Stack
Python, OpenCV, YOLO, OCR, Deep Learning, Flask/FastAPI, CCTV Cameras
Problem Type
Deeptech And System Based
