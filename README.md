# ADAS Pedestrian Automatic Emergency Braking (AEB)

## Overview

This project implements a computer vision–based Advanced Driver Assistance System (ADAS) focused on pedestrian safety.  
A YOLO deep learning model is trained to detect pedestrians in real-world driving scenes and forms the perception layer for an Automatic Emergency Braking (AEB) system.

The system is designed as a modular research pipeline:
- Detection
- Risk estimation
- Temporal decision logic
- Future AEB integration

This repository contains the perception + inference stage.

---

## Quick Start

git clone <repo-url>
cd ADAS-Pedestrian-AEB
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/detection/infer.py --source data/sample/test.jpg

---

## Features

- YOLO-based pedestrian detection
- Custom-trained model
- Image & video inference pipeline
- Saved evaluation artifacts
- Reproducible training setup
- Modular code structure
- Ready for integration with control logic

---

## Repository Structure

```
configs/        → dataset + model configs
data/           → sample demo inputs (no raw dataset)
models/         → trained weights
src/            → core source code
notebooks/      → training & experimentation notebooks
results/        → logs, training plots, predictions
reports/        → documentation & writeups
```

Large datasets are excluded from version control.

---

## Setup

### 1. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements-local.txt
```

---

## Training

Training was performed in Google Colab using GPU.

Model used:
```
YOLOv8n
```

Dataset:
```
BDD pedestrian subset
```

Training artifacts are saved in:

```
results/train_logs/
```

---

## Inference Demo

Run pedestrian detection on an image:

```
python src/detection/infer.py --source data/sample/test.jpg
```

Run detection on a video:

```
python src/detection/infer.py --source data/sample/test_video.mp4
```

Outputs are saved to:

```
results/inference/
```

Bounding boxes are automatically drawn.

---

## Demo Output

Example pedestrian detection:

results/inference/images/image_output.jpg
results/inference/videos/video_output.mp4

---

## Results

The trained model achieves:

- mAP50 ≈ 0.61
- mAP50-95 ≈ 0.30
- Precision ≈ 0.73
- Recall ≈ 0.52

Evaluation plots include:

- Precision–Recall curves
- Loss curves
- Confusion matrix
- Label distribution

These are stored in:

```
results/train_logs/
```

---

## Hardware Used

Training performed on NVIDIA Tesla T4 GPU (Google Colab)
Inference runs on CPU or GPU

---

## Reproducibility

Training notebook:
notebooks/yolo_training_pipeline.ipynb

Dataset config:
configs/data.yaml

Model weights:
models/best.pt

---

## Future Work

- Temporal pedestrian tracking
- Collision risk prediction
- Confidence-aware AEB logic
- Real-time system integration
- Edge deployment optimization

---

## Team

- Atharv
- Chehek
- Arnav
- Debangan

---

## License

Research & educational use only.
