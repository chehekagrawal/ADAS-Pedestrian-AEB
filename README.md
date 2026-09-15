# A Physics-Aware Vision-Based Automatic Emergency Braking System for Vulnerable Road User Protection

> **Physics-aware vision-based Automatic Emergency Braking (AEB) system for vulnerable road users — couples YOLOv8n perception with Pacejka tyre dynamics, brake thermodynamics, driver monitoring, and Euro NCAP/HIC validation. Dept. of Mechanical and Industrial Engineering, IIT Roorkee.**

<!-- Repository Search Discovery Topics -->
[![Topic: adas](https://shields.io)](https://github.com)
[![Topic: autonomous-driving](https://shields.io)](https://github.com)
[![Topic: computer-vision](https://shields.io)](https://github.com)
[![Topic: yolov8](https://shields.io)](https://github.com)
[![Topic: object-detection](https://shields.io)](https://github.com)
[![Topic: pedestrian-detection](https://shields.io)](https://github.com)
[![Topic: autonomous-emergency-braking](https://shields.io)](https://github.com)
[![Topic: vehicle-dynamics](https://shields.io)](https://github.com)
[![Topic: driver-monitoring](https://shields.io)](https://github.com)
[![Topic: euro-ncap](https://shields.io)](https://github.com)
[![Topic: pytorch](https://shields.io)](https://github.com)

---

A monocular-vision Automatic Emergency Braking (AEB) pipeline that goes beyond the standard kinematic stopping-distance model (`d = v²/2µg`) by coupling real-time perception with non-linear vehicle dynamics, brake thermodynamics, driver monitoring, and biomechanical injury modeling.

Full method, derivations, and evaluation: see the paper.

## Highlights
 
- **Perception:** YOLOv8n on BDD100K (4 VRU-critical classes) — 40.0% mAP@0.5, 19.2 fps on Tesla T4
- **Dynamics:** Pacejka '92 + ABS + load transfer — corrects kinematic stopping-distance underestimates of 15–30%
- **Thermodynamics:** rotor thermal fade dynamically dilates the TTC threshold (up to +3.0s)
- **Driver state:** EAR + head pose scale reaction buffers from 0.7s to 1.6s, with microsleep override
- **Decision:** dual-channel arbitration (deterministic physics + Random Forest, 98.67% ISO 22839 agreement)
- **Validation:** full 36-scenario Euro NCAP sweep — 97% dry / 92% wet compliance; HIC/AIS injury modeling
## Setup
 
```bash
git clone https://github.com/chehekagrawal/ADAS-Pedestrian-AEB.git
cd ADAS-Pedestrian-AEB
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
 
## Usage
 
```bash
# perception only
python src/detection/infer.py --source data/sample/test_multiclass.png --model models/yolo_multiclass_best.pt
 
# full pipeline
python run_pipeline.py --source data/sample/test_video.mp4
 
# reproduce all verified results
python verification/run_all_verifications.py
```
 
## Structure
 
```
src/            perception -> decision -> actuation pipeline (see paper, Fig. 1)
models/         YOLOv8n detector, behavior + adaptive-AEB classifiers
configs/        dataset, vehicle dynamics, driver-state threshold configs
verification/   scripts + audit outputs backing every reported number
notebooks/      training pipeline (incl. BDD100K -> YOLO conversion)
results/        training curves, tracking runs, inference outputs
```
 
## Citation
 
This repository accompanies a manuscript currently in preparation for submission to a peer-reviewed journal:
 
> Priyadarshi, A., Gopal, A. C., Sarkar, D., Garg, A., & Singh, I. V.   *A Physics-Aware Vision-Based Automatic Emergency Braking System for Vulnerable Road User Protection.* Manuscript in preparation. Dept. of Mechanical and Industrial Engineering, IIT Roorkee.
 
(see also [`CITATION.cff`](./CITATION.cff))
 
## Contributors
 
Atharv Priyadarshi · Agrawal Chehek Gopal · Debangan Sarkar · Arnav Garg · Prof. Indra Vir Singh
 
## License
 
MIT — see [LICENSE](./LICENSE)




































<!-- # ADAS Perception System for Automatic Emergency Braking (AEB)

## Overview

This project implements a **computer vision–based perception system for Advanced Driver Assistance Systems (ADAS)**, with a focus on **Automatic Emergency Braking (AEB)**.

A **YOLOv8 deep learning model** is trained to detect **multiple road agents** in real-world driving scenes.  
The perception output is designed to serve as the foundation for **risk estimation and decision-making modules** in an AEB pipeline.

### Detected Classes
- Pedestrian
- Car
- Bicycle
- Motorcycle

This repository currently contains the **dataset preparation, training, evaluation, and inference pipeline**.

---

## Quick Start (Local Inference)

```bash
git clone <repo-url>
cd ADAS-Pedestrian-AEB

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/detection/infer.py --source data/sample/test_multiclass.png
```

---

## Key Features

* Multiclass object detection for ADAS
* YOLOv8-based custom training
* Clean dataset conversion pipeline (BDD100K → YOLO)
* Image & video inference support
* Saved training metrics and visual artifacts
* Modular, extensible codebase
* CPU-compatible inference (GPU optional)

---

## Repository Structure

```text
configs/        → dataset & training configs
data/           → sample images/videos for demo
models/         → trained model weights
src/            → source code (training, inference)
notebooks/      → Colab & experimentation notebooks
results/        → training logs, plots, predictions
reports/        → documentation & analysis
```

> Raw datasets (BDD100K) are intentionally excluded from version control.

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

* Dataset: **BDD100K**
* Images used: **6000**
* Classes: pedestrian, car, bicycle, motorcycle
* Labels converted from BDD JSON format to YOLO format

Dataset configuration:

```text
configs/data.yaml
```

---

## Training

Training was performed using **YOLOv8n** with pretrained weights.

* Platform: **Google Colab**
* GPU: **NVIDIA Tesla T4**
* Epochs: 30
* Image size: 640×640

Training artifacts are stored in:

```text
results/training_multiclass/
```

These include:

* loss curves
* precision–recall plots
* confusion matrix
* label distribution

---

## Inference

### Image Inference

```bash
python src/detection/infer.py \
  --source data/sample/test_multiclass.png \
  --model models/yolo_multiclass_best.pt
```

### Video Inference

```bash
python src/detection/infer.py \
  --source data/sample/test_video_multiclass.mp4 \
  --model models/yolo_multiclass_best.pt
```

### Output Locations

```text
results/inference/images_multiclass/
results/inference/videos_multiclass/
```

Bounding boxes and class labels are rendered automatically.

---

## Results (Multiclass Model)

| Class      | mAP50 |
| ---------- | ----- |
| Pedestrian | ~0.43 |
| Car        | ~0.68 |
| Bicycle    | ~0.23 |
| Motorcycle | ~0.27 |

Overall:

* **mAP50 ≈ 0.40**
* **mAP50–95 ≈ 0.21**

---

## Hardware

* Training: NVIDIA Tesla T4 (Google Colab)
* Inference: CPU or GPU (local machine)

---

## Reproducibility

* Training notebook:  
  `notebooks/yolo_training_pipeline.ipynb`

* Dataset conversion scripts:  
  `src/data/`

* Inference script:  
  `src/detection/infer.py`

* Model weights:  
  `models/yolo_multiclass_best.pt`

---

## Future Work

* Temporal object tracking
* Collision risk estimation (TTC-based)
* Confidence-aware AEB logic
* Sensor fusion (camera + radar)
* Real-time deployment & optimization

---

## Team

* Atharv
* Chehek
* Arnav
* Debangan

---

## License

For research and educational use only. -->
