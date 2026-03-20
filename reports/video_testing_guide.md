# ADAS Pedestrian AEB: New Video Testing Guide

This guide provides a step-by-step process for testing new videos using the ADAS-Pedestrian-AEB system.

## 📁 1. Video Placement
Place your new video file in the `data/sample/` directory.
- **Example**: `data/sample/new_test_video.mp4`

---

## 🔍 2. Part 2: Detection & Tracking
You need to generate object trajectories before the collision engine can make decisions.

### Step 2.1: Generate Detections
Run the detection script to create frame-by-frame JSON objects.
```powershell
python src/tracking/generate_detections.py --source data/sample/new_test_video.mp4 --output results/detections
```

### Step 2.2: Run Tracking
Run the tracking script to link detections across frames and create a `trajectories.json` file.
```powershell
python src/tracking/run_tracking.py --detections results/detections --output results/tracking/new_test
```

---

## 🧠 3. Part 3: Collision Prediction & AEB
Now process the trajectories to see the safety decisions.

### Step 3.1: Visual Results (TTC Overlay)
Generate a video with Time-To-Collision (TTC) labels and bounding boxes.
```powershell
python src/collision/overlay.py --video data/sample/new_test_video.mp4 --trajectories results/tracking/new_test/trajectories.json --output results/collision/plots/new_test_overlay.mp4
```

### Step 3.2: Quantitative Results (Safety Logs)
Run the simulation to generate structured JSON logs containing collision probabilities and AEB triggers.
```powershell
python src/collision/simulator.py --trajectories results/tracking/new_test/trajectories.json --output results/collision/logs/new_test_run.json
```

---

## 📊 4. Output Summary
| Output File | Description |
| :--- | :--- |
| `results/collision/plots/new_test_overlay.mp4` | Visual proof of TTC and pedestrian tracking. |
| `results/collision/logs/new_test_run.json` | Detailed safety metrics (Distance, TTC, AEB Trigger status). |

> [!TIP]
> If the TTC labels seem inaccurate, you can adjust the ego speed using the `--speed` argument in `overlay.py` (default is 12.0 px/frame).
