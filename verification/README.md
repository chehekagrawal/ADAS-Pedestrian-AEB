# Verification & Reproducibility Package

This directory contains the independent verification scripts, logs, and artifacts for the research paper:
**"A Physics-Aware Vision-Based Automatic Emergency Braking System for Vulnerable Road User Protection: Multi-Class Perception, Adaptive Time-to-Collision Control, Brake Thermodynamics, and Biomechanical Injury Mitigation"**

Submitted for peer review to **Elsevier (ScienceDirect)**.

---

## 1. Ground Truth Verification Policy
To prevent any fabricated or unverified claims, every number in the revised manuscript is traced directly to one of three sources:
1. An existing training log or model file in the repository (`results/training_multiclass/results.csv`, `models/`).
2. An independently executed script in this directory that simulates the physical model end-to-end.
3. An engineering constant defined directly in the configuration and physics modules (`configs/`, `src/vehicle_dynamics/`).

---

## 2. Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| `run_all_verifications.py` | Python script | Master test runner that executes all verification scripts sequentially. |
| `verify_detection.py` | Python script | Extracts exact epoch 30 metrics from `results/training_multiclass/results.csv`. |
| `detection_metrics_verified.json` | Data | Final epoch metrics: Precision (53.6%), Recall (37.9%), mAP@0.5 (40.0%), mAP@0.5:0.95 (21.3%). |
| `detection_metrics_verified.csv` | Data | Table comparing verified metrics against earlier draft claims. |
| `verify_ncap.py` | Python script | Runs Euro NCAP test suite across 4 scenarios (CPFA, CPNA, CPNC, CPLA) and 9 speeds (20–60 km/h). |
| `ncap_matrix_dry.csv` | Data | 36-test results on dry surface (34 avoided, 1 partial, 1 collision; 97% overall score). |
| `ncap_matrix_wet.csv` | Data | 36-test results on wet surface (32 avoided, 1 partial, 3 collisions; 92% overall score). |
| `ncap_summary.json` | Data | High-level summary of NCAP test outcomes. |
| `verify_hic.py` | Python script | Evaluates Head Injury Criterion (HIC) across speeds 0–80 km/h and compares multi-zone and half-sine models. |
| `hic_biomechanics_verified.csv` | Data | Tabulated HIC values, impact zones, contact durations, and AIS classifications. |
| `verify_thermo.py` | Python script | Evaluates kinetic energy conversion, rotor temperature rise, friction fade multiplier, and TTC penalty. |
| `thermo_sweep_verified.csv` | Data | Temperature sweep from 20°C to 800°C with exact friction loss and TTC buffer penalty. |
| `verify_dynamics.py` | Python script | Compares kinematic vs Pacejka stopping distances across 5 surfaces (dry, wet, gravel, snow, ice). |
| `stopping_distance_verified.csv`| Data | Detailed stopping distance comparison with and without 0.33s system delay. |
| `verify_ml_models.py` | Python script | Audits `models/adaptive_aeb_model.pkl` and `models/behavior_classifier.pkl`. |
| `ml_models_verified.json` | Data | Model architectures, estimators, class distributions, and feature importances. |
| `verify_driver_monitoring.py` | Python script | Audits EAR thresholds and reaction time delays from `configs/part5_thresholds.yaml`. |
| `driver_monitoring_verified.json`| Data | Exact verified thresholds for driver state transitions. |

---

## 3. How to Reproduce All Results

To run the entire suite from the workspace root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the master verification suite
python verification/run_all_verifications.py
```

All scripts execute in less than 15 seconds on a modern CPU.
