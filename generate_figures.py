"""
Consolidates ALL project figures into one directory:
  reports/final_deliverables/figures/

Generates and saves every matplotlib figure (plots + flowcharts) that
the PPT and report use, then copies the useful training/inference images.

Run:
    python3 generate_figures.py
"""

import sys, shutil
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB')

import matplotlib
matplotlib.use('Agg')

OUT = '/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables/figures'

# ── Import all figure-making functions from the PPT generator ─────────────────
from generate_ppt_v2 import (
    make_detection_fig,
    make_dynamics_fig,
    make_thermo_fig,
    make_aeb_driver_fig,
    make_ncap_hic_fig,
    make_pipeline_flowchart,
    make_aeb_decision_flowchart,
    make_timeline_fig,
    make_bev_radar_fig,
)

# ── 1. Matplotlib figures ─────────────────────────────────────────────────────

FIGURES = [
    (make_detection_fig,        'fig_01_detection_mAP_training.png',      150),
    (make_dynamics_fig,         'fig_02_dynamics_stopping_distance.png',   150),
    (make_thermo_fig,           'fig_03_thermo_rotor_fade.png',            150),
    (make_aeb_driver_fig,       'fig_04_aeb_threshold_ear_drowsiness.png', 150),
    (make_ncap_hic_fig,         'fig_05_ncap_passfail_hic_curve.png',      150),
    (make_pipeline_flowchart,   'fig_06_pipeline_flowchart.png',           150),
    (make_aeb_decision_flowchart,'fig_07_aeb_decision_flowchart.png',      150),
    (make_timeline_fig,         'fig_08_timeline_gantt.png',               150),
    (make_bev_radar_fig,        'fig_09_bev_radar_matplotlib.png',         150),
]

print("── Saving matplotlib figures ────────────────────────────────")
for fn, name, dpi in FIGURES:
    import matplotlib.pyplot as plt
    fig = fn()
    path = f'{OUT}/{name}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {name}")

# ── 2. Copy training / inference images ──────────────────────────────────────

COPIES = [
    # (source, dest_filename)
    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/results.png',
     'training_01_results_curves.png'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/confusion_matrix_normalized.png',
     'training_02_confusion_matrix_normalized.png'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/confusion_matrix.png',
     'training_03_confusion_matrix.png'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/BoxPR_curve.png',
     'training_04_PR_curve.png'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/BoxF1_curve.png',
     'training_05_F1_curve.png'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/val_batch0_pred.jpg',
     'training_06_val_batch0_predictions.jpg'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/training_multiclass/val_batch0_labels.jpg',
     'training_07_val_batch0_labels.jpg'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/inference/images_multiclass/image_output_multiclass.jpg',
     'inference_01_output_multiclass.jpg'),

    ('/home/atharv/ADAS-Pedestrian-AEB/results/inference/images/image_output.jpg',
     'inference_02_output_pedestrian.jpg'),
]

import os
print("\n── Copying training / inference images ─────────────────────")
for src, dst_name in COPIES:
    dst = f'{OUT}/{dst_name}'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✓  {dst_name}")
    else:
        print(f"  ✗  MISSING: {src}")

# ── Summary ───────────────────────────────────────────────────────────────────
all_files = sorted(f for f in os.listdir(OUT)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
print(f"\n── All figures in {OUT} ─────")
for f in all_files:
    size_kb = os.path.getsize(f'{OUT}/{f}') // 1024
    print(f"  {f:55s}  {size_kb:>4d} KB")
print(f"\nTotal: {len(all_files)} files")
