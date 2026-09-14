"""
Generate BEV radar variant images for visual selection.
Saves 4 options to reports/final_deliverables/figures/
"""

import sys
import os
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from src.visualization.bev_radar import BEVRadarMapper

OUT = '/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables/figures'

mapper = BEVRadarMapper(focal_length_px=700, img_width=1920, img_height=1080)

# Shared detections: safe(25m), warning(13m), critical(6m)
detections = [
    (1, (870, 492, 918, 540)),   # 25 m — SAFE
    (2, (980, 474, 1071, 566)),  # 13 m — WARNING
    (3, (880, 441, 1078, 639)),  #  6 m — CRITICAL
    (4, (600, 490, 640, 538)),   # 25 m left — SAFE
    (5, (1300, 476, 1370, 558)), # 12 m right — WARNING
]
tracks = mapper.process_frame(detections, ego_speed_ms=5.0, aeb_threshold=2.0)

# ── Variant A: Raw BEV canvas (OpenCV, 720×720) ─────────────────────────────
bev_a = mapper.render_bev_frame(tracks, canvas_hw=(720, 720), range_m=50, lateral_m=20)
cv2.imwrite(f'{OUT}/bev_variant_A_radar_canvas.png', bev_a)
print("Variant A saved → bev_variant_A_radar_canvas.png")


# ── Variant B: Wider BEV canvas widescreen (720×1280) ───────────────────────
bev_b = mapper.render_bev_frame(tracks, canvas_hw=(480, 1280), range_m=50, lateral_m=20)
cv2.imwrite(f'{OUT}/bev_variant_B_widescreen.png', bev_b)
print("Variant B saved → bev_variant_B_widescreen.png")


# ── Variant C: Matplotlib Polar-style BEV with styled zones ─────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#050d05')
ax.set_facecolor('#050d05')
ax.set_xlim(-22, 22)
ax.set_ylim(-2, 42)
ax.set_aspect('equal')
ax.axis('off')

# Grid lines (horizontal depth ticks)
for d in [10, 20, 30, 40]:
    ax.plot([-22, 22], [d, d], color='#1a3a1a', lw=0.7, ls='--')
    ax.text(21.5, d + 0.5, f'{d}m', color='#2a6a2a', fontsize=7, ha='right', va='bottom')

# Vertical centerline
ax.plot([0, 0], [0, 42], color='#1a3a1a', lw=0.5, ls=':')

# Lateral grid
for x in [-15, -10, -5, 5, 10, 15]:
    ax.plot([x, x], [0, 42], color='#0d200d', lw=0.5, ls=':')

# Risk zone bands (horizontal, depth-based)
ax.fill_betweenx([0, 10], -22, 22, color='#3a0000', alpha=0.55, zorder=1)
ax.fill_betweenx([10, 22], -22, 22, color='#1e1400', alpha=0.45, zorder=1)
ax.fill_betweenx([22, 42], -22, 22, color='#001a00', alpha=0.35, zorder=1)

# Zone text
ax.text(20, 5, 'CRITICAL', color='#ff4444', fontsize=7.5, ha='right', va='center',
        fontweight='bold', alpha=0.8)
ax.text(20, 16, 'WARNING', color='#ffcc00', fontsize=7.5, ha='right', va='center',
        fontweight='bold', alpha=0.8)
ax.text(20, 32, 'SAFE', color='#44ff44', fontsize=7.5, ha='right', va='center',
        fontweight='bold', alpha=0.8)

# Ego vehicle
ego = mpatches.FancyBboxPatch((-1, 0), 2, 3.5, boxstyle='round,pad=0.2',
                               facecolor='#3060c0', edgecolor='#80a0ff', lw=1.5, zorder=5)
ax.add_patch(ego)
ax.text(0, 1.8, 'EGO', color='white', fontsize=6.5, ha='center', va='center',
        fontweight='bold', zorder=6)

# FOV lines
fov_half = np.radians(55)
for sign in [-1, 1]:
    fx = sign * 42 * np.sin(fov_half)
    ax.plot([0, fx], [0, 42], color='#1e4e1e', lw=1, ls='-', alpha=0.6, zorder=2)

# Color maps
color_map = {'SAFE': '#44ff44', 'WARNING': '#ffcc00', 'CRITICAL': '#ff3333'}

# Pedestrian tracks
for t in tracks:
    c = color_map[t.risk_label]
    # Glow
    ax.scatter(t.X_m, t.Z_m, s=320, c=c, alpha=0.18, zorder=3)
    ax.scatter(t.X_m, t.Z_m, s=160, c=c, alpha=0.35, zorder=4)
    ax.scatter(t.X_m, t.Z_m, s=60, c=c, edgecolors='white', linewidths=0.8, zorder=5)

    # Label
    ttc_str = f'TTC={t.ttc_s:.1f}s' if np.isfinite(t.ttc_s) else 'TTC=∞'
    lbl = f'ID:{t.track_id}  {t.Z_m:.0f}m\n{ttc_str} [{t.risk_label}]'
    offx = 1.5 if t.X_m >= 0 else -1.5
    ha = 'left' if t.X_m >= 0 else 'right'
    ax.text(t.X_m + offx, t.Z_m + 0.4, lbl, color=c, fontsize=6.5,
            ha=ha, va='bottom', linespacing=1.3, zorder=6)

# AEB annotation for critical
for t in tracks:
    if t.risk_label == 'CRITICAL':
        ax.text(t.X_m, t.Z_m - 2.2, '⚠ AEB TRIGGER', color='#ff3333',
                fontsize=8, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a0000',
                          edgecolor='#ff3333', lw=1.2), zorder=7)

# Title bar
ax.text(0, 41.5, 'BEV RADAR  —  ADAS Pedestrian AEB',
        color='#66dd66', fontsize=10, ha='center', va='top', fontweight='bold',
        fontfamily='monospace')
ax.text(-21, 41.5, '● SAFE  ● WARN  ● CRIT',
        color='#888', fontsize=7, va='top')

# Legend
legend_elems = [
    mpatches.Patch(facecolor='#44ff44', label='SAFE (TTC > 4s)'),
    mpatches.Patch(facecolor='#ffcc00', label='WARNING (2–4s)'),
    mpatches.Patch(facecolor='#ff3333', label='CRITICAL / AEB (< 2s)'),
]
ax.legend(handles=legend_elems, loc='lower left', fontsize=6.5,
          facecolor='#0a1a0a', edgecolor='#2a6a2a', labelcolor='white',
          framealpha=0.85, bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.2)
plt.savefig(f'{OUT}/bev_variant_C_matplotlib_polar.png', dpi=150,
            bbox_inches='tight', facecolor='#050d05')
plt.close()
print("Variant C saved → bev_variant_C_matplotlib_polar.png")


# ── Variant D: Split-screen — synthetic dashcam + BEV radar ─────────────────
# Create a synthetic dashcam scene (dark road + ped boxes)
H, W = 480, 640
dashcam = np.zeros((H, W, 3), dtype=np.uint8)
# Sky gradient
for y in range(H // 2):
    v = int(20 + y * 0.3)
    dashcam[y, :] = (v, v + 5, v + 8)
# Road (dark grey trapezoid)
pts = np.array([[0, H], [W, H], [int(W * 0.75), H // 2], [int(W * 0.25), H // 2]])
cv2.fillPoly(dashcam, [pts], (45, 45, 50))
# Lane markings
for lx in [W // 2 - 2, W // 2 + 2]:
    for seg in range(0, H, 40):
        cv2.line(dashcam, (lx, seg + 10), (lx, seg + 25), (200, 200, 180), 1)

# Draw pedestrian bboxes on dashcam
color_map_cv = {'SAFE': (40, 210, 50), 'WARNING': (10, 200, 255), 'CRITICAL': (30, 30, 230)}
for t in tracks:
    # Scale from 1920x1080 to 640x480
    x1, y1, x2, y2 = t.bbox
    sx1 = int(x1 * 640 / 1920)
    sx2 = int(x2 * 640 / 1920)
    sy1 = int(y1 * 480 / 1080)
    sy2 = int(y2 * 480 / 1080)
    c = color_map_cv[t.risk_label]
    cv2.rectangle(dashcam, (sx1, sy1), (sx2, sy2), c, 2)
    cv2.rectangle(dashcam, (sx1, sy1 - 22), (sx2, sy1), c, -1)
    cv2.putText(dashcam, f'ID:{t.track_id} {t.Z_m:.0f}m', (sx1 + 2, sy1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    ttc_str = f'TTC:{t.ttc_s:.1f}s' if np.isfinite(t.ttc_s) else 'inf'
    cv2.putText(dashcam, f'{ttc_str} {t.risk_label}', (sx1 + 2, sy1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

# BEV radar panel
bev_panel = mapper.render_bev_frame(tracks, canvas_hw=(480, 640), range_m=50, lateral_m=20)

# Combine side-by-side
split = np.hstack([dashcam, bev_panel])
# Header bar
cv2.rectangle(split, (0, 0), (1280, 28), (10, 30, 10), -1)
cv2.putText(split, 'DASHCAM (YOLO + SORT Detection)', (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)
cv2.putText(split, 'BEV RADAR MAP (Pinhole Depth + TTC)', (660, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
# Divider
cv2.line(split, (640, 0), (640, 480), (40, 100, 40), 2)

cv2.imwrite(f'{OUT}/bev_variant_D_split_screen.png', split)
print("Variant D saved → bev_variant_D_split_screen.png")

print(f"\nAll 4 variants saved in:\n  {OUT}/")
print("A = Raw OpenCV radar canvas (square)")
print("B = Widescreen OpenCV radar (1280x480)")
print("C = Matplotlib polar-style radar (styled zones, glow, legend)")
print("D = Split-screen: synthetic dashcam + BEV radar side-by-side")
