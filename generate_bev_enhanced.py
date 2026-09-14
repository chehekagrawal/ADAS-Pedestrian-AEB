"""
Enhanced Dashcam + BEV split-screen variants.
Perspective-correct pedestrian placement, clean labels, no clutter.
"""

import sys
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = '/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables/figures'

# ── Shared scene definition ──────────────────────────────────────────────────
# Focal length 700px, dashcam 960×540, cx=480, cy=270
F      = 700.0
CX     = 480
W, H   = 960, 540
HORIZ  = 240          # horizon line y in image

# Three pedestrians: (track_id, depth_m, lateral_m, risk)
PEDS = [
    dict(tid=1, Z=25.0, X=-5.0, risk='SAFE',    ttc=5.0),
    dict(tid=2, Z=13.0, X= 2.5, risk='WARNING',  ttc=2.6),
    dict(tid=3, Z= 6.0, X=-1.0, risk='CRITICAL', ttc=1.2),
]

# Colors
COL = {
    'SAFE':     (40,  210,  50),   # green
    'WARNING':  (10,  200, 255),   # yellow
    'CRITICAL': (30,   30, 230),   # red
}
COL_MPL = {
    'SAFE':     '#28d232',
    'WARNING':  '#ffc800',
    'CRITICAL': '#e01010',
}

H_REAL = 1.7   # pedestrian height metres

def ped_bbox(Z, X, img_w=W, img_h=H, horiz=HORIZ, f=F, cx=CX):
    """Compute perspective-correct bbox for a pedestrian at (X, Z)."""
    h_px = max(12, int(f * H_REAL / Z))
    w_px = max(8,  int(h_px * 0.45))
    x_cen = int(cx + X * f / Z)
    # feet touch the ground plane: map ground-plane y from depth
    ground_px_range = img_h - horiz          # pixels from horizon to bottom
    # at Z metres: feet y = horiz + ground_range * (1/Z) / (1/Z_min)
    Z_min = 3.0
    y_feet = int(horiz + ground_px_range * (Z_min / Z) ** 0.55 * 3.2)
    y_feet = min(y_feet, img_h - 4)
    y_top  = y_feet - h_px
    x1, y1 = x_cen - w_px // 2, y_top
    x2, y2 = x_cen + w_px // 2, y_feet
    return (x1, y1, x2, y2)

def draw_road(canvas, w=W, h=H, horiz=HORIZ):
    """Draw a clean road scene on canvas."""
    # Sky
    for y in range(horiz):
        r = int(60 + y * 0.35)
        g = int(80 + y * 0.45)
        b = int(110 + y * 0.50)
        canvas[y, :] = (b, g, r)

    # Side environment (grass/pavement)
    canvas[horiz:, :] = (42, 45, 42)

    # Road trapezoid
    vp_x = w // 2            # vanishing point x
    road_top_half = int(w * 0.16)   # half-width at horizon
    road_bot_half = int(w * 0.48)   # half-width at bottom
    pts = np.array([
        [vp_x - road_top_half, horiz],
        [vp_x + road_top_half, horiz],
        [vp_x + road_bot_half, h],
        [vp_x - road_bot_half, h],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (55, 58, 62))

    # Road edge markings (white lines)
    for side in [-1, 1]:
        top_x = vp_x + side * road_top_half
        bot_x = vp_x + side * road_bot_half
        cv2.line(canvas, (top_x, horiz), (bot_x, h), (200, 200, 195), 2, cv2.LINE_AA)

    # Centre lane dashes (perspective-scaled)
    num_dash = 10
    for i in range(num_dash):
        t0 = i / num_dash
        t1 = (i + 0.45) / num_dash
        y0 = int(horiz + (h - horiz) * t0)
        y1 = int(horiz + (h - horiz) * t1)
        # x narrows linearly toward vanishing point
        x0 = int(vp_x + (vp_x - (w // 2)) * (1 - t0) * 0)  # = vp_x (centre)
        cv2.line(canvas, (vp_x, y0), (vp_x, y1), (200, 200, 190), 2, cv2.LINE_AA)

    # Horizon line (subtle)
    cv2.line(canvas, (0, horiz), (w, horiz), (80, 90, 95), 1)

    # Simple tree silhouettes on sides
    for tx in [80, 180, 760, 870]:
        tree_h = np.random.randint(55, 90)
        ty = horiz - tree_h
        cv2.ellipse(canvas, (tx, ty), (18, tree_h // 2),
                    0, 0, 360, (20, 55, 20), -1, cv2.LINE_AA)
        cv2.line(canvas, (tx, horiz), (tx, ty + tree_h // 2), (30, 22, 12), 2)

    return canvas

def draw_ped_clean(canvas, bbox, color, tid, Z, ttc, risk):
    """Draw a clean, non-cluttered pedestrian box."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1

    # Body stick figure (simple silhouette)
    mid_x = (x1 + x2) // 2
    head_r = max(4, bw // 6)
    head_y = y1 + head_r + 1
    body_y1 = head_y + head_r
    body_y2 = y1 + (y2 - y1) * 2 // 3
    leg_y   = y2

    cv2.circle(canvas, (mid_x, head_y), head_r, (210, 195, 175), -1, cv2.LINE_AA)
    cv2.line(canvas, (mid_x, body_y1), (mid_x, body_y2), (200, 185, 165), 2, cv2.LINE_AA)
    cv2.line(canvas, (mid_x, body_y2), (mid_x - bw // 3, leg_y), (190, 175, 155), 2, cv2.LINE_AA)
    cv2.line(canvas, (mid_x, body_y2), (mid_x + bw // 3, leg_y), (190, 175, 155), 2, cv2.LINE_AA)

    # Bbox outline — thick colored border
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    # Corner brackets (cleaner look)
    clen = max(6, bw // 4)
    for (cx_, cy_) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        sx = 1 if cx_ == x1 else -1
        sy = 1 if cy_ == y1 else -1
        cv2.line(canvas, (cx_, cy_), (cx_ + sx * clen, cy_), color, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx_, cy_), (cx_, cy_ + sy * clen), color, 3, cv2.LINE_AA)

    # Bottom badge: distance + TTC
    badge_txt = f'{Z:.0f}m   TTC:{ttc:.1f}s'
    badge_w   = max(bw + 10, 90)
    bx1 = mid_x - badge_w // 2
    by1 = y2 + 4
    by2 = y2 + 20
    bx1 = max(0, min(bx1, canvas.shape[1] - badge_w))
    cv2.rectangle(canvas, (bx1, by1), (bx1 + badge_w, by2), color, -1)
    cv2.putText(canvas, badge_txt, (bx1 + 4, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    # Top badge: Track ID + risk
    tbadge = f'ID:{tid}  {risk}'
    cv2.rectangle(canvas, (x1, y1 - 20), (x1 + 90, y1), color, -1)
    cv2.putText(canvas, tbadge, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    # AEB flash for critical
    if risk == 'CRITICAL':
        glow_col = (0, 0, 255)
        for thick in [8, 5]:
            cv2.rectangle(canvas, (x1 - thick, y1 - thick),
                          (x2 + thick, y2 + thick), glow_col, 1, cv2.LINE_AA)
    return canvas

def make_dashcam(peds, w=W, h=H, horiz=HORIZ):
    """Build a clean synthetic dashcam frame with pedestrians."""
    np.random.seed(42)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    draw_road(canvas, w, h, horiz)

    # Draw pedestrians back-to-front (farthest first)
    for p in sorted(peds, key=lambda x: -x['Z']):
        bbox = ped_bbox(p['Z'], p['X'])
        color = COL[p['risk']]
        draw_ped_clean(canvas, bbox, color, p['tid'], p['Z'], p['ttc'], p['risk'])

    # HUD bar at bottom
    cv2.rectangle(canvas, (0, h - 40), (w, h), (8, 12, 8), -1)
    cv2.putText(canvas, 'DASHCAM  |  YOLOv8 + SORT  |  Pinhole Depth Estimation',
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)
    ego_spd = 'v=18 km/h'
    cv2.putText(canvas, ego_spd, (w - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)

    return canvas

def make_bev_opencv(peds, cw=640, ch=540, range_m=35, lat_m=15):
    """Clean OpenCV BEV radar panel."""
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:] = (6, 10, 6)

    EGO_PAD = int(ch * 0.10)
    z_sc = (ch - EGO_PAD) / range_m
    x_sc = (cw / 2.0) / lat_m
    ox, oy = cw // 2, ch - EGO_PAD

    def wp(X, Z):
        return int(ox + X * x_sc), int(oy - Z * z_sc)

    # Risk zone bands (horizontal stripes behind rings)
    crit_y = oy - int(10 * z_sc)
    warn_y = oy - int(22 * z_sc)
    if crit_y > 0:
        canvas[max(0, crit_y):oy, :] = (22, 4, 4)
    if warn_y > 0 and crit_y > 0:
        canvas[max(0, warn_y):max(0, crit_y), :] = (18, 14, 4)
    canvas[0:max(0, warn_y), :] = (4, 18, 4)

    # Range rings
    for r in [5, 10, 15, 20, 25, 30]:
        if r > range_m: break
        r_px = int(r * z_sc)
        cv2.ellipse(canvas, (ox, oy), (r_px, r_px), 0, 180, 360,
                    (30, 65, 30), 1, cv2.LINE_AA)
        lbl_y = oy - r_px + 4
        if 2 < lbl_y < ch:
            cv2.putText(canvas, f'{r}m', (ox + 4, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (45, 110, 45), 1)

    # FOV cone
    fov = np.radians(55)
    for sgn in [-1, 1]:
        flen = min(range_m, 34)
        tx = ox + int(sgn * flen * x_sc * np.sin(fov))
        ty = oy - int(flen * z_sc * np.cos(fov))
        cv2.line(canvas, (ox, oy), (tx, ty), (20, 55, 20), 1, cv2.LINE_AA)

    # Zone labels
    labels = [('CRITICAL', (0, 0, 200), oy - int(5 * z_sc)),
              ('WARNING',  (0, 150, 200), oy - int(16 * z_sc)),
              ('SAFE',     (0, 160, 0),  oy - int(28 * z_sc))]
    for txt, col, ly in labels:
        if 0 < ly < ch:
            cv2.putText(canvas, txt, (cw - 75, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    # Ego vehicle
    ev_w, ev_h = 14, 22
    cv2.rectangle(canvas, (ox - ev_w // 2, oy - ev_h), (ox + ev_w // 2, oy),
                  (180, 100, 30), -1)
    cv2.rectangle(canvas, (ox - ev_w // 2, oy - ev_h), (ox + ev_w // 2, oy),
                  (220, 160, 60), 1)
    cv2.putText(canvas, 'EGO', (ox - 14, oy - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 220, 100), 1)

    # Pedestrian dots
    col_bgr = {'SAFE': (40, 210, 50), 'WARNING': (10, 200, 255), 'CRITICAL': (30, 30, 230)}
    for p in peds:
        px_, py_ = wp(p['X'], p['Z'])
        if not (0 <= px_ < cw and 0 <= py_ < ch): continue
        c = col_bgr[p['risk']]
        r_dot = max(7, min(16, int(14 - p['Z'] * 0.25)))
        # Glow rings
        for glow_r, alpha in [(r_dot + 8, 60), (r_dot + 4, 120)]:
            overlay = canvas.copy()
            cv2.circle(overlay, (px_, py_), glow_r, c, -1)
            cv2.addWeighted(overlay, alpha / 255.0, canvas, 1 - alpha / 255.0, 0, canvas)
        cv2.circle(canvas, (px_, py_), r_dot, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px_, py_), r_dot, (240, 240, 240), 1, cv2.LINE_AA)

        # Label chip
        lbl = f'{p["Z"]:.0f}m  {p["ttc"]:.1f}s'
        lx = min(px_ + r_dot + 4, cw - 80)
        ly = py_
        cv2.rectangle(canvas, (lx - 2, ly - 12), (lx + 78, ly + 4), (10, 20, 10), -1)
        cv2.putText(canvas, lbl, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'ID:{p["tid"]} {p["risk"]}', (lx, ly + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1, cv2.LINE_AA)

        if p['risk'] == 'CRITICAL':
            cv2.putText(canvas, 'AEB!', (px_ - 14, py_ - r_dot - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 230), 1, cv2.LINE_AA)

    # Header
    cv2.rectangle(canvas, (0, 0), (cw, 22), (6, 22, 6), -1)
    cv2.putText(canvas, 'BEV RADAR MAP', (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 190, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Pinhole Depth + TTC', (cw - 145, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 140, 40), 1, cv2.LINE_AA)

    return canvas

def make_bev_matplotlib(peds, figw=6.0, figh=5.2):
    """Matplotlib styled BEV with zone bands."""
    fig, ax = plt.subplots(figsize=(figw, figh), facecolor='#050d05')
    ax.set_facecolor('#050d05')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-1.5, 33)
    ax.set_aspect('equal')
    ax.axis('off')

    # Zone bands
    ax.fill_betweenx([0, 10], -16, 16, color='#350000', alpha=0.65, zorder=1)
    ax.fill_betweenx([10, 22], -16, 16, color='#201200', alpha=0.50, zorder=1)
    ax.fill_betweenx([22, 33], -16, 16, color='#001800', alpha=0.40, zorder=1)

    # Depth grid lines
    for d in [5, 10, 15, 20, 25, 30]:
        ax.plot([-16, 16], [d, d], color='#1a3a1a', lw=0.6, ls='--', zorder=2)
        ax.text(15.5, d + 0.3, f'{d}m', color='#2a6a2a', fontsize=6.5, ha='right', va='bottom')

    # Lateral grid
    for x in [-10, -5, 5, 10]:
        ax.plot([x, x], [0, 33], color='#0d1f0d', lw=0.5, ls=':', zorder=2)

    # Centre line
    ax.plot([0, 0], [0, 33], color='#1e3e1e', lw=0.7, ls='--', zorder=2)

    # Zone labels
    ax.text(14.5, 5,  'CRITICAL', color='#ff4444', fontsize=7.5, ha='right',
            fontweight='bold', va='center', zorder=3)
    ax.text(14.5, 16, 'WARNING',  color='#ffcc00', fontsize=7.5, ha='right',
            fontweight='bold', va='center', zorder=3)
    ax.text(14.5, 27, 'SAFE',     color='#44ff44', fontsize=7.5, ha='right',
            fontweight='bold', va='center', zorder=3)

    # FOV cone
    fov = np.radians(55)
    for sgn in [-1, 1]:
        fx = sgn * 32 * np.sin(fov)
        ax.plot([0, fx], [0, 32], color='#1e4a1e', lw=1.0, alpha=0.7, zorder=2)

    # Ego
    ego = mpatches.FancyBboxPatch((-1, 0), 2, 3.0, boxstyle='round,pad=0.2',
                                   facecolor='#2855b8', edgecolor='#6080ff', lw=1.5, zorder=5)
    ax.add_patch(ego)
    ax.text(0, 1.5, 'EGO', color='white', fontsize=6, ha='center', va='center',
            fontweight='bold', zorder=6)

    # Pedestrians
    for p in peds:
        c = COL_MPL[p['risk']]
        # Glow layers
        for s, a in [(350, 0.12), (180, 0.25), (80, 0.70)]:
            ax.scatter(p['X'], p['Z'], s=s, c=c, alpha=a, zorder=4)
        ax.scatter(p['X'], p['Z'], s=55, c=c,
                   edgecolors='white', linewidths=0.8, zorder=5)

        # Clean label
        offx = 1.5 if p['X'] >= 0 else -1.5
        ha   = 'left' if p['X'] >= 0 else 'right'
        lbl  = f"ID:{p['tid']}  {p['Z']:.0f}m\nTTC={p['ttc']:.1f}s"
        ax.text(p['X'] + offx, p['Z'] + 0.5, lbl, color=c, fontsize=6.5,
                ha=ha, va='bottom', linespacing=1.3, zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#0a140a',
                          edgecolor=c, lw=0.8, alpha=0.85))

        # AEB annotation
        if p['risk'] == 'CRITICAL':
            ax.text(p['X'], p['Z'] - 2.5, '⚠ AEB TRIGGER',
                    color='#ff3333', fontsize=8, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#280000',
                              edgecolor='#ff3333', lw=1.2), zorder=7)

    # Title
    ax.text(0, 32.5, 'BEV RADAR  —  ADAS Pedestrian AEB',
            color='#66dd66', fontsize=9, ha='center', va='top',
            fontweight='bold', fontfamily='monospace', zorder=6)

    # Legend
    leg = [mpatches.Patch(facecolor='#28d232', label='SAFE (TTC > 4s)'),
           mpatches.Patch(facecolor='#ffc800', label='WARNING (2–4s)'),
           mpatches.Patch(facecolor='#e01010', label='CRITICAL / AEB (< 2s)')]
    ax.legend(handles=leg, loc='lower left', fontsize=6,
              facecolor='#0a1a0a', edgecolor='#2a6a2a', labelcolor='white',
              framealpha=0.9, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout(pad=0.1)
    return fig


# ── Variant E: Clean split — perspective dashcam + OpenCV BEV ────────────────
dashcam_e = make_dashcam(PEDS, W, H)
bev_e     = make_bev_opencv(PEDS, cw=W, ch=H)
split_e   = np.hstack([dashcam_e, bev_e])
# Separator
cv2.line(split_e, (W, 0), (W, H), (50, 120, 50), 3)
cv2.imwrite(f'{OUT}/bev_E_splitscreen_opencv.png', split_e)
print("E saved → bev_E_splitscreen_opencv.png")


# ── Variant F: Clean dashcam only (wider, 1280×540) ─────────────────────────
dashcam_f = make_dashcam(PEDS, 1280, H, horiz=240)
# Add HUD panel on right side of dashcam
hud_w = 280
hud = np.zeros((H, hud_w, 3), dtype=np.uint8)
hud[:] = (8, 14, 8)
y_cur = 30
cv2.putText(hud, 'AEB STATUS', (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 80), 1)
y_cur += 25
cv2.line(hud, (5, y_cur), (hud_w - 5, y_cur), (30, 80, 30), 1)
y_cur += 15
for p in PEDS:
    c = COL[p['risk']]
    cv2.circle(hud, (20, y_cur), 7, c, -1, cv2.LINE_AA)
    cv2.putText(hud, f'ID:{p["tid"]}  {p["Z"]:.0f}m  {p["ttc"]:.1f}s',
                (35, y_cur + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
    y_cur += 28
# AEB trigger
cv2.rectangle(hud, (10, 160), (hud_w - 10, 200), (30, 30, 200), -1)
cv2.putText(hud, 'AEB TRIGGERED', (18, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
dashcam_wide = np.hstack([dashcam_f[:, :1000], hud])
dashcam_wide = cv2.resize(dashcam_wide, (1280, H))
cv2.imwrite(f'{OUT}/bev_F_dashcam_with_hud.png', dashcam_wide)
print("F saved → bev_F_dashcam_with_hud.png")


# ── Variant G: Matplotlib BEV rendered and split with dashcam ────────────────
fig_g = make_bev_matplotlib(PEDS)
fig_g.savefig('/tmp/bev_mpl_tmp.png', dpi=140, bbox_inches='tight',
              facecolor='#050d05')
plt.close(fig_g)
bev_mpl = cv2.imread('/tmp/bev_mpl_tmp.png')
# Scale dashcam to match BEV height
bev_h, bev_w = bev_mpl.shape[:2]
dashcam_g = make_dashcam(PEDS, W, H)
dash_g    = cv2.resize(dashcam_g, (int(bev_h * W / H), bev_h))
split_g   = np.hstack([dash_g, bev_mpl])
cv2.imwrite(f'{OUT}/bev_G_splitscreen_matplotlib.png', split_g)
print("G saved → bev_G_splitscreen_matplotlib.png")


# ── Variant H: Full-width BEV only (matplotlib), no dashcam ─────────────────
fig_h = make_bev_matplotlib(PEDS, figw=10.0, figh=5.0)
fig_h.savefig(f'{OUT}/bev_H_bev_only_wide.png', dpi=150,
              bbox_inches='tight', facecolor='#050d05')
plt.close(fig_h)
print("H saved → bev_H_bev_only_wide.png")

print(f"\nAll 4 enhanced variants → {OUT}/")
print("E = Split: perspective dashcam + OpenCV radar")
print("F = Dashcam only + AEB status HUD panel")
print("G = Split: perspective dashcam + matplotlib BEV radar")
print("H = Matplotlib BEV only (wide, for full-slide use)")
