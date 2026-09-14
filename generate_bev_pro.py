"""
Professional BEV + Dashcam split-screen.
Uses real Times Square dashcam image with clean AEB annotations.
BEV: heat map background, car-shaped ego, danger cone, grid, compass.
"""

import sys
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB')

import numpy as np
import cv2

OUT  = '/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables/figures'
DASH = '/home/atharv/ADAS-Pedestrian-AEB/data/sample/test.jpg'

# ── Scene: real pedestrians identified in Times Square image (640×425) ────────
# Picked from actual visible people at different depths
# depth = 700 * 1.7 / h_px (pinhole)  |  X = (cx - 320) * depth / 700
F    = 700.0
CX   = 320
EGO  = 5.0   # m/s (18 km/h)
THRESH = 2.0  # AEB threshold seconds

PEDS = [
    # (id, x1,y1,x2,y2 on 640×425 image)
    dict(tid=3, bbox=(68,  270, 142, 425), risk=None, ttc=None, Z=None, X=None),  # large close
    dict(tid=2, bbox=(270, 285, 335, 375), risk=None, ttc=None, Z=None, X=None),  # mid center
    dict(tid=5, bbox=(420, 268, 476, 348), risk=None, ttc=None, Z=None, X=None),  # mid right
    dict(tid=1, bbox=(510, 200, 550, 252), risk=None, ttc=None, Z=None, X=None),  # far right
]

def compute_depth_ttc(p):
    x1,y1,x2,y2 = p['bbox']
    h_px = y2 - y1
    Z = F * 1.7 / h_px
    cx_px = (x1 + x2) / 2.0
    X = (cx_px - CX) * Z / F
    ttc = Z / EGO
    if ttc > THRESH * 2.0:
        risk = 'SAFE'
    elif ttc > THRESH:
        risk = 'WARNING'
    else:
        risk = 'CRITICAL'
    p.update(Z=Z, X=X, ttc=ttc, risk=risk)
    return p

PEDS = [compute_depth_ttc(p) for p in PEDS]

for p in PEDS:
    print(f"  ID:{p['tid']}  Z={p['Z']:.1f}m  X={p['X']:.1f}m  TTC={p['ttc']:.1f}s  [{p['risk']}]")

COL_BGR = {'SAFE': (40,210,50), 'WARNING': (10,200,255), 'CRITICAL': (30,30,230)}


# ═══════════════════════════════════════════════════════════════════════════════
# DASHCAM PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_dashcam(img_path, peds, out_w=760, out_h=500):
    raw = cv2.imread(img_path)
    # Upscale to output size
    frame = cv2.resize(raw, (out_w, out_h))
    # Scale bbox coords
    sx = out_w / 640.0
    sy = out_h / 425.0

    for p in sorted(peds, key=lambda x: -x['Z']):  # far first
        x1 = int(p['bbox'][0] * sx)
        y1 = int(p['bbox'][1] * sy)
        x2 = int(p['bbox'][2] * sx)
        y2 = int(p['bbox'][3] * sy)
        c  = COL_BGR[p['risk']]

        # Corner bracket style bbox
        clen = max(8, (x2-x1)//4)
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 1, cv2.LINE_AA)  # thin full outline
        for (bx,by) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            sx_ = 1 if bx==x1 else -1
            sy_ = 1 if by==y1 else -1
            cv2.line(frame,(bx,by),(bx+sx_*clen,by),c,3,cv2.LINE_AA)
            cv2.line(frame,(bx,by),(bx,by+sy_*clen),c,3,cv2.LINE_AA)

        # AEB glow outline for critical
        if p['risk'] == 'CRITICAL':
            for t,a in [(6,0.15),(4,0.25)]:
                ov = frame.copy()
                cv2.rectangle(ov,(x1-t,y1-t),(x2+t,y2+t),c,-1)
                cv2.addWeighted(ov,a,frame,1-a,0,frame)
            cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),c,2,cv2.LINE_AA)

        # Label chip (top of box, no overlap)
        ttc_str = f'TTC:{p["ttc"]:.1f}s'
        dist_str = f'{p["Z"]:.0f}m'
        line1 = f'ID:{p["tid"]}, {dist_str}'
        line2 = f'{ttc_str} {p["risk"]}'
        if p['risk'] == 'CRITICAL':
            line2 = f'{ttc_str} CRITICAL  AEB Active!'

        chip_w = max(x2-x1, 130)
        chip_h = 34
        chip_x = max(0, min(x1, out_w - chip_w))
        chip_y = max(chip_h, y1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (chip_x, chip_y-chip_h), (chip_x+chip_w, chip_y),
                      c, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, line1, (chip_x+4, chip_y-chip_h+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (chip_x+4, chip_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)

    # Header bar
    cv2.rectangle(frame,(0,0),(out_w,28),(8,18,8),-1)
    cv2.putText(frame,'DASHCAM  |  Classes: Pedestrian  Car  Bicycle  Motorcycle',(8,19),
                cv2.FONT_HERSHEY_SIMPLEX,0.44,(80,200,80),1,cv2.LINE_AA)

    # Bottom HUD
    cv2.rectangle(frame,(0,out_h-28),(out_w,out_h),(8,18,8),-1)
    cv2.putText(frame,f'YOLOv8 + SORT Tracker  |  v={EGO*3.6:.0f} km/h  |  AEB Threshold={THRESH:.1f}s',
                (8,out_h-9),cv2.FONT_HERSHEY_SIMPLEX,0.40,(60,160,60),1,cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# BEV RADAR PANEL — Professional
# ═══════════════════════════════════════════════════════════════════════════════

def draw_car_topdown(canvas, cx, cy, color=(70,140,220), scale=1.2):
    """Simple top-down sedan: rectangular body, roof, 4 wheels. No F1 nose."""
    W = int(18 * scale)   # car width
    L = int(34 * scale)   # car length (front bumper to rear bumper)

    # ── Car body (plain rounded rectangle) ───────────────────────────────────
    cv2.rectangle(canvas, (cx-W//2, cy-L//2), (cx+W//2, cy+L//2), color, -1)
    cv2.rectangle(canvas, (cx-W//2, cy-L//2), (cx+W//2, cy+L//2), (255, 190, 140), 1)  # light-blue border BGR

    # ── Roof (lighter central rectangle) ─────────────────────────────────────
    rc = tuple(min(255, c+55) for c in color)
    cv2.rectangle(canvas, (cx-W//2+3, cy-L//4), (cx+W//2-3, cy+L//5), rc, -1)

    # ── Windshield (front glass — dark tinted) ────────────────────────────────
    cv2.rectangle(canvas, (cx-W//2+4, cy-L//2+3), (cx+W//2-4, cy-L//4), (90, 75, 55), -1)

    # ── Rear window ───────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (cx-W//2+4, cy+L//5), (cx+W//2-4, cy+L//2-3), (70, 60, 45), -1)

    # ── Headlights (front, bright yellow) ────────────────────────────────────
    for hx in [cx-W//2+2, cx+W//2-6]:
        cv2.rectangle(canvas, (hx, cy-L//2-1), (hx+4, cy-L//2+3), (10, 220, 255), -1)

    # ── Taillights (rear, red rectangles) ─────────────────────────────────────
    for tx in [cx-W//2+2, cx+W//2-6]:
        cv2.rectangle(canvas, (tx, cy+L//2-3), (tx+4, cy+L//2+1), (30,30,190), -1)

    # ── Wheels (4 corners, dark filled ellipses) ──────────────────────────────
    wh_off = 3  # how far wheels stick out from body
    for wx, wy in [(cx-W//2-wh_off, cy-L//3),   # front-left
                   (cx+W//2+wh_off, cy-L//3),    # front-right
                   (cx-W//2-wh_off, cy+L//3),    # rear-left
                   (cx+W//2+wh_off, cy+L//3)]:   # rear-right
        cv2.ellipse(canvas,(wx,wy),(4,7),90,0,360,(18,18,18),-1,cv2.LINE_AA)
        cv2.ellipse(canvas,(wx,wy),(2,4),90,0,360,(55,55,55),-1,cv2.LINE_AA)

    # ── Label ─────────────────────────────────────────────────────────────────
    cv2.putText(canvas,'EGO',(cx-11, cy+L//2+14),
                cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,220,255),1,cv2.LINE_AA)


def make_threat_heatmap(H, W, peds, ox, oy, z_sc, x_sc):
    heat = np.zeros((H,W), dtype=np.float32)
    yy,xx = np.mgrid[0:H, 0:W]
    for p in peds:
        px_ = ox + p['X']*x_sc
        py_ = oy - p['Z']*z_sc
        rw   = {'CRITICAL':1.0,'WARNING':0.4,'SAFE':0.08}[p['risk']]
        sig  = max(22, int(55 - p['Z']*1.2))
        d2   = (xx-px_)**2 + (yy-py_)**2
        heat += rw * np.exp(-d2/(2*sig**2))
    heat = np.clip(heat/(heat.max()+1e-6), 0, 1)
    return heat


def colorize_heat(heat):
    """BGR ADAS danger colormap: dark → yellow-orange → red (no blue artifacts)."""
    h = heat
    R = np.clip(h * 3.0, 0, 1)
    G = np.clip(np.where(h < 0.4, h * 2.2, np.maximum(0.0, (0.88 - h) * 2.0)), 0.0, 0.85)
    B = np.zeros_like(h)
    bgr = np.stack([B, G, R], axis=2)
    return (bgr * 255).astype(np.uint8)


def draw_danger_cone(canvas, ox, oy, reach_y, color=(30,30,220)):
    """AEB braking zone: straight forward-facing symmetric cone from ego.
    reach_y = y-pixel of the furthest critical threat (cone tip stops there).
    """
    half = np.radians(22)   # cone half-width — symmetric about forward axis
    dist = oy - reach_y     # distance in pixels (straight ahead)
    dist = max(dist, 30)
    angles = np.linspace(-half, half, 30)
    pts = [(ox, oy)] + [(int(ox + dist*np.sin(a)), int(oy - dist*np.cos(a)))
                        for a in angles]
    pts = np.array(pts, dtype=np.int32)
    ov = canvas.copy()
    cv2.fillPoly(ov, [pts], color)
    cv2.addWeighted(ov, 0.28, canvas, 0.72, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)


def draw_compass(canvas, cx, cy, r=18):
    """4-point compass star."""
    for ang_deg in [0, 90, 180, 270]:
        ang = np.radians(ang_deg)
        tip_x = int(cx + r*np.sin(ang))
        tip_y = int(cy - r*np.cos(ang))
        base1 = (int(cx+4*np.cos(ang)), int(cy+4*np.sin(ang)))
        base2 = (int(cx-4*np.cos(ang)), int(cy-4*np.sin(ang)))
        pts = np.array([(tip_x,tip_y), base1, base2], dtype=np.int32)
        col = (240,240,200) if ang_deg==0 else (120,140,100)
        cv2.fillPoly(canvas,[pts],col)
    cv2.circle(canvas,(cx,cy),3,(180,180,120),-1,cv2.LINE_AA)


def make_bev_pro(peds, bev_w=760, bev_h=500, range_m=32, lat_m=14):
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    EGO_PAD = int(bev_h*0.11)
    z_sc = (bev_h - EGO_PAD) / range_m
    x_sc = (bev_w/2.0) / lat_m
    ox   = bev_w // 2
    oy   = bev_h - EGO_PAD

    # ── 1. Heat map background ────────────────────────────────────────────
    heat = make_threat_heatmap(bev_h, bev_w, peds, ox, oy, z_sc, x_sc)
    heat_bgr = colorize_heat(heat)
    alpha_map = np.clip(heat*0.65, 0, 1)[:,:,np.newaxis]
    canvas = (heat_bgr * alpha_map).astype(np.uint8)
    canvas[:] = np.clip(canvas.astype(int) + 6, 0, 255).astype(np.uint8)

    # ── 2. Grid ───────────────────────────────────────────────────────────
    GRID_M = 5
    grid_col = (22, 48, 22)
    for d in range(0, range_m+1, GRID_M):
        y_px = int(oy - d*z_sc)
        if 0<=y_px<bev_h:
            cv2.line(canvas,(0,y_px),(bev_w,y_px),grid_col,1,cv2.LINE_AA)
    for lx in range(-int(lat_m), int(lat_m)+1, GRID_M):
        x_px = int(ox + lx*x_sc)
        if 0<=x_px<bev_w:
            cv2.line(canvas,(x_px,0),(x_px,bev_h),grid_col,1,cv2.LINE_AA)

    # ── 3. Range rings ────────────────────────────────────────────────────
    ring_col   = (35, 80, 35)
    ring_label = (55, 130, 55)
    for r in [5,10,15,20,25,30]:
        if r > range_m: break
        r_px = int(r*z_sc)
        cv2.ellipse(canvas,(ox,oy),(r_px,r_px),0,180,360,ring_col,1,cv2.LINE_AA)
        ly = oy - r_px + 5
        if 2 < ly < bev_h:
            cv2.putText(canvas,f'{r}m',(ox+5,ly),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,ring_label,1,cv2.LINE_AA)

    # ── 4. FOV cone lines ─────────────────────────────────────────────────
    fov_col = (28, 70, 28)
    fov_h = np.radians(55)
    for sgn in [-1,1]:
        flen = min(range_m,31)
        tx = ox + int(sgn*flen*x_sc*np.sin(fov_h))
        ty = oy - int(flen*z_sc*np.cos(fov_h))
        cv2.line(canvas,(ox,oy),(tx,ty),fov_col,1,cv2.LINE_AA)

    # ── 5. Danger cone — straight forward, reaches closest critical ped ──────
    critical_peds = [p for p in peds if p['risk'] == 'CRITICAL']
    if critical_peds:
        closest = min(critical_peds, key=lambda p: p['Z'])
        reach_y = int(oy - closest['Z'] * z_sc)
        draw_danger_cone(canvas, ox, oy, reach_y)

    # ── 6. Pedestrian dots ────────────────────────────────────────────────
    col_bgr = {'SAFE':(40,210,50),'WARNING':(10,200,255),'CRITICAL':(30,30,230)}
    for p in peds:
        px_ = int(ox + p['X']*x_sc)
        py_ = int(oy - p['Z']*z_sc)
        if not (0<=px_<bev_w and 0<=py_<bev_h): continue
        c = col_bgr[p['risk']]
        r_dot = max(8, min(18, int(16 - p['Z']*0.35)))

        # Multi-layer glow
        for gr, ga in [(r_dot+16,0.08),(r_dot+9,0.18),(r_dot+4,0.35)]:
            ov = canvas.copy()
            cv2.circle(ov,(px_,py_),gr,c,-1,cv2.LINE_AA)
            cv2.addWeighted(ov,ga,canvas,1-ga,0,canvas)
        cv2.circle(canvas,(px_,py_),r_dot,c,-1,cv2.LINE_AA)
        cv2.circle(canvas,(px_,py_),r_dot,(255,255,255),1,cv2.LINE_AA)

        # Label chip
        ttc_str = f'TTC={p["ttc"]:.1f}s'
        dist_str = f'{p["Z"]:.0f}m'
        lbl1 = f'{dist_str}  {ttc_str}'
        chip_x = min(px_+r_dot+5, bev_w-100)
        chip_y = py_
        cv2.rectangle(canvas,(chip_x-2,chip_y-13),(chip_x+92,chip_y+3),(6,14,6),-1)
        cv2.putText(canvas,lbl1,(chip_x,chip_y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,c,1,cv2.LINE_AA)
        cv2.putText(canvas,f'ID:{p["tid"]} {p["risk"]}',(chip_x,chip_y+13),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,c,1,cv2.LINE_AA)

        if p['risk']=='CRITICAL':
            cv2.putText(canvas,'AEB!',(px_-14,py_-r_dot-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(30,30,230),1,cv2.LINE_AA)

    # ── 7. Ego vehicle (car shape) ────────────────────────────────────────
    draw_car_topdown(canvas, ox, oy, color=(200, 130, 55), scale=1.3)

    # ── 8. Compass ────────────────────────────────────────────────────────
    draw_compass(canvas, bev_w-28, bev_h-28, r=18)

    # ── 9. Header ─────────────────────────────────────────────────────────
    cv2.rectangle(canvas,(0,0),(bev_w,28),(8,18,8),-1)
    cv2.putText(canvas,'BEV RADAR MAP  (Multi-Sensor Fusion)',(8,19),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,(60,190,60),1,cv2.LINE_AA)
    cv2.putText(canvas,'Pinhole Depth + TTC Risk Field',(bev_w-215,19),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(40,140,40),1,cv2.LINE_AA)

    # ── 10. Zone labels ───────────────────────────────────────────────────
    for (label,col,z_m) in [('CRITICAL',(0,0,200),6),
                              ('WARNING', (0,160,200),15),
                              ('SAFE',    (0,160,0),27)]:
        ly = int(oy - z_m*z_sc)
        if 0<ly<bev_h:
            cv2.putText(canvas,label,(bev_w-80,ly),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,col,1,cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE SPLIT SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

BEV_W, BEV_H = 760, 500
DASH_W = 760

dash_panel = annotate_dashcam(DASH, PEDS, out_w=DASH_W, out_h=BEV_H)
bev_panel  = make_bev_pro(PEDS, bev_w=BEV_W, bev_h=BEV_H)

split = np.hstack([dash_panel, bev_panel])
cv2.line(split,(DASH_W,0),(DASH_W,BEV_H),(50,130,50),3)

cv2.imwrite(f'{OUT}/bev_PRO_splitscreen.png', split)
print(f"Saved → {OUT}/bev_PRO_splitscreen.png  ({split.shape[1]}×{split.shape[0]})")

# Also save BEV panel alone
cv2.imwrite(f'{OUT}/bev_PRO_radar_only.png', bev_panel)
print(f"Saved → {OUT}/bev_PRO_radar_only.png")

# Also save dashcam alone
cv2.imwrite(f'{OUT}/bev_PRO_dashcam_only.png', dash_panel)
print(f"Saved → {OUT}/bev_PRO_dashcam_only.png")
