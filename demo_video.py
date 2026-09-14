"""
ADAS AEB Demo Video Generator

Processes a dashcam MP4 through the full AEB pipeline and renders:
  Left  panel : annotated dashcam (YOLO + SORT, all 4 classes)
  Right panel : live BEV radar (pinhole depth, TTC, risk colour)
  Bottom-left : live Ego Speed + Brake Force graph
  Bottom-right: live Brake Rotor Temperature graph (thermodynamics)

Pipeline per frame:
  1. Resize 4K input to 1280x720 for inference
  2. CLAHE preprocessing
  3. YOLOv8 multi-class detection (4 classes)
  4. SORT tracking (Kalman + Hungarian)
  5. BEVRadarMapper — pinhole depth, TTC, risk label per track
  6. Adaptive AEB decision (weather-adjusted, all-class, ego-corridor gated)
  7. Brake thermodynamics + ego-speed model (live telemetry)
  8. Render 1280x720 split-screen + graph panels
  9. Write output MP4

Usage:
    python3 demo_video.py
    python3 demo_video.py --input path/to/video.mp4 --speed 40
    python3 demo_video.py --weather clear
"""

import sys, os, argparse, collections
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB')
sys.path.insert(0, '/home/atharv/ADAS-Pedestrian-AEB/src/tracking')
sys.path.insert(0, '/home/atharv/.local/lib/python3.10/site-packages')

import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from src.visualization.bev_radar import BEVRadarMapper, PedestrianTrack

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH  = '/home/atharv/ADAS-Pedestrian-AEB/models/yolo_multiclass_best.pt'
INPUT_PATH  = '/home/atharv/ADAS-Pedestrian-AEB/data/sample/BEV_input_video.mp4'
OUTPUT_DIR  = '/home/atharv/ADAS-Pedestrian-AEB/reports/final_deliverables'
OUTPUT_PATH = f'{OUTPUT_DIR}/aeb_demo_video.mp4'

# ── Layout ────────────────────────────────────────────────────────
INFER_W, INFER_H = 1280, 720
PANEL_W, PANEL_H = 640, 480
GRAPH_H          = 240          # bottom graph section height
OUT_W = PANEL_W * 2            # 1280
OUT_H = PANEL_H + GRAPH_H      # 720  (clean 720p output)

# ── AEB / Detection config ────────────────────────────────────────
CONF_THRESH   = 0.30
AEB_BASE_TTC  = 1.5
DEFAULT_SPEED_KMH = 30.0
CORRIDOR_M    = 4.0             # ±4 m lateral ego corridor

CLASS_NAMES   = {0: 'Ped', 1: 'Car', 2: 'Bike', 3: 'Moto'}
CLASS_HEIGHTS = {0: 1.70, 1: 1.50, 2: 1.10, 3: 1.20}
RISK_BGR      = {
    'SAFE':     (50,  200,  50),
    'WARNING':  (0,   190, 255),
    'CRITICAL': (30,   30, 230),
}

HISTORY_LEN = 150   # rolling graph window (~5 s at 30 fps)

_WEATHER_DELTAS = {'clear': 0.0, 'night': 0.2, 'rain': 0.3, 'fog': 0.5}


# ── Physics models ────────────────────────────────────────────────

class BrakeThermo:
    """Simple Newton cooling model for a single brake rotor."""
    T_AMB  = 25.0    # °C ambient
    K_COOL = 0.04    # Newton cooling rate  (1/s)
    K_HEAT = 20.0    # heating rate °C/s at full AEB braking

    def __init__(self):
        self.temp = self.T_AMB

    def update(self, intensity: float, dt: float = 1/30) -> float:
        """intensity 0.0 = free rolling, 1.0 = full AEB stop."""
        self.temp += self.K_HEAT * intensity * dt
        self.temp -= self.K_COOL * (self.temp - self.T_AMB) * dt
        return self.temp


class EgoSpeed:
    DECEL_MS2   = 0.8 * 9.81   # 0.8 g deceleration
    RECOVER_MS2 = 2.0           # gentle re-acceleration

    def __init__(self, initial_kmh: float):
        self.speed_ms  = initial_kmh / 3.6
        self.target_ms = initial_kmh / 3.6

    def update(self, intensity: float, dt: float = 1/30):
        """Returns (speed_kmh, brake_g)."""
        if intensity > 0:
            self.speed_ms = max(0.0, self.speed_ms - self.DECEL_MS2 * intensity * dt)
        else:
            self.speed_ms = min(self.target_ms, self.speed_ms + self.RECOVER_MS2 * dt)
        return self.speed_ms * 3.6, intensity * 0.8


# ── Weather ───────────────────────────────────────────────────────

def classify_weather(frame_bgr, override=None):
    if override:
        key = override.lower()
        return override.capitalize(), _WEATHER_DELTAS.get(key, 0.0)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 55:
        return 'Night', 0.2
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    if lap_var < 500 and sat < 50:
        return 'Fog', 0.5
    if brightness < 85 and lap_var < 800 and sat > 20:
        return 'Rain', 0.3
    return 'Clear', 0.0


# ── CLAHE ─────────────────────────────────────────────────────────

def clahe_enhance(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Dashcam annotation ────────────────────────────────────────────

def annotate_dashcam(frame_infer, tracks_raw, bev_tracks, sx, sy):
    panel = cv2.resize(frame_infer, (PANEL_W, PANEL_H))
    bev_map = {bt.track_id: bt for bt in bev_tracks}

    for trk in tracks_raw:
        x1, y1, x2, y2, tid, cls = trk
        cls = int(cls); tid = int(tid)
        x1s = int(x1 * sx); y1s = int(y1 * sy)
        x2s = int(x2 * sx); y2s = int(y2 * sy)
        class_name = CLASS_NAMES.get(cls, f'cls{cls}')

        bt = bev_map.get(tid)
        if bt is None:
            cv2.rectangle(panel, (x1s, y1s), (x2s, y2s), (90, 90, 90), 1)
            continue

        c = RISK_BGR[bt.risk_label]
        thickness = 3 if bt.risk_label == 'CRITICAL' else 2
        cv2.rectangle(panel, (x1s, y1s), (x2s, y2s), c, thickness)

        ttc_str = f'{bt.ttc_s:.1f}s' if np.isfinite(bt.ttc_s) else 'inf'
        label = f'{class_name} ID:{tid} {bt.Z_m:.0f}m TTC:{ttc_str}'
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ly = max(y1s - 1, lh + 4)
        cv2.rectangle(panel, (x1s, ly - lh - 4), (x1s + lw + 4, ly), c, -1)
        cv2.putText(panel, label, (x1s + 2, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        if bt.risk_label == 'CRITICAL':
            cv2.putText(panel, f'!! AEB: {class_name} !!', (x1s, y2s + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 230), 2)

    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (10, 28, 10), -1)
    cv2.putText(panel, 'DASHCAM  |  YOLOv8 + SORT  (Ped / Car / Bike / Moto)', (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 210, 150), 1)
    return panel


# ── Live graph rendering ──────────────────────────────────────────

def _norm_y(val, vmin, vmax, y_top, h):
    frac = (val - vmin) / max(vmax - vmin, 1e-6)
    frac = max(0.0, min(1.0, frac))
    return int(y_top + h - frac * h)


def draw_graph(canvas, x0, y0, w, h,
               hist_a, label_a, unit_a, color_a, vmin_a, vmax_a,
               hist_b=None, label_b=None, unit_b=None, color_b=None,
               vmin_b=None, vmax_b=None,
               title='', aeb_events=None):
    """
    Render a dual-line scrolling graph.
    hist_a / hist_b: deques of float values (len up to HISTORY_LEN)
    aeb_events: deque of bool — True frames get a red vertical tick
    """
    PAD_L, PAD_R, PAD_T, PAD_B = 42, 8, 26, 20

    # Background + border
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (8, 14, 8), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), (35, 70, 35), 1)

    gx0 = x0 + PAD_L
    gy0 = y0 + PAD_T
    gw  = w - PAD_L - PAD_R
    gh  = h - PAD_T - PAD_B

    # Horizontal grid + Y-axis labels (primary axis)
    N_GRID = 4
    for i in range(N_GRID + 1):
        yg = gy0 + int(gh * i / N_GRID)
        cv2.line(canvas, (gx0, yg), (gx0 + gw, yg), (20, 40, 20), 1)
        val = vmax_a - (vmax_a - vmin_a) * i / N_GRID
        cv2.putText(canvas, f'{val:.0f}', (x0 + 2, yg + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (70, 120, 70), 1)

    # AEB event ticks (red vertical lines)
    if aeb_events:
        for i, ev in enumerate(aeb_events):
            if ev:
                xp = gx0 + int(i * gw / max(HISTORY_LEN - 1, 1))
                cv2.line(canvas, (xp, gy0), (xp, gy0 + gh), (50, 50, 200), 1)

    # Plot helper
    def _plot(hist, vmin, vmax, col):
        if len(hist) < 2:
            return
        pts = []
        for i, v in enumerate(hist):
            xp = gx0 + int(i * gw / max(HISTORY_LEN - 1, 1))
            yp = _norm_y(v, vmin, vmax, gy0, gh)
            pts.append((xp, yp))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 3, col, -1, cv2.LINE_AA)

    _plot(hist_a, vmin_a, vmax_a, color_a)
    if hist_b is not None:
        _plot(hist_b, vmin_b, vmax_b, color_b)

    # Title bar
    cv2.putText(canvas, title, (x0 + PAD_L, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 220, 140), 1)

    # Current value readouts
    if hist_a:
        txt = f'{hist_a[-1]:.1f} {unit_a}'
        cv2.putText(canvas, txt, (x0 + PAD_L, y0 + h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color_a, 1)
    if hist_b and label_b:
        txt2 = f'{hist_b[-1]:.2f} {unit_b}'
        cv2.putText(canvas, txt2, (x0 + PAD_L + 120, y0 + h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color_b, 1)

    # Legend dots
    cv2.circle(canvas, (x0 + PAD_L, y0 + h - 6 - 10), 4, color_a, -1)
    if label_a:
        cv2.putText(canvas, label_a, (x0 + PAD_L + 7, y0 + h - 6 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color_a, 1)
    if hist_b and label_b:
        cv2.circle(canvas, (x0 + PAD_L + 120, y0 + h - 6 - 10), 4, color_b, -1)
        cv2.putText(canvas, label_b, (x0 + PAD_L + 128, y0 + h - 6 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color_b, 1)

    # Time axis label
    cv2.putText(canvas, '5 s window',
                (gx0 + gw // 2 - 22, y0 + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (45, 80, 45), 1)

    # Right-side secondary Y labels
    if hist_b and vmin_b is not None:
        for i in range(N_GRID + 1):
            yg = gy0 + int(gh * i / N_GRID)
            val2 = vmax_b - (vmax_b - vmin_b) * i / N_GRID
            cv2.putText(canvas, f'{val2:.1f}', (x0 + w - PAD_R - 30, yg + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, color_b, 1)


def prescan_crash_frame(input_path, model, speed_kmh, weather_override=None):
    """
    Quick first pass: find the single frame with the global minimum in-corridor TTC.
    Returns (crash_frame_idx, min_ttc).
    """
    from tracker import Tracker as _Tracker, KalmanBoxTracker as _KBT
    _KBT.count = 0
    ps_tracker = _Tracker(max_age=3, min_hits=2, iou_threshold=0.3)
    ps_mapper  = BEVRadarMapper(focal_length_px=700, img_width=INFER_W, img_height=INFER_H)
    ego_ms = speed_kmh / 3.6

    cap = cv2.VideoCapture(input_path)
    global_min_ttc   = float('inf')
    crash_frame_idx  = 1
    frame_idx        = 0

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame     = cv2.resize(frame_raw, (INFER_W, INFER_H))
        frame_enh = clahe_enhance(frame)
        _, w_delta = classify_weather(frame_enh, override=weather_override)
        threshold  = AEB_BASE_TTC + w_delta

        results = model(frame_enh, conf=CONF_THRESH, verbose=False)[0]
        boxes   = results.boxes
        dets    = []
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

        dets_arr   = np.array(dets, dtype=float) if dets else np.empty((0, 6))
        tracks_raw = ps_tracker.update(dets_arr)

        for t in tracks_raw:
            x1, y1, x2, y2, tid, cls = t
            h_px = float(y2 - y1)
            if h_px <= 0:
                continue
            depth = ps_mapper.focal_length_px * CLASS_HEIGHTS.get(int(cls), 1.5) / h_px
            if not np.isfinite(depth) or depth <= 0:
                continue
            bbox = (int(x1), int(y1), int(x2), int(y2))
            X, _ = ps_mapper.project_to_bev(bbox, depth)
            if abs(X) > CORRIDOR_M:
                continue
            ttc = ps_mapper.compute_ttc(depth, ego_ms)
            if ttc < global_min_ttc:
                global_min_ttc  = ttc
                crash_frame_idx = frame_idx

    cap.release()
    _KBT.count = 0   # reset so main pass gets consistent IDs
    return crash_frame_idx, global_min_ttc


def draw_collision_warning(canvas, aeb_status, frame_idx,
                            crash_frame=None, fps=30.0):
    """Overlay a big flashing warning on the dashcam panel.
    Shows for exactly 1 second before the crash frame and stops after it.
    """
    is_aeb  = 'TRIGGER' in aeb_status
    is_coll = 'COLL' in aeb_status
    if not (is_aeb or is_coll):
        return
    if crash_frame is not None:
        pre_frames = int(fps)          # 1 second before
        post_frames = int(fps * 0.1)   # ~3 frames after (visual closure)
        if frame_idx < crash_frame - pre_frames or frame_idx > crash_frame + post_frames:
            return

    # Flash: visible every other ~8-frame block
    if (frame_idx // 8) % 2 == 0:
        # Semi-transparent red fill over dashcam panel
        overlay = canvas[:PANEL_H, :PANEL_W].copy()
        overlay[:] = (0, 0, 200)
        cv2.addWeighted(overlay, 0.25, canvas[:PANEL_H, :PANEL_W], 0.75, 0,
                        canvas[:PANEL_H, :PANEL_W])

        # Red pulsing border around the dashcam panel
        for t in range(1, 5):
            cv2.rectangle(canvas, (t, t), (PANEL_W - t, PANEL_H - t), (0, 0, 255), 1)

    # Always-on big centred text
    label   = '!! AEB TRIGGERED !!' if is_aeb else '!! COLLISION WARNING !!'
    color   = (30, 30, 255) if is_aeb else (0, 100, 255)
    font    = cv2.FONT_HERSHEY_DUPLEX
    scale   = 0.75
    thick   = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tx = (PANEL_W - tw) // 2
    ty = PANEL_H // 2 + th // 2

    # Dark backing rectangle
    cv2.rectangle(canvas, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                  (0, 0, 0), -1)
    cv2.rectangle(canvas, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                  color, 2)
    cv2.putText(canvas, label, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def draw_info_strip(canvas, aeb_status, weather, threshold, frame_idx, total):
    """Thin info bar at the very bottom of the graph section."""
    y = PANEL_H + GRAPH_H - 18
    strip_color = (30, 30, 200) if 'TRIGGER' in aeb_status else \
                  (0, 140, 255) if 'COLL' in aeb_status else \
                  (0, 180, 240) if 'WARN' in aeb_status else (40, 160, 40)
    cv2.rectangle(canvas, (0, y), (OUT_W, OUT_H), (12, 20, 12), -1)
    cv2.putText(canvas, f'AEB: {aeb_status}',
                (8, y + 13), cv2.FONT_HERSHEY_DUPLEX, 0.50, strip_color, 1)
    cv2.putText(canvas, f'Thr:{threshold:.1f}s  Weather:{weather}  Frame:{frame_idx}/{total}',
                (260, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 170), 1)


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',   default=INPUT_PATH)
    ap.add_argument('--output',  default=OUTPUT_PATH)
    ap.add_argument('--speed',   type=float, default=DEFAULT_SPEED_KMH,
                    help='Assumed ego speed in km/h')
    ap.add_argument('--weather', default=None,
                    choices=['clear', 'night', 'rain', 'fog'],
                    help='Force weather label (skips auto-detection)')
    args = ap.parse_args()

    ego_ms = args.speed / 3.6
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading YOLOv8 model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # Pre-scan: find the frame with the global minimum corridor TTC
    print("Pre-scanning to locate crash frame ...")
    crash_frame, crash_ttc = prescan_crash_frame(
        args.input, model, args.speed, weather_override=args.weather)
    print(f"  → Crash frame: {crash_frame}  (TTC = {crash_ttc:.2f} s)")

    tracker = Tracker(max_age=3, min_hits=2, iou_threshold=0.3)
    KalmanBoxTracker_count_reset = __import__(
        'tracker', fromlist=['KalmanBoxTracker']).KalmanBoxTracker
    KalmanBoxTracker_count_reset.count = 0

    mapper = BEVRadarMapper(focal_length_px=700,
                            img_width=INFER_W, img_height=INFER_H)

    thermo = BrakeThermo()
    speedo = EgoSpeed(args.speed)

    # Rolling history deques
    hist_speed  = collections.deque(maxlen=HISTORY_LEN)
    hist_brake  = collections.deque(maxlen=HISTORY_LEN)
    hist_temp   = collections.deque(maxlen=HISTORY_LEN)
    hist_aeb_ev = collections.deque(maxlen=HISTORY_LEN)  # bool per frame

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt    = 1.0 / fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (OUT_W, OUT_H))

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Frames: {total}  |  FPS: {fps:.1f}  |  Ego speed: {args.speed} km/h")
    print("-" * 60)

    sx = PANEL_W / INFER_W
    sy = PANEL_H / INFER_H
    frame_idx = 0
    max_temp_seen = BrakeThermo.T_AMB + 10.0   # grows dynamically

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Resize → inference resolution
        frame = cv2.resize(frame_raw, (INFER_W, INFER_H))

        # 2. CLAHE
        frame_enh = clahe_enhance(frame)

        # 3. Weather
        weather, w_delta = classify_weather(frame, override=args.weather)
        threshold = AEB_BASE_TTC + w_delta

        # 4. YOLO detection
        results = model(frame_enh, conf=CONF_THRESH, verbose=False)[0]
        boxes = results.boxes

        all_dets = []
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0])
                cls   = int(box.cls[0])
                all_dets.append([x1, y1, x2, y2, score, cls])

        dets_array = np.array(all_dets, dtype=float) if all_dets \
                     else np.empty((0, 6))

        # 5. SORT tracking
        tracks_raw = tracker.update(dets_array)

        # 6. Build BEV tracks for all 4 classes
        bev_tracks = []
        for t in tracks_raw:
            x1, y1, x2, y2, tid, cls = t
            tid = int(tid); cls = int(cls)
            bbox  = (int(x1), int(y1), int(x2), int(y2))
            h_px  = float(y2 - y1)
            if h_px <= 0:
                continue
            h_real = CLASS_HEIGHTS.get(cls, 1.5)
            depth  = mapper.focal_length_px * h_real / h_px
            if not np.isfinite(depth) or depth <= 0:
                continue
            X, Z = mapper.project_to_bev(bbox, depth)
            ttc  = mapper.compute_ttc(depth, ego_ms)
            color_bgr, risk_label = mapper.ttc_to_color(ttc, threshold)
            bev_tracks.append(PedestrianTrack(
                track_id=tid, bbox=bbox, depth_m=depth,
                X_m=X, Z_m=Z, ttc_s=ttc,
                color_bgr=color_bgr, risk_label=risk_label,
            ))

        # 7. AEB decision
        #    Pedestrian CRITICAL  → AEB TRIGGERED  (full brake, intensity 1.0)
        #    Vehicle CRITICAL     → COLL. WARNING   (partial brake, intensity 0.5)
        #    Any WARNING in path  → WARNING          (light brake, intensity 0.2)
        in_path = [bt for bt in bev_tracks if abs(bt.X_m) <= CORRIDOR_M]

        # Build a track_id → class lookup from tracks_raw
        tid_to_cls = {int(t[4]): int(t[5]) for t in tracks_raw}

        ped_critical = any(
            bt.risk_label == 'CRITICAL' and tid_to_cls.get(bt.track_id, -1) == 0
            for bt in in_path
        )
        veh_critical = any(
            bt.risk_label == 'CRITICAL' and tid_to_cls.get(bt.track_id, -1) != 0
            for bt in in_path
        )
        has_warning = any(bt.risk_label == 'WARNING' for bt in in_path)

        valid_ttcs = [bt.ttc_s for bt in in_path if np.isfinite(bt.ttc_s)]
        min_ttc    = min(valid_ttcs) if valid_ttcs else 999.0

        if ped_critical:
            aeb_status = 'AEB TRIGGERED'
            intensity  = 1.0
        elif veh_critical:
            aeb_status = 'COLL. WARNING'
            intensity  = 0.5
        elif has_warning:
            aeb_status = 'WARNING'
            intensity  = 0.2
        else:
            aeb_status = 'SAFE'
            intensity  = 0.0
        speed_kmh, brake_g = speedo.update(intensity, dt)
        temp_c = thermo.update(intensity, dt)
        max_temp_seen = max(max_temp_seen, temp_c)

        hist_speed.append(speed_kmh)
        hist_brake.append(brake_g)
        hist_temp.append(temp_c)
        hist_aeb_ev.append(ped_critical)

        # 9. Dashcam panel
        dash_panel = annotate_dashcam(frame, tracks_raw, bev_tracks, sx, sy)

        # 10. BEV radar panel
        bev_panel = mapper.render_bev_frame(bev_tracks,
                                            canvas_hw=(PANEL_H, PANEL_W),
                                            range_m=50, lateral_m=15)

        # 11. Compose canvas (1280 x 720)
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        canvas[:PANEL_H, :PANEL_W]  = dash_panel
        canvas[:PANEL_H, PANEL_W:]  = bev_panel
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (60, 120, 60), 2)

        # Separator between panels and graphs
        cv2.line(canvas, (0, PANEL_H), (OUT_W, PANEL_H), (40, 80, 40), 1)

        # 12. Speed + Brake Force graph (bottom-left)
        STRIP_H = 18   # thin info strip height at the very bottom
        graph_h = GRAPH_H - STRIP_H
        draw_graph(
            canvas,
            x0=0, y0=PANEL_H, w=PANEL_W, h=graph_h,
            hist_a=hist_speed,
            label_a='Speed', unit_a='km/h', color_a=(60, 220, 100),
            vmin_a=0, vmax_a=max(args.speed * 1.1, 10),
            hist_b=hist_brake,
            label_b='Brake', unit_b='g', color_b=(60, 80, 230),
            vmin_b=0, vmax_b=1.0,
            title='EGO SPEED  +  BRAKE FORCE',
            aeb_events=hist_aeb_ev,
        )

        # 13. Brake Temperature graph (bottom-right)
        temp_ceil = max(max_temp_seen * 1.15, BrakeThermo.T_AMB + 50)
        draw_graph(
            canvas,
            x0=PANEL_W, y0=PANEL_H, w=PANEL_W, h=graph_h,
            hist_a=hist_temp,
            label_a='Rotor Temp', unit_a='°C', color_a=(40, 140, 255),
            vmin_a=BrakeThermo.T_AMB, vmax_a=temp_ceil,
            title='BRAKE ROTOR TEMPERATURE',
            aeb_events=hist_aeb_ev,
        )

        # 14. Thin info strip
        draw_info_strip(canvas, aeb_status, weather, threshold, frame_idx, total)

        # 15. Big collision warning overlay (dashcam panel) — 1 s before crash frame
        draw_collision_warning(canvas, aeb_status, frame_idx, crash_frame, fps)

        writer.write(canvas)

        if frame_idx % 60 == 0 or frame_idx == 1:
            pct = frame_idx / max(total, 1) * 100
            print(f'  [{pct:5.1f}%] Frame {frame_idx:4d}/{total}'
                  f'  AEB:{aeb_status:<14s}  TTC:{min_ttc:5.1f}s'
                  f'  Spd:{speed_kmh:5.1f}km/h  T:{temp_c:5.1f}°C  {weather}')

    cap.release()
    writer.release()
    print("-" * 60)
    print(f"Done — {frame_idx} frames written to:\n  {args.output}")


if __name__ == '__main__':
    main()
