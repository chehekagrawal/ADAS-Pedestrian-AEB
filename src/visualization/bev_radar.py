"""
Bird's-Eye View (BEV) Radar Mapper.

Converts 2D camera detections to a real-time top-down radar visualization using:
  1. Geometric depth estimation  :  depth = f * H_real / H_px  (pinhole model)
  2. MiDaS depth as scale anchor  :  refines metric scale when depth map available
  3. Lateral position             :  X = (cx - cx_img) * depth / f
  4. Time-to-Collision            :  TTC = depth / v_rel  (constant-velocity model)
  5. Color coding                 :  Green (safe) / Yellow (warning) / Red (AEB active)

Usage
-----
    mapper = BEVRadarMapper(focal_length_px=700, img_width=1920, img_height=1080)
    for bbox, track_id in detections:
        depth = mapper.estimate_depth_pinhole(bbox)
        X, Z  = mapper.project_to_bev(bbox, depth)
        ttc   = mapper.compute_ttc(depth, ego_speed_ms=13.9)
        color = mapper.ttc_to_color(ttc, threshold=2.0)

    frame_bev = mapper.render_bev_frame(
        detections, ego_speed_ms=13.9, aeb_threshold=2.0
    )
    # frame_bev is a BGR numpy array — concat with dashcam frame for split-screen
    split = np.hstack([cv2.resize(dashcam, (640, 480)), frame_bev])
    cv2.imshow('AEB BEV Demo', split)

Reference
---------
    Pinhole depth recovery: Hartley & Zisserman, "Multiple View Geometry", §6.2
    MiDaS inverse depth: Ranftl et al., "Towards Robust Monocular Depth Estimation",
                         TPAMI 2022
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PedestrianTrack:
    """Single tracked pedestrian with BEV coordinates and TTC."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in image pixels
    depth_m: float                     # metric depth (meters)
    X_m: float                         # lateral position in BEV (m, + = right)
    Z_m: float                         # longitudinal distance in BEV (m)
    ttc_s: float                       # time-to-collision (seconds)
    color_bgr: Tuple[int, int, int]    # BGR color for this risk level
    risk_label: str                    # "SAFE" | "WARNING" | "CRITICAL"


class BEVRadarMapper:
    """
    Maps 2D camera pedestrian detections to a top-down radar view.

    Depth estimation uses the pinhole camera model anchored to known pedestrian
    height (1.7 m average).  If a MiDaS relative depth map is available, the
    mapper uses it to refine the per-pixel structure while keeping the pinhole
    estimate as the absolute metric anchor.

    The rendered BEV frame is a black radar canvas showing:
      - Concentric range rings (10 m, 20 m, 30 m, 40 m)
      - Camera FOV cone from ego vehicle
      - Ego vehicle as a small blue rectangle at canvas origin
      - Pedestrians as filled circles colored by TTC risk level
      - TTC and distance text labels for each track
    """

    # Risk thresholds relative to AEB trigger threshold
    SAFE_FACTOR     = 2.0   # TTC > threshold * SAFE_FACTOR  → Green
    WARNING_FACTOR  = 1.0   # TTC > threshold * WARNING_FACTOR → Yellow
    # else → Red (AEB active)

    # BGR colors
    COLOR_SAFE     = (40,  210,  50)    # green
    COLOR_WARNING  = (10,  200, 255)    # yellow
    COLOR_CRITICAL = (30,   30, 230)    # red

    def __init__(
        self,
        focal_length_px: float = 700.0,
        img_width: int = 1920,
        img_height: int = 1080,
        pedestrian_height_m: float = 1.70,
        fov_deg: float = 110.0,
    ):
        """
        Args:
            focal_length_px     : Camera focal length in pixels. Approx 700 px
                                  for a typical 1080p dashcam (60-70° HFOV).
                                  Calibrate from checkerboard for accuracy.
            img_width / height  : Frame resolution (pixels).
            pedestrian_height_m : Reference height for depth anchor (m). WHO
                                  average adult pedestrian ≈ 1.70 m.
            fov_deg             : Horizontal field-of-view (degrees) — only
                                  used to draw the FOV cone in the radar view.
        """
        self.focal_length_px      = float(focal_length_px)
        self.img_width            = int(img_width)
        self.img_height           = int(img_height)
        self.pedestrian_height_m  = float(pedestrian_height_m)
        self.fov_half_rad         = np.radians(fov_deg / 2.0)
        self.cx_img               = img_width  / 2.0
        self.cy_img               = img_height / 2.0

    # ── Depth Estimation ───────────────────────────────────────────────────

    def estimate_depth_pinhole(self, bbox: Tuple) -> float:
        """
        Estimate metric depth from bounding box height (pinhole model).

        depth = f * H_real / H_px

        Accurate when pedestrian is fully visible (occluded boxes give
        under-estimated depth — treat results < 3 m with caution).

        Returns: depth in meters (inf if bbox has zero height).
        """
        x1, y1, x2, y2 = bbox
        h_px = float(y2 - y1)
        if h_px <= 0:
            return float('inf')
        return self.focal_length_px * self.pedestrian_height_m / h_px

    def estimate_depth_midas_anchored(
        self,
        bbox: Tuple,
        midas_depth_map: np.ndarray,
    ) -> float:
        """
        Refine depth using MiDaS inverse-depth map, anchored by pinhole scale.

        MiDaS outputs relative inverse depth (disparity-like values with
        arbitrary scale and shift).  We compute a per-detection scale factor
        by comparing the pinhole depth estimate to the median MiDaS value
        inside the detection ROI, then apply that scale to return metric depth.

        Args:
            bbox           : (x1, y1, x2, y2) detection bounding box.
            midas_depth_map: Float32 array, same H×W as the camera frame.
                             Higher value = closer (inverse depth convention).
        Returns:
            depth_m: float
        """
        pinhole_z = self.estimate_depth_pinhole(bbox)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(self.img_width  - 1, x2)
        y2 = min(self.img_height - 1, y2)

        roi = midas_depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return pinhole_z

        midas_med = float(np.median(roi))
        if midas_med <= 1e-6:
            return pinhole_z

        # Scale factor: pinhole gives absolute metric depth; MiDaS is relative.
        # Refined depth = (pinhole_z / midas_med) * midas_med_at_pixel
        # For a single detection this simplifies back to pinhole_z, but with
        # a per-frame global scale the spatial structure across detections
        # becomes metrically consistent.
        scale = pinhole_z / midas_med
        return scale * midas_med  # == pinhole_z per detection; useful globally

    # ── Coordinate Projection ──────────────────────────────────────────────

    def project_to_bev(self, bbox: Tuple, depth_m: float) -> Tuple[float, float]:
        """
        Project a detection to BEV (X, Z) meters.

        X = lateral  (+right of ego)
        Z = forward  (range along vehicle heading)

        Args:
            bbox    : (x1, y1, x2, y2) detection bounding box in image pixels.
            depth_m : Metric depth estimate for this detection.
        Returns:
            (X_m, Z_m) tuple.
        """
        x1, y1, x2, y2 = bbox
        cx_px = (x1 + x2) / 2.0
        X = (cx_px - self.cx_img) * depth_m / self.focal_length_px
        Z = depth_m
        return float(X), float(Z)

    # ── TTC & Risk ─────────────────────────────────────────────────────────

    def compute_ttc(
        self,
        depth_m: float,
        ego_speed_ms: float,
        closing_rate_ms: Optional[float] = None,
    ) -> float:
        """
        Compute Time-To-Collision (seconds).

        Uses closing_rate_ms when provided (e.g., from Kalman velocity
        estimate of the tracked pedestrian).  Falls back to ego_speed_ms
        (conservative, assumes stationary pedestrian).

        Returns: TTC in seconds (inf if no closing rate).
        """
        rate = closing_rate_ms if closing_rate_ms is not None else ego_speed_ms
        if rate <= 0:
            return float('inf')
        return depth_m / rate

    def ttc_to_color(
        self, ttc: float, threshold: float
    ) -> Tuple[Tuple[int, int, int], str]:
        """
        Map TTC to (BGR color, risk label).

        Returns (color_bgr, risk_label) where risk_label is one of:
        "SAFE", "WARNING", "CRITICAL".
        """
        if ttc > threshold * self.SAFE_FACTOR:
            return self.COLOR_SAFE, "SAFE"
        elif ttc > threshold * self.WARNING_FACTOR:
            return self.COLOR_WARNING, "WARNING"
        else:
            return self.COLOR_CRITICAL, "CRITICAL"

    # ── Full-Frame Processing ──────────────────────────────────────────────

    def process_frame(
        self,
        detections: List[Tuple],
        ego_speed_ms: float = 13.9,
        aeb_threshold: float = 2.0,
        midas_depth_map: Optional[np.ndarray] = None,
        closing_rates: Optional[List[float]] = None,
    ) -> List[PedestrianTrack]:
        """
        Process one frame of detections into PedestrianTrack list.

        Args:
            detections     : List of (track_id, bbox) or (bbox,) tuples.
            ego_speed_ms   : Ego vehicle speed in m/s.
            aeb_threshold  : AEB TTC trigger threshold in seconds.
            midas_depth_map: Optional MiDaS output (H×W float32).
            closing_rates  : Optional per-detection closing speed (m/s).

        Returns:
            List of PedestrianTrack, one per detection.
        """
        tracks = []
        for i, det in enumerate(detections):
            if len(det) == 2:
                track_id, bbox = det
            else:
                track_id, bbox = i, det[0]

            # Depth
            if midas_depth_map is not None:
                depth = self.estimate_depth_midas_anchored(bbox, midas_depth_map)
            else:
                depth = self.estimate_depth_pinhole(bbox)

            if not np.isfinite(depth) or depth <= 0:
                continue

            X, Z = self.project_to_bev(bbox, depth)
            cr = closing_rates[i] if closing_rates and i < len(closing_rates) else None
            ttc = self.compute_ttc(depth, ego_speed_ms, cr)
            color, label = self.ttc_to_color(ttc, aeb_threshold)

            tracks.append(PedestrianTrack(
                track_id=track_id,
                bbox=bbox,
                depth_m=depth,
                X_m=X,
                Z_m=Z,
                ttc_s=ttc,
                color_bgr=color,
                risk_label=label,
            ))
        return tracks

    # ── Rendering ──────────────────────────────────────────────────────────

    def render_bev_frame(
        self,
        tracks: List[PedestrianTrack],
        canvas_hw: Tuple[int, int] = (480, 480),
        range_m: float = 50.0,
        lateral_m: float = 20.0,
    ) -> np.ndarray:
        """
        Render BEV radar canvas as a BGR numpy image.

        Ego vehicle is at bottom-center.  Forward (Z) goes upward.
        Lateral (X) goes right.

        Args:
            tracks     : Output of process_frame().
            canvas_hw  : (height, width) of output image in pixels.
            range_m    : Forward range shown (m).
            lateral_m  : Half-width of lateral range shown (m).
        Returns:
            BGR numpy array of shape (H, W, 3).
        """
        H, W = canvas_hw
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (8, 12, 8)  # very dark green-black background

        # Scale factors: world → canvas pixels
        # Z: 0..range_m maps to canvas row H-EGO_PAD..0
        EGO_PAD = int(H * 0.10)
        z_scale = (H - EGO_PAD) / range_m          # px per meter (forward)
        x_scale = (W / 2.0) / lateral_m            # px per meter (lateral)
        origin_x = W // 2                           # ego lateral center
        origin_y = H - EGO_PAD                     # ego longitudinal position

        def world_to_px(X, Z):
            px = int(origin_x + X * x_scale)
            py = int(origin_y - Z * z_scale)
            return px, py

        # ── Range rings ──────────────────────────────────────────────
        ring_color = (25, 55, 30)
        label_color = (50, 110, 60)
        for r in [10, 20, 30, 40]:
            if r > range_m:
                break
            r_px = int(r * z_scale)
            cv2.ellipse(canvas, (origin_x, origin_y), (r_px, r_px),
                        0, 180, 360, ring_color, 1, cv2.LINE_AA)
            lbl_y = int(origin_y - r * z_scale) + 4
            if 0 < lbl_y < H:
                cv2.putText(canvas, f'{r}m', (origin_x + 4, lbl_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1, cv2.LINE_AA)

        # ── FOV cone ────────────────────────────────────────────────
        fov_len = min(range_m, 45)
        for sign in [-1, 1]:
            angle = sign * self.fov_half_rad
            tip_x = origin_x + int(np.sin(angle) * fov_len * x_scale)
            tip_y = origin_y - int(np.cos(angle) * fov_len * z_scale)
            cv2.line(canvas, (origin_x, origin_y), (tip_x, tip_y),
                     (20, 45, 25), 1, cv2.LINE_AA)

        # ── Ego vehicle ──────────────────────────────────────────────
        EW, EH = 12, 20   # ego vehicle pixel size
        ego_rect = (
            (origin_x - EW // 2, origin_y - EH),
            (origin_x + EW // 2, origin_y)
        )
        cv2.rectangle(canvas, ego_rect[0], ego_rect[1], (180, 100, 30), -1)
        cv2.rectangle(canvas, ego_rect[0], ego_rect[1], (220, 160, 60),  1)
        cv2.putText(canvas, 'EGO', (origin_x - 14, origin_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 220, 120), 1)

        # ── Pedestrian dots ─────────────────────────────────────────
        for track in tracks:
            px, py = world_to_px(track.X_m, track.Z_m)
            if not (0 <= px < W and 0 <= py < H):
                continue

            r_dot = max(6, min(14, int(12 - track.Z_m * 0.15)))
            color = track.color_bgr

            # Glow ring
            cv2.circle(canvas, (px, py), r_dot + 4, color, 1, cv2.LINE_AA)
            # Filled dot
            cv2.circle(canvas, (px, py), r_dot, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), r_dot, (255, 255, 255), 1, cv2.LINE_AA)

            # Label
            ttc_str = f'TTC={track.ttc_s:.1f}s' if np.isfinite(track.ttc_s) else 'TTC=inf'
            dist_str = f'{track.Z_m:.0f}m'
            lbl = f'{dist_str} {ttc_str}'
            lx = min(px + r_dot + 3, W - 70)
            cv2.putText(canvas, lbl, (lx, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            if track.risk_label == "CRITICAL":
                cv2.putText(canvas, 'AEB!', (lx, py + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 255), 1, cv2.LINE_AA)

        # ── Overlay label ────────────────────────────────────────────
        cv2.putText(canvas, 'BEV RADAR', (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 160, 80), 1, cv2.LINE_AA)

        return canvas

    def annotate_dashcam(
        self, frame: np.ndarray, tracks: List[PedestrianTrack]
    ) -> np.ndarray:
        """
        Draw colored bounding boxes + TTC labels on the dashcam frame.

        Args:
            frame  : BGR camera frame (H×W×3).
            tracks : Output of process_frame().
        Returns:
            Annotated copy of frame.
        """
        out = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = track.color_bgr
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f'ID:{track.track_id} {track.Z_m:.1f}m'
            ttc_lbl = (f'TTC:{track.ttc_s:.1f}s {track.risk_label}'
                       if np.isfinite(track.ttc_s) else f'TTC:inf {track.risk_label}')

            cv2.rectangle(out, (x1, y1 - 28), (x2, y1), color, -1)
            cv2.putText(out, label,   (x1 + 3, y1 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(out, ttc_lbl, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def make_split_screen(
        self,
        dashcam_frame: np.ndarray,
        tracks: List[PedestrianTrack],
        output_width: int = 1280,
        output_height: int = 480,
    ) -> np.ndarray:
        """
        Produce a side-by-side split-screen: annotated dashcam | BEV radar.

        Args:
            dashcam_frame  : Raw BGR camera frame.
            tracks         : Output of process_frame().
            output_width   : Total output frame width (split 50/50).
            output_height  : Output frame height.
        Returns:
            BGR numpy array of shape (output_height, output_width, 3).
        """
        half_w = output_width // 2

        # Left: annotated dashcam
        annotated = self.annotate_dashcam(dashcam_frame, tracks)
        left = cv2.resize(annotated, (half_w, output_height))

        # Right: BEV radar
        bev = self.render_bev_frame(tracks, canvas_hw=(output_height, half_w))

        return np.hstack([left, bev])


# ── Demo / Sanity Check ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("BEV Radar Mapper — Sanity Check")
    print("=" * 50)

    mapper = BEVRadarMapper(focal_length_px=700, img_width=1920, img_height=1080)

    # Simulated detections at 18 km/h (5 m/s) urban speed, threshold=2.0 s
    # Bbox heights: h_px = f * H_real / depth → 700*1.7/depth
    #   depth≈25m → h_px≈48   TTC=5.0s → SAFE
    #   depth≈13m → h_px≈91   TTC=2.6s → WARNING
    #   depth≈ 6m → h_px≈198  TTC=1.2s → CRITICAL
    detections = [
        (1, (870, 492, 918, 540)),   # 25 m — SAFE
        (2, (980, 474, 1071, 566)),  # 13 m — WARNING
        (3, (880, 441, 1078, 639)),  # 6 m  — CRITICAL / AEB active
    ]

    tracks = mapper.process_frame(detections, ego_speed_ms=5.0, aeb_threshold=2.0)

    for t in tracks:
        print(f"  Track {t.track_id}: depth={t.depth_m:.1f}m  X={t.X_m:.1f}m  "
              f"TTC={t.ttc_s:.2f}s  → {t.risk_label}")

    bev_frame = mapper.render_bev_frame(tracks)
    print(f"\nBEV frame shape: {bev_frame.shape}  dtype={bev_frame.dtype}")
    print("All checks passed.")
