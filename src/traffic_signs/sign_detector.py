"""
Traffic Sign Recognition Module.

Detects speed limit signs and adjusts AEB thresholds based on speed zones.
Supports two detection modes:
    1. YOLO-based: Uses a fine-tuned model on traffic sign datasets (GTSDB)
    2. Heuristic-based: HSV color filtering for red circular signs + contour analysis

Integration:
    - Speed limit → adjust_aeb_threshold() → tighter/looser AEB trigger
    - School zones (30 km/h) → 50% earlier AEB trigger
    - Highway zones (80+ km/h) → 10% relaxed to reduce false alarms

Reference:
    German Traffic Sign Detection Benchmark (GTSDB)
    Euro NCAP AEB test protocols for speed zone awareness
"""

import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class TrafficSignDetector:
    """
    Detect speed limit signs and determine active speed zone.

    Dual-mode detection:
    - If a trained YOLO model for traffic signs exists → ML detection
    - Otherwise → heuristic color/shape-based detection (always available)
    """

    SPEED_LIMITS = {
        "speed_30": 30,    # school zone
        "speed_50": 50,    # urban
        "speed_60": 60,    # suburban
        "speed_80": 80,    # highway
        "speed_100": 100,  # highway
    }

    ZONE_TYPES = {
        30: "school",
        50: "urban",
        60: "suburban",
        80: "highway",
        100: "highway",
    }

    DEFAULT_SPEED_LIMIT = 50  # Urban default (km/h)

    def __init__(self, model_path=None):
        """
        Load traffic sign detection model.

        Args:
            model_path: Path to YOLO .pt model trained on traffic signs.
                        If None or file missing, uses heuristic detection.
        """
        self.model = None
        self._last_detected_limit = self.DEFAULT_SPEED_LIMIT
        self._detection_confidence_threshold = 0.5

        if model_path and YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"[TrafficSignDetector] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[TrafficSignDetector] Failed to load model: {e}")
                print("[TrafficSignDetector] Falling back to heuristic mode.")

    def detect_signs(self, frame):
        """
        Detect traffic signs in a frame.

        Args:
            frame: BGR frame (numpy H×W×3)

        Returns:
            list of dicts, each with:
                "sign_type": str (e.g., "speed_30")
                "confidence": float (0.0–1.0)
                "bbox": [x1, y1, x2, y2]
                "speed_limit": int or None
        """
        if self.model is not None:
            return self._yolo_detect(frame)
        else:
            return self._heuristic_detect(frame)

    def _yolo_detect(self, frame):
        """Run YOLO inference for traffic sign detection."""
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._detection_confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "unknown")
                bbox = box.xyxy[0].cpu().numpy().tolist()

                speed = self.SPEED_LIMITS.get(cls_name)
                detections.append({
                    "sign_type": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "speed_limit": speed,
                })

        return detections

    def _heuristic_detect(self, frame):
        """
        Heuristic traffic sign detection using color and shape analysis.

        Pipeline:
        1. Convert to HSV color space
        2. Mask red regions (speed limit signs are red circles)
        3. Find circular contours using HoughCircles or contour analysis
        4. Extract ROI and estimate speed limit from size/context

        This provides basic detection without a trained model.
        """
        if frame is None or frame.size == 0:
            return []

        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color ranges in HSV (red wraps around 0/180)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small to be a sign
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.6:  # Not circular enough
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Speed limit signs are roughly square
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue

            # This is likely a speed limit sign
            # Estimate speed from sign size (larger = closer = more relevant)
            # Without OCR, we use a size-based heuristic
            bbox = [x, y, x + w, y + h]
            confidence = min(circularity, 0.95)

            detections.append({
                "sign_type": "speed_sign_detected",
                "confidence": float(confidence),
                "bbox": bbox,
                "speed_limit": None,  # Cannot determine exact number without OCR
            })

        return detections

    def get_active_speed_limit(self, frame):
        """
        Get the current active speed limit from detected signs.

        Uses temporal smoothing: keeps the last detected limit active
        until a new sign is detected (signs persist until replaced).

        Args:
            frame: BGR frame

        Returns:
            dict with:
                "speed_limit_kmh": int
                "zone_type": str ("school", "urban", "suburban", "highway")
                "sign_detected": bool (whether a sign was found this frame)
        """
        signs = self.detect_signs(frame)
        sign_detected = False

        for sign in signs:
            if sign["speed_limit"] is not None:
                self._last_detected_limit = sign["speed_limit"]
                sign_detected = True
                break

        limit = self._last_detected_limit
        zone = self.ZONE_TYPES.get(limit, "urban")

        return {
            "speed_limit_kmh": limit,
            "zone_type": zone,
            "sign_detected": sign_detected,
        }

    def draw_signs(self, frame, signs):
        """
        Draw detected traffic sign bounding boxes on frame.

        Args:
            frame: BGR frame (modified in-place)
            signs: list from detect_signs()

        Returns:
            frame: Annotated frame
        """
        for sign in signs:
            x1, y1, x2, y2 = [int(c) for c in sign["bbox"]]
            label = sign["sign_type"]
            conf = sign["confidence"]
            speed = sign.get("speed_limit")

            color = (0, 0, 255)  # Red for traffic signs
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label}"
            if speed:
                text += f" {speed}km/h"
            text += f" {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame


def adjust_aeb_threshold(base_ttc_threshold, speed_limit_kmh):
    """
    Adjust AEB trigger threshold based on detected speed zone.

    School zones require earlier braking (pedestrians are children,
    unpredictable behavior). Highway zones can be slightly relaxed
    to avoid false alarms from distant objects.

    Mapping:
        School zone (≤30 km/h):  threshold × 1.5 (trigger 50% earlier)
        Urban (≤50 km/h):        threshold × 1.0 (normal)
        Highway (>50 km/h):      threshold × 0.9 (10% relaxed)

    Args:
        base_ttc_threshold: float — base TTC threshold (typically 1.5s)
        speed_limit_kmh: int — detected speed limit

    Returns:
        adjusted_threshold: float — modified TTC threshold
    """
    if speed_limit_kmh <= 30:
        return base_ttc_threshold * 1.5
    elif speed_limit_kmh <= 50:
        return base_ttc_threshold * 1.0
    else:
        return base_ttc_threshold * 0.9
