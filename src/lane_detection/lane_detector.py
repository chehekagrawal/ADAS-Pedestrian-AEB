"""
Classical Lane Detection using Hough Transform.

Detects left and right lane lines from dashcam footage using a traditional
computer vision pipeline: grayscale → blur → Canny edges → ROI masking →
Hough Line Transform → slope-based classification → line extrapolation.

This feeds into the ego corridor module for in-path pedestrian filtering.

Limitations:
    - Works best on straight roads with clear lane markings
    - Curved roads require polynomial fitting (future enhancement)
    - Heavily degraded markings may not be detected
"""

import cv2
import numpy as np


def detect_lanes(frame):
    """
    Detect left and right lane lines using Hough Transform.

    Pipeline:
    1. Convert to grayscale
    2. Gaussian blur (5×5 kernel) to reduce noise
    3. Canny edge detection (thresholds: 50, 150)
    4. Apply trapezoidal ROI mask (road area only)
    5. Probabilistic Hough Line Transform
    6. Classify lines into left lane (negative slope) and right lane (positive slope)
    7. Average and extrapolate each group into a single line

    Args:
        frame: BGR frame (numpy H×W×3)

    Returns:
        list of lane lines, each as [(x1, y1), (x2, y2)].
        Typically returns 0, 1, or 2 lines (left and/or right).
        Empty list if no lanes detected.
    """
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]

    # Step 1: Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Apply ROI mask (only process road area)
    roi_mask = _create_roi_mask(frame.shape)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # Step 5: Hough Line Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,                    # Distance resolution (pixels)
        theta=np.pi / 180,       # Angle resolution (radians)
        threshold=50,             # Min votes to detect a line
        minLineLength=40,         # Min line length (pixels)
        maxLineGap=150            # Max gap between segments to merge
    )

    if lines is None:
        return []

    # Step 6: Classify into left and right lanes
    left_lines, right_lines = _classify_lines(lines, w)

    # Step 7: Average and extrapolate each group
    result = []
    left_lane = _average_and_extrapolate(left_lines, h)
    right_lane = _average_and_extrapolate(right_lines, h)

    if left_lane is not None:
        result.append(left_lane)
    if right_lane is not None:
        result.append(right_lane)

    return result


def _create_roi_mask(frame_shape):
    """
    Create a trapezoidal ROI covering the road area.

    The trapezoid narrows from the full frame width at the bottom
    to a narrow strip around the vanishing point near the horizon.

    Args:
        frame_shape: (height, width, channels)

    Returns:
        Binary mask (H×W, dtype uint8, 0 or 255)
    """
    h, w = frame_shape[:2]
    vertices = np.array([[
        (int(w * 0.1), h),              # bottom-left
        (int(w * 0.45), int(h * 0.6)),  # top-left (near vanishing point)
        (int(w * 0.55), int(h * 0.6)),  # top-right (near vanishing point)
        (int(w * 0.9), h),              # bottom-right
    ]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def _classify_lines(lines, frame_width):
    """
    Separate Hough lines into left lane and right lane by slope.

    Convention (image coordinates, y increases downward):
    - Left lane: negative slope (line goes from bottom-left to top-right)
    - Right lane: positive slope (line goes from bottom-right to top-left)

    Filters out near-horizontal lines (|slope| < 0.3) as they are typically
    road markings, shadows, or noise — not lane boundaries.

    Args:
        lines: Output from cv2.HoughLinesP — shape (N, 1, 4)
        frame_width: Width of the frame (used for position filtering)

    Returns:
        (left_lines, right_lines): Two lists of (slope, intercept) tuples
    """
    left_lines = []
    right_lines = []
    center_x = frame_width / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip vertical lines (would cause division by zero)
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter near-horizontal lines
        if abs(slope) < 0.3:
            continue

        # Classify by slope sign AND position relative to frame center
        # This prevents misclassification when lanes cross the center
        midpoint_x = (x1 + x2) / 2

        if slope < 0 and midpoint_x < center_x:
            left_lines.append((slope, intercept))
        elif slope > 0 and midpoint_x > center_x:
            right_lines.append((slope, intercept))

    return left_lines, right_lines


def _average_and_extrapolate(line_params, frame_height):
    """
    Average a group of (slope, intercept) pairs and extrapolate to full lane length.

    The extrapolated line runs from the bottom of the frame (y = frame_height)
    to the top of the ROI region (y ≈ 60% of frame_height).

    Args:
        line_params: list of (slope, intercept) tuples
        frame_height: Height of the frame

    Returns:
        [(x1, y1), (x2, y2)] or None if no valid lines
    """
    if not line_params:
        return None

    slopes, intercepts = zip(*line_params)
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)

    # Avoid division by zero for degenerate cases
    if abs(avg_slope) < 1e-6:
        return None

    # Extrapolate: y1 = bottom of frame, y2 = top of ROI
    y1 = frame_height
    y2 = int(frame_height * 0.6)

    x1 = int((y1 - avg_intercept) / avg_slope)
    x2 = int((y2 - avg_intercept) / avg_slope)

    return [(x1, y1), (x2, y2)]


def draw_lanes(frame, lanes, color=(0, 255, 0), thickness=3):
    """
    Overlay detected lane lines on a frame.

    Args:
        frame: BGR frame to draw on (modified in-place)
        lanes: List of lane lines, each as [(x1, y1), (x2, y2)]
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels

    Returns:
        frame: The annotated frame
    """
    overlay = frame.copy()

    for lane in lanes:
        (x1, y1), (x2, y2) = lane
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

    # Semi-transparent overlay for better visibility
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


def draw_roi(frame, alpha=0.3):
    """
    Draw the ROI trapezoid on the frame for debugging/visualization.

    Args:
        frame: BGR frame
        alpha: Transparency of the overlay (0.0–1.0)

    Returns:
        frame: The annotated frame
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    vertices = np.array([[
        (int(w * 0.1), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.9), h),
    ]], dtype=np.int32)

    cv2.fillPoly(overlay, vertices, (0, 255, 255))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
