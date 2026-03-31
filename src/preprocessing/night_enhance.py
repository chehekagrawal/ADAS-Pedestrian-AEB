"""
CLAHE Night Vision Enhancement Module.

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve
YOLO detection and dlib face detection accuracy in low-light conditions.

CLAHE operates on the L (lightness) channel in LAB color space, enhancing
local contrast without amplifying noise — unlike standard histogram equalization.

Reference:
    Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization"
    Graphics Gems IV, Academic Press, 1994.
"""

import cv2
import numpy as np


def enhance_frame(frame, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves YOLO detection and dlib face detection in low-light.

    Steps:
    1. Convert BGR → LAB color space
    2. Apply CLAHE to the L (lightness) channel only
    3. Merge back and convert to BGR

    Args:
        frame: BGR frame (numpy H×W×3, dtype uint8)
        clip_limit: Contrast limiting threshold (higher = more contrast, more noise)
        grid_size: Tile grid size for adaptive equalization (smaller = more local)

    Returns:
        Enhanced BGR frame (same shape and dtype as input)
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def is_low_light(frame, threshold=80):
    """
    Auto-detect if frame is low-light by checking mean brightness.

    Uses the grayscale mean intensity as a proxy for overall scene brightness.
    If mean brightness < threshold → frame is low-light and should be enhanced.

    Args:
        frame: BGR frame (numpy H×W×3)
        threshold: Brightness threshold (0-255). Default 80 works well for
                   typical dashcam footage.

    Returns:
        bool — True if frame is low-light (should apply CLAHE)
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def get_brightness(frame):
    """
    Get the mean brightness value of a frame.

    Useful for logging / dashboard display.

    Args:
        frame: BGR frame

    Returns:
        float — mean brightness (0.0–255.0)
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def adaptive_enhance(frame, threshold=80, clip_limit=2.0, grid_size=(8, 8)):
    """
    Conditionally apply CLAHE only if the frame is low-light.

    This is the recommended entry point for pipeline integration —
    it automatically decides whether enhancement is needed.

    Args:
        frame: BGR frame
        threshold: Low-light brightness threshold
        clip_limit: CLAHE clip limit
        grid_size: CLAHE tile grid size

    Returns:
        tuple: (output_frame, was_enhanced: bool)
    """
    if is_low_light(frame, threshold):
        return enhance_frame(frame, clip_limit, grid_size), True
    return frame, False


def generate_comparison(frame, save_path):
    """
    Generate side-by-side before/after comparison image for documentation.

    Left half = original frame, Right half = CLAHE-enhanced frame.

    Args:
        frame: BGR frame (original, possibly dark)
        save_path: File path to save the comparison image

    Returns:
        comparison: The side-by-side BGR image (also saved to disk)
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None")

    enhanced = enhance_frame(frame)
    comparison = np.hstack([frame, enhanced])
    cv2.imwrite(save_path, comparison)
    return comparison
