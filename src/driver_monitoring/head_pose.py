"""
Head Pose & Gaze Estimation Module.

Uses cv2.solvePnP to estimate 3D head orientation (yaw, pitch, roll) from
dlib's 68-point facial landmarks. Determines if the driver is distracted
by looking away from the road.

Integration:
    - Receives landmarks from DrowsinessDetector (Part 5)
    - Feeds DISTRACTED state into AlertnessState → increased reaction_time → earlier AEB trigger

Reference:
    Perspective-n-Point (PnP) pose estimation using OpenCV's solvePnP.
    6-point face model based on standard anthropometric measurements.
"""

import cv2
import numpy as np

# ─── 3D Model Points ──────────────────────────────────────────────
# Standard 3D reference points of a generic face model (in mm).
# These correspond to specific dlib 68-landmark indices:
#   Nose tip (30), Chin (8), Left eye corner (36),
#   Right eye corner (45), Left mouth corner (48), Right mouth corner (54)

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip       — landmark 30
    (0.0, -330.0, -65.0),       # Chin            — landmark 8
    (-225.0, 170.0, -135.0),    # Left eye corner — landmark 36
    (225.0, 170.0, -135.0),     # Right eye corner — landmark 45
    (-150.0, -150.0, -125.0),   # Left mouth corner — landmark 48
    (150.0, -150.0, -125.0),    # Right mouth corner — landmark 54
], dtype=np.float64)

# Corresponding dlib landmark indices
LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


def estimate_head_pose(landmarks_68, frame_shape):
    """
    Estimate head pose (yaw, pitch, roll) using Perspective-n-Point.

    Steps:
    1. Extract 6 key 2D landmark points from the 68-point set
    2. Approximate camera intrinsics from frame dimensions
    3. Solve PnP to get rotation and translation vectors
    4. Convert rotation vector → rotation matrix → Euler angles

    Args:
        landmarks_68: Array-like of 68 (x, y) facial landmark points from dlib.
                      Can be list of tuples, list of lists, or numpy array (68, 2).
        frame_shape: Tuple (height, width[, channels]) of the video frame.

    Returns:
        dict with keys:
            "yaw":   float — horizontal head rotation (degrees, + = looking right)
            "pitch": float — vertical head rotation (degrees, + = looking down)
            "roll":  float — head tilt (degrees, + = tilting right)
            "rotation_vector": np.array (3,1) — raw rotation vector from solvePnP
            "translation_vector": np.array (3,1) — raw translation vector

    Raises:
        ValueError: If landmarks array doesn't have 68 points or solvePnP fails.
    """
    landmarks = np.array(landmarks_68, dtype=np.float64)
    if landmarks.shape[0] < 55:
        raise ValueError(f"Expected at least 55 landmarks, got {landmarks.shape[0]}")

    h, w = frame_shape[:2]

    # Extract the 6 key 2D image points
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye corner
        landmarks[45],  # Right eye corner
        landmarks[48],  # Left mouth corner
        landmarks[54],  # Right mouth corner
    ], dtype=np.float64)

    # Approximate camera intrinsic matrix
    # Focal length ≈ frame width (reasonable for standard webcams/dashcams)
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP — find rotation and translation that maps 3D model → 2D image
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise ValueError("solvePnP failed to find a valid pose solution")

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    # Build the 3×4 projection matrix for decomposition
    projection_matrix = np.hstack((rotation_matrix, translation_vec))

    # Decompose into Euler angles using OpenCV
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    # euler_angles is (3, 1): [pitch, yaw, roll] in degrees
    pitch_deg = float(euler_angles[0, 0])
    yaw_deg = float(euler_angles[1, 0])
    roll_deg = float(euler_angles[2, 0])

    return {
        "yaw": yaw_deg,
        "pitch": pitch_deg,
        "roll": roll_deg,
        "rotation_vector": rotation_vec,
        "translation_vector": translation_vec,
    }


def is_distracted(yaw, pitch, yaw_threshold=30.0, pitch_threshold=25.0):
    """
    Determine if the driver is distracted based on head pose angles.

    A driver is considered DISTRACTED if:
    - Looking sideways: |yaw| > yaw_threshold (default 30°)
    - Looking down (phone, console): |pitch| > pitch_threshold (default 25°)

    These thresholds are based on ergonomics research showing that head
    rotations beyond 30° significantly reduce peripheral road awareness.

    Args:
        yaw: Horizontal rotation in degrees (from estimate_head_pose)
        pitch: Vertical rotation in degrees (from estimate_head_pose)
        yaw_threshold: Max acceptable yaw angle (default 30°)
        pitch_threshold: Max acceptable pitch angle (default 25°)

    Returns:
        bool — True if driver is distracted (looking away from road)
    """
    return abs(yaw) > yaw_threshold or abs(pitch) > pitch_threshold


def draw_head_pose(frame, landmarks_68, pose_result):
    """
    Visualize head pose by drawing a nose direction line on the frame.

    Projects the nose tip direction vector into 2D and draws an arrow
    showing where the driver is looking.

    Args:
        frame: BGR frame to draw on (modified in-place)
        landmarks_68: 68-point landmark array
        pose_result: dict from estimate_head_pose()

    Returns:
        frame: The annotated frame
    """
    landmarks = np.array(landmarks_68, dtype=np.float64)
    h, w = frame.shape[:2]

    # Camera matrix (same as in estimation)
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Project a point along the nose direction (500mm forward from nose tip)
    nose_end_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
    nose_end_2d, _ = cv2.projectPoints(
        nose_end_3d,
        pose_result["rotation_vector"],
        pose_result["translation_vector"],
        camera_matrix,
        dist_coeffs
    )

    # Draw the direction arrow from nose tip
    nose_tip = tuple(landmarks[30].astype(int))
    nose_end = tuple(nose_end_2d[0][0].astype(int))

    # Determine color based on distraction
    distracted = is_distracted(pose_result["yaw"], pose_result["pitch"])
    color = (0, 0, 255) if distracted else (0, 255, 0)  # Red if distracted, Green if OK
    label = "DISTRACTED" if distracted else "ATTENTIVE"

    cv2.arrowedLine(frame, nose_tip, nose_end, color, 3)
    cv2.putText(frame, label, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Yaw: {pose_result['yaw']:.1f} Pitch: {pose_result['pitch']:.1f}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
