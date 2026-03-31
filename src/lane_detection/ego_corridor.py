"""
Ego Corridor Computation & In-Path Detection Filtering.

Computes the drivable corridor polygon from detected lane lines and
determines which detected objects (pedestrians) are actually in the
vehicle's path — filtering out sidewalk detections to reduce false AEB triggers.

Integration:
    - Uses shapely (same library as collision_engine.py) for polygon operations
    - Receives detections from YOLO (Part 1)
    - Filtered detections feed into AEB decision pipeline

Key concept:
    An object is "in-path" if its FOOT POINT (bottom-center of bounding box)
    falls inside the ego corridor polygon. This is more accurate than using
    the bbox center, as it represents where the person is standing on the road.
"""

import numpy as np
from shapely.geometry import Polygon, Point


def compute_ego_corridor(left_lane, right_lane, frame_height):
    """
    Create a polygon representing the ego vehicle's driving corridor.

    The corridor is the area between left and right lane lines,
    extending from the bottom of the frame to the vanishing point.

    The polygon is constructed by connecting:
    - Left lane bottom → Left lane top → Right lane top → Right lane bottom

    Args:
        left_lane: [(x1, y1), (x2, y2)] — left lane line (bottom to top)
        right_lane: [(x1, y1), (x2, y2)] — right lane line (bottom to top)
        frame_height: int — height of the video frame

    Returns:
        corridor: shapely Polygon representing the drivable area
        Returns None if either lane is missing.
    """
    if left_lane is None or right_lane is None:
        return None

    (lx1, ly1), (lx2, ly2) = left_lane    # left bottom, left top
    (rx1, ry1), (rx2, ry2) = right_lane    # right bottom, right top

    # Build polygon: left-bottom → left-top → right-top → right-bottom
    corridor_points = [
        (lx1, ly1),   # left lane bottom
        (lx2, ly2),   # left lane top (near vanishing point)
        (rx2, ry2),   # right lane top (near vanishing point)
        (rx1, ry1),   # right lane bottom
    ]

    corridor = Polygon(corridor_points)

    # Validate the polygon (can be invalid if lanes cross)
    if not corridor.is_valid:
        corridor = corridor.buffer(0)  # Fix self-intersections

    return corridor


def is_in_path(bbox, corridor):
    """
    Check if a detected object's foot point is inside the ego corridor.

    The foot point = bottom-center of the bounding box, representing
    where the pedestrian's feet meet the road surface. This is more
    accurate than using the bbox center for ground-plane projection.

    Args:
        bbox: [x1, y1, x2, y2] — bounding box coordinates
        corridor: shapely Polygon from compute_ego_corridor()

    Returns:
        bool — True if the object is in the ego lane (AEB should consider it)
    """
    if corridor is None:
        # If no corridor available, assume all detections are in-path (safety-first)
        return True

    x1, y1, x2, y2 = bbox
    foot_point = Point((x1 + x2) / 2, y2)  # bottom center

    return corridor.contains(foot_point)


def filter_detections(detections, corridor):
    """
    Partition detections into in-path and off-path groups.

    In-path detections are candidates for AEB activation.
    Off-path detections (sidewalk pedestrians, etc.) are logged but not acted upon.

    Args:
        detections: list of dicts, each with at least a "bbox" key
                    e.g. [{"bbox": [x1,y1,x2,y2], "class": "person", "conf": 0.9}, ...]
        corridor: shapely Polygon from compute_ego_corridor()

    Returns:
        tuple: (in_path_detections, off_path_detections)
            - in_path_detections: list of dicts (objects in the ego lane)
            - off_path_detections: list of dicts (objects outside the ego lane)
    """
    in_path = []
    off_path = []

    for det in detections:
        bbox = det.get("bbox")
        if bbox is None:
            off_path.append(det)
            continue

        if is_in_path(bbox, corridor):
            in_path.append(det)
        else:
            off_path.append(det)

    return in_path, off_path


def draw_corridor(frame, corridor, color=(0, 255, 0), alpha=0.25):
    """
    Draw the ego corridor polygon on a frame as a semi-transparent overlay.

    Args:
        frame: BGR frame to draw on (modified in-place)
        corridor: shapely Polygon
        color: BGR fill color (default: green)
        alpha: Transparency (0.0 = transparent, 1.0 = opaque)

    Returns:
        frame: Annotated frame
    """
    if corridor is None:
        return frame

    import cv2

    overlay = frame.copy()

    # Extract polygon exterior coordinates
    coords = np.array(corridor.exterior.coords, dtype=np.int32)
    cv2.fillPoly(overlay, [coords], color)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
