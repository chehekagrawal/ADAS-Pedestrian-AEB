"""
MiDaS Depth Estimation Module for ADAS-Pedestrian-AEB.

Uses Intel MiDaS v3.1 to convert monocular camera frames into depth maps,
then extracts metric distance estimates at detected object locations.

This is the bridge between the ML perception layer and mechanical physics modules.
"""

import torch
import cv2
import numpy as np
import yaml
import os


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS.
    Converts pixel-space detections into approximate real-world meter distances.
    """

    def __init__(self, model_type="MiDaS_small", device=None, config_path=None):
        """
        Load MiDaS model from torch hub.

        Args:
            model_type: "MiDaS_small" (fast, CPU-friendly) or "DPT_Large" (accurate, needs GPU)
            device: "cpu" or "cuda" (auto-detected if None)
            config_path: path to depth_config.yaml (optional)
        """
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            model_type = cfg.get("model_type", model_type)
            self.calibration_factor = cfg.get("calibration_factor", 500.0)
            self.patch_size = cfg.get("patch_size", 10)
            device = cfg.get("device", device)
        else:
            self.calibration_factor = 500.0
            self.patch_size = 10

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_type = model_type

        # Load MiDaS model
        print(f"[DepthEstimator] Loading MiDaS model: {model_type} on {self.device}")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load appropriate transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print(f"[DepthEstimator] Model loaded successfully.")

    def estimate_depth(self, frame):
        """
        Compute depth map from a single BGR frame.

        Args:
            frame: numpy array (H, W, 3) in BGR format

        Returns:
            depth_map: numpy array (H, W) float32, higher value = closer to camera
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = self.transform(img_rgb).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        return depth_map.astype(np.float32)

    def get_depth_at_bbox(self, depth_map, bbox):
        """
        Extract depth value at the center of a bounding box.
        Uses median of a small patch for robustness against noise.

        Args:
            depth_map: (H, W) float array from estimate_depth()
            bbox: [x1, y1, x2, y2] bounding box coordinates

        Returns:
            depth_value: float (raw MiDaS depth units, higher = closer)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape[:2]

        # Center of bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Define patch around center
        half = self.patch_size // 2
        px1 = max(0, cx - half)
        px2 = min(w, cx + half)
        py1 = max(0, cy - half)
        py2 = min(h, cy + half)

        # Extract patch and take median (robust to outliers)
        patch = depth_map[py1:py2, px1:px2]

        if patch.size == 0:
            return 0.0

        return float(np.median(patch))

    def depth_to_meters(self, raw_depth):
        """
        Convert MiDaS relative depth to approximate meters.

        MiDaS outputs inverse relative depth (higher = closer).
        We approximate: distance_m ≈ calibration_factor / raw_depth

        Calibration procedure:
        1. Place a person at a known distance (e.g., 5 meters)
        2. Run MiDaS, get raw_depth at that person
        3. Set calibration_factor = raw_depth × 5.0

        Args:
            raw_depth: float from get_depth_at_bbox()

        Returns:
            distance_meters: float (approximate distance in meters)
        """
        if raw_depth <= 0:
            return float("inf")

        return self.calibration_factor / raw_depth

    def get_distance_at_bbox(self, depth_map, bbox):
        """
        Convenience: get distance in meters at a bounding box location.

        Args:
            depth_map: from estimate_depth()
            bbox: [x1, y1, x2, y2]

        Returns:
            distance_meters: float
        """
        raw_depth = self.get_depth_at_bbox(depth_map, bbox)
        return self.depth_to_meters(raw_depth)

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_MAGMA):
        """
        Create a colorized visualization of the depth map.

        Args:
            depth_map: (H, W) float array
            colormap: OpenCV colormap

        Returns:
            colored_depth: (H, W, 3) BGR image for display/saving
        """
        # Normalize to 0-255
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)

        return colored

    def set_calibration(self, known_distance_m, raw_depth_at_known):
        """
        Calibrate the depth-to-meters conversion using a known reference.

        Args:
            known_distance_m: actual distance to reference object in meters
            raw_depth_at_known: MiDaS raw depth value at that object
        """
        self.calibration_factor = raw_depth_at_known * known_distance_m
        print(f"[DepthEstimator] Calibration factor set to {self.calibration_factor:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MiDaS depth estimation")
    parser.add_argument("--image", type=str, default="data/sample/test.jpg", help="Input image")
    parser.add_argument("--output", type=str, default="results/depth/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load and process
    estimator = DepthEstimator()
    frame = cv2.imread(args.image)

    if frame is None:
        print(f"Error: could not load image {args.image}")
        exit(1)

    depth_map = estimator.estimate_depth(frame)

    # Save visualization
    vis = estimator.visualize_depth(depth_map)
    cv2.imwrite(os.path.join(args.output, "depth_map_sample.png"), vis)

    # Test depth at center
    h, w = frame.shape[:2]
    center_bbox = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
    distance = estimator.get_distance_at_bbox(depth_map, center_bbox)
    print(f"Approximate distance at center: {distance:.2f} meters")
    print(f"Depth map saved to {args.output}")
