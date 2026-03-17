import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_TRAJECTORIES_PATH = "results/tracking/tracking_multiclass/trajectories.json"
DEFAULT_COLLISION_PATH = "results/collision/collision_events.json"
DEFAULT_OUTPUT_DIR = "results/intelligence/datasets"

INPUT_LEN = 5
PRED_LEN = 5
FPS = 30.0

# For image based ego approximation.
# If you know exact frame size, pass via CLI.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def bbox_to_center(bbox: List[float]) -> Tuple[float, float]:
    """
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def bbox_to_size(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-8:
        return default
    return a / b


def train_val_test_split(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


# ============================================================
# Collision log loading and indexing
# ============================================================

def load_collision_logs(collision_path: str) -> List[Dict[str, Any]]:
    """
    Supports:
    1. A direct JSON file containing a list of collision/risk events
    2. A directory containing multiple .json files
    """
    if os.path.isfile(collision_path):
        data = load_json(collision_path)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If wrapped in a dict, try common keys or flatten values
            for key in ["events", "collision_events", "logs", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # fallback: flatten list values
            merged = []
            for v in data.values():
                if isinstance(v, list):
                    merged.extend(v)
            return merged
        else:
            return []

    elif os.path.isdir(collision_path):
        merged = []
        for fname in sorted(os.listdir(collision_path)):
            if fname.lower().endswith(".json"):
                fpath = os.path.join(collision_path, fname)
                try:
                    data = load_json(fpath)
                    if isinstance(data, list):
                        merged.extend(data)
                    elif isinstance(data, dict):
                        for key in ["events", "collision_events", "logs", "data"]:
                            if key in data and isinstance(data[key], list):
                                merged.extend(data[key])
                                break
                except Exception as e:
                    print(f"[WARNING] Failed to read collision file {fpath}: {e}")
        return merged

    else:
        print(f"[WARNING] Collision path not found: {collision_path}")
        return []


def build_collision_index(
    collision_logs: List[Dict[str, Any]]
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Key: (track_id, frame)
    """
    index = {}
    for item in collision_logs:
        track_id = str(item.get("track_id"))
        frame = item.get("frame")
        if frame is None:
            continue
        index[(track_id, int(frame))] = item
    return index


# ============================================================
# Trajectory preprocessing
# ============================================================

def load_trajectories(trajectories_path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = load_json(trajectories_path)

    if not isinstance(data, dict):
        raise ValueError("trajectories.json must be a dict of track_id -> detections list")

    normalized = {}
    for track_id, detections in data.items():
        if not isinstance(detections, list):
            continue

        clean_dets = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            if "frame" not in det or "bbox" not in det:
                continue
            clean_dets.append(det)

        clean_dets = sorted(clean_dets, key=lambda x: int(x["frame"]))
        normalized[str(track_id)] = clean_dets

    return normalized


def preprocess_track(
    track_id: str,
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    fps: float
) -> List[Dict[str, Any]]:
    """
    Converts bbox detections into enriched per-frame motion data.
    """
    ego_x = frame_width / 2.0
    ego_y = float(frame_height)

    processed = []
    prev = None

    for det in detections:
        frame = int(det["frame"])
        bbox = det["bbox"]
        obj_class = det.get("class", "unknown")

        cx, cy = bbox_to_center(bbox)
        w, h = bbox_to_size(bbox)

        if prev is None:
            dx = 0.0
            dy = 0.0
            dt_frames = 1
        else:
            dt_frames = max(1, frame - prev["frame"])
            dx = cx - prev["cx"]
            dy = cy - prev["cy"]

        vx = dx / dt_frames
        vy = dy / dt_frames
        speed_px_per_frame = math.sqrt(vx ** 2 + vy ** 2)
        speed_px_per_sec = speed_px_per_frame * fps

        dist_to_ego = euclidean_distance(cx, cy, ego_x, ego_y)

        processed.append({
            "track_id": str(track_id),
            "frame": frame,
            "class": obj_class,
            "bbox": bbox,
            "cx": cx,
            "cy": cy,
            "width": w,
            "height": h,
            "vx": vx,
            "vy": vy,
            "speed_px_per_frame": speed_px_per_frame,
            "speed_px_per_sec": speed_px_per_sec,
            "distance_to_ego_px": dist_to_ego,
        })

        prev = processed[-1]

    return processed


# ============================================================
# Behavior labeling
# ============================================================

def heuristic_behavior_label(
    speed_px_per_frame: float,
    vx: float,
    vy: float,
    distance_to_ego: float,
    lateral_ratio_threshold: float = 1.2
) -> str:
    """
    Simple heuristic behavior labeling.

    Notes:
    - Since your current coordinates are image pixels, thresholds are heuristic.
    - You should tune these based on your videos.
    """
    abs_vx = abs(vx)
    abs_vy = abs(vy)

    if speed_px_per_frame < 1.0:
        return "standing"
    elif speed_px_per_frame < 4.0:
        if abs_vx > lateral_ratio_threshold * max(abs_vy, 1e-6):
            return "crossing"
        return "walking"
    else:
        if distance_to_ego < 250:
            return "approaching_road"
        return "running"


def behavior_to_id(label: str) -> int:
    mapping = {
        "standing": 0,
        "walking": 1,
        "running": 2,
        "crossing": 3,
        "approaching_road": 4,
    }
    return mapping.get(label, -1)


# ============================================================
# Dataset generation
# ============================================================

def create_prediction_samples(
    processed_tracks: Dict[str, List[Dict[str, Any]]],
    collision_index: Dict[Tuple[str, int], Dict[str, Any]],
    input_len: int,
    pred_len: int
) -> Dict[str, Any]:
    """
    Creates sliding-window trajectory prediction samples.

    X shape: [N, input_len, feature_dim]
    Y shape: [N, pred_len, 2]
    """
    X = []
    Y = []
    meta = []

    for track_id, frames in processed_tracks.items():
        # Usually Part 4 trajectory prediction focuses on pedestrians,
        # but keep it generic for now. You can filter to person later.
        if len(frames) < input_len + pred_len:
            continue

        for i in range(0, len(frames) - input_len - pred_len + 1):
            input_seq = frames[i:i + input_len]
            target_seq = frames[i + input_len:i + input_len + pred_len]

            # Features for each input timestep
            # [cx, cy, vx, vy, speed, distance_to_ego, width, height]
            x_seq = []
            for item in input_seq:
                x_seq.append([
                    item["cx"],
                    item["cy"],
                    item["vx"],
                    item["vy"],
                    item["speed_px_per_frame"],
                    item["distance_to_ego_px"],
                    item["width"],
                    item["height"],
                ])

            # Future target coordinates only
            y_seq = []
            for item in target_seq:
                y_seq.append([item["cx"], item["cy"]])

            last_input = input_seq[-1]
            coll = collision_index.get((track_id, last_input["frame"]), {})

            X.append(x_seq)
            Y.append(y_seq)
            meta.append({
                "track_id": track_id,
                "class": last_input["class"],
                "start_frame": input_seq[0]["frame"],
                "end_input_frame": last_input["frame"],
                "end_target_frame": target_seq[-1]["frame"],
                "distance": float(coll.get("distance", last_input["distance_to_ego_px"])),
                "ttc": float(coll.get("ttc", -1.0)) if coll else -1.0,
                "collision_probability": float(coll.get("collision_probability", 0.0)) if coll else 0.0,
                "collision_detected": bool(coll.get("collision_detected", False)) if coll else False,
                "AEB_triggered": bool(coll.get("AEB_triggered", False)) if coll else False,
            })

    return {
        "X": np.array(X, dtype=np.float32),
        "Y": np.array(Y, dtype=np.float32),
        "meta": meta,
    }


def create_behavior_dataset(
    processed_tracks: Dict[str, List[Dict[str, Any]]],
    window_len: int = 6
) -> pd.DataFrame:
    """
    Creates a behavior classification dataset using heuristic labels.
    One row per sliding window.
    """
    rows = []

    for track_id, frames in processed_tracks.items():
        if len(frames) < window_len:
            continue

        for i in range(0, len(frames) - window_len + 1):
            window = frames[i:i + window_len]

            cxs = np.array([w["cx"] for w in window], dtype=np.float32)
            cys = np.array([w["cy"] for w in window], dtype=np.float32)
            vxs = np.array([w["vx"] for w in window], dtype=np.float32)
            vys = np.array([w["vy"] for w in window], dtype=np.float32)
            speeds = np.array([w["speed_px_per_frame"] for w in window], dtype=np.float32)
            dists = np.array([w["distance_to_ego_px"] for w in window], dtype=np.float32)

            start = window[0]
            end = window[-1]

            total_dx = float(cxs[-1] - cxs[0])
            total_dy = float(cys[-1] - cys[0])
            displacement = math.sqrt(total_dx ** 2 + total_dy ** 2)

            mean_speed = float(np.mean(speeds))
            max_speed = float(np.max(speeds))
            std_speed = float(np.std(speeds))

            mean_vx = float(np.mean(vxs))
            mean_vy = float(np.mean(vys))
            mean_dist = float(np.mean(dists))

            label = heuristic_behavior_label(
                speed_px_per_frame=mean_speed,
                vx=mean_vx,
                vy=mean_vy,
                distance_to_ego=mean_dist
            )

            rows.append({
                "track_id": track_id,
                "class": end["class"],
                "start_frame": start["frame"],
                "end_frame": end["frame"],
                "mean_cx": float(np.mean(cxs)),
                "mean_cy": float(np.mean(cys)),
                "mean_speed": mean_speed,
                "max_speed": max_speed,
                "std_speed": std_speed,
                "mean_vx": mean_vx,
                "mean_vy": mean_vy,
                "total_dx": total_dx,
                "total_dy": total_dy,
                "displacement": displacement,
                "mean_distance_to_ego": mean_dist,
                "behavior_label": label,
                "behavior_id": behavior_to_id(label),
            })

    return pd.DataFrame(rows)


def create_aeb_dataset(
    processed_tracks: Dict[str, List[Dict[str, Any]]],
    collision_index: Dict[Tuple[str, int], Dict[str, Any]]
) -> pd.DataFrame:
    """
    Creates frame-level AEB learning dataset.
    This is a useful starting point for adaptive_aeb.py later.
    """
    rows = []

    for track_id, frames in processed_tracks.items():
        for item in frames:
            coll = collision_index.get((track_id, item["frame"]), {})

            label = heuristic_behavior_label(
                speed_px_per_frame=item["speed_px_per_frame"],
                vx=item["vx"],
                vy=item["vy"],
                distance_to_ego=item["distance_to_ego_px"]
            )

            rows.append({
                "track_id": track_id,
                "frame": item["frame"],
                "class": item["class"],
                "cx": item["cx"],
                "cy": item["cy"],
                "vx": item["vx"],
                "vy": item["vy"],
                "speed_px_per_frame": item["speed_px_per_frame"],
                "speed_px_per_sec": item["speed_px_per_sec"],
                "distance_to_ego_px": item["distance_to_ego_px"],
                "width": item["width"],
                "height": item["height"],
                "behavior_label": label,
                "behavior_id": behavior_to_id(label),
                "distance": float(coll.get("distance", item["distance_to_ego_px"])) if coll else item["distance_to_ego_px"],
                "ttc": float(coll.get("ttc", -1.0)) if coll else -1.0,
                "collision_probability": float(coll.get("collision_probability", 0.0)) if coll else 0.0,
                "collision_detected": int(bool(coll.get("collision_detected", False))) if coll else 0,
                "AEB_triggered": int(bool(coll.get("AEB_triggered", False))) if coll else 0,
            })

    return pd.DataFrame(rows)


# ============================================================
# Save functions
# ============================================================

def save_prediction_dataset(
    pred_data: Dict[str, Any],
    output_dir: str,
    seed: int = 42
) -> None:
    X = pred_data["X"]
    Y = pred_data["Y"]
    meta = pred_data["meta"]

    if len(X) == 0:
        print("[WARNING] No trajectory prediction samples generated.")
        return

    train_idx, val_idx, test_idx = train_val_test_split(len(X), seed=seed)

    pred_dir = os.path.join(output_dir, "trajectory_prediction")
    ensure_dir(pred_dir)

    # Save arrays
    np.savez_compressed(
        os.path.join(pred_dir, "train.npz"),
        X=X[train_idx],
        Y=Y[train_idx]
    )
    np.savez_compressed(
        os.path.join(pred_dir, "val.npz"),
        X=X[val_idx],
        Y=Y[val_idx]
    )
    np.savez_compressed(
        os.path.join(pred_dir, "test.npz"),
        X=X[test_idx],
        Y=Y[test_idx]
    )

    # Save metadata separately
    meta_train = [meta[i] for i in train_idx]
    meta_val = [meta[i] for i in val_idx]
    meta_test = [meta[i] for i in test_idx]

    save_json(meta_train, os.path.join(pred_dir, "train_meta.json"))
    save_json(meta_val, os.path.join(pred_dir, "val_meta.json"))
    save_json(meta_test, os.path.join(pred_dir, "test_meta.json"))

    print(f"[INFO] Saved trajectory prediction dataset to: {pred_dir}")
    print(f"       Train: {len(train_idx)} samples")
    print(f"       Val:   {len(val_idx)} samples")
    print(f"       Test:  {len(test_idx)} samples")
    print(f"       X shape: {X.shape}, Y shape: {Y.shape}")


def save_behavior_dataset(df: pd.DataFrame, output_dir: str) -> None:
    out_dir = os.path.join(output_dir, "behavior_classification")
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "behavior_labels.csv")
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved behavior classification dataset to: {csv_path}")
    print(f"       Rows: {len(df)}")


def save_aeb_dataset(df: pd.DataFrame, output_dir: str) -> None:
    out_dir = os.path.join(output_dir, "aeb_learning")
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "aeb_features.csv")
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved AEB learning dataset to: {csv_path}")
    print(f"       Rows: {len(df)}")


def save_dataset_summary(
    processed_tracks: Dict[str, List[Dict[str, Any]]],
    pred_data: Dict[str, Any],
    behavior_df: pd.DataFrame,
    aeb_df: pd.DataFrame,
    output_dir: str
) -> None:
    class_counts = {}
    for _, frames in processed_tracks.items():
        if len(frames) == 0:
            continue
        cls = frames[0]["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    summary = {
        "num_tracks": len(processed_tracks),
        "track_class_counts": class_counts,
        "trajectory_prediction_samples": int(len(pred_data["X"])),
        "behavior_samples": int(len(behavior_df)),
        "aeb_samples": int(len(aeb_df)),
    }

    save_json(summary, os.path.join(output_dir, "dataset_summary.json"))
    print(f"[INFO] Saved dataset summary to: {os.path.join(output_dir, 'dataset_summary.json')}")


# ============================================================
# Main pipeline
# ============================================================

def generate_datasets(
    trajectories_path: str,
    collision_path: str,
    output_dir: str,
    input_len: int,
    pred_len: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    only_person: bool = False
) -> None:
    print("[INFO] Loading trajectories...")
    trajectories = load_trajectories(trajectories_path)
    print(f"[INFO] Loaded {len(trajectories)} tracks from {trajectories_path}")

    print("[INFO] Loading collision logs...")
    collision_logs = load_collision_logs(collision_path)
    collision_index = build_collision_index(collision_logs)
    print(f"[INFO] Loaded {len(collision_logs)} collision log entries")

    print("[INFO] Preprocessing tracks...")
    processed_tracks = {}

    for track_id, dets in trajectories.items():
        processed = preprocess_track(
            track_id=track_id,
            detections=dets,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps
        )

        if not processed:
            continue

        if only_person and processed[0]["class"] != "person":
            continue

        processed_tracks[track_id] = processed

    print(f"[INFO] Retained {len(processed_tracks)} processed tracks")

    print("[INFO] Creating trajectory prediction dataset...")
    pred_data = create_prediction_samples(
        processed_tracks=processed_tracks,
        collision_index=collision_index,
        input_len=input_len,
        pred_len=pred_len
    )

    print("[INFO] Creating behavior classification dataset...")
    behavior_df = create_behavior_dataset(
        processed_tracks=processed_tracks,
        window_len=input_len
    )

    print("[INFO] Creating AEB learning dataset...")
    aeb_df = create_aeb_dataset(
        processed_tracks=processed_tracks,
        collision_index=collision_index
    )

    print("[INFO] Saving datasets...")
    ensure_dir(output_dir)
    save_prediction_dataset(pred_data, output_dir)
    save_behavior_dataset(behavior_df, output_dir)
    save_aeb_dataset(aeb_df, output_dir)
    save_dataset_summary(processed_tracks, pred_data, behavior_df, aeb_df, output_dir)

    print("[INFO] Dataset generation completed successfully.")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ML datasets for Part 4 intelligence module.")

    parser.add_argument(
        "--trajectories",
        type=str,
        default=DEFAULT_TRAJECTORIES_PATH,
        help="Path to trajectories.json"
    )
    parser.add_argument(
        "--collision",
        type=str,
        default=DEFAULT_COLLISION_PATH,
        help="Path to collision JSON file or collision logs directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=INPUT_LEN,
        help="Input sequence length"
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=PRED_LEN,
        help="Prediction sequence length"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="Video FPS"
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Frame width in pixels"
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="Frame height in pixels"
    )
    parser.add_argument(
        "--only_person",
        action="store_true",
        help="Use only person tracks"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_datasets(
        trajectories_path=args.trajectories,
        collision_path=args.collision,
        output_dir=args.output,
        input_len=args.input_len,
        pred_len=args.pred_len,
        fps=args.fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        only_person=args.only_person
    )