import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Defaults
# ============================================================

DEFAULT_VIDEO_PATH = "data/sample/youtube_sample.mp4"
DEFAULT_TRAJECTORIES_PATH = "results/tracking/tracking_multiclass/trajectories.json"
DEFAULT_COLLISION_PATH = "results/collision/logs"
DEFAULT_BEHAVIOR_MODEL_PATH = "models/behavior_classifier.pkl"
DEFAULT_TRAJECTORY_MODEL_PATH = "models/trajectory_predictor.pt"
DEFAULT_AEB_MODEL_PATH = "models/adaptive_aeb_model.pkl"
DEFAULT_OUTPUT_PATH = "results/intelligence/part4_output_video.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization
MC_SAMPLES = 20
PRED_COLOR = (0, 255, 0)         # green
GTEXT_COLOR = (255, 255, 255)    # white
SAFE_COLOR = (0, 200, 0)         # green
WATCH_COLOR = (0, 255, 255)      # yellow
BRAKE_COLOR = (0, 0, 255)        # red
UNC_COLOR = (180, 180, 180)      # gray


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_to_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_to_size(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def denormalize_targets(Y_rel_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Y_rel_norm * std) + mean


def convert_relative_to_absolute(last_pos: np.ndarray, Y_rel: np.ndarray) -> np.ndarray:
    return Y_rel + last_pos[None, :]


def load_collision_logs(collision_path: str) -> List[Dict[str, Any]]:
    if os.path.isfile(collision_path):
        data = load_json(collision_path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["events", "collision_events", "logs", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            out = []
            for v in data.values():
                if isinstance(v, list):
                    out.extend(v)
            return out
        return []

    if os.path.isdir(collision_path):
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
                    print(f"[WARNING] Could not read collision log {fpath}: {e}")
        return merged

    print(f"[WARNING] Collision path not found: {collision_path}")
    return []


def build_collision_index(logs: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    idx = {}
    for item in logs:
        track_id = str(item.get("track_id"))
        frame = item.get("frame")
        if frame is None:
            continue
        idx[(track_id, int(frame))] = item
    return idx


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def put_text_block(
    frame: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    line_gap: int = 18
) -> None:
    for i, line in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            frame, line, (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA
        )


# ============================================================
# Trajectory model
# ============================================================

class LSTMTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        pred_len: int = 5,
        output_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.pred_len = pred_len
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred = self.head(last_hidden)
        pred = pred.view(-1, self.pred_len, self.output_size)
        return pred


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def load_trajectory_model(model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    config = checkpoint["config"]
    feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
    target_mean = np.array(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.array(checkpoint["target_std"], dtype=np.float32)

    model = LSTMTrajectoryPredictor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        pred_len=config["pred_len"],
        output_size=2,
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, feature_mean, feature_std, target_mean, target_std, config


# ============================================================
# Tracking prep
# ============================================================

def load_trajectories(path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("trajectories.json must be dict of track_id -> list[detection]")

    out = {}
    for track_id, dets in data.items():
        if not isinstance(dets, list):
            continue
        clean = []
        for d in dets:
            if isinstance(d, dict) and "frame" in d and "bbox" in d:
                clean.append(d)
        clean = sorted(clean, key=lambda x: int(x["frame"]))
        out[str(track_id)] = clean
    return out


def build_frame_index(trajectories: Dict[str, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    frame_index: Dict[int, List[Dict[str, Any]]] = {}
    for track_id, dets in trajectories.items():
        for d in dets:
            item = dict(d)
            item["track_id"] = str(track_id)
            frame_id = int(item["frame"])
            frame_index.setdefault(frame_id, []).append(item)
    return frame_index


def build_track_histories(
    trajectories: Dict[str, List[Dict[str, Any]]],
    frame_width: int,
    frame_height: int
) -> Dict[str, List[Dict[str, Any]]]:
    ego_x = frame_width / 2.0
    ego_y = float(frame_height)

    histories = {}
    for track_id, dets in trajectories.items():
        seq = []
        prev = None
        for d in dets:
            frame = int(d["frame"])
            bbox = d["bbox"]
            cls = d.get("class", "unknown")
            cx, cy = bbox_to_center(bbox)
            w, h = bbox_to_size(bbox)

            if prev is None:
                dt_frames = 1
                dx = 0.0
                dy = 0.0
            else:
                dt_frames = max(1, frame - prev["frame"])
                dx = cx - prev["cx"]
                dy = cy - prev["cy"]

            vx = dx / dt_frames
            vy = dy / dt_frames
            speed = float(np.sqrt(vx * vx + vy * vy))
            dist_ego = euclidean_distance(cx, cy, ego_x, ego_y)

            seq.append({
                "track_id": str(track_id),
                "frame": frame,
                "bbox": bbox,
                "class": cls,
                "cx": cx,
                "cy": cy,
                "width": w,
                "height": h,
                "vx": vx,
                "vy": vy,
                "speed_px_per_frame": speed,
                "distance_to_ego_px": dist_ego,
            })
            prev = seq[-1]
        histories[str(track_id)] = seq
    return histories


def get_history_window(
    track_history: List[Dict[str, Any]],
    current_frame: int,
    input_len: int
) -> Optional[np.ndarray]:
    """
    Returns X_raw of shape [input_len, 8] ending at current_frame.
    """
    upto = [x for x in track_history if int(x["frame"]) <= int(current_frame)]
    if len(upto) < input_len:
        return None
    window = upto[-input_len:]

    x_seq = []
    for item in window:
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
    return np.array(x_seq, dtype=np.float32)


# ============================================================
# Behavior prediction
# ============================================================

def predict_behavior_from_aeb_style_features(
    behavior_bundle: Dict[str, Any],
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    speed: float,
    distance_to_ego_px: float
) -> Tuple[int, str]:
    model = behavior_bundle["model"]
    imputer = behavior_bundle["imputer"]
    feature_columns = behavior_bundle["feature_columns"]
    label_map = behavior_bundle.get("label_map", {})

    row = {
        "mean_cx": cx,
        "mean_cy": cy,
        "mean_speed": speed,
        "max_speed": speed,
        "std_speed": 0.0,
        "mean_vx": vx,
        "mean_vy": vy,
        "total_dx": vx,
        "total_dy": vy,
        "displacement": float(np.sqrt(vx * vx + vy * vy)),
        "mean_distance_to_ego": distance_to_ego_px,
    }

    X = np.array([[row[c] for c in feature_columns]], dtype=np.float32)
    X = imputer.transform(X)
    pred = int(model.predict(X)[0])

    try:
        label = label_map.get(pred, label_map.get(str(pred), str(pred)))
    except Exception:
        label = str(pred)

    return pred, str(label)


# ============================================================
# AEB risk prediction
# ============================================================

def compute_aeb_features(
    det_info: Dict[str, Any],
    behavior_id_pred: int,
    collision_info: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    cx = float(det_info["cx"])
    cy = float(det_info["cy"])
    vx = float(det_info["vx"])
    vy = float(det_info["vy"])
    speed = float(det_info["speed_px_per_frame"])
    distance_to_ego_px = float(det_info["distance_to_ego_px"])
    width = float(det_info["width"])
    height = float(det_info["height"])

    distance = float(collision_info.get("distance", distance_to_ego_px)) if collision_info else distance_to_ego_px
    ttc = float(collision_info.get("ttc", -1.0)) if collision_info else -1.0
    collision_probability = float(collision_info.get("collision_probability", 0.0)) if collision_info else 0.0

    ttc_valid = 1 if (ttc > 0 and np.isfinite(ttc)) else 0
    inv_ttc = (1.0 / max(ttc, 1e-6)) if ttc_valid else 0.0
    speed_mag = float(np.sqrt(vx * vx + vy * vy))
    bbox_area = width * height
    toward_ego_proxy = 1 if vy > 0 else 0
    uncertainty_proxy = (
        0.4 * abs(vx) +
        0.4 * abs(vy) +
        0.2 * np.clip(collision_probability, 0.0, 1.0) * 10.0
    )

    behavior_risk_map = {
        0: 0.10,
        1: 0.30,
        2: 0.55,
        3: 0.70,
        4: 0.80,
    }
    behavior_risk_prior = behavior_risk_map.get(int(behavior_id_pred), 0.2)

    return {
        "cx": cx,
        "cy": cy,
        "vx": vx,
        "vy": vy,
        "speed_px_per_frame": speed,
        "speed_px_per_sec": speed * 30.0,
        "distance_to_ego_px": distance_to_ego_px,
        "width": width,
        "height": height,
        "distance": distance,
        "ttc": ttc,
        "ttc_valid": ttc_valid,
        "inv_ttc": inv_ttc,
        "collision_probability": collision_probability,
        "bbox_area": bbox_area,
        "speed_mag": speed_mag,
        "toward_ego_proxy": toward_ego_proxy,
        "uncertainty_proxy": uncertainty_proxy,
        "behavior_id_pred": int(behavior_id_pred),
        "behavior_risk_prior": behavior_risk_prior,
    }


def predict_adaptive_aeb(
    aeb_bundle: Dict[str, Any],
    feat_dict: Dict[str, float]
) -> Tuple[float, int]:
    model = aeb_bundle["model"]
    imputer = aeb_bundle["imputer"]
    feature_columns = aeb_bundle["feature_columns"]

    X = np.array([[feat_dict[c] for c in feature_columns]], dtype=np.float32)
    X = imputer.transform(X)

    pred = int(model.predict(X)[0])

    score = float(pred)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            learned_class = int(model.classes_[0])
            score = 1.0 if learned_class == 1 else 0.0
        else:
            class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
            score = float(proba[0, class_to_idx[1]]) if 1 in class_to_idx else 0.0

    return score, pred


# ============================================================
# Trajectory + uncertainty prediction
# ============================================================

@torch.no_grad()
def mc_predict_single(
    model: nn.Module,
    X_raw: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    mc_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_raw: [input_len, 8]
    Returns:
    mean_pred_abs: [pred_len, 2]
    std_pred_abs:  [pred_len, 2]
    """
    X_norm = normalize_features(X_raw[None, :, :], feat_mean, feat_std)
    X_tensor = torch.from_numpy(X_norm).float().to(DEVICE)

    enable_mc_dropout(model)

    preds_abs = []
    last_pos = X_raw[-1, 0:2]

    for _ in range(mc_samples):
        pred_rel_norm = model(X_tensor).cpu().numpy()[0]       # [pred_len, 2]
        pred_rel = denormalize_targets(pred_rel_norm, target_mean, target_std)
        pred_abs = convert_relative_to_absolute(last_pos, pred_rel)
        preds_abs.append(pred_abs)

    preds_abs = np.array(preds_abs, dtype=np.float32)          # [S, pred_len, 2]
    mean_pred_abs = np.mean(preds_abs, axis=0)
    std_pred_abs = np.std(preds_abs, axis=0)
    return mean_pred_abs, std_pred_abs


# ============================================================
# Drawing
# ============================================================

def draw_predicted_path(
    frame: np.ndarray,
    mean_pred_abs: np.ndarray,
    std_pred_abs: np.ndarray
) -> None:
    pts = []
    for i in range(len(mean_pred_abs)):
        x = int(round(mean_pred_abs[i, 0]))
        y = int(round(mean_pred_abs[i, 1]))
        pts.append((x, y))

        # uncertainty circle
        radius = int(max(3, np.linalg.norm(std_pred_abs[i])))
        cv2.circle(frame, (x, y), radius, UNC_COLOR, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, PRED_COLOR, -1, cv2.LINE_AA)

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], PRED_COLOR, 2, cv2.LINE_AA)


def choose_box_color(aeb_pred: int, risk_score: float) -> Tuple[int, int, int]:
    if aeb_pred == 1 or risk_score >= 0.70:
        return BRAKE_COLOR
    if risk_score >= 0.40:
        return WATCH_COLOR
    return SAFE_COLOR


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Part 4 ADAS output video.")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--trajectories", type=str, default=DEFAULT_TRAJECTORIES_PATH, help="Path to trajectories.json")
    parser.add_argument("--collision", type=str, default=DEFAULT_COLLISION_PATH, help="Path to collision log file or directory")
    parser.add_argument("--behavior_model", type=str, default=DEFAULT_BEHAVIOR_MODEL_PATH, help="Path to behavior classifier pkl")
    parser.add_argument("--trajectory_model", type=str, default=DEFAULT_TRAJECTORY_MODEL_PATH, help="Path to trajectory predictor pt")
    parser.add_argument("--aeb_model", type=str, default=DEFAULT_AEB_MODEL_PATH, help="Path to adaptive AEB model pkl")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output video path")
    parser.add_argument("--only_person", action="store_true", help="Visualize only person tracks")
    parser.add_argument("--mc_samples", type=int, default=MC_SAMPLES, help="Number of MC dropout samples")
    args = parser.parse_args()

    print(f"[INFO] Using device: {DEVICE}")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.trajectories):
        raise FileNotFoundError(f"Trajectories not found: {args.trajectories}")
    if not os.path.exists(args.behavior_model):
        raise FileNotFoundError(f"Behavior model not found: {args.behavior_model}")
    if not os.path.exists(args.trajectory_model):
        raise FileNotFoundError(f"Trajectory model not found: {args.trajectory_model}")
    if not os.path.exists(args.aeb_model):
        raise FileNotFoundError(f"AEB model not found: {args.aeb_model}")

    print("[INFO] Loading trajectories...")
    trajectories = load_trajectories(args.trajectories)
    frame_index = build_frame_index(trajectories)

    print("[INFO] Opening video...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height}, fps={fps:.2f}, frames={frame_count}")

    print("[INFO] Preparing track histories...")
    track_histories = build_track_histories(trajectories, width, height)

    print("[INFO] Loading collision logs...")
    collision_logs = load_collision_logs(args.collision)
    collision_index = build_collision_index(collision_logs)
    print(f"[INFO] Loaded {len(collision_logs)} collision log entries")

    print("[INFO] Loading behavior classifier...")
    behavior_bundle = joblib.load(args.behavior_model)

    print("[INFO] Loading adaptive AEB model...")
    aeb_bundle = joblib.load(args.aeb_model)

    print("[INFO] Loading trajectory predictor...")
    (
        traj_model,
        feat_mean,
        feat_std,
        target_mean,
        target_std,
        traj_cfg
    ) = load_trajectory_model(args.trajectory_model)
    input_len = int(traj_cfg["input_size"] and traj_cfg["config"]["input_size"] if False else 5)
    # safer: derive from expected dataset structure at runtime
    input_len = 5
    pred_len = int(traj_cfg["pred_len"])

    # Output writer
    ensure_dir(os.path.dirname(args.output))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {args.output}")

    print("[INFO] Starting visualization...")
    frame_id_zero_based = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Your tracking/collision data may use frame indexing starting from 0 or 1.
        # We try both and use whichever exists.
        candidate_frames = [frame_id_zero_based, frame_id_zero_based + 1]
        detections = []
        used_frame_key = None
        for cf in candidate_frames:
            if cf in frame_index:
                detections = frame_index[cf]
                used_frame_key = cf
                break
        if used_frame_key is None:
            detections = []

        # Global header
        cv2.putText(
            frame,
            f"Part 4 ADAS Intelligence Output | Frame {frame_id_zero_based}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            GTEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        active_count = 0

        for det in detections:
            track_id = str(det["track_id"])
            cls = str(det.get("class", "unknown"))

            if args.only_person and cls != "person":
                continue

            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]

            history = track_histories.get(track_id, [])
            X_raw = get_history_window(history, int(used_frame_key), input_len=input_len)

            # current motion info from matching processed history
            current_hist = None
            for h in history:
                if int(h["frame"]) == int(used_frame_key):
                    current_hist = h
                    break
            if current_hist is None and history:
                prev_hist = [h for h in history if int(h["frame"]) <= int(used_frame_key)]
                current_hist = prev_hist[-1] if prev_hist else None

            if current_hist is None:
                continue

            cx = current_hist["cx"]
            cy = current_hist["cy"]
            vx = current_hist["vx"]
            vy = current_hist["vy"]
            speed = current_hist["speed_px_per_frame"]
            distance_to_ego_px = current_hist["distance_to_ego_px"]

            collision_info = collision_index.get((track_id, int(used_frame_key)), {})
            if not collision_info:
                # try alternate frame id
                alt = frame_id_zero_based if used_frame_key == frame_id_zero_based + 1 else frame_id_zero_based + 1
                collision_info = collision_index.get((track_id, int(alt)), {})

            # Behavior
            behavior_id, behavior_label = predict_behavior_from_aeb_style_features(
                behavior_bundle,
                cx=cx,
                cy=cy,
                vx=vx,
                vy=vy,
                speed=speed,
                distance_to_ego_px=distance_to_ego_px
            )

            # Predict trajectory + uncertainty if enough history
            mean_pred_abs = None
            std_pred_abs = None
            overall_unc = 0.0
            if X_raw is not None:
                mean_pred_abs, std_pred_abs = mc_predict_single(
                    model=traj_model,
                    X_raw=X_raw,
                    feat_mean=feat_mean,
                    feat_std=feat_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    mc_samples=args.mc_samples
                )
                overall_unc = float(np.mean(np.linalg.norm(std_pred_abs, axis=-1)))

            # Adaptive AEB
            feat_dict = compute_aeb_features(
                det_info=current_hist,
                behavior_id_pred=behavior_id,
                collision_info=collision_info
            )

            # Replace uncertainty proxy with actual uncertainty if available
            if mean_pred_abs is not None and std_pred_abs is not None:
                feat_dict["uncertainty_proxy"] = overall_unc

            risk_score, aeb_pred = predict_adaptive_aeb(aeb_bundle, feat_dict)
            box_color = choose_box_color(aeb_pred, risk_score)

            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # draw future path
            if mean_pred_abs is not None and std_pred_abs is not None:
                draw_predicted_path(frame, mean_pred_abs, std_pred_abs)

            # text
            ttc_val = feat_dict["ttc"]
            ttc_str = f"{ttc_val:.2f}" if ttc_val > 0 else "NA"
            unc_str = f"{overall_unc:.2f}" if mean_pred_abs is not None else "NA"
            risk_str = f"{risk_score:.2f}"

            status = "BRAKE" if aeb_pred == 1 else ("WATCH" if risk_score >= 0.40 else "SAFE")

            lines = [
                f"ID: {track_id}  {cls}",
                f"Behavior: {behavior_label}",
                f"TTC: {ttc_str}  Risk: {risk_str}",
                f"Unc: {unc_str}  AEB: {status}",
            ]

            text_x = max(5, x1)
            text_y = max(20, y1 - 55)
            put_text_block(frame, lines, text_x, text_y, box_color, scale=0.45, thickness=1, line_gap=16)

            active_count += 1

        # frame footer
        cv2.putText(
            frame,
            f"Active objects shown: {active_count}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GTEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        writer.write(frame)
        frame_id_zero_based += 1

        if frame_id_zero_based % 50 == 0:
            print(f"[INFO] Processed {frame_id_zero_based}/{frame_count} frames")

    cap.release()
    writer.release()

    print("\n[INFO] Part 4 visualization completed.")
    print(f"[INFO] Output video saved to: {args.output}")


if __name__ == "__main__":
    main()