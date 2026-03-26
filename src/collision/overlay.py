import os
import json
import cv2
import numpy as np
import argparse

# Class map for your project
CLASS_MAP = {
    "person": "person",
    "car": "car",
    "bike": "bike",
    "motor": "motor",
    0: "person",
    1: "car",
    2: "bike",
    3: "motor",
}


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def estimate_velocity_from_centers(centers, frames):
    if len(centers) < 2:
        return np.array([0.0, 0.0])
    dx = centers[-1][0] - centers[0][0]
    dy = centers[-1][1] - centers[0][1]
    dt = frames[-1] - frames[0]
    if dt == 0:
        return np.array([0.0, 0.0])
    return np.array([dx / dt, dy / dt], dtype=float)


def compute_ttc(vehicle_pos, obj_pos, v_rel):
    dist = np.linalg.norm(obj_pos - vehicle_pos)
    rel_speed = np.linalg.norm(v_rel)
    if rel_speed <= 1e-6:
        return float("inf"), dist
    return dist / rel_speed, dist


def main(
    video_path="data/sample/youtube_sample.mp4",
    trajectories_path="results/tracking/tracking_multiclass/trajectories.json",
    output_path="results/collision/plots/ttc_overlay.mp4",
    ego_speed_px_per_frame=12.0,
):
    # Load trajectories from Part 2
    if not os.path.exists(trajectories_path):
        print(f"Error: Trajectories file not found at {trajectories_path}")
        return

    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Ego vehicle assumed at image center (more reasonable than 0,0)
    ego_pos = np.array([W / 2.0, H / 2.0], dtype=float)

    # Scaling settings
    font_scale = max(0.6, W / 1600.0)
    thickness = max(2, int(W / 640.0))

    # Build quick lookup: for each frame, which tracks exist with bbox+class
    frame_index = {}
    for track_id, dets in trajectories.items():
        for det in dets:
            fr = int(det["frame"])
            frame_index.setdefault(fr, []).append((track_id, det))

    frame_num = 0
    print(f"Generating overlay for {video_path}...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # If your tracking frames start at 1, use frame_num+1
        fr_key = frame_num + 1

        # Draw ego reference point
        cv2.circle(frame, (int(ego_pos[0]), int(ego_pos[1])), thickness * 2, (255, 255, 255), -1)
        cv2.putText(
            frame,
            "EGO",
            (int(ego_pos[0] + 12), int(ego_pos[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        if fr_key in frame_index:
            for track_id, det in frame_index[fr_key]:
                bbox = det["bbox"]
                cls_name = CLASS_MAP.get(det.get("class", "unknown"), "unknown")

                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Estimate velocity
                track_history = trajectories[track_id]
                hist = [d for d in track_history if int(d["frame"]) <= fr_key]
                hist = hist[-5:]
                centers = [bbox_center(d["bbox"]) for d in hist]
                frames_hist = [int(d["frame"]) for d in hist]
                v_obj = estimate_velocity_from_centers(centers, frames_hist)

                obj_pos = np.array([float(cx), float(cy)])
                los = obj_pos - ego_pos
                los_norm = np.linalg.norm(los)
                v_ego = (los / los_norm * ego_speed_px_per_frame) if los_norm > 1e-6 else np.array([0.0, 0.0])
                v_rel = v_obj - v_ego
                ttc, dist = compute_ttc(ego_pos, obj_pos, v_rel)

                # Draw bbox
                color = (0, 255, 255) # Bright Yellow
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Text overlay — now includes depth in meters when available
                ttc_text = "TTC: inf" if (np.isinf(ttc) or ttc > 999) else f"TTC: {ttc:.2f}s"
                # depth_m is injected by the pipeline; falls back to pixel dist
                depth_text = f"Depth: {dist:.0f}px"
                label = f"ID {track_id} | {cls_name} | {depth_text} | {ttc_text}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                )

        if output_path:
            out.write(frame)
        
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"Processed {frame_num} frames...")

    cap.release()
    if output_path:
        out.release()
    print(f"Saved overlay video to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay TTC on video based on trajectories")
    parser.add_argument("--video", default="data/sample/youtube_sample.mp4", help="Input video path")
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json", help="Path to trajectories.json")
    parser.add_argument("--output", default="results/collision/plots/ttc_overlay.mp4", help="Output video path")
    parser.add_argument("--speed", type=float, default=12.0, help="Ego speed in px/frame")
    
    args = parser.parse_args()
    
    main(
        video_path=args.video,
        trajectories_path=args.trajectories,
        output_path=args.output,
        ego_speed_px_per_frame=args.speed
    )