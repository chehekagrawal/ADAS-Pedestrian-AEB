import os
import json
import cv2
import numpy as np

# Class map for your project
CLASS_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
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
    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Ego vehicle assumed at image center (more reasonable than 0,0)
    ego_pos = np.array([W / 2.0, H / 2.0], dtype=float)

    # Build quick lookup: for each frame, which tracks exist with bbox+class
    frame_index = {}
    for track_id, dets in trajectories.items():
        for det in dets:
            fr = int(det["frame"])
            frame_index.setdefault(fr, []).append((track_id, det))

    frame_num = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # If your tracking frames start at 1, use frame_num+1
        fr_key = frame_num + 1

        # Draw ego reference point
        cv2.circle(frame, (int(ego_pos[0]), int(ego_pos[1])), 5, (255, 255, 255), -1)
        cv2.putText(
            frame,
            "EGO",
            (int(ego_pos[0] + 8), int(ego_pos[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if fr_key in frame_index:
            for track_id, det in frame_index[fr_key]:
                bbox = det["bbox"]
                cls = det.get("class", "unknown")
                cls_name = CLASS_MAP.get(cls, str(cls))

                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Estimate velocity from last K positions within this track
                # Collect historical centers up to current frame
                dets = trajectories[track_id]
                hist = [d for d in dets if int(d["frame"]) <= fr_key]
                hist = hist[-5:]  # last 5 samples

                centers = [bbox_center(d["bbox"]) for d in hist]
                frames_hist = [int(d["frame"]) for d in hist]
                v_obj = estimate_velocity_from_centers(centers, frames_hist)

                # Simplified relative velocity: ego moves forward magnitude ego_speed_px_per_frame
                # We do not have ego direction, so treat ego velocity as towards object line-of-sight
                obj_pos = np.array([float(cx), float(cy)])
                los = obj_pos - ego_pos
                los_norm = np.linalg.norm(los)
                if los_norm > 1e-6:
                    v_ego = (los / los_norm) * ego_speed_px_per_frame
                else:
                    v_ego = np.array([0.0, 0.0])

                v_rel = v_obj - v_ego
                ttc, dist = compute_ttc(ego_pos, obj_pos, v_rel)

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Text overlay
                if np.isinf(ttc) or ttc > 999:
                    ttc_text = "TTC: inf"
                else:
                    ttc_text = f"TTC: {ttc:.2f}s"

                label = f"ID {track_id} | {cls_name} | {ttc_text}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"Saved overlay video to: {output_path}")


if __name__ == "__main__":
    main()