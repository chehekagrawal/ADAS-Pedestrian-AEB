import os
import json
import cv2
import numpy as np

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


def is_reasonable_box(x1, y1, x2, y2, W, H):
    # malformed
    if x2 <= x1 or y2 <= y1:
        return False

    # completely outside frame
    if x2 < 0 or x1 > W or y2 < 0 or y1 > H:
        return False

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh

    # too tiny
    if bw < 10 or bh < 10:
        return False

    # absurdly huge
    if bw > 0.6 * W or bh > 0.7 * H:
        return False

    # tiny noisy blobs
    if area < 200:
        return False

    # floating sky boxes quick hack:
    # if object is very high in image and small, skip it
    cy = (y1 + y2) / 2.0
    if cy < 0.45 * H and area < 12000:
        return False

    return True


def main(
    video_path="data/sample/part_3_final_input_1.mp4",
    trajectories_path="results/tracking/tracking_multiclass/trajectories.json",
    output_path="results/collision/plots/ttc_overlay_1.mp4",
    ego_speed_px_per_frame=12.0,
):
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

    ego_pos = np.array([W / 2.0, H / 2.0], dtype=float)

    frame_index = {}
    for track_id, dets in trajectories.items():
        dets = sorted(dets, key=lambda d: int(d["frame"]))
        trajectories[track_id] = dets
        for det in dets:
            fr = int(det["frame"])
            frame_index.setdefault(fr, []).append((track_id, det))

    frame_num = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # try frame_num first; if needed later we can switch back to +1
        fr_key = frame_num

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

                x1, y1, x2, y2 = bbox

                if not is_reasonable_box(x1, y1, x2, y2, W, H):
                    continue

                # clamp to frame
                x1 = max(0, min(int(x1), W - 1))
                y1 = max(0, min(int(y1), H - 1))
                x2 = max(0, min(int(x2), W - 1))
                y2 = max(0, min(int(y2), H - 1))

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                dets = trajectories[track_id]
                hist = [d for d in dets if int(d["frame"]) <= fr_key]
                hist = sorted(hist, key=lambda d: int(d["frame"]))
                hist = hist[-5:]

                if len(hist) < 2:
                    continue

                centers = [bbox_center(d["bbox"]) for d in hist]
                frames_hist = [int(d["frame"]) for d in hist]
                v_obj = estimate_velocity_from_centers(centers, frames_hist)

                # hardcoded cleanup for drifting tracks
                if np.linalg.norm(v_obj) > 80:
                    continue

                obj_pos = np.array([float(cx), float(cy)])
                los = obj_pos - ego_pos
                los_norm = np.linalg.norm(los)

                if los_norm > 1e-6:
                    v_ego = (los / los_norm) * ego_speed_px_per_frame
                else:
                    v_ego = np.array([0.0, 0.0])

                v_rel = v_obj - v_ego
                ttc, dist = compute_ttc(ego_pos, obj_pos, v_rel)

                # quick practical filters
                if dist > 1200:
                    continue

                if not np.isinf(ttc) and ttc > 60:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
    print("Overlay video resolution:", W, H)
    print("FPS:", fps)


if __name__ == "__main__":
    main()