import cv2
import json

video_path = "data/sample/part_3_final_input_2.mp4"
trajectories_path = "results/tracking/tracking_multiclass/trajectories.json"

# choose a specific known sample
track_id_to_test = "350"
frame_to_test = 156

with open(trajectories_path, "r") as f:
    trajectories = json.load(f)

# find detection
target_det = None
for det in trajectories[track_id_to_test]:
    if int(det["frame"]) == frame_to_test:
        target_det = det
        break

if target_det is None:
    raise ValueError("Detection not found")

bbox = target_det["bbox"]
print("Testing bbox:", bbox)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(video_path)

# OpenCV frame indexing is usually zero-based
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_test)
ok, frame = cap.read()

if not ok:
    raise ValueError("Could not read frame")

x1, y1, x2, y2 = map(int, bbox)
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("debug_frame.png", frame)
print("Saved debug_frame.png")
cap.release()