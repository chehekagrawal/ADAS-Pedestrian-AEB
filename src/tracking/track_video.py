import argparse
import cv2
import yaml
import numpy as np
import time
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from tracker import Tracker

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Color mapping (BGR)
COLORS = {
    0: (0, 0, 255),    # Pedestrian: Red
    1: (255, 0, 0),    # Car: Blue
    2: (0, 255, 0),    # Bicycle: Green
    3: (0, 165, 255),  # Motorcycle: Orange
    "default": (255, 255, 255)
}

def track_video(source, model_path, output_path, config_path, classes=None):
    # Load config
    cfg = load_config(config_path)
    params = cfg['tracking_parameters']
    target_classes_map = cfg['target_classes']
    
    # Initialize Tracker
    tracker = Tracker(
        max_age=params['max_age'],
        min_hits=params['min_hits'],
        iou_threshold=params['iou_threshold']
    )
    
    # Initialize Model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open Video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open {source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    # Output Video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Trajectories for visualization
    trajectories = {}
    
    frame_id = 0
    print(f"Starting tracking on {source}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # 1. Detection
        results = model(frame, verbose=False, classes=classes)
        
        dets_array = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter if classes are specified (model call usually handles this, but good safety)
                if classes and cls not in classes:
                    continue
                    
                dets_array.append([b[0], b[1], b[2], b[3], conf, cls])
        
        dets_array = np.array(dets_array)
        if len(dets_array) == 0:
            dets_array = np.empty((0, 6))
            
        # 2. Tracking
        track_results = tracker.update(dets_array)
        
        # 3. Visualization
        for trk in track_results:
            # trk: [x1, y1, x2, y2, track_id, class_id]
            x1, y1, x2, y2, track_id, cls_id = trk
            track_id = int(track_id)
            cls_id = int(cls_id)
            
            cls_name = target_classes_map.get(cls_id, "unknown")
            color = COLORS.get(cls_id, COLORS["default"])
            
            # Draw BBox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw Label
            label = f"ID:{track_id} {cls_name}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update Trajectory
            center = (int((x1 + x2) / 2), int(y2))
            if track_id not in trajectories:
                trajectories[track_id] = deque(maxlen=20)
            trajectories[track_id].append(center)
            
            # Draw Tail
            path = list(trajectories[track_id])
            for i in range(1, len(path)):
                cv2.line(frame, path[i-1], path[i], color, 2)
        
        # Draw Frame Info
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")
            
    cap.release()
    out.release()
    print(f"Done! Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--output", default="results/tracking/videos/tracking_direct.mp4", help="Output video path")
    parser.add_argument("--config", default="configs/tracker_config.yaml", help="Tracker config")
    parser.add_argument("--classes", nargs="+", type=int, help="Classes to track (e.g. 0 1 2 3)")
    
    args = parser.parse_args()
    
    track_video(args.source, args.model, args.output, args.config, args.classes)
