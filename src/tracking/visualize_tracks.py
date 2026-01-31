import argparse
import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from collections import deque

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

def visualize_tracks(video_source, tracks_dir, output_file, config_path):
    # Load config for class names
    cfg = load_config(config_path)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open {video_source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Video Writer
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    tracks_dir = Path(tracks_dir)
    frame_id = 0
    
    # Store history for tails: {track_id: deque of points}
    trajectories = {}
    
    print("Generating tracking video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # Load tracks for this frame
        json_file = tracks_dir / f"frame_{frame_id:04d}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                tracks = data.get('tracks', [])
                
            for t in tracks:
                tid = t['track_id']
                cid = t['class_id']
                cls_name = t['class']
                bbox = t['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                color = COLORS.get(cid, COLORS["default"])
                
                # Draw BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                label = f"ID:{tid} {cls_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update Trajectory
                center = (int((x1 + x2) / 2), int(y2))
                if tid not in trajectories:
                    trajectories[tid] = deque(maxlen=20)
                trajectories[tid].append(center)
                
                # Draw Tail
                path = list(trajectories[tid])
                for i in range(1, len(path)):
                    cv2.line(frame, path[i-1], path[i], color, 2)
                    
        # Draw Frame Counter
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_id % 50 == 0:
            print(f"Drawing frame {frame_id}...")
            
    cap.release()
    out.release()
    print(f"Saved video to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", default="results/tracking/tracks")
    parser.add_argument("--output", default="results/tracking/videos/tracking_demo.mp4")
    parser.add_argument("--config", default="configs/tracker_config.yaml")
    args = parser.parse_args()
    
    visualize_tracks(args.video, args.tracks, args.output, args.config)
