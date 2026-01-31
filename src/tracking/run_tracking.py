import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from tracker import Tracker

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_tracking(detection_dir, output_dir, config_path):
    # Load config
    cfg = load_config(config_path)
    params = cfg['tracking_parameters']
    target_classes = cfg['target_classes']
    
    # Initialize Tracker
    tracker = Tracker(
        max_age=params['max_age'],
        min_hits=params['min_hits'],
        iou_threshold=params['iou_threshold']
    )
    
    detection_dir = Path(detection_dir)
    output_dir = Path(output_dir)
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all json files sorted
    json_files = sorted(list(detection_dir.glob("frame_*.json")))
    
    if not json_files:
        print(f"No detection files found in {detection_dir}")
        return

    full_trajectories = {} # Format: {track_id: [{"frame": 1, "bbox": [], "class": ""}, ...]}

    print(f"Starting tracking on {len(json_files)} frames...")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        frame_id = data['frame_id']
        detections = data['detections']
        
        # Prepare detections for SORT: [x1, y1, x2, y2, score, class_id]
        dets_array = []
        for d in detections:
            # Check if class is one we want to track (if filtering is needed)
            # For now, we track everything passed to us
            bbox = d['bbox']
            score = d['confidence']
            cls_id = d.get('class_id', 0) # Default to 0 (pedestrian) if missing
            
            dets_array.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])
            
        dets_array = np.array(dets_array)
        if len(dets_array) == 0:
            dets_array = np.empty((0, 6))
            
        # Update Tracker
        track_results = tracker.update(dets_array)
        
        # Serialize output
        output_tracks = []
        for trk in track_results:
            # trk: [x1, y1, x2, y2, track_id, class_id]
            x1, y1, x2, y2, track_id, cls_id = trk
            track_id = int(track_id)
            cls_id = int(cls_id)
            cls_name = target_classes.get(cls_id, "unknown")
            
            track_obj = {
                "track_id": track_id,
                "class_id": cls_id,
                "class": cls_name,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            }
            output_tracks.append(track_obj)
            
            # Update full trajectories
            if track_id not in full_trajectories:
                full_trajectories[track_id] = []
            full_trajectories[track_id].append({
                "frame": frame_id,
                "bbox": track_obj["bbox"],
                "class": cls_name
            })
            
        # Save frame output
        with open(tracks_dir / f"frame_{frame_id:04d}.json", "w") as f:
            json.dump({
                "frame_id": frame_id,
                "tracks": output_tracks
            }, f, indent=2)
            
    # Save full trajectories
    with open(output_dir / "trajectories.json", "w") as f:
        json.dump(full_trajectories, f, indent=2)
        
    print(f"Tracking complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default="results/detections")
    parser.add_argument("--output", default="results/tracking")
    parser.add_argument("--config", default="configs/tracker_config.yaml")
    args = parser.parse_args()
    
    run_tracking(args.detections, args.output, args.config)
