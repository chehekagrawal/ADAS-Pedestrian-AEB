import argparse
import cv2
import json
import logging
from pathlib import Path
from ultralytics import YOLO

def generate_detections(source, model_path, output_dir):
    """
    Runs YOLO inference on a video and saves detections as JSON files (one per frame).
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {source}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        results = model(frame, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class
                # Box format: [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "bbox": b,
                    "confidence": conf,
                    "class_id": cls
                })
        
        # Save JSON for this frame
        frame_file = output_dir / f"frame_{frame_id:04d}.json"
        with open(frame_file, "w") as f:
            json.dump({
                "frame_id": frame_id,
                "detections": detections
            }, f, indent=2)
            
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")
            
    cap.release()
    print(f"Done! Detections saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON detections from video")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="results/detections", help="Output directory for JSONs")
    
    args = parser.parse_args()
    
    generate_detections(args.source, args.model, args.output)
