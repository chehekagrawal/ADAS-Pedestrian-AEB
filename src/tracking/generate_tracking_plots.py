import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def generate_plots(trajectories_path, video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
        
    # 1. Trajectory Plot (X-Y)
    plt.figure(figsize=(10, 6))
    for tid, points in trajectories.items():
        if len(points) < 5: continue # Skip short tracks
        xs = [p['bbox'][0] for p in points]
        ys = [p['bbox'][1] for p in points]
        plt.plot(xs, ys, label=f'ID {tid}')
    
    plt.gca().invert_yaxis()
    plt.title('Vehicle & Pedestrian Trajectories')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectories_plot.png')
    plt.close()
    
    # 2. Sample Gallery (Grid of frames)
    cap = cv2.VideoCapture(video_path)
    frames_to_save = [100, 200, 300, 400, 500, 600]
    gallery_images = []
    
    for f_id in frames_to_save:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_id - 1)
        ret, frame = cap.read()
        if not ret: break
        
        # Draw tracks for this frame
        for tid, points in trajectories.items():
            for p in points:
                if p['frame'] == f_id:
                    x1, y1, x2, y2 = map(int, p['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Downsample for gallery
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // 4, h // 4))
        gallery_images.append(small_frame)
        
    if gallery_images:
        # Create a grid
        rows = 2
        cols = (len(gallery_images) + 1) // 2
        
        h, w = gallery_images[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        for idx, img in enumerate(gallery_images):
            r = idx // cols
            c = idx % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
            
        cv2.imwrite(str(output_dir / 'tracking_gallery.jpg'), grid)
        print(f"Gallery saved to {output_dir / 'tracking_gallery.jpg'}")

    cap.release()
    print(f"Trajectory plot saved to {output_dir / 'trajectories_plot.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/tracking/plots")
    args = parser.parse_args()
    
    generate_plots(args.trajectories, args.video, args.output)
