import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def generate_detection_analytics(trajectories_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
        
    all_detections = []
    for tid, points in trajectories.items():
        all_detections.extend(points)
        
    if not all_detections:
        print("No detections found.")
        return

    # 1. Plot: Confidence Distribution
    confidences = [d.get('confidence', 0) for d in all_detections if 'confidence' in d]
    if confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Detection Confidence Distribution', fontsize=15)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(output_dir / 'detection_confidence_hist.png', dpi=150)
        plt.close()

    # 2. Plot: Spatial Distribution (Bbox Centers)
    centers_x = [(d['bbox'][0] + d['bbox'][2]) / 2 for d in all_detections]
    centers_y = [(d['bbox'][1] + d['bbox'][3]) / 2 for d in all_detections]
    
    plt.figure(figsize=(12, 8))
    hb = plt.hexbin(centers_x, centers_y, gridsize=40, cmap='Blues', mincnt=1)
    plt.colorbar(hb, label='Detection Density')
    plt.title('Spatial Detection Density (Frame Map)', fontsize=15)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.gca().invert_yaxis() # Match image coords
    plt.savefig(output_dir / 'detection_spatial_heatmap.png', dpi=150)
    plt.close()

    # 3. Plot: Bbox Size vs Aspect Ratio
    widths = [(d['bbox'][2] - d['bbox'][0]) for d in all_detections]
    heights = [(d['bbox'][3] - d['bbox'][1]) for d in all_detections]
    areas = [w * h for w, h in zip(widths, heights)]
    aspect_ratios = [w / h if h > 0 else 0 for w, h in zip(widths, heights)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(areas, aspect_ratios, alpha=0.3, c='purple', s=5)
    plt.xscale('log')
    plt.title('Bbox Characteristic Distribution', fontsize=15)
    plt.xlabel('Bbox Area (pixels^2, log scale)')
    plt.ylabel('Aspect Ratio (W/H)')
    plt.grid(True, alpha=0.2)
    plt.savefig(output_dir / 'detection_size_aspect_scatter.png', dpi=150)
    plt.close()

    # 4. Plot: Class Distribution (Total Instances)
    class_counts = {}
    for d in all_detections:
        cls = d.get('class', 'unknown')
        class_counts[cls] = class_counts.get(cls, 0) + 1
        
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='mediumseagreen')
    plt.title('Total Detection Instances by Class', fontsize=15)
    plt.ylabel('Frame-level Instance Count')
    plt.savefig(output_dir / 'detection_class_bar.png', dpi=150)
    plt.close()

    print(f"Advanced detection reports generated at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json")
    parser.add_argument("--output", default="results/detection/plots/advanced")
    args = parser.parse_args()
    generate_detection_analytics(args.trajectories, args.output)
