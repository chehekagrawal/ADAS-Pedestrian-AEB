import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def generate_dashboard(trajectories_path, simulation_path, output_path):
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
    
    with open(simulation_path, 'r') as f:
        sim_data = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle('Tracking & Safety Systematic Analytics', fontsize=20)
    
    # 1. Class Distribution (Unique Tracks)
    class_counts = {}
    for tid, points in trajectories.items():
        cls_name = points[0].get('class', 'unknown')
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    axes[0, 0].bar(classes, counts, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    axes[0, 0].set_title('Unique Object Count by Class')
    axes[0, 0].set_ylabel('Number of Unique Tracks')
    
    # 2. Track Longevity (Top 10)
    longevity = sorted([(tid, len(points)) for tid, points in trajectories.items()], key=lambda x: x[1], reverse=True)[:10]
    tids, lens = zip(*longevity)
    axes[0, 1].bar([f"ID {t}" for t in tids], lens, color='plum')
    axes[0, 1].set_title('Top 10 Longest Active Tracks')
    axes[0, 1].set_ylabel('Duration (frames)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Collision Risk (Min TTC Timeline)
    # Extract min TTC per frame manually without pandas
    frame_min_ttc = {}
    for entry in sim_data:
        fr = entry['frame']
        ttc = float(entry['ttc'])
        if fr not in frame_min_ttc or ttc < frame_min_ttc[fr]:
            frame_min_ttc[fr] = ttc
            
    sorted_frames = sorted(frame_min_ttc.keys())
    min_ttcs = [min(30, frame_min_ttc[fr]) for fr in sorted_frames] # clip to 30 for visualization

    axes[1, 0].plot(sorted_frames, min_ttcs, color='red', linewidth=1)
    axes[1, 0].axhline(y=2.0, color='darkred', linestyle='--', label='AEB Threshold (2s)')
    axes[1, 0].set_title('Minimum Time-to-Collision (TTC) Timeline')
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('TTC (seconds)')
    axes[1, 0].legend()
    
    # 4. Spatial Activity Heatmap (Histogram 2D)
    all_centers_x = []
    all_centers_y = []
    for tid, points in trajectories.items():
        for p in points:
            bbox = p['bbox']
            all_centers_x.append((bbox[0] + bbox[2]) / 2)
            all_centers_y.append((bbox[1] + bbox[3]) / 2)
            
    h = axes[1, 1].hist2d(all_centers_x, all_centers_y, bins=30, cmap='inferno')
    axes[1, 1].set_title('Spatial Distribution Heatmap')
    axes[1, 1].invert_yaxis()
    fig.colorbar(h[3], ax=axes[1, 1], label='Detection Count')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Systematic dashboard saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json")
    parser.add_argument("--simulation", default="results/collision/logs/new_test_run.json")
    parser.add_argument("--output", default="results/tracking/plots/systematic_analytics.png")
    args = parser.parse_args()
    
    generate_dashboard(args.trajectories, args.simulation, args.output)
