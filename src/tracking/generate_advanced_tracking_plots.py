import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def generate_tracking_analytics(trajectories_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
        
    if not trajectories:
        print("No tracking data found.")
        return

    # 1. Plot: Track Longevity (Survival Histogram)
    durations = [len(points) for points in trajectories.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, color='plum', edgecolor='black', alpha=0.7)
    plt.title('Track Longevity (Survival Time)', fontsize=15)
    plt.xlabel('Duration (frames)')
    plt.ylabel('Number of Tracks')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / 'tracking_longevity_hist.png', dpi=150)
    plt.close()

    # 2. Plot: Active Tracks Timeline
    frame_track_count = {}
    for tid, points in trajectories.items():
        for p in points:
            f = p['frame']
            frame_track_count[f] = frame_track_count.get(f, 0) + 1
            
    sorted_frames = sorted(frame_track_count.keys())
    counts = [frame_track_count[f] for f in sorted_frames]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_frames, counts, color='darkorange', linewidth=2)
    plt.fill_between(sorted_frames, counts, color='orange', alpha=0.2)
    plt.title('Concurrent Active Tracks Over Time', fontsize=15)
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Active Tracks')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'tracking_concurrency_timeline.png', dpi=150)
    plt.close()

    # 3. Plot: Estimated Motion Velocity (Pixel/Frame)
    velocities = []
    for tid, points in trajectories.items():
        if len(points) < 2: continue
        for i in range(1, len(points)):
            p1 = points[i-1]['bbox']
            p2 = points[i]['bbox']
            c1 = np.array([(p1[0]+p1[2])/2, (p1[1]+p1[3])/2])
            c2 = np.array([(p2[0]+p2[2])/2, (p2[1]+p2[3])/2])
            dist = np.linalg.norm(c2 - c1)
            velocities.append(dist)
            
    if velocities:
        plt.figure(figsize=(10, 6))
        plt.hist(velocities, bins=50, range=(0, 50), color='salmon', alpha=0.7)
        plt.title('Object Motion Velocity Distribution', fontsize=15)
        plt.xlabel('Pixel Velocity (px/frame)')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'tracking_velocity_hist.png', dpi=150)
        plt.close()

    # 4. Plot: Track Path Curvature vs Length
    plt.figure(figsize=(10, 6))
    # Curvature approximation: Total distance vs Displacement
    for tid, points in trajectories.items():
        if len(points) < 5: continue
        p_start = points[0]['bbox']
        p_end = points[-1]['bbox']
        c_start = np.array([(p_start[0]+p_start[2])/2, (p_start[1]+p_start[3])/2])
        c_end = np.array([(p_end[0]+p_end[2])/2, (p_end[1]+p_end[3])/2])
        displacement = np.linalg.norm(c_end - c_start)
        
        total_path = 0
        for i in range(1, len(points)):
            pa = points[i-1]['bbox']
            pb = points[i]['bbox']
            ca = np.array([(pa[0]+pa[2])/2, (pa[1]+pa[3])/2])
            cb = np.array([(pb[0]+pb[2])/2, (pb[1]+pb[3])/2])
            total_path += np.linalg.norm(cb - ca)
        
        # Plot Path Efficiency (Displacement / Path)
        efficiency = displacement / total_path if total_path > 0 else 0
        plt.scatter(len(points), efficiency, alpha=0.4, c='navy', s=20)
        
    plt.title('Track Motion Efficiency (Straightness)', fontsize=15)
    plt.xlabel('Track Duration (frames)')
    plt.ylabel('Displacement / Total Path Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'tracking_efficiency_scatter.png', dpi=150)
    plt.close()

    print(f"Advanced tracking reports generated at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json")
    parser.add_argument("--output", default="results/tracking/plots/advanced")
    args = parser.parse_args()
    generate_tracking_analytics(args.trajectories, args.output)
