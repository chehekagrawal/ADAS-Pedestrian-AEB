import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import cv2

def generate_collision_reports(simulation_path, trajectories_path, video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(simulation_path, 'r') as f:
        sim_data = json.load(f)
        
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
        
    # Filter valid TTCs
    valid_data = [e for e in sim_data if not np.isinf(e['ttc']) and e['ttc'] > 0]
    
    if not valid_data:
        print("No valid collision risk data found to plot.")
        return

    # 1. Plot: Risk Density (Distance vs TTC Heatmap)
    plt.figure(figsize=(10, 6))
    # Use hexbin for a more "scientific/engaging" look than just scatter
    hb = plt.hexbin([e['distance'] for e in valid_data], 
                   [e['ttc'] for e in valid_data], 
                   gridsize=35, cmap='YlOrRd', bins='log', mincnt=1)
    cb = plt.colorbar(hb, label='Detection Frequency (log scale)')
    plt.axhline(y=2.0, color='red', linestyle='--', label='Critical Risk (2s)')
    plt.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='Warning (5s)')
    plt.title('Collision Risk Density Map (TTC vs Proximity)', fontsize=15, pad=15)
    plt.xlabel('Estimated Distance (pixels)')
    plt.ylabel('Time to Collision (seconds)')
    plt.ylim(0, 15)
    plt.grid(True, alpha=0.15)
    plt.legend(loc='upper right')
    plt.savefig(output_dir / 'risk_density_map.png', dpi=150)
    plt.close()
    
    # 2. Plot: Multi-Object Risk Timelines (Top 5 Riskiest)
    id_min_ttc = {}
    for e in valid_data:
        tid = e['track_id']
        if tid not in id_min_ttc or e['ttc'] < id_min_ttc[tid]:
            id_min_ttc[tid] = e['ttc']
    
    top_risky_ids = sorted(id_min_ttc, key=id_min_ttc.get)[:5]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_risky_ids)))
    for i, tid in enumerate(top_risky_ids):
        track_data = [e for e in valid_data if e['track_id'] == tid]
        track_data.sort(key=lambda x: x['frame'])
        frames = [e['frame'] for e in track_data]
        ttcs = [e['ttc'] for e in track_data]
        plt.plot(frames, ttcs, label=f'Object ID {tid}', linewidth=2.5, color=colors[i], marker='o', markersize=3, alpha=0.8)
    
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, label='AEB Trigger')
    plt.fill_between([min(frames) if frames else 0, max(frames) if frames else 1000], 0, 2.0, color='red', alpha=0.05)
    plt.title('Dynamic Risk Profiles: Top 5 High-Risk Objects', fontsize=15, pad=15)
    plt.xlabel('Video frame index')
    plt.ylabel('Time-to-Collision (s)')
    plt.ylim(0, 15)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_dir / 'multi_object_risk_timeline.png', dpi=150)
    plt.close()
    
    # 3. Plot: Ego-Centric Radar Risk Map (Top-Down Representation)
    plt.figure(figsize=(10, 10))
    # Draw Ego Car (simplified)
    plt.fill([-100, 100, 100, -100], [-100, -100, 100, 100], color='blue', alpha=0.2, label='Ego Vehicle')
    plt.text(0, 0, 'EGO', ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue')
    
    # Draw Radar Concentric Circles (for scale)
    for dist in [500, 1000, 1500, 2000, 2500]:
        circle = plt.Circle((0, 0), dist, color='gray', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.text(dist, 50, f'{dist}px', color='gray', alpha=0.5, fontsize=8)

    # Plot objects
    for e in valid_data:
        if e['ttc'] < 12.0: # Moderate to High risk only for clarity
            tid = str(e['track_id'])
            frame_idx = e['frame']
            rel_x = 0
            if tid in trajectories:
                for p in trajectories[tid]:
                    if p['frame'] == frame_idx:
                        bbox = p['bbox']
                        rel_x = (bbox[0] + bbox[2]) / 2 - 1280 # Relative to center of 2560 width
                        break
            
            # Map TTC to color (Red=Dangerous, Yellow=Caution, Green=Clear)
            # Use a diverging colormap or custom logic
            if e['ttc'] < 3.0: color = 'red'
            elif e['ttc'] < 7.0: color = 'orange'
            else: color = 'forestgreen'
            
            plt.scatter(rel_x, e['distance'], c=color, s=25, alpha=0.3)
            
    plt.title('Top-Down Potential Collision Radar', fontsize=18, pad=20)
    plt.xlabel('Lateral Offset from Center (pixels)', fontsize=12)
    plt.ylabel('Forward Object Distance (pixels)', fontsize=12)
    plt.xlim(-1280, 1280)
    plt.ylim(-200, 3200)
    plt.grid(True, alpha=0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_dir / 'ego_radar_risk.png', dpi=150)
    plt.close()
    
    # 4. Plot: High-Risk Detections by Class (Engaging Bar Chart)
    class_risk_counts = {}
    for e in valid_data:
        if e['ttc'] < 10.0:
            tid = str(e['track_id'])
            if tid in trajectories:
                cls_name = trajectories[tid][0].get('class', 'unknown')
                class_risk_counts[cls_name] = class_risk_counts.get(cls_name, 0) + 1
                
    if class_risk_counts:
        plt.figure(figsize=(10, 6))
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        plt.bar(class_risk_counts.keys(), class_risk_counts.values(), color=colors[:len(class_risk_counts)])
        plt.title('Collision Danger Breakdown by Object Type (TTC < 10s)', fontsize=15, pad=15)
        plt.ylabel('Aggregate Frames with Potential Risk')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(output_dir / 'class_risk_distribution.png', dpi=150)
        plt.close()
    
    # Final Summary: Capture highest-risk frame
    sim_data_sorted = sorted(valid_data, key=lambda x: x['ttc'])
    if sim_data_sorted:
        risk_entry = sim_data_sorted[0]
        best_frame = risk_entry['frame']
        min_ttc_val = risk_entry['ttc']
        target_id = str(risk_entry['track_id'])
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            if target_id in trajectories:
                for p in trajectories[target_id]:
                    if p['frame'] == best_frame:
                        x1, y1, x2, y2 = map(int, p['bbox'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6) # Yellow thick
                        cv2.putText(frame, f"TARGET ID {target_id} [TTC {min_ttc_val:.1f}s]", (x1, y1-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 5)
                        break

            cv2.putText(frame, "CRITICAL PROTECTION EVENT ANALYZED", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 8)
            cv2.imwrite(str(output_dir / 'critical_risk_capture.jpg'), frame)

    print(f"Advanced collision reports generated at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", default="results/collision/logs/new_test_run.json")
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/collision/plots/analytics")
    args = parser.parse_args()
    
    generate_collision_reports(args.simulation, args.trajectories, args.video, args.output)
