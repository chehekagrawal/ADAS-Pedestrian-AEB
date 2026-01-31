import argparse
import json
import numpy as np
from pathlib import Path

def evaluate_tracking(trajectories_file, output_report):
    with open(trajectories_file, 'r') as f:
        data = json.load(f)
        
    # data format: {track_id: [list of frames]}
    
    total_tracks = len(data)
    durations = []
    class_counts = {}
    
    for tid, frames in data.items():
        durations.append(len(frames))
        
        # Assume class is constant, check first frame
        cls = frames[0]['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        
    avg_duration = np.mean(durations) if durations else 0
    max_duration = np.max(durations) if durations else 0
    min_duration = np.min(durations) if durations else 0
    
    report = []
    report.append("=== Tracking Evaluation Report ===")
    report.append(f"Total Unique Tracks: {total_tracks}")
    report.append(f"Average Track Duration: {avg_duration:.2f} frames")
    report.append(f"Max Track Duration: {max_duration} frames")
    report.append(f"Min Track Duration: {min_duration} frames")
    report.append("")
    report.append("--- Class Breakdown ---")
    for cls, count in class_counts.items():
        report.append(f"{cls}: {count}")
        
    report_content = "\n".join(report)
    print(report_content)
    
    with open(output_report, "w") as f:
        f.write(report_content)
        
    print(f"\nReport saved to {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/tracking/trajectories.json")
    parser.add_argument("--report", default="results/tracking/tracking_report.txt")
    args = parser.parse_args()
    
    evaluate_tracking(args.trajectories, args.report)
