import json
import os
import numpy as np
import argparse

from src.collision.motion_model import predict_future, EgoVehicle
from src.collision.collision_engine import (
    create_vehicle_box,
    compute_distance,
    compute_relative_speed,
    compute_ttc,
    check_collision,
    collision_probability,
)
from src.collision.aeb_controller import AEBController


def run_simulation(trajectories_path="results/tracking/tracking_multiclass/trajectories.json", output_log="results/collision/logs/run_01.json"):
    # Load trajectories from Part 2 tracking output
    if not os.path.exists(trajectories_path):
        print(f"Error: Trajectories file not found at {trajectories_path}")
        return

    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    ego = EgoVehicle(speed=12.0, deceleration=6.0, reaction_time=1.0)
    aeb = AEBController()

    logs = []
    vehicle_position = [0, 0]  # assume car at origin

    print(f"Running simulation using {trajectories_path}...")

    for track_id, detections in trajectories.items():

        # Convert bbox detections into [x_center, y_center, time]
        track = []

        for det in detections:
            # Optional: only process pedestrians
            # if det["class"] != "person":
            #     continue

            x1, y1, x2, y2 = det["bbox"]
            frame = det["frame"]

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            track.append([x_center, y_center, frame])

        # Skip very short tracks
        if len(track) < 2:
            continue

        future_positions, ped_velocity = predict_future(track)

        for i, ped_pos in enumerate(future_positions):
            distance = compute_distance(vehicle_position, ped_pos)
            relative_speed = compute_relative_speed(ego.speed, ped_velocity)
            ttc = compute_ttc(distance, relative_speed)

            vehicle_poly = create_vehicle_box(0, 0)
            collision = check_collision(vehicle_poly, ped_pos)

            prob = collision_probability(ttc)
            triggered = aeb.evaluate(ttc)

            logs.append({
                "frame": i,
                "track_id": track_id,
                "distance": distance,
                "ttc": ttc,
                "collision_probability": prob,
                "collision_detected": collision,
                "AEB_triggered": triggered
            })

    # Create output directory
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    # Save logs
    with open(output_log, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"Simulation complete. Logs saved to {output_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AEB simulation based on trajectories")
    parser.add_argument("--trajectories", default="results/tracking/tracking_multiclass/trajectories.json", help="Path to trajectories.json")
    parser.add_argument("--output", default="results/collision/logs/run_01.json", help="Output JSON log path")
    
    args = parser.parse_args()
    
    run_simulation(trajectories_path=args.trajectories, output_log=args.output)
