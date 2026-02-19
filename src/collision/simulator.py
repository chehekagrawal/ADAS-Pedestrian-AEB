import json
import os
import numpy as np

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


def run_simulation():
    # Load trajectories from Part 2 tracking output
    with open("results/tracking/tracking_multiclass/trajectories.json", "r") as f:
        trajectories = json.load(f)

    ego = EgoVehicle(speed=12.0, deceleration=6.0, reaction_time=1.0)
    aeb = AEBController()

    logs = []
    vehicle_position = [0, 0]  # assume car at origin

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
    os.makedirs("results/collision/logs", exist_ok=True)

    # Save logs
    with open("results/collision/logs/run_01.json", "w") as f:
        json.dump(logs, f, indent=4)

    print("Simulation complete. Logs saved.")


if __name__ == "__main__":
    run_simulation()
