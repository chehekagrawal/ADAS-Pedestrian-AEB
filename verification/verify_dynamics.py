"""
Verification Script: Vehicle Longitudinal Dynamics & Stopping Distance
Evaluates the Pacejka Magic Formula braking simulation (src/vehicle_dynamics/dynamics_sim.py)
and compares stopping distances with kinematic models and ISO 21994 reference corridors.
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_dynamics.dynamics_sim import simulate_braking, braking_distance
from src.vehicle_dynamics.tire_model import SURFACES

# Nominal surface friction values
SURFACE_MU = {
    "dry": 0.85,
    "wet": 0.55,
    "gravel": 0.45,
    "snow": 0.25,
    "ice": 0.10
}

# ISO 21994 / standard braking corridors at 50 km/h
ISO_CORRIDORS_50 = {
    "dry": (13.0, 22.0),
    "wet": (22.0, 35.0),
    "gravel": (28.0, 40.0),
    "snow": (55.0, 75.0),
    "ice": (60.0, 150.0)
}

def kinematic_distance(speed_kmh, mu):
    v = speed_kmh / 3.6
    g = 9.81
    return (v ** 2) / (2 * mu * g)

def main():
    os.makedirs("verification", exist_ok=True)

    records = []
    speeds = [30, 50, 80, 100]

    print("=== Stopping Distance Verification: Kinematic vs Pacejka ===")
    print(f"{'Surface':<8} {'Speed':<6} {'Kinematic':<12} {'Pacejka (pure)':<16} {'With 0.33s delay':<18} {'ISO Range':<15}")
    print("-" * 75)

    for surf in ["dry", "wet", "gravel", "snow", "ice"]:
        mu = SURFACE_MU[surf]
        for v in speeds:
            d_kin = kinematic_distance(v, mu)
            d_pure = braking_distance(v, surf)
            
            # Reaction delay component (0.33s typical sensor + hydraulic lag)
            v_ms = v / 3.6
            d_delay = v_ms * 0.33
            d_total = d_pure + d_delay

            iso_range = ISO_CORRIDORS_50.get(surf, (None, None)) if v == 50 else (None, None)
            iso_str = f"{iso_range[0]:.0f}–{iso_range[1]:.0f}m" if iso_range[0] is not None else "-"

            print(f"{surf:<8} {v:<6} {d_kin:<12.1f} {d_pure:<16.1f} {d_total:<18.1f} {iso_str:<15}")

            records.append({
                "surface": surf,
                "mu_nominal": mu,
                "speed_kmh": v,
                "kinematic_m": round(d_kin, 2),
                "pacejka_pure_m": round(d_pure, 2),
                "delay_0_33s_m": round(d_delay, 2),
                "pacejka_total_m": round(d_total, 2),
                "kinematic_underestimation_pct": round(((d_total - d_kin) / d_total) * 100, 1),
                "iso_range": iso_str
            })

    with open("verification/stopping_distance_verified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with open("verification/stopping_distance_summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print("\nSaved verification/stopping_distance_verified.csv and verification/stopping_distance_summary.json")

if __name__ == "__main__":
    main()
