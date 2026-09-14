"""
Verification Script: Euro NCAP Test Suite
Executes the Euro NCAP AEB simulation suite (src/ncap_testing/ncap_runner.py)
for both Dry and Wet road conditions across all standard scenarios and speeds.
Outputs CSV matrices and verified JSON records.
"""

import os
import sys
import json
import csv

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncap_testing.ncap_runner import run_full_test_matrix
from src.ncap_testing.ncap_scorer import score_single_test, score_scenario

def export_matrix(results, filename):
    fieldnames = [
        "scenario_code", "vehicle_speed_kmh", "surface",
        "detection_distance_m", "braking_distance_needed_m",
        "available_braking_distance_m", "outcome",
        "impact_speed_kmh", "speed_reduction_pct",
        "hic", "ais_level", "injury_description", "score"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scenario_code": r.scenario_code,
                "vehicle_speed_kmh": r.vehicle_speed_kmh,
                "surface": r.surface,
                "detection_distance_m": round(r.detection_distance_m, 2),
                "braking_distance_needed_m": round(r.braking_distance_needed_m, 2),
                "available_braking_distance_m": round(r.available_braking_distance_m, 2),
                "outcome": r.outcome,
                "impact_speed_kmh": round(r.impact_speed_kmh, 2),
                "speed_reduction_pct": round(r.speed_reduction_pct, 1),
                "hic": round(r.hic, 1),
                "ais_level": r.ais_level,
                "injury_description": r.injury_description,
                "score": round(r.score, 4)
            })

def main():
    os.makedirs("verification", exist_ok=True)
    print("Executing Euro NCAP Full Simulation Suite (Dry Surface)...")
    res_dry = run_full_test_matrix("dry")
    export_matrix(res_dry, "verification/ncap_matrix_dry.csv")

    print("Executing Euro NCAP Full Simulation Suite (Wet Surface)...")
    res_wet = run_full_test_matrix("wet")
    export_matrix(res_wet, "verification/ncap_matrix_wet.csv")

    # Generate summary JSON
    def summarize(res, surf):
        avoided = sum(1 for r in res if r.outcome == "avoided")
        partial = sum(1 for r in res if r.outcome == "partial")
        collision = sum(1 for r in res if r.outcome == "collision")
        avg_score = sum(r.score for r in res) / len(res)
        
        # Breakdown by scenario at standard 50 km/h
        at_50 = {r.scenario_code: {"outcome": r.outcome, "impact_speed": r.impact_speed_kmh, "hic": r.hic}
                 for r in res if r.vehicle_speed_kmh == 50}
        
        return {
            "surface": surf,
            "total_tests": len(res),
            "avoided": avoided,
            "partial": partial,
            "collision": collision,
            "overall_score_pct": round(avg_score * 100, 1),
            "at_50kmh": at_50
        }

    summary = {
        "dry": summarize(res_dry, "dry"),
        "wet": summarize(res_wet, "wet")
    }

    with open("verification/ncap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nEuro NCAP verification outputs generated:")
    print("  - verification/ncap_matrix_dry.csv")
    print("  - verification/ncap_matrix_wet.csv")
    print("  - verification/ncap_summary.json")

if __name__ == "__main__":
    main()
