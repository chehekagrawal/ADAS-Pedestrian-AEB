"""
Verification Script: Head Injury Criterion (HIC) & Biomechanical Injury
Evaluates the biomechanical impact models in src/safety_analysis/impact_model.py
and verifies the relationship between impact velocity, deceleration pulse,
HIC value, and AIS severity classification.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safety_analysis.impact_model import ImpactAnalyzer, AIS_THRESHOLDS

def compute_closed_form_hic(v_kmh, k=35000.0, m=4.5):
    """
    Closed-form half-sine acceleration pulse HIC formulation:
    t_c = pi * sqrt(m / k)
    a_peak = pi * v / (2 * t_c)
    HIC = t_c * (a_peak / g)^2.5
    """
    if v_kmh <= 0.5:
        return 0.0, 0.0, 0.0
    v = v_kmh / 3.6
    tc = np.pi * np.sqrt(m / k)
    ap = np.pi * v / (2 * tc)
    g = 9.81
    hic = tc * (ap / g) ** 2.5
    return float(hic), float(tc), float(ap / g)

def main():
    os.makedirs("verification", exist_ok=True)
    analyzer = ImpactAnalyzer()
    
    speeds = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    records = []

    print("=== HIC Biomechanics Verification Sweep ===")
    print(f"{'Speed (km/h)':<12} {'HIC (Model)':<12} {'Zone':<12} {'AIS':<5} {'HIC (k=35kN)':<14} {'Orig Paper HIC':<14}")
    print("-" * 72)

    orig_paper_hic = {20: 45, 30: 280, 40: 760, 50: 2200}

    for v in speeds:
        # 1. Multi-zone model from impact_model.py
        hic_model = analyzer.compute_hic(v, ped_height_m=1.75)
        wad, zone = analyzer.compute_wrap_around_distance(1.75, v)
        sev = analyzer.injury_severity(hic_model)

        # 2. Closed-form half-sine model (k=35 kN/m)
        hic_35k, tc, a_peak_g = compute_closed_form_hic(v, k=35000.0)

        paper_val = orig_paper_hic.get(v, "-")
        print(f"{v:<12} {hic_model:<12.1f} {zone:<12} {sev['ais_level']:<5} {hic_35k:<14.1f} {str(paper_val):<14}")

        records.append({
            "speed_kmh": v,
            "hic_geometry_model": round(hic_model, 1),
            "impact_zone": zone,
            "wad_m": round(wad, 2),
            "ais_level": sev["ais_level"],
            "injury_description": sev["description"],
            "hic_closed_form_35k": round(hic_35k, 1),
            "contact_time_ms": round(tc * 1000, 2),
            "peak_accel_g": round(a_peak_g, 1),
            "original_paper_claim": paper_val
        })

    with open("verification/hic_biomechanics_verified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with open("verification/hic_biomechanics_verified.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print("\nSaved verification/hic_biomechanics_verified.csv and .json")

if __name__ == "__main__":
    main()
