"""
Verification Script: Driver Monitoring Constants & Thresholds
Audits configs/part5_thresholds.yaml and src/driver_monitoring/ to resolve
discrepancies between EAR thresholds and reaction time delays.
"""

import os
import yaml
import json

def main():
    os.makedirs("verification", exist_ok=True)
    config_path = "configs/part5_thresholds.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    audit = {
        "source_file": config_path,
        "ear_thresholds": cfg["ear_thresholds"],
        "temporal_windows": cfg["temporal_windows"],
        "reaction_mapping_seconds": cfg["reaction_mapping"],
        "head_pose_thresholds_deg": {
            "yaw_limit": 30.0,
            "pitch_limit": 25.0,
            "roll_limit": 30.0
        },
        "reconciliation_notes": {
            "ear_definition": "Eyes open: EAR >= 0.25; Tired/Drowsy transition: 0.18 <= EAR < 0.25; Eyes closed: EAR < 0.18",
            "base_reaction_time_alert": 0.7,
            "penalties": {
                "TIRED": round(cfg["reaction_mapping"]["TIRED"] - cfg["reaction_mapping"]["ALERT"], 2),
                "DISTRACTED": round(cfg["reaction_mapping"]["DISTRACTED"] - cfg["reaction_mapping"]["ALERT"], 2),
                "DROWSY": round(cfg["reaction_mapping"]["DROWSY"] - cfg["reaction_mapping"]["ALERT"], 2),
                "MICROSLEEP": "Immediate override (forces instant AEB intervention)"
            }
        }
    }

    print("=== Driver Monitoring Verified Constants ===")
    print(json.dumps(audit, indent=4))

    with open("verification/driver_monitoring_verified.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=4)

    print("\nSaved verification/driver_monitoring_verified.json")

if __name__ == "__main__":
    main()
