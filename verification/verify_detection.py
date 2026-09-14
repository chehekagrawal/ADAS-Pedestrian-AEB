"""
Verification Script: Detection Metrics
Parses the true Ultralytics training log from results/training_multiclass/results.csv
and outputs exact numerical verification artifacts using standard library csv and json.
"""

import os
import json
import csv

def verify_detection():
    csv_path = "results/training_multiclass/results.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from headers
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    if not rows:
        raise ValueError("Empty results.csv")

    final_row = rows[-1]
    
    verified = {
        "final_epoch": int(final_row["epoch"]),
        "time_seconds": float(final_row["time"]),
        "train_box_loss": float(final_row["train/box_loss"]),
        "train_cls_loss": float(final_row["train/cls_loss"]),
        "train_dfl_loss": float(final_row["train/dfl_loss"]),
        "val_box_loss": float(final_row["val/box_loss"]),
        "val_cls_loss": float(final_row["val/cls_loss"]),
        "val_dfl_loss": float(final_row["val/dfl_loss"]),
        "precision_B": float(final_row["metrics/precision(B)"]),
        "recall_B": float(final_row["metrics/recall(B)"]),
        "mAP50_B": float(final_row["metrics/mAP50(B)"]),
        "mAP50_95_B": float(final_row["metrics/mAP50-95(B)"]),
        "lr_pg0": float(final_row["lr/pg0"]),
        "lr_pg1": float(final_row["lr/pg1"]),
        "lr_pg2": float(final_row["lr/pg2"])
    }

    print("=== Verified Detection Metrics (Epoch 30 Final) ===")
    for k, v in verified.items():
        print(f"  {k:20s}: {v}")

    os.makedirs("verification", exist_ok=True)
    with open("verification/detection_metrics_verified.json", "w") as f:
        json.dump(verified, f, indent=4)

    # Save summary table
    summary_rows = [
        {
            "Metric": "Precision (Box)",
            "Verified Value": f"{verified['precision_B']*100:.1f}%",
            "Raw Decimal": verified["precision_B"],
            "Original Paper Claim": "78.5%",
            "Discrepancy": f"{verified['precision_B']*100 - 78.5:.1f}%"
        },
        {
            "Metric": "Recall (Box)",
            "Verified Value": f"{verified['recall_B']*100:.1f}%",
            "Raw Decimal": verified["recall_B"],
            "Original Paper Claim": "75.9%",
            "Discrepancy": f"{verified['recall_B']*100 - 75.9:.1f}%"
        },
        {
            "Metric": "mAP@0.5 (Box)",
            "Verified Value": f"{verified['mAP50_B']*100:.1f}%",
            "Raw Decimal": verified["mAP50_B"],
            "Original Paper Claim": "77.1%",
            "Discrepancy": f"{verified['mAP50_B']*100 - 77.1:.1f}%"
        },
        {
            "Metric": "mAP@0.5:0.95 (Box)",
            "Verified Value": f"{verified['mAP50_95_B']*100:.1f}%",
            "Raw Decimal": verified["mAP50_95_B"],
            "Original Paper Claim": "44.8%",
            "Discrepancy": f"{verified['mAP50_95_B']*100 - 44.8:.1f}%"
        }
    ]

    with open("verification/detection_metrics_verified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Verified Value", "Raw Decimal", "Original Paper Claim", "Discrepancy"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved verification/detection_metrics_verified.json and verification/detection_metrics_verified.csv")

if __name__ == "__main__":
    verify_detection()
