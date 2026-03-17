import os
import json
import numpy as np

from src.evaluation.metrics import (
    trajectory_metrics,
    constant_velocity_predict
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = "results/intelligence/datasets/trajectory_prediction"
OUTPUT_DIR = "results/intelligence"


# ============================================================
# Utility
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(split):
    path = os.path.join(DATA_DIR, f"{split}.npz")
    data = np.load(path)

    X = data["X"]
    Y = data["Y"]

    print(f"{split.upper()} DATA")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print()

    return X, Y


# ============================================================
# Baseline Evaluation
# ============================================================

def evaluate_split(split):

    X, Y = load_dataset(split)

    pred_len = Y.shape[1]

    Y_pred = constant_velocity_predict(X, pred_len)

    metrics = trajectory_metrics(Y, Y_pred)

    print(f"{split.upper()} METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print()

    return metrics, Y_pred


# ============================================================
# Main
# ============================================================

def main():

    ensure_dir(OUTPUT_DIR)

    results = {}

    predictions = {}

    for split in ["train", "val", "test"]:

        metrics, preds = evaluate_split(split)

        results[split] = metrics

        predictions[split] = preds.tolist()

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "baseline_results.json")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Saved metrics to:", metrics_path)

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "baseline_predictions.json")

    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    print("Saved predictions to:", pred_path)


if __name__ == "__main__":
    main()