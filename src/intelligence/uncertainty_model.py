import os
import json
from typing import Dict, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.evaluation.metrics import trajectory_metrics


# ============================================================
# Config
# ============================================================

TEST_DATA_PATH = "results/intelligence/datasets/trajectory_prediction/test.npz"
MODEL_PATH = "models/trajectory_predictor.pt"
OUTPUT_DIR = "results/intelligence"

UNCERTAINTY_RESULTS_PATH = os.path.join(OUTPUT_DIR, "uncertainty_results.json")
UNCERTAINTY_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "uncertainty_predictions.npz")
UNCERTAINTY_PLOT_PATH = os.path.join(OUTPUT_DIR, "uncertainty_cone.png")

NUM_SAMPLES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"].astype(np.float32), data["Y"].astype(np.float32)


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def denormalize_targets(Y_rel_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Y_rel_norm * std) + mean


def convert_relative_to_absolute(X_raw: np.ndarray, Y_rel: np.ndarray) -> np.ndarray:
    last_pos = X_raw[:, -1, 0:2]
    return (Y_rel + last_pos[:, None, :]).astype(np.float32)


# ============================================================
# Model
# ============================================================

class LSTMTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        pred_len: int = 5,
        output_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.pred_len = pred_len
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred = self.head(last_hidden)
        pred = pred.view(-1, self.pred_len, self.output_size)
        return pred


# ============================================================
# MC Dropout
# ============================================================

def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout during inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    X_tensor: torch.Tensor,
    num_samples: int = 30
) -> np.ndarray:
    """
    Returns:
    preds_rel_norm shape = [S, N, T, 2]
    """
    preds = []

    for _ in range(num_samples):
        pred = model(X_tensor).cpu().numpy()
        preds.append(pred)

    return np.stack(preds, axis=0)


def compute_uncertainty_statistics(
    preds_abs_samples: np.ndarray,
    y_true_abs: np.ndarray
) -> Dict:
    """
    preds_abs_samples: [S, N, T, 2]
    y_true_abs: [N, T, 2]
    """
    mean_pred = np.mean(preds_abs_samples, axis=0)      # [N, T, 2]
    std_pred = np.std(preds_abs_samples, axis=0)        # [N, T, 2]
    var_pred = np.var(preds_abs_samples, axis=0)        # [N, T, 2]

    metrics = trajectory_metrics(y_true_abs, mean_pred)

    per_sample_uncertainty = np.mean(std_pred, axis=(1, 2)) if std_pred.ndim == 3 else []
    timestep_uncertainty = np.mean(std_pred, axis=(0, 2))   # [T]
    overall_uncertainty = float(np.mean(std_pred))

    return {
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "var_prediction": var_pred,
        "trajectory_metrics": metrics,
        "overall_uncertainty": overall_uncertainty,
        "timestep_uncertainty": timestep_uncertainty,
    }


# ============================================================
# Visualization
# ============================================================

def plot_uncertainty_cone(
    X_raw: np.ndarray,
    y_true_abs: np.ndarray,
    mean_pred_abs: np.ndarray,
    std_pred_abs: np.ndarray,
    save_path: str,
    sample_index: int = 0
) -> None:
    """
    Plot one example trajectory with uncertainty cone.
    """
    ensure_dir(os.path.dirname(save_path))

    past_xy = X_raw[sample_index, :, 0:2]
    true_xy = y_true_abs[sample_index]
    pred_xy = mean_pred_abs[sample_index]
    std_xy = std_pred_abs[sample_index]

    pred_std = np.linalg.norm(std_xy, axis=-1)

    plt.figure(figsize=(8, 6))

    # Past trajectory
    plt.plot(
        past_xy[:, 0], past_xy[:, 1],
        marker="o",
        label="Past trajectory"
    )

    # Ground truth future
    plt.plot(
        true_xy[:, 0], true_xy[:, 1],
        marker="o",
        label="Ground truth future"
    )

    # Predicted future
    plt.plot(
        pred_xy[:, 0], pred_xy[:, 1],
        marker="o",
        label="Predicted mean future"
    )

    # Uncertainty circles
    for i in range(len(pred_xy)):
        circle = plt.Circle(
            (pred_xy[i, 0], pred_xy[i, 1]),
            radius=max(1.0, pred_std[i]),
            fill=False,
            alpha=0.5
        )
        plt.gca().add_patch(circle)

    plt.gca().invert_yaxis()
    plt.title("Trajectory Prediction with Uncertainty Cone")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] Using device: {DEVICE}")

    # Load saved checkpoint
    print("[INFO] Loading trained trajectory predictor...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    config = checkpoint["config"]
    feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
    target_mean = np.array(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.array(checkpoint["target_std"], dtype=np.float32)

    model = LSTMTrajectoryPredictor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        pred_len=config["pred_len"],
        output_size=2,
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test data
    print("[INFO] Loading test dataset...")
    X_raw, y_true_abs = load_npz(TEST_DATA_PATH)
    X_norm = normalize_features(X_raw, feature_mean, feature_std)

    X_tensor = torch.from_numpy(X_norm).float().to(DEVICE)

    print("[INFO] Running Monte Carlo dropout inference...")
    enable_mc_dropout(model)
    preds_rel_norm_samples = mc_dropout_predict(
        model=model,
        X_tensor=X_tensor,
        num_samples=NUM_SAMPLES
    )

    print("[INFO] Converting predictions back to absolute coordinates...")
    preds_abs_samples = []
    for s in range(NUM_SAMPLES):
        pred_rel = denormalize_targets(preds_rel_norm_samples[s], target_mean, target_std)
        pred_abs = convert_relative_to_absolute(X_raw, pred_rel)
        preds_abs_samples.append(pred_abs)

    preds_abs_samples = np.stack(preds_abs_samples, axis=0)   # [S, N, T, 2]

    print("[INFO] Computing uncertainty statistics...")
    stats = compute_uncertainty_statistics(preds_abs_samples, y_true_abs)

    mean_pred = stats["mean_prediction"]
    std_pred = stats["std_prediction"]
    var_pred = stats["var_prediction"]

    results = {
        "num_mc_samples": NUM_SAMPLES,
        "trajectory_metrics_mean_prediction": {
            "ADE": float(stats["trajectory_metrics"]["ADE"]),
            "FDE": float(stats["trajectory_metrics"]["FDE"]),
            "RMSE": float(stats["trajectory_metrics"]["RMSE"]),
        },
        "overall_uncertainty": float(stats["overall_uncertainty"]),
        "timestep_uncertainty": [float(x) for x in stats["timestep_uncertainty"]],
        "model_config": config,
    }

    print("[INFO] Saving uncertainty outputs...")
    save_json(results, UNCERTAINTY_RESULTS_PATH)

    np.savez_compressed(
        UNCERTAINTY_PREDICTIONS_PATH,
        x_test_raw=X_raw,
        y_true_abs=y_true_abs,
        preds_abs_samples=preds_abs_samples,
        mean_pred_abs=mean_pred,
        std_pred_abs=std_pred,
        var_pred_abs=var_pred,
    )

    print("[INFO] Creating uncertainty cone plot...")
    plot_uncertainty_cone(
        X_raw=X_raw,
        y_true_abs=y_true_abs,
        mean_pred_abs=mean_pred,
        std_pred_abs=std_pred,
        save_path=UNCERTAINTY_PLOT_PATH,
        sample_index=0
    )

    print("\n[INFO] Uncertainty modeling completed.")
    print(f"[INFO] Saved results to: {UNCERTAINTY_RESULTS_PATH}")
    print(f"[INFO] Saved predictions to: {UNCERTAINTY_PREDICTIONS_PATH}")
    print(f"[INFO] Saved plot to: {UNCERTAINTY_PLOT_PATH}")

    print("\n[RESULTS]")
    print(f"ADE (mean prediction): {results['trajectory_metrics_mean_prediction']['ADE']:.4f}")
    print(f"FDE (mean prediction): {results['trajectory_metrics_mean_prediction']['FDE']:.4f}")
    print(f"RMSE (mean prediction): {results['trajectory_metrics_mean_prediction']['RMSE']:.4f}")
    print(f"Overall uncertainty: {results['overall_uncertainty']:.4f}")
    print("Timestep uncertainty:", [round(x, 4) for x in results["timestep_uncertainty"]])


if __name__ == "__main__":
    main()