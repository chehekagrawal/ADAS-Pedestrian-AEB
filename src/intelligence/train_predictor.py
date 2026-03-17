import os
import json
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.evaluation.metrics import trajectory_metrics, constant_velocity_predict


# ============================================================
# Config
# ============================================================

DATA_DIR = "results/intelligence/datasets/trajectory_prediction"
OUTPUT_DIR = "results/intelligence"
MODEL_DIR = "models"

TRAIN_PATH = os.path.join(DATA_DIR, "train.npz")
VAL_PATH = os.path.join(DATA_DIR, "val.npz")
TEST_PATH = os.path.join(DATA_DIR, "test.npz")

MODEL_PATH = os.path.join(MODEL_DIR, "trajectory_predictor.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "trajectory_training_history.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "trajectory_predictor_results.json")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "trajectory_predictor_test_predictions.npz")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"].astype(np.float32), data["Y"].astype(np.float32)


def compute_feature_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def convert_targets_to_relative(X_raw: np.ndarray, Y_abs: np.ndarray) -> np.ndarray:
    """
    Convert absolute future coordinates into offsets relative to the
    last observed input position.

    X_raw: [N, input_len, F]
    Y_abs: [N, pred_len, 2]
    returns: [N, pred_len, 2]
    """
    last_pos = X_raw[:, -1, 0:2]          # [N, 2]
    Y_rel = Y_abs - last_pos[:, None, :]  # broadcast to [N, pred_len, 2]
    return Y_rel.astype(np.float32)


def convert_relative_to_absolute(X_raw: np.ndarray, Y_rel: np.ndarray) -> np.ndarray:
    """
    Convert predicted relative offsets back to absolute coordinates.
    """
    last_pos = X_raw[:, -1, 0:2]          # [N, 2]
    Y_abs = Y_rel + last_pos[:, None, :]
    return Y_abs.astype(np.float32)


def compute_target_stats(Y_rel_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = Y_rel_train.reshape(-1, Y_rel_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_targets(Y_rel: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Y_rel - mean) / std


def denormalize_targets(Y_rel_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Y_rel_norm * std) + mean


# ============================================================
# Dataset
# ============================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


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
        """
        x: [B, input_len, input_size]
        returns: [B, pred_len, 2] relative offsets
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred = self.head(last_hidden)
        pred = pred.view(-1, self.pred_len, self.output_size)
        return pred


# ============================================================
# Training / Eval
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, Y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def predict_all_relative(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> np.ndarray:
    model.eval()
    all_preds = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def evaluate_model_absolute_metrics(
    model: nn.Module,
    loader: DataLoader,
    X_raw: np.ndarray,
    Y_abs_true: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str
) -> Dict[str, float]:
    """
    Model predicts normalized relative offsets.
    Convert:
    pred_norm -> pred_rel -> pred_abs
    then compare against absolute ground truth.
    """
    pred_rel_norm = predict_all_relative(model, loader, device)
    pred_rel = denormalize_targets(pred_rel_norm, target_mean, target_std)
    pred_abs = convert_relative_to_absolute(X_raw, pred_rel)

    return trajectory_metrics(Y_abs_true, pred_abs)


def evaluate_baseline_split(X_raw: np.ndarray, Y_abs: np.ndarray) -> Dict[str, float]:
    pred_len = Y_abs.shape[1]
    Y_pred = constant_velocity_predict(X_raw, pred_len=pred_len)
    return trajectory_metrics(Y_abs, Y_pred)


def compare_results(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in ["ADE", "FDE", "RMSE"]:
        b = float(baseline_metrics[key])
        m = float(model_metrics[key])
        improvement = b - m
        improvement_pct = (improvement / b * 100.0) if abs(b) > 1e-8 else 0.0

        out[key] = {
            "baseline": b,
            "model": m,
            "absolute_improvement": improvement,
            "improvement_percent": improvement_pct,
        }
    return out


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    print(f"[INFO] Using device: {DEVICE}")

    # Load raw datasets
    X_train_raw, Y_train_abs = load_npz(TRAIN_PATH)
    X_val_raw, Y_val_abs = load_npz(VAL_PATH)
    X_test_raw, Y_test_abs = load_npz(TEST_PATH)

    print("[INFO] Loaded datasets")
    print("       Train X:", X_train_raw.shape, "Y:", Y_train_abs.shape)
    print("       Val   X:", X_val_raw.shape, "Y:", Y_val_abs.shape)
    print("       Test  X:", X_test_raw.shape, "Y:", Y_test_abs.shape)

    # Normalize X using train stats
    feat_mean, feat_std = compute_feature_stats(X_train_raw)
    X_train = normalize_features(X_train_raw, feat_mean, feat_std)
    X_val = normalize_features(X_val_raw, feat_mean, feat_std)
    X_test = normalize_features(X_test_raw, feat_mean, feat_std)

    # Convert Y absolute -> relative offsets
    Y_train_rel = convert_targets_to_relative(X_train_raw, Y_train_abs)
    Y_val_rel = convert_targets_to_relative(X_val_raw, Y_val_abs)
    Y_test_rel = convert_targets_to_relative(X_test_raw, Y_test_abs)

    # Normalize relative targets using train stats
    target_mean, target_std = compute_target_stats(Y_train_rel)
    Y_train = normalize_targets(Y_train_rel, target_mean, target_std)
    Y_val = normalize_targets(Y_val_rel, target_mean, target_std)
    Y_test = normalize_targets(Y_test_rel, target_mean, target_std)

    print("[INFO] Prepared relative target prediction")
    print("       Train target rel shape:", Y_train.shape)
    print("       Target mean:", target_mean)
    print("       Target std :", target_std)

    # Datasets / loaders
    train_ds = TrajectoryDataset(X_train, Y_train)
    val_ds = TrajectoryDataset(X_val, Y_val)
    test_ds = TrajectoryDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[-1]
    pred_len = Y_train.shape[1]

    model = LSTMTrajectoryPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=pred_len,
        output_size=2,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    checkpoint = {
        "model_state_dict": None,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "config": {
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "pred_len": pred_len,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "predict_mode": "relative_offsets_normalized",
        }
    }

    print("[INFO] Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint["model_state_dict"] = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"[INFO] Saved best model at epoch {epoch} -> {MODEL_PATH}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    print(f"[INFO] Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")

    # Load best model
    saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])

    # Evaluate baseline on raw absolute coordinates
    baseline_train = evaluate_baseline_split(X_train_raw, Y_train_abs)
    baseline_val = evaluate_baseline_split(X_val_raw, Y_val_abs)
    baseline_test = evaluate_baseline_split(X_test_raw, Y_test_abs)

    # Evaluate model by reconstructing absolute coords
    train_metrics = evaluate_model_absolute_metrics(
        model, train_loader, X_train_raw, Y_train_abs, target_mean, target_std, DEVICE
    )
    val_metrics = evaluate_model_absolute_metrics(
        model, val_loader, X_val_raw, Y_val_abs, target_mean, target_std, DEVICE
    )
    test_metrics = evaluate_model_absolute_metrics(
        model, test_loader, X_test_raw, Y_test_abs, target_mean, target_std, DEVICE
    )

    comparison = {
        "train": compare_results(baseline_train, train_metrics),
        "val": compare_results(baseline_val, val_metrics),
        "test": compare_results(baseline_test, test_metrics),
    }

    results = {
        "device": DEVICE,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "baseline_metrics": {
            "train": baseline_train,
            "val": baseline_val,
            "test": baseline_test,
        },
        "model_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "comparison": comparison,
        "config": checkpoint["config"],
    }

    save_json(history, HISTORY_PATH)
    save_json(results, RESULTS_PATH)

    # Save test predictions in absolute coordinate form
    pred_test_rel_norm = predict_all_relative(model, test_loader, DEVICE)
    pred_test_rel = denormalize_targets(pred_test_rel_norm, target_mean, target_std)
    pred_test_abs = convert_relative_to_absolute(X_test_raw, pred_test_rel)

    np.savez_compressed(
        PREDICTIONS_PATH,
        y_true=Y_test_abs,
        y_pred=pred_test_abs,
        x_test_raw=X_test_raw,
        y_pred_rel=pred_test_rel,
    )

    print("\n[INFO] Training completed.")
    print(f"[INFO] Saved history to: {HISTORY_PATH}")
    print(f"[INFO] Saved results to: {RESULTS_PATH}")
    print(f"[INFO] Saved test predictions to: {PREDICTIONS_PATH}")

    print("\n[RESULTS] Baseline vs LSTM")
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        for metric_name in ["ADE", "FDE", "RMSE"]:
            b = results["baseline_metrics"][split][metric_name]
            m = results["model_metrics"][split][metric_name]
            imp = results["comparison"][split][metric_name]["improvement_percent"]
            print(
                f"  {metric_name}: baseline={b:.4f}, model={m:.4f}, "
                f"improvement={imp:.2f}%"
            )


if __name__ == "__main__":
    main()