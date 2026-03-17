import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, List[float], List[List[float]]]


# ============================================================
# Helpers
# ============================================================

def _to_numpy(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


# ============================================================
# Trajectory Prediction Metrics
# ============================================================

def average_displacement_error(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    ADE = mean Euclidean distance over all predicted timesteps and samples.

    Expected shape:
    y_true: [N, T, 2]
    y_pred: [N, T, 2]
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    errors = np.linalg.norm(y_true - y_pred, axis=-1)   # [N, T]
    return float(np.mean(errors))


def final_displacement_error(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    FDE = mean Euclidean distance at final predicted timestep.

    Expected shape:
    y_true: [N, T, 2]
    y_pred: [N, T, 2]
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    final_errors = np.linalg.norm(y_true[:, -1, :] - y_pred[:, -1, :], axis=-1)  # [N]
    return float(np.mean(final_errors))


def trajectory_rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    RMSE over all coordinates.

    Expected shape:
    y_true: [N, T, 2]
    y_pred: [N, T, 2]
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def trajectory_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> Dict[str, float]:
    """
    Returns all major trajectory metrics together.
    """
    return {
        "ADE": average_displacement_error(y_true, y_pred),
        "FDE": final_displacement_error(y_true, y_pred),
        "RMSE": trajectory_rmse(y_true, y_pred),
    }


# ============================================================
# Constant Velocity Baseline Support
# ============================================================

def constant_velocity_predict(
    x_input: ArrayLike,
    pred_len: int
) -> np.ndarray:
    """
    Constant velocity baseline predictor.

    Input:
    x_input: [N, input_len, F]
        Feature layout expected from your dataset_generator:
        [cx, cy, vx, vy, speed, distance_to_ego, width, height]

    Output:
    y_pred: [N, pred_len, 2]
    """
    x_input = _to_numpy(x_input)

    if x_input.ndim != 3:
        raise ValueError(f"x_input must be 3D [N, input_len, F], got {x_input.shape}")

    if x_input.shape[-1] < 4:
        raise ValueError("x_input must contain at least cx, cy, vx, vy as first 4 features")

    last_state = x_input[:, -1, :]  # [N, F]
    cx = last_state[:, 0]
    cy = last_state[:, 1]
    vx = last_state[:, 2]
    vy = last_state[:, 3]

    preds = []
    for step in range(1, pred_len + 1):
        next_x = cx + vx * step
        next_y = cy + vy * step
        preds.append(np.stack([next_x, next_y], axis=1))  # [N, 2]

    return np.stack(preds, axis=1)  # [N, pred_len, 2]


def evaluate_constant_velocity_baseline(
    x_input: ArrayLike,
    y_true: ArrayLike
) -> Dict[str, float]:
    """
    Predicts with constant velocity baseline and evaluates against ground truth.
    """
    x_input = _to_numpy(x_input)
    y_true = _to_numpy(y_true)

    pred_len = y_true.shape[1]
    y_pred = constant_velocity_predict(x_input, pred_len=pred_len)
    return trajectory_metrics(y_true, y_pred)


# ============================================================
# Classification Metrics
# ============================================================

def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Returns confusion matrix of shape [C, C]
    rows = true classes
    cols = predicted classes
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of elements")

    if num_classes is None:
        max_label = int(max(np.max(y_true), np.max(y_pred))) if len(y_true) > 0 else 0
        num_classes = max_label + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t < 0 or p < 0:
            continue
        if t >= num_classes or p >= num_classes:
            continue
        cm[t, p] += 1

    return cm


def classification_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_recall_f1_from_confusion(cm: np.ndarray) -> Dict[str, Union[float, List[float]]]:
    """
    Computes per-class and macro metrics from confusion matrix.
    """
    num_classes = cm.shape[0]

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        precision = _safe_divide(tp, tp + fp, default=0.0)
        recall = _safe_divide(tp, tp + fn, default=0.0)
        f1 = _safe_divide(2 * precision * recall, precision + recall, default=0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    macro_precision = float(np.mean(precisions)) if num_classes > 0 else 0.0
    macro_recall = float(np.mean(recalls)) if num_classes > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if num_classes > 0 else 0.0

    total_support = sum(supports)
    weighted_precision = _safe_divide(
        float(sum(p * s for p, s in zip(precisions, supports))),
        total_support,
        default=0.0
    )
    weighted_recall = _safe_divide(
        float(sum(r * s for r, s in zip(recalls, supports))),
        total_support,
        default=0.0
    )
    weighted_f1 = _safe_divide(
        float(sum(f * s for f, s in zip(f1s, supports))),
        total_support,
        default=0.0
    )

    return {
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "support": supports,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: Optional[int] = None
) -> Dict[str, Union[float, List[float], np.ndarray]]:
    """
    General classification metrics for behavior classifier.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    prf = precision_recall_f1_from_confusion(cm)
    acc = classification_accuracy(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_precision": prf["macro_precision"],
        "macro_recall": prf["macro_recall"],
        "macro_f1": prf["macro_f1"],
        "weighted_precision": prf["weighted_precision"],
        "weighted_recall": prf["weighted_recall"],
        "weighted_f1": prf["weighted_f1"],
        "per_class_precision": prf["per_class_precision"],
        "per_class_recall": prf["per_class_recall"],
        "per_class_f1": prf["per_class_f1"],
        "support": prf["support"],
        "confusion_matrix": cm,
    }


# ============================================================
# Binary / AEB Metrics
# ============================================================

def binary_confusion_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> Dict[str, int]:
    """
    For AEB trigger or collision event prediction.
    Labels are assumed binary: 0 or 1
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have same length")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def binary_precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> Dict[str, float]:
    counts = binary_confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

    precision = _safe_divide(tp, tp + fp, default=0.0)
    recall = _safe_divide(tp, tp + fn, default=0.0)
    f1 = _safe_divide(2 * precision * recall, precision + recall, default=0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def false_braking_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    Fraction of non-dangerous cases where brake was incorrectly triggered.
    Interpreting y_true=1 as 'brake should happen' or 'danger present'
    """
    counts = binary_confusion_counts(y_true, y_pred)
    fp, tn = counts["FP"], counts["TN"]
    return _safe_divide(fp, fp + tn, default=0.0)


def collision_miss_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    Fraction of dangerous cases missed by braking logic.
    """
    counts = binary_confusion_counts(y_true, y_pred)
    fn, tp = counts["FN"], counts["TP"]
    return _safe_divide(fn, fn + tp, default=0.0)


def collision_detection_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    Same as recall for positive/dangerous class.
    """
    counts = binary_confusion_counts(y_true, y_pred)
    tp, fn = counts["TP"], counts["FN"]
    return _safe_divide(tp, tp + fn, default=0.0)


def average_ttc_at_trigger(
    ttc_values: ArrayLike,
    brake_predictions: ArrayLike
) -> float:
    """
    Average TTC for frames where brake was triggered.
    Ignores invalid TTC values <= 0
    """
    ttc_values = _to_numpy(ttc_values).reshape(-1)
    brake_predictions = np.asarray(brake_predictions, dtype=np.int64).reshape(-1)

    mask = (brake_predictions == 1) & (ttc_values > 0)
    if not np.any(mask):
        return -1.0

    return float(np.mean(ttc_values[mask]))


def average_reaction_time(
    ttc_values: ArrayLike,
    brake_predictions: ArrayLike
) -> float:
    """
    Proxy for reaction time using TTC at trigger.
    Higher TTC at trigger means earlier reaction.
    """
    return average_ttc_at_trigger(ttc_values, brake_predictions)


def stopping_distance_proxy(
    distances: ArrayLike,
    brake_predictions: ArrayLike
) -> float:
    """
    Average distance when brake is triggered.
    This is a proxy, not a physical braking distance model.
    """
    distances = _to_numpy(distances).reshape(-1)
    brake_predictions = np.asarray(brake_predictions, dtype=np.int64).reshape(-1)

    mask = brake_predictions == 1
    if not np.any(mask):
        return -1.0

    return float(np.mean(distances[mask]))


def aeb_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ttc_values: Optional[ArrayLike] = None,
    distances: Optional[ArrayLike] = None
) -> Dict[str, float]:
    """
    AEB evaluation metrics.

    Interprets:
    y_true = ground truth danger / should-brake label
    y_pred = predicted brake decision
    """
    counts = binary_confusion_counts(y_true, y_pred)
    prf = binary_precision_recall_f1(y_true, y_pred)
    acc = classification_accuracy(y_true, y_pred)

    results = {
        "accuracy": acc,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "false_braking_rate": false_braking_rate(y_true, y_pred),
        "collision_miss_rate": collision_miss_rate(y_true, y_pred),
        "collision_detection_rate": collision_detection_rate(y_true, y_pred),
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
    }

    if ttc_values is not None:
        results["average_ttc_at_trigger"] = average_ttc_at_trigger(ttc_values, y_pred)
        results["average_reaction_time_proxy"] = average_reaction_time(ttc_values, y_pred)

    if distances is not None:
        results["average_trigger_distance"] = stopping_distance_proxy(distances, y_pred)

    return results


# ============================================================
# Comparison Utility
# ============================================================

def compare_metric_tables(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Creates side-by-side comparison for benchmarking.
    """
    keys = sorted(set(baseline_metrics.keys()).intersection(set(model_metrics.keys())))
    comparison = {}

    for key in keys:
        base_val = baseline_metrics[key]
        model_val = model_metrics[key]

        if isinstance(base_val, (int, float)) and isinstance(model_val, (int, float)):
            comparison[key] = {
                "baseline": float(base_val),
                "model": float(model_val),
                "delta": float(model_val - base_val),
            }

    return comparison


# ============================================================
# Example self-test
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # Trajectory test
    # -----------------------------
    y_true = np.array([
        [[10, 10], [11, 11], [12, 12]],
        [[20, 20], [21, 21], [22, 22]]
    ], dtype=np.float32)

    y_pred = np.array([
        [[10, 10], [12, 11], [13, 12]],
        [[20, 19], [21, 20], [23, 22]]
    ], dtype=np.float32)

    print("Trajectory Metrics:")
    print(trajectory_metrics(y_true, y_pred))

    # -----------------------------
    # Classification test
    # -----------------------------
    cls_true = np.array([0, 1, 2, 1, 0, 2, 2])
    cls_pred = np.array([0, 1, 1, 1, 0, 2, 0])

    print("\nClassification Metrics:")
    cls_results = classification_metrics(cls_true, cls_pred, num_classes=3)
    for k, v in cls_results.items():
        print(f"{k}: {v}")

    # -----------------------------
    # AEB test
    # -----------------------------
    aeb_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    aeb_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    ttc = np.array([5.0, 3.5, 1.2, 0.9, 4.5, 1.8, 6.0, 0.7])
    dist = np.array([100, 80, 25, 20, 110, 35, 120, 18])

    print("\nAEB Metrics:")
    print(aeb_metrics(aeb_true, aeb_pred, ttc_values=ttc, distances=dist))