import os
import json
import joblib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import classification_metrics


# ============================================================
# Config
# ============================================================

DATA_PATH = "results/intelligence/datasets/behavior_classification/behavior_labels.csv"
OUTPUT_DIR = "results/intelligence"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "behavior_classifier.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "behavior_classifier_results.json")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "behavior_classifier_predictions.csv")

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

LABEL_MAP = {
    0: "standing",
    1: "walking",
    2: "running",
    3: "crossing",
    4: "approaching_road",
}

FEATURE_COLUMNS = [
    "mean_cx",
    "mean_cy",
    "mean_speed",
    "max_speed",
    "std_speed",
    "mean_vx",
    "mean_vy",
    "total_dx",
    "total_dy",
    "displacement",
    "mean_distance_to_ego",
]


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_behavior_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Behavior dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + ["behavior_id", "behavior_label"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in behavior dataset: {missing}")

    return df


def serialize_confusion_matrix(cm: np.ndarray) -> List[List[int]]:
    return cm.astype(int).tolist()


def summarize_labels(y: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(y, return_counts=True)
    out = {}
    for u, c in zip(unique, counts):
        name = LABEL_MAP.get(int(u), f"class_{int(u)}")
        out[name] = int(c)
    return out


# ============================================================
# Data Prep
# ============================================================

def prepare_splits(
    df: pd.DataFrame,
    random_state: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train, val, test.
    """
    # First split off test
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=df["behavior_id"] if df["behavior_id"].nunique() > 1 else None,
    )

    # Then split train/val
    val_ratio_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df["behavior_id"] if train_val_df["behavior_id"].nunique() > 1 else None,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns].copy().values.astype(np.float32)
    y = df["behavior_id"].values.astype(np.int64)
    return X, y


# ============================================================
# Training
# ============================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_split(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str
) -> Dict:
    y_pred = model.predict(X)

    num_classes = len(LABEL_MAP)
    metrics = classification_metrics(y, y_pred, num_classes=num_classes)

    print(f"\n{split_name.upper()} METRICS")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {metrics['weighted_f1']:.4f}")

    cm = metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print(cm)

    result = {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "per_class_precision": [float(x) for x in metrics["per_class_precision"]],
        "per_class_recall": [float(x) for x in metrics["per_class_recall"]],
        "per_class_f1": [float(x) for x in metrics["per_class_f1"]],
        "support": [int(x) for x in metrics["support"]],
        "confusion_matrix": serialize_confusion_matrix(cm),
    }

    return result, y_pred


def save_predictions_csv(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str
) -> None:
    out_df = df.copy()
    out_df["y_true"] = y_true
    out_df["y_pred"] = y_pred
    out_df["y_true_label"] = [LABEL_MAP.get(int(x), str(int(x))) for x in y_true]
    out_df["y_pred_label"] = [LABEL_MAP.get(int(x), str(int(x))) for x in y_pred]
    out_df.to_csv(path, index=False)


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    print("[INFO] Loading behavior dataset...")
    df = load_behavior_data(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows from {DATA_PATH}")

    print("[INFO] Preparing train/val/test splits...")
    train_df, val_df, test_df = prepare_splits(df)

    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Val rows:   {len(val_df)}")
    print(f"[INFO] Test rows:  {len(test_df)}")

    X_train, y_train = prepare_features_and_labels(train_df, FEATURE_COLUMNS)
    X_val, y_val = prepare_features_and_labels(val_df, FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_labels(test_df, FEATURE_COLUMNS)

    print("[INFO] Handling missing values...")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    print("[INFO] Training Random Forest behavior classifier...")
    model = train_model(X_train, y_train)

    print("[INFO] Evaluating on train/val/test...")
    train_results, train_pred = evaluate_split(model, X_train, y_train, "train")
    val_results, val_pred = evaluate_split(model, X_val, y_val, "val")
    test_results, test_pred = evaluate_split(model, X_test, y_test, "test")

    feature_importance = {
        FEATURE_COLUMNS[i]: float(model.feature_importances_[i])
        for i in range(len(FEATURE_COLUMNS))
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    results = {
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_label_distribution": summarize_labels(y_train),
            "val_label_distribution": summarize_labels(y_val),
            "test_label_distribution": summarize_labels(y_test),
        },
        "feature_columns": FEATURE_COLUMNS,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": SEED,
        },
        "feature_importance": feature_importance,
        "train_metrics": train_results,
        "val_metrics": val_results,
        "test_metrics": test_results,
    }

    print("[INFO] Saving model...")
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_columns": FEATURE_COLUMNS,
            "label_map": LABEL_MAP,
        },
        MODEL_PATH
    )

    print("[INFO] Saving results...")
    save_json(results, RESULTS_PATH)

    print("[INFO] Saving test predictions...")
    save_predictions_csv(test_df, y_test, test_pred, PREDICTIONS_PATH)

    print("\n[INFO] Behavior classifier training completed.")
    print(f"[INFO] Saved model to: {MODEL_PATH}")
    print(f"[INFO] Saved results to: {RESULTS_PATH}")
    print(f"[INFO] Saved predictions to: {PREDICTIONS_PATH}")

    print("\n[RESULTS] TEST SUMMARY")
    print(f"Accuracy:    {test_results['accuracy']:.4f}")
    print(f"Macro F1:    {test_results['macro_f1']:.4f}")
    print(f"Weighted F1: {test_results['weighted_f1']:.4f}")

    print("\n[TOP FEATURES]")
    for k, v in list(feature_importance.items())[:8]:
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()