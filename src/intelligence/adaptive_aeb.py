import os
import json
import joblib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import aeb_metrics


# ============================================================
# Config
# ============================================================

AEB_DATA_PATH = "results/intelligence/datasets/aeb_learning/aeb_features.csv"
BEHAVIOR_MODEL_PATH = "models/behavior_classifier.pkl"
OUTPUT_DIR = "results/intelligence"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "adaptive_aeb_model.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "adaptive_aeb_results.json")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "adaptive_aeb_predictions.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "braking_decisions.png")

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Conservative fallback label when no positive collision labels exist
RISK_TTC_THRESHOLD = 3.0
RISK_PROB_THRESHOLD = 0.1
RISK_DISTANCE_THRESHOLD = 300.00

BEHAVIOR_LABEL_MAP = {
    0: "standing",
    1: "walking",
    2: "running",
    3: "crossing",
    4: "approaching_road",
}


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def safe_bool_int(x) -> int:
    if pd.isna(x):
        return 0
    return int(bool(x))


def summarize_labels(y: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def serialize_feature_importance(feature_names: List[str], importances: np.ndarray) -> Dict[str, float]:
    mapping = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
    return dict(sorted(mapping.items(), key=lambda kv: kv[1], reverse=True))


# ============================================================
# Label construction
# ============================================================

def build_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the learning target for adaptive AEB.

    Priority:
    1. Use collision_detected if it contains positive labels
    2. Else use AEB_triggered if it contains positive labels
    3. Else derive a proxy risk label using TTC, collision probability, distance
    """
    df = df.copy()

    if "collision_detected" in df.columns:
        df["collision_detected"] = df["collision_detected"].apply(safe_bool_int)
    else:
        df["collision_detected"] = 0

    if "AEB_triggered" in df.columns:
        df["AEB_triggered"] = df["AEB_triggered"].apply(safe_bool_int)
    else:
        df["AEB_triggered"] = 0

    if df["collision_detected"].sum() > 0:
        df["risk_label"] = df["collision_detected"].astype(int)
        df["risk_label_source"] = "collision_detected"
        return df

    if df["AEB_triggered"].sum() > 0:
        df["risk_label"] = df["AEB_triggered"].astype(int)
        df["risk_label_source"] = "AEB_triggered"
        return df

    # Derived proxy risk label
    ttc_cond = (df["ttc"] > 0) & (df["ttc"] < RISK_TTC_THRESHOLD)
    prob_cond = df["collision_probability"] >= RISK_PROB_THRESHOLD
    dist_cond = df["distance"] < RISK_DISTANCE_THRESHOLD

    df["risk_label"] = ((ttc_cond & dist_cond) | prob_cond).astype(int)
    df["risk_label_source"] = "derived_proxy"
    return df


# ============================================================
# Behavior feature enrichment
# ============================================================

def attach_behavior_predictions(df: pd.DataFrame, behavior_bundle: Dict) -> pd.DataFrame:
    """
    Use the trained behavior classifier to add behavior_id_pred as an input feature.
    """
    df = df.copy()

    model = behavior_bundle["model"]
    imputer = behavior_bundle["imputer"]
    feature_columns = behavior_bundle["feature_columns"]

    rename_map = {
        "cx": "mean_cx",
        "cy": "mean_cy",
        "speed_px_per_frame": "mean_speed",
        "speed_px_per_frame_max": "max_speed",
        "speed_px_per_frame_std": "std_speed",
        "vx": "mean_vx",
        "vy": "mean_vy",
        "vx_total": "total_dx",
        "vy_total": "total_dy",
        "motion_displacement": "displacement",
        "distance_to_ego_px": "mean_distance_to_ego",
    }

    # Build approximate behavior feature frame from AEB data
    tmp = pd.DataFrame(index=df.index)
    tmp["mean_cx"] = df["cx"]
    tmp["mean_cy"] = df["cy"]
    tmp["mean_speed"] = df["speed_px_per_frame"]
    tmp["max_speed"] = df["speed_px_per_frame"]
    tmp["std_speed"] = 0.0
    tmp["mean_vx"] = df["vx"]
    tmp["mean_vy"] = df["vy"]
    tmp["total_dx"] = df["vx"]
    tmp["total_dy"] = df["vy"]
    tmp["displacement"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    tmp["mean_distance_to_ego"] = df["distance_to_ego_px"]

    Xb = tmp[feature_columns].values.astype(np.float32)
    Xb = imputer.transform(Xb)
    behavior_pred = model.predict(Xb)

    df["behavior_id_pred"] = behavior_pred.astype(int)
    return df


# ============================================================
# Feature engineering
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ttc_valid"] = ((df["ttc"] > 0) & np.isfinite(df["ttc"])).astype(int)
    df["inv_ttc"] = np.where(df["ttc"] > 0, 1.0 / np.maximum(df["ttc"], 1e-6), 0.0)
    df["speed_mag"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    df["bbox_area"] = df["width"] * df["height"]
    df["toward_ego_proxy"] = np.where(df["vy"] > 0, 1, 0)

    # Basic heuristic uncertainty proxy if actual uncertainty per frame is unavailable
    df["uncertainty_proxy"] = (
        0.4 * np.abs(df["vx"]) +
        0.4 * np.abs(df["vy"]) +
        0.2 * np.clip(df["collision_probability"], 0, 1) * 10.0
    )

    # Behavior risk prior
    behavior_risk_map = {
        0: 0.10,  # standing
        1: 0.30,  # walking
        2: 0.55,  # running
        3: 0.70,  # crossing
        4: 0.80,  # approaching_road
    }
    df["behavior_risk_prior"] = df["behavior_id_pred"].map(behavior_risk_map).fillna(0.2)

    # Rule-based reference brake decision from Part 3
    df["rule_based_brake"] = ((df["ttc"] > 0) & (df["ttc"] < RISK_TTC_THRESHOLD)).astype(int)

    return df


def get_feature_columns() -> List[str]:
    return [
        "cx",
        "cy",
        "vx",
        "vy",
        "speed_px_per_frame",
        "speed_px_per_sec",
        "distance_to_ego_px",
        "width",
        "height",
        "distance",
        "ttc",
        "ttc_valid",
        "inv_ttc",
        "collision_probability",
        "bbox_area",
        "speed_mag",
        "toward_ego_proxy",
        "uncertainty_proxy",
        "behavior_id_pred",
        "behavior_risk_prior",
    ]


# ============================================================
# Splitting
# ============================================================

def prepare_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_col = df["risk_label"] if df["risk_label"].nunique() > 1 else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=stratify_col,
    )

    val_ratio_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)
    stratify_train_val = train_val_df["risk_label"] if train_val_df["risk_label"].nunique() > 1 else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=SEED,
        stratify=stratify_train_val,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_xy(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns].values.astype(np.float32)
    y = df["risk_label"].values.astype(np.int64)
    return X, y


# ============================================================
# Model
# ============================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=250,
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
    y_true: np.ndarray,
    df_split: pd.DataFrame
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # Handle single-class training case safely
        if proba.shape[1] == 1:
            learned_class = int(model.classes_[0])
            if learned_class == 1:
                risk_score = np.ones(len(X), dtype=np.float32)
            else:
                risk_score = np.zeros(len(X), dtype=np.float32)
        else:
            class_to_index = {int(c): i for i, c in enumerate(model.classes_)}
            if 1 in class_to_index:
                risk_score = proba[:, class_to_index[1]]
            else:
                risk_score = np.zeros(len(X), dtype=np.float32)
    else:
        risk_score = y_pred.astype(np.float32)

    metrics = aeb_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ttc_values=df_split["ttc"].values,
        distances=df_split["distance"].values,
    )

    return metrics, y_pred, risk_score


# ============================================================
# Plotting
# ============================================================

def plot_braking_decisions(
    df_test: pd.DataFrame,
    rule_pred: np.ndarray,
    adaptive_pred: np.ndarray,
    save_path: str,
    max_points: int = 120
) -> None:
    ensure_dir(os.path.dirname(save_path))

    plot_df = df_test.copy().reset_index(drop=True)
    plot_df["rule_pred"] = rule_pred
    plot_df["adaptive_pred"] = adaptive_pred

    # sort for a meaningful visual
    sort_cols = ["track_id", "frame"] if "track_id" in plot_df.columns and "frame" in plot_df.columns else None
    if sort_cols:
        plot_df = plot_df.sort_values(sort_cols).reset_index(drop=True)

    plot_df = plot_df.iloc[:max_points]

    x = np.arange(len(plot_df))

    plt.figure(figsize=(11, 5))
    plt.plot(x, plot_df["ttc"].clip(lower=0, upper=10), label="TTC")
    plt.plot(x, plot_df["risk_label"], label="Ground truth risk")
    plt.plot(x, plot_df["rule_pred"], label="Rule-based AEB")
    plt.plot(x, plot_df["adaptive_pred"], label="Adaptive AEB")
    plt.xlabel("Test sample index")
    plt.ylabel("Value")
    plt.title("Rule-based vs Adaptive Braking Decisions")
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
    ensure_dir(MODEL_DIR)

    print("[INFO] Loading AEB dataset...")
    df = load_csv(AEB_DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows")

    print("[INFO] Building adaptive AEB target label...")
    df = build_risk_label(df)
    print("[INFO] Risk label source:", df["risk_label_source"].iloc[0])
    print("[INFO] Risk label distribution:")
    print(df["risk_label"].value_counts(dropna=False))

    print("[INFO] Loading behavior classifier...")
    behavior_bundle = joblib.load(BEHAVIOR_MODEL_PATH)

    print("[INFO] Attaching behavior predictions...")
    df = attach_behavior_predictions(df, behavior_bundle)

    print("[INFO] Engineering features...")
    df = add_engineered_features(df)

    feature_columns = get_feature_columns()

    

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    print("[INFO] Preparing train/val/test splits...")
    train_df, val_df, test_df = prepare_splits(df)

    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Val rows:   {len(val_df)}")
    print(f"[INFO] Test rows:  {len(test_df)}")

    print("[INFO] Risk label source:", df["risk_label_source"].iloc[0])
    print("[INFO] Risk label distribution:")
    print(df["risk_label"].value_counts(dropna=False))

    X_train, y_train = prepare_xy(train_df, feature_columns)
    X_val, y_val = prepare_xy(val_df, feature_columns)
    X_test, y_test = prepare_xy(test_df, feature_columns)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    print("[INFO] Training adaptive AEB model...")
    model = train_model(X_train, y_train)

    print("[INFO] Evaluating adaptive AEB...")
    train_metrics, train_pred, train_score = evaluate_split(model, X_train, y_train, train_df)
    val_metrics, val_pred, val_score = evaluate_split(model, X_val, y_val, val_df)
    test_metrics, test_pred, test_score = evaluate_split(model, X_test, y_test, test_df)

    print("[INFO] Evaluating rule-based AEB reference...")
    rule_train_metrics = aeb_metrics(
        y_true=y_train,
        y_pred=train_df["rule_based_brake"].values,
        ttc_values=train_df["ttc"].values,
        distances=train_df["distance"].values,
    )
    rule_val_metrics = aeb_metrics(
        y_true=y_val,
        y_pred=val_df["rule_based_brake"].values,
        ttc_values=val_df["ttc"].values,
        distances=val_df["distance"].values,
    )
    rule_test_metrics = aeb_metrics(
        y_true=y_test,
        y_pred=test_df["rule_based_brake"].values,
        ttc_values=test_df["ttc"].values,
        distances=test_df["distance"].values,
    )

    feature_importance = serialize_feature_importance(feature_columns, model.feature_importances_)

    results = {
        "dataset": {
            "total_rows": int(len(df)),
            "risk_label_source": str(df["risk_label_source"].iloc[0]) if len(df) > 0 else "unknown",
            "total_positive_labels": int(df["risk_label"].sum()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_label_distribution": summarize_labels(y_train),
            "val_label_distribution": summarize_labels(y_val),
            "test_label_distribution": summarize_labels(y_test),
        },
        "feature_columns": feature_columns,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 250,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": SEED,
        },
        "feature_importance": feature_importance,
        "rule_based_metrics": {
            "train": rule_train_metrics,
            "val": rule_val_metrics,
            "test": rule_test_metrics,
        },
        "adaptive_aeb_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    print("[INFO] Saving model...")
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_columns": feature_columns,
            "risk_ttc_threshold": RISK_TTC_THRESHOLD,
            "behavior_label_map": BEHAVIOR_LABEL_MAP,
        },
        MODEL_PATH
    )

    print("[INFO] Saving results...")
    save_json(results, RESULTS_PATH)

    print("[INFO] Saving test predictions...")
    pred_df = test_df.copy()
    pred_df["rule_based_brake"] = test_df["rule_based_brake"].values
    pred_df["adaptive_brake"] = test_pred
    pred_df["adaptive_risk_score"] = test_score
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    print("[INFO] Saving comparison plot...")
    plot_braking_decisions(
        df_test=test_df,
        rule_pred=test_df["rule_based_brake"].values,
        adaptive_pred=test_pred,
        save_path=PLOT_PATH,
    )

    print("\n[INFO] Adaptive AEB training completed.")
    print(f"[INFO] Saved model to: {MODEL_PATH}")
    print(f"[INFO] Saved results to: {RESULTS_PATH}")
    print(f"[INFO] Saved predictions to: {PREDICTIONS_PATH}")
    print(f"[INFO] Saved plot to: {PLOT_PATH}")

    print("\n[RESULTS] TEST COMPARISON")
    print("Rule-based AEB:")
    print(f"  Accuracy:            {rule_test_metrics['accuracy']:.4f}")
    print(f"  Precision:           {rule_test_metrics['precision']:.4f}")
    print(f"  Recall:              {rule_test_metrics['recall']:.4f}")
    print(f"  F1:                  {rule_test_metrics['f1']:.4f}")
    print(f"  False braking rate:  {rule_test_metrics['false_braking_rate']:.4f}")
    print(f"  Collision miss rate: {rule_test_metrics['collision_miss_rate']:.4f}")

    print("\nAdaptive AEB:")
    print(f"  Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"  Precision:           {test_metrics['precision']:.4f}")
    print(f"  Recall:              {test_metrics['recall']:.4f}")
    print(f"  F1:                  {test_metrics['f1']:.4f}")
    print(f"  False braking rate:  {test_metrics['false_braking_rate']:.4f}")
    print(f"  Collision miss rate: {test_metrics['collision_miss_rate']:.4f}")

    print("\n[TOP FEATURES]")
    for k, v in list(feature_importance.items())[:10]:
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()