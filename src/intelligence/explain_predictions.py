import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = "results/intelligence"
MODEL_DIR = "models"

BEHAVIOR_RESULTS = os.path.join(RESULTS_DIR, "behavior_classifier_results.json")
ADAPTIVE_RESULTS = os.path.join(RESULTS_DIR, "adaptive_aeb_results.json")

ADAPTIVE_MODEL = os.path.join(MODEL_DIR, "adaptive_aeb_model.pkl")

AEB_PREDICTIONS = os.path.join(RESULTS_DIR, "adaptive_aeb_predictions.csv")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "explainability")

FEATURE_IMPORTANCE_PLOT = os.path.join(OUTPUT_DIR, "feature_importance.png")
RISK_SCORE_PLOT = os.path.join(OUTPUT_DIR, "risk_score_distribution.png")
DECISION_PLOT = os.path.join(OUTPUT_DIR, "decision_explanation.png")


# ============================================================
# Utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Feature Importance Visualization
# ============================================================

def plot_feature_importance(model_bundle):
    model = model_bundle["model"]
    feature_names = model_bundle["feature_columns"]

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8,6))

    plt.barh(sorted_features[:10][::-1], sorted_importances[:10][::-1])

    plt.title("Adaptive AEB Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    plt.close()

    print("[INFO] Saved feature importance plot")


# ============================================================
# Risk Score Visualization
# ============================================================

def plot_risk_score_distribution(pred_df):

    if "adaptive_risk_score" not in pred_df.columns:
        print("[WARNING] Risk score column missing")
        return

    plt.figure(figsize=(7,5))

    plt.hist(pred_df["adaptive_risk_score"], bins=20)

    plt.title("Adaptive AEB Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(RISK_SCORE_PLOT)
    plt.close()

    print("[INFO] Saved risk score distribution plot")


# ============================================================
# Decision Explanation Plot
# ============================================================

def plot_decision_example(pred_df):

    df = pred_df.head(60).copy()

    plt.figure(figsize=(10,5))

    x = np.arange(len(df))

    plt.plot(x, df["ttc"], label="TTC")

    if "adaptive_risk_score" in df.columns:
        plt.plot(x, df["adaptive_risk_score"], label="Adaptive Risk Score")

    plt.plot(x, df["adaptive_brake"], label="Adaptive Brake Decision")

    plt.xlabel("Frame Index")
    plt.ylabel("Value")

    plt.title("Example AEB Decision Explanation")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(DECISION_PLOT)
    plt.close()

    print("[INFO] Saved decision explanation plot")


# ============================================================
# Main
# ============================================================

def main():

    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading adaptive AEB model")

    model_bundle = joblib.load(ADAPTIVE_MODEL)

    print("[INFO] Loading prediction dataset")

    pred_df = pd.read_csv(AEB_PREDICTIONS)

    print("[INFO] Generating feature importance plot")
    plot_feature_importance(model_bundle)

    print("[INFO] Generating risk score distribution")
    plot_risk_score_distribution(pred_df)

    print("[INFO] Generating decision explanation plot")
    plot_decision_example(pred_df)

    print("\n[INFO] Explainability module completed")

    print("\nGenerated files:")
    print(FEATURE_IMPORTANCE_PLOT)
    print(RISK_SCORE_PLOT)
    print(DECISION_PLOT)


if __name__ == "__main__":
    main()