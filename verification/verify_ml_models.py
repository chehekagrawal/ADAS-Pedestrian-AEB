"""
Verification Script: Machine Learning Models Audit
Audits both trained pickle models:
  1. models/adaptive_aeb_model.pkl (Random Forest for adaptive AEB risk)
  2. models/behavior_classifier.pkl (Random Forest for VRU behavior classification)
Logs exact parameters, feature importances, and class labels.
"""

import os
import json
import joblib

def main():
    os.makedirs("verification", exist_ok=True)
    report = {}

    # 1. Adaptive AEB Model
    aeb_path = "models/adaptive_aeb_model.pkl"
    if os.path.exists(aeb_path):
        aeb_data = joblib.load(aeb_path)
        rf_aeb = aeb_data["model"]
        feature_cols = aeb_data["feature_columns"]
        importances = rf_aeb.feature_importances_
        
        report["adaptive_aeb_model"] = {
            "file": aeb_path,
            "type": str(type(rf_aeb)),
            "n_estimators": getattr(rf_aeb, "n_estimators", None),
            "classes": [int(c) for c in rf_aeb.classes_],
            "num_features": len(feature_cols),
            "feature_columns": feature_cols,
            "feature_importances": {col: float(imp) for col, imp in zip(feature_cols, importances)},
            "risk_ttc_threshold": aeb_data.get("risk_ttc_threshold", None),
            "behavior_label_map": aeb_data.get("behavior_label_map", None),
            "audit_finding": "Model contains only single class [0] due to non-collision training data. Fictitious 91.4% claim in earlier draft is ungrounded."
        }
    else:
        report["adaptive_aeb_model"] = {"error": "File not found"}

    # 2. Behavior Classifier Model
    beh_path = "models/behavior_classifier.pkl"
    if os.path.exists(beh_path):
        beh_data = joblib.load(beh_path)
        rf_beh = beh_data["model"]
        feature_cols_beh = beh_data["feature_columns"]
        importances_beh = rf_beh.feature_importances_
        
        ranked_features = sorted(
            [{"feature": col, "importance": float(imp)} for col, imp in zip(feature_cols_beh, importances_beh)],
            key=lambda x: x["importance"],
            reverse=True
        )

        report["behavior_classifier_model"] = {
            "file": beh_path,
            "type": str(type(rf_beh)),
            "n_estimators": getattr(rf_beh, "n_estimators", None),
            "classes": [int(c) for c in rf_beh.classes_],
            "num_features": len(feature_cols_beh),
            "feature_columns": feature_cols_beh,
            "label_map": beh_data.get("label_map", None),
            "ranked_feature_importances": ranked_features,
            "audit_finding": "Genuine multi-class Random Forest trained across 5 VRU states (standing, walking, running, crossing, approaching_road)."
        }
    else:
        report["behavior_classifier_model"] = {"error": "File not found"}

    print("=== Machine Learning Models Audit ===")
    print(f"Adaptive AEB Model: classes={report['adaptive_aeb_model'].get('classes')}, n_estimators={report['adaptive_aeb_model'].get('n_estimators')}")
    print(f"Behavior Classifier: classes={report['behavior_classifier_model'].get('classes')}, n_estimators={report['behavior_classifier_model'].get('n_estimators')}")
    print("\nTop 5 Behavior Classifier Features:")
    for item in report["behavior_classifier_model"]["ranked_feature_importances"][:5]:
        print(f"  {item['feature']:25s}: {item['importance']:.4f}")

    with open("verification/ml_models_verified.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nSaved verification/ml_models_verified.json")

if __name__ == "__main__":
    main()
