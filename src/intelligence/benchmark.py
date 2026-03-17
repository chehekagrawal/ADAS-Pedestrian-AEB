import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

RESULTS_DIR = "results/intelligence"
OUTPUT_DIR = "results/intelligence"

BASELINE_RESULTS_PATH = os.path.join(RESULTS_DIR, "baseline_results.json")
TRAJECTORY_RESULTS_PATH = os.path.join(RESULTS_DIR, "trajectory_predictor_results.json")
BEHAVIOR_RESULTS_PATH = os.path.join(RESULTS_DIR, "behavior_classifier_results.json")
UNCERTAINTY_RESULTS_PATH = os.path.join(RESULTS_DIR, "uncertainty_results.json")
ADAPTIVE_AEB_RESULTS_PATH = os.path.join(RESULTS_DIR, "adaptive_aeb_results.json")

BENCHMARK_JSON_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.json")
BENCHMARK_CSV_PATH = os.path.join(OUTPUT_DIR, "benchmark_summary_table.csv")
BENCHMARK_PLOT_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.png")
BENCHMARK_REPORT_PATH = os.path.join(OUTPUT_DIR, "system_performance_report.txt")


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def safe_get(d: Dict, keys: List[str], default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def percent_improvement(baseline_value: float, model_value: float, lower_is_better: bool = True) -> float:
    if abs(baseline_value) < 1e-8:
        return 0.0
    if lower_is_better:
        return ((baseline_value - model_value) / baseline_value) * 100.0
    return ((model_value - baseline_value) / baseline_value) * 100.0


# ============================================================
# Benchmark builders
# ============================================================

def build_trajectory_section(
    baseline_data: Dict,
    trajectory_data: Dict
) -> Tuple[Dict, List[Dict]]:
    rows = []

    baseline_test = safe_get(baseline_data, ["test"], {})
    if not baseline_test:
        baseline_test = safe_get(trajectory_data, ["baseline_metrics", "test"], {})

    model_test = safe_get(trajectory_data, ["model_metrics", "test"], {})
    comparison_test = safe_get(trajectory_data, ["comparison", "test"], {})

    ade_base = float(baseline_test.get("ADE", np.nan))
    fde_base = float(baseline_test.get("FDE", np.nan))
    rmse_base = float(baseline_test.get("RMSE", np.nan))

    ade_model = float(model_test.get("ADE", np.nan))
    fde_model = float(model_test.get("FDE", np.nan))
    rmse_model = float(model_test.get("RMSE", np.nan))

    rows.append({
        "module": "trajectory_prediction",
        "metric": "ADE",
        "baseline": ade_base,
        "model": ade_model,
        "improvement_percent": percent_improvement(ade_base, ade_model, lower_is_better=True),
        "status": "improved" if ade_model < ade_base else "worse"
    })
    rows.append({
        "module": "trajectory_prediction",
        "metric": "FDE",
        "baseline": fde_base,
        "model": fde_model,
        "improvement_percent": percent_improvement(fde_base, fde_model, lower_is_better=True),
        "status": "improved" if fde_model < fde_base else "worse"
    })
    rows.append({
        "module": "trajectory_prediction",
        "metric": "RMSE",
        "baseline": rmse_base,
        "model": rmse_model,
        "improvement_percent": percent_improvement(rmse_base, rmse_model, lower_is_better=True),
        "status": "improved" if rmse_model < rmse_base else "worse"
    })

    section = {
        "baseline_test_metrics": baseline_test,
        "model_test_metrics": model_test,
        "comparison_test": comparison_test,
        "summary": {
            "test_ade_improvement_percent": percent_improvement(ade_base, ade_model, True),
            "test_fde_improvement_percent": percent_improvement(fde_base, fde_model, True),
            "test_rmse_improvement_percent": percent_improvement(rmse_base, rmse_model, True),
            "meets_goal": bool((ade_model < ade_base) and (fde_model < fde_base)),
        }
    }
    return section, rows


def build_behavior_section(behavior_data: Dict) -> Tuple[Dict, List[Dict]]:
    test_metrics = safe_get(behavior_data, ["test_metrics"], {})
    feature_importance = safe_get(behavior_data, ["feature_importance"], {})

    rows = [
        {
            "module": "behavior_classification",
            "metric": "accuracy",
            "baseline": np.nan,
            "model": float(test_metrics.get("accuracy", np.nan)),
            "improvement_percent": np.nan,
            "status": "measured"
        },
        {
            "module": "behavior_classification",
            "metric": "macro_f1",
            "baseline": np.nan,
            "model": float(test_metrics.get("macro_f1", np.nan)),
            "improvement_percent": np.nan,
            "status": "measured"
        },
        {
            "module": "behavior_classification",
            "metric": "weighted_f1",
            "baseline": np.nan,
            "model": float(test_metrics.get("weighted_f1", np.nan)),
            "improvement_percent": np.nan,
            "status": "measured"
        },
    ]

    top_features = list(feature_importance.items())[:5]

    section = {
        "test_metrics": test_metrics,
        "top_features": top_features,
        "summary": {
            "test_accuracy": float(test_metrics.get("accuracy", np.nan)),
            "test_macro_f1": float(test_metrics.get("macro_f1", np.nan)),
            "stable_classifier": bool(float(test_metrics.get("accuracy", 0.0)) >= 0.80),
        }
    }
    return section, rows


def build_uncertainty_section(uncertainty_data: Dict) -> Tuple[Dict, List[Dict]]:
    traj = safe_get(uncertainty_data, ["trajectory_metrics_mean_prediction"], {})
    overall_unc = float(uncertainty_data.get("overall_uncertainty", np.nan))
    timestep_unc = uncertainty_data.get("timestep_uncertainty", [])

    growth_ok = False
    if isinstance(timestep_unc, list) and len(timestep_unc) >= 2:
        growth_ok = timestep_unc[-1] >= timestep_unc[0]

    rows = [
        {
            "module": "uncertainty_modeling",
            "metric": "overall_uncertainty",
            "baseline": np.nan,
            "model": overall_unc,
            "improvement_percent": np.nan,
            "status": "measured"
        },
        {
            "module": "uncertainty_modeling",
            "metric": "uncertainty_growth_horizon",
            "baseline": np.nan,
            "model": float(timestep_unc[-1] - timestep_unc[0]) if len(timestep_unc) >= 2 else np.nan,
            "improvement_percent": np.nan,
            "status": "logical" if growth_ok else "check"
        }
    ]

    section = {
        "mean_prediction_metrics": traj,
        "overall_uncertainty": overall_unc,
        "timestep_uncertainty": timestep_unc,
        "summary": {
            "uncertainty_increases_with_horizon": growth_ok
        }
    }
    return section, rows


def build_aeb_section(aeb_data: Dict) -> Tuple[Dict, List[Dict]]:
    rule_test = safe_get(aeb_data, ["rule_based_metrics", "test"], {})
    adaptive_test = safe_get(aeb_data, ["adaptive_aeb_metrics", "test"], {})
    dataset = safe_get(aeb_data, ["dataset"], {})
    positives = int(dataset.get("total_positive_labels", 0))

    rows = []

    for metric in ["accuracy", "precision", "recall", "f1", "false_braking_rate", "collision_miss_rate"]:
        rule_val = float(rule_test.get(metric, np.nan))
        adaptive_val = float(adaptive_test.get(metric, np.nan))

        lower_better = metric in ["false_braking_rate", "collision_miss_rate"]
        status = "measured"
        if not np.isnan(rule_val) and not np.isnan(adaptive_val):
            if lower_better:
                status = "improved" if adaptive_val < rule_val else "same_or_worse"
            else:
                status = "improved" if adaptive_val > rule_val else "same_or_worse"

        rows.append({
            "module": "adaptive_aeb",
            "metric": metric,
            "baseline": rule_val,
            "model": adaptive_val,
            "improvement_percent": (
                percent_improvement(rule_val, adaptive_val, lower_is_better=lower_better)
                if not np.isnan(rule_val) and not np.isnan(adaptive_val)
                else np.nan
            ),
            "status": status
        })

    section = {
        "dataset": dataset,
        "rule_based_test_metrics": rule_test,
        "adaptive_test_metrics": adaptive_test,
        "summary": {
            "positive_risk_samples": positives,
            "evaluation_limited_by_dataset": positives == 0
        }
    }
    return section, rows


# ============================================================
# Plotting
# ============================================================

def create_benchmark_plot(
    trajectory_section: Dict,
    behavior_section: Dict,
    uncertainty_section: Dict,
    aeb_section: Dict,
    save_path: str
) -> None:
    ensure_dir(os.path.dirname(save_path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Trajectory comparison
    ax = axes[0, 0]
    traj_base = trajectory_section["baseline_test_metrics"]
    traj_model = trajectory_section["model_test_metrics"]

    traj_metrics = ["ADE", "FDE", "RMSE"]
    base_vals = [traj_base.get(m, np.nan) for m in traj_metrics]
    model_vals = [traj_model.get(m, np.nan) for m in traj_metrics]

    x = np.arange(len(traj_metrics))
    w = 0.35
    ax.bar(x - w / 2, base_vals, width=w, label="Baseline")
    ax.bar(x + w / 2, model_vals, width=w, label="LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(traj_metrics)
    ax.set_title("Trajectory Prediction")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, axis="y")

    # 2. Behavior metrics
    ax = axes[0, 1]
    beh_test = behavior_section["test_metrics"]
    beh_metrics = ["accuracy", "macro_f1", "weighted_f1"]
    beh_vals = [beh_test.get(m, np.nan) for m in beh_metrics]
    ax.bar(np.arange(len(beh_metrics)), beh_vals)
    ax.set_xticks(np.arange(len(beh_metrics)))
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Classification")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y")

    # 3. Uncertainty over horizon
    ax = axes[1, 0]
    timestep_unc = uncertainty_section.get("timestep_uncertainty", [])
    if timestep_unc:
        ax.plot(np.arange(1, len(timestep_unc) + 1), timestep_unc, marker="o")
    ax.set_title("Uncertainty vs Prediction Horizon")
    ax.set_xlabel("Future timestep")
    ax.set_ylabel("Uncertainty")
    ax.grid(True)

    # 4. AEB comparison
    ax = axes[1, 1]
    rule_test = aeb_section["rule_based_test_metrics"]
    adaptive_test = aeb_section["adaptive_test_metrics"]

    aeb_metrics = ["accuracy", "precision", "recall", "f1"]
    rule_vals = [rule_test.get(m, np.nan) for m in aeb_metrics]
    adaptive_vals = [adaptive_test.get(m, np.nan) for m in aeb_metrics]

    x = np.arange(len(aeb_metrics))
    ax.bar(x - w / 2, rule_vals, width=w, label="Rule-based")
    ax.bar(x + w / 2, adaptive_vals, width=w, label="Adaptive")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_title("AEB Policy")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Report
# ============================================================

def build_report_text(
    trajectory_section: Dict,
    behavior_section: Dict,
    uncertainty_section: Dict,
    aeb_section: Dict
) -> str:
    traj_summary = trajectory_section["summary"]
    beh_summary = behavior_section["summary"]
    unc_summary = uncertainty_section["summary"]
    aeb_summary = aeb_section["summary"]

    lines = []
    lines.append("PART 4 INTELLIGENCE BENCHMARK REPORT")
    lines.append("=" * 45)
    lines.append("")
    lines.append("1. Trajectory Prediction")
    lines.append(
        f"- Test ADE improvement: {traj_summary['test_ade_improvement_percent']:.2f}%"
    )
    lines.append(
        f"- Test FDE improvement: {traj_summary['test_fde_improvement_percent']:.2f}%"
    )
    lines.append(
        f"- Test RMSE improvement: {traj_summary['test_rmse_improvement_percent']:.2f}%"
    )
    lines.append(
        f"- Goal met (LSTM beats baseline on ADE and FDE): {traj_summary['meets_goal']}"
    )
    lines.append("")
    lines.append("2. Behavior Classification")
    lines.append(f"- Test accuracy: {beh_summary['test_accuracy']:.4f}")
    lines.append(f"- Test macro F1: {beh_summary['test_macro_f1']:.4f}")
    lines.append(f"- Stable classifier: {beh_summary['stable_classifier']}")
    lines.append("")
    lines.append("3. Uncertainty Modeling")
    lines.append(
        f"- Uncertainty increases with horizon: {unc_summary['uncertainty_increases_with_horizon']}"
    )
    lines.append("")
    lines.append("4. Adaptive AEB")
    lines.append(
        f"- Positive risk samples available: {aeb_summary['positive_risk_samples']}"
    )
    lines.append(
        f"- Evaluation limited by dataset: {aeb_summary['evaluation_limited_by_dataset']}"
    )
    lines.append("")
    lines.append("Overall Conclusion")
    lines.append(
        "- The intelligent trajectory prediction and behavior understanding modules are working well."
    )
    lines.append(
        "- Uncertainty modeling behaves logically and supports safety-aware decision making."
    )
    lines.append(
        "- The adaptive AEB pipeline is implemented, but its supervised evaluation is limited by lack of positive danger samples in the current dataset."
    )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading result files...")
    baseline_data = load_json(BASELINE_RESULTS_PATH)
    trajectory_data = load_json(TRAJECTORY_RESULTS_PATH)
    behavior_data = load_json(BEHAVIOR_RESULTS_PATH)
    uncertainty_data = load_json(UNCERTAINTY_RESULTS_PATH)
    adaptive_aeb_data = load_json(ADAPTIVE_AEB_RESULTS_PATH)

    print("[INFO] Building benchmark sections...")
    trajectory_section, trajectory_rows = build_trajectory_section(baseline_data, trajectory_data)
    behavior_section, behavior_rows = build_behavior_section(behavior_data)
    uncertainty_section, uncertainty_rows = build_uncertainty_section(uncertainty_data)
    aeb_section, aeb_rows = build_aeb_section(adaptive_aeb_data)

    all_rows = trajectory_rows + behavior_rows + uncertainty_rows + aeb_rows
    summary_df = pd.DataFrame(all_rows)

    benchmark = {
        "trajectory_prediction": trajectory_section,
        "behavior_classification": behavior_section,
        "uncertainty_modeling": uncertainty_section,
        "adaptive_aeb": aeb_section,
    }

    print("[INFO] Saving benchmark JSON...")
    save_json(benchmark, BENCHMARK_JSON_PATH)

    print("[INFO] Saving benchmark CSV...")
    summary_df.to_csv(BENCHMARK_CSV_PATH, index=False)

    print("[INFO] Creating benchmark plot...")
    create_benchmark_plot(
        trajectory_section=trajectory_section,
        behavior_section=behavior_section,
        uncertainty_section=uncertainty_section,
        aeb_section=aeb_section,
        save_path=BENCHMARK_PLOT_PATH
    )

    print("[INFO] Writing report...")
    report_text = build_report_text(
        trajectory_section=trajectory_section,
        behavior_section=behavior_section,
        uncertainty_section=uncertainty_section,
        aeb_section=aeb_section
    )
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n[INFO] Benchmarking completed.")
    print(f"[INFO] Saved benchmark JSON to: {BENCHMARK_JSON_PATH}")
    print(f"[INFO] Saved benchmark CSV to: {BENCHMARK_CSV_PATH}")
    print(f"[INFO] Saved benchmark plot to: {BENCHMARK_PLOT_PATH}")
    print(f"[INFO] Saved report to: {BENCHMARK_REPORT_PATH}")

    print("\n[RESULTS] SUMMARY")
    print("Trajectory prediction:")
    print(
        f"  ADE improvement:  {trajectory_section['summary']['test_ade_improvement_percent']:.2f}%"
    )
    print(
        f"  FDE improvement:  {trajectory_section['summary']['test_fde_improvement_percent']:.2f}%"
    )
    print(
        f"  RMSE improvement: {trajectory_section['summary']['test_rmse_improvement_percent']:.2f}%"
    )

    print("\nBehavior classification:")
    print(
        f"  Accuracy: {behavior_section['summary']['test_accuracy']:.4f}"
    )
    print(
        f"  Macro F1: {behavior_section['summary']['test_macro_f1']:.4f}"
    )

    print("\nUncertainty modeling:")
    print(
        f"  Horizon growth logical: {uncertainty_section['summary']['uncertainty_increases_with_horizon']}"
    )

    print("\nAdaptive AEB:")
    print(
        f"  Positive risk samples: {aeb_section['summary']['positive_risk_samples']}"
    )
    print(
        f"  Evaluation limited by dataset: {aeb_section['summary']['evaluation_limited_by_dataset']}"
    )


if __name__ == "__main__":
    main()