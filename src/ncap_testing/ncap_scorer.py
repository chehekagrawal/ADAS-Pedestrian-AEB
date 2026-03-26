"""
Euro NCAP AEB Scoring Module.

Computes scores following the Euro NCAP methodology:
- Per-test scores based on collision avoidance
- Per-scenario aggregate scores
- Overall system score and star rating
"""

import numpy as np
from typing import List
from src.ncap_testing.ncap_runner import TestResult


# ─── Scoring Weights ──────────────────────────────────────────────
# Euro NCAP weights each scenario equally (simplified)
SCENARIO_WEIGHTS = {
    "CPFA-50": 0.25,
    "CPNA-25": 0.25,
    "CPNC-50": 0.25,
    "CPLA": 0.25,
}


def score_single_test(result):
    """
    Score one test case (Euro NCAP simplified scoring).

    Scoring rules:
        Full avoidance (impact speed = 0)       → 1.00 (100%)
        Speed reduction ≥ 50%                    → proportional (0.50 – 0.99)
        Speed reduction ≥ 20% but < 50%          → proportional (0.10 – 0.49)
        Speed reduction < 20%                    → 0.00

    Args:
        result: TestResult

    Returns:
        score: float (0.0 – 1.0)
    """
    if result.outcome == "avoided" or result.impact_speed_kmh < 1.0:
        return 1.0

    reduction = result.speed_reduction_pct / 100.0  # normalize to 0-1

    if reduction >= 0.50:
        return reduction
    elif reduction >= 0.20:
        return reduction * 0.5  # penalized
    else:
        return 0.0


def score_scenario(results, scenario_code):
    """
    Compute aggregate score for one scenario across all test speeds.

    Euro NCAP weights different speeds differently — lower speeds
    generally contribute more because they're more common in urban areas.

    Args:
        results: list of TestResult
        scenario_code: str (e.g., "CPFA-50")

    Returns:
        dict: {"scenario": code, "score": float, "detail": list}
    """
    scenario_results = [r for r in results if r.scenario_code == scenario_code]
    if not scenario_results:
        return {"scenario": scenario_code, "score": 0.0, "detail": []}

    # Speed-based weighting (lower speeds = higher weight)
    speed_weights = {
        20: 0.15, 25: 0.15, 30: 0.13, 35: 0.12, 40: 0.11,
        45: 0.10, 50: 0.09, 55: 0.08, 60: 0.07
    }

    weighted_sum = 0.0
    total_weight = 0.0
    detail = []

    for r in scenario_results:
        test_score = score_single_test(r)
        weight = speed_weights.get(int(r.vehicle_speed_kmh), 0.10)
        weighted_sum += test_score * weight
        total_weight += weight
        detail.append({
            "speed": r.vehicle_speed_kmh,
            "score": test_score,
            "outcome": r.outcome,
            "weight": weight,
        })

    scenario_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "scenario": scenario_code,
        "score": scenario_score,
        "tests_count": len(scenario_results),
        "avoided": sum(1 for r in scenario_results if r.outcome == "avoided"),
        "partial": sum(1 for r in scenario_results if r.outcome == "partial"),
        "collision": sum(1 for r in scenario_results if r.outcome == "collision"),
        "detail": detail,
    }


def overall_score(results):
    """
    Compute overall Euro NCAP AEB-Pedestrian score.

    Returns:
        dict: {
            "overall_score": float (0.0 – 1.0),
            "overall_percent": float (0 – 100),
            "star_rating": int (0 – 5),
            "grade": str ("A" – "F"),
            "scenarios": dict of per-scenario scores
        }
    """
    scenario_scores = {}
    weighted_total = 0.0

    for code, weight in SCENARIO_WEIGHTS.items():
        sc = score_scenario(results, code)
        scenario_scores[code] = sc
        weighted_total += sc["score"] * weight

    star_rating = _compute_stars(weighted_total)
    grade = _compute_grade(weighted_total)

    return {
        "overall_score": weighted_total,
        "overall_percent": weighted_total * 100,
        "star_rating": star_rating,
        "grade": grade,
        "scenarios": scenario_scores,
    }


def _compute_stars(score):
    """Simplified Euro NCAP star mapping."""
    if score >= 0.90:
        return 5
    elif score >= 0.75:
        return 4
    elif score >= 0.60:
        return 3
    elif score >= 0.40:
        return 2
    elif score >= 0.20:
        return 1
    else:
        return 0


def _compute_grade(score):
    """Letter grade mapping."""
    if score >= 0.90:
        return "A"
    elif score >= 0.80:
        return "B"
    elif score >= 0.70:
        return "C"
    elif score >= 0.60:
        return "D"
    else:
        return "F"


def format_scorecard(score_data):
    """
    Format score data as a text scorecard.

    Args:
        score_data: dict from overall_score()

    Returns:
        str: formatted scorecard
    """
    stars = "★" * score_data["star_rating"] + "☆" * (5 - score_data["star_rating"])

    lines = [
        "┌──────────────────────────────────────────────────┐",
        "│        Euro NCAP AEB-Pedestrian Scorecard        │",
        "├──────────────────────────────────────────────────┤",
        f"│  Overall: {score_data['overall_percent']:.0f}%  ({score_data['grade']})  {stars}  │",
        "├──────────────────────────────────────────────────┤",
    ]

    for code, sc in score_data["scenarios"].items():
        pct = sc['score'] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"│  {code:<10} {bar} {pct:>5.0f}%  │")

    lines.append("└──────────────────────────────────────────────────┘")

    return "\n".join(lines)


def print_full_scorecard(results):
    """Compute and print the complete scorecard."""
    score_data = overall_score(results)
    print(format_scorecard(score_data))

    # Per-scenario details
    for code, sc in score_data["scenarios"].items():
        print(f"\n  {code}: {sc['avoided']}✓ avoided | "
              f"{sc['partial']}◐ partial | {sc['collision']}✗ collision")


if __name__ == "__main__":
    from src.ncap_testing.ncap_runner import run_full_test_matrix

    print("Running full Euro NCAP test matrix (dry surface)...\n")
    results = run_full_test_matrix("dry")
    print_full_scorecard(results)
