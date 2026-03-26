"""
AEB Effectiveness Evaluation.

Quantifies the safety benefit of AEB by comparing outcomes
with and without the system at different TTC thresholds.

This produces THE headline result of the project:
    "AEB reduces HIC from ~1800 (fatal) to ~200 (minor)"

Uses the vehicle dynamics simulator for realistic braking distances
and the HIC model for injury severity classification.
"""

import numpy as np
import json
import os

from src.vehicle_dynamics.dynamics_sim import simulate_braking, braking_distance
from src.safety_analysis.impact_model import ImpactAnalyzer


def evaluate_aeb_benefit(initial_speed_kmh=50.0, distance_to_ped_m=30.0,
                         surface="dry", ped_height_m=1.75):
    """
    Compare pedestrian outcomes with different AEB trigger timings.

    For each TTC threshold, computes:
    1. At what distance AEB triggers
    2. How much braking distance is available
    3. Whether vehicle stops in time or residual speed
    4. HIC value and injury severity at residual speed

    Args:
        initial_speed_kmh: vehicle approach speed
        distance_to_ped_m: initial distance to pedestrian
        surface: road surface condition
        ped_height_m: pedestrian height

    Returns:
        list of dicts, one per TTC threshold
    """
    analyzer = ImpactAnalyzer()
    v_ms = initial_speed_kmh / 3.6

    # Get actual braking distance from dynamics simulator
    brake_dist = braking_distance(initial_speed_kmh, surface)

    # TTC thresholds to evaluate
    ttc_thresholds = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = []

    for ttc in ttc_thresholds:
        if ttc is None:
            # No AEB — full speed impact
            label = "No AEB"
            trigger_dist = 0.0
            available_dist = 0.0
            impact_speed = initial_speed_kmh
        else:
            # AEB triggers when TTC = threshold
            # trigger_distance = v × TTC
            trigger_dist = v_ms * ttc

            if trigger_dist > distance_to_ped_m:
                # Would trigger before reaching detection range
                # Use detection distance instead
                trigger_dist = distance_to_ped_m

            # Available braking distance
            available_dist = trigger_dist

            # Compute residual impact speed
            impact_speed = analyzer.compute_residual_speed(
                initial_speed_kmh, brake_dist, available_dist
            )

        # Compute HIC at impact speed
        hic = analyzer.compute_hic(impact_speed, ped_height_m)
        severity = analyzer.injury_severity(hic)

        # Speed reduction percentage
        speed_reduction = ((initial_speed_kmh - impact_speed) / initial_speed_kmh * 100
                           if initial_speed_kmh > 0 else 0)

        result = {
            "label": label if ttc is None else f"TTC = {ttc:.1f}s",
            "ttc_threshold": ttc,
            "initial_speed_kmh": initial_speed_kmh,
            "trigger_distance_m": trigger_dist,
            "available_braking_m": available_dist,
            "total_braking_needed_m": brake_dist,
            "impact_speed_kmh": impact_speed,
            "speed_reduction_pct": speed_reduction,
            "collision_avoided": impact_speed < 1.0,
            "hic": hic,
            "ais_level": severity["ais_level"],
            "injury_description": severity["description"],
            "ncap_color": severity["ncap_color"],
        }
        results.append(result)

    return results


def compute_hic_reduction(results):
    """
    Compute % HIC reduction compared to no-AEB baseline.

    Args:
        results: list from evaluate_aeb_benefit()

    Returns:
        list of dicts with "ttc_threshold" and "hic_reduction_pct"
    """
    # Find no-AEB baseline
    baseline_hic = None
    for r in results:
        if r["ttc_threshold"] is None:
            baseline_hic = r["hic"]
            break

    if baseline_hic is None or baseline_hic == 0:
        return []

    reductions = []
    for r in results:
        if r["ttc_threshold"] is not None:
            reduction = (1 - r["hic"] / baseline_hic) * 100 if baseline_hic > 0 else 0
            reductions.append({
                "ttc_threshold": r["ttc_threshold"],
                "hic_reduction_pct": reduction,
                "hic_value": r["hic"],
                "baseline_hic": baseline_hic,
            })

    return reductions


def print_effectiveness_table(results):
    """Print a formatted AEB effectiveness comparison table."""
    print(f"\n{'AEB Setting':<14} {'Trigger':>10} {'Impact ':>10} {'Speed↓':>8} "
          f"{'HIC':>8} {'AIS':>5} {'Injury':>12} {'Avoided?':>10}")
    print("-" * 85)

    for r in results:
        trigger = f"{r['trigger_distance_m']:.1f} m" if r['ttc_threshold'] else "—"
        avoided = "✓ YES" if r['collision_avoided'] else "✗ No"
        print(f"{r['label']:<14} {trigger:>10} {r['impact_speed_kmh']:>8.1f} km/h "
              f"{r['speed_reduction_pct']:>7.0f}% {r['hic']:>8.0f} "
              f"{r['ais_level']:>5} {r['injury_description']:>12} {avoided:>10}")


def multi_speed_analysis(speeds=None, surface="dry", save_path=None):
    """
    Run AEB effectiveness analysis across multiple approach speeds.

    Returns:
        dict: {speed: results_list}
    """
    if speeds is None:
        speeds = [30, 40, 50, 60, 70, 80]

    all_results = {}
    for speed in speeds:
        results = evaluate_aeb_benefit(speed, distance_to_ped_m=40.0, surface=surface)
        all_results[speed] = results

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert to serializable format
        serializable = {}
        for speed, results in all_results.items():
            serializable[str(speed)] = results
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Results saved to {save_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AEB Effectiveness Evaluation")
    parser.add_argument("--speed", type=float, default=50.0, help="Approach speed (km/h)")
    parser.add_argument("--distance", type=float, default=30.0, help="Distance to pedestrian (m)")
    parser.add_argument("--surface", type=str, default="dry", help="Road surface")
    parser.add_argument("--multi", action="store_true", help="Run multi-speed analysis")
    args = parser.parse_args()

    if args.multi:
        print("\n" + "=" * 85)
        print("  AEB Effectiveness — Multi-Speed Analysis")
        print("=" * 85)
        all_results = multi_speed_analysis(surface=args.surface,
                                           save_path="results/safety_analysis/effectiveness.json")
        for speed, results in all_results.items():
            print(f"\n─── {speed} km/h ───")
            print_effectiveness_table(results)
    else:
        print(f"\n{'=' * 85}")
        print(f"  AEB Effectiveness at {args.speed} km/h — {args.surface.capitalize()} Surface")
        print(f"{'=' * 85}")
        results = evaluate_aeb_benefit(args.speed, args.distance, args.surface)
        print_effectiveness_table(results)

        reductions = compute_hic_reduction(results)
        if reductions:
            print(f"\nHIC Reduction vs. No-AEB (baseline HIC = {reductions[0]['baseline_hic']:.0f}):")
            for r in reductions:
                print(f"  TTC {r['ttc_threshold']:.1f}s → HIC {r['hic_value']:.0f} "
                      f"({r['hic_reduction_pct']:.0f}% reduction)")
