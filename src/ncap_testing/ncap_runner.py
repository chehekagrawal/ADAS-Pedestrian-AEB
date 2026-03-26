"""
Euro NCAP AEB Test Runner.

Executes the standardized test scenarios by simulating:
1. Vehicle approach at specified speed
2. Pedestrian crossing/walking as defined by scenario
3. System detection at estimated range
4. AEB triggering and braking (using vehicle dynamics simulator)
5. Outcome determination (avoided / partial / collision + HIC)

Each test produces a TestResult with all relevant metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from src.ncap_testing.ncap_scenarios import NCAPScenario, ALL_SCENARIOS
from src.vehicle_dynamics.dynamics_sim import simulate_braking, braking_distance
from src.vehicle_dynamics.vehicle_params import VehicleParameters, SEDAN_DEFAULT
from src.safety_analysis.impact_model import ImpactAnalyzer


@dataclass
class TestResult:
    """Result of one Euro NCAP AEB test case."""
    scenario_code: str
    vehicle_speed_kmh: float
    surface: str

    detection_distance_m: float
    ttc_at_detection_s: float
    reaction_time_s: float

    braking_distance_needed_m: float
    available_braking_distance_m: float

    outcome: str            # "avoided", "partial", "collision"
    impact_speed_kmh: float
    speed_reduction_pct: float

    hic: float
    ais_level: int
    injury_description: str

    score: float            # 0.0 – 1.0


def compute_detection_distance(vehicle_speed_kmh, scenario):
    """
    Estimate the distance at which the AEB system first detects the pedestrian.

    Factors:
    - Scenario detection range (max capability)
    - Obstructed view reduces detection distance significantly
    - Overlap percentage affects lateral detection confidence

    Args:
        vehicle_speed_kmh: approach speed
        scenario: NCAPScenario

    Returns:
        detection_distance_m: float
    """
    base_range = scenario.detection_range_m

    # Obstructed scenarios: detection only possible when ped is visible
    if scenario.is_obstructed:
        # Child appears from behind parked car at ~12-18m
        base_range = min(base_range, 15.0 + vehicle_speed_kmh * 0.05)

    # Lower overlap = pedestrian at edge of FOV = slightly later detection
    overlap_factor = 0.85 + 0.15 * (scenario.overlap_percent / 100.0)
    detection_distance = base_range * overlap_factor

    # At very high speeds, detection distance matters more
    # Ensure detection distance is at least reaction_distance * 1.5
    reaction_distance = vehicle_speed_kmh / 3.6 * 0.8  # ~0.8s system delay
    detection_distance = max(detection_distance, reaction_distance * 1.2)

    return detection_distance


def compute_ttc(vehicle_speed_kmh, ped_speed_kmh, detection_distance_m, scenario):
    """
    Compute Time-to-Collision at the moment of detection.

    For crossing scenarios: TTC depends on relative geometry
    For longitudinal: TTC = distance / (vehicle_speed - ped_speed)
    """
    v_vehicle = vehicle_speed_kmh / 3.6  # m/s
    v_ped = ped_speed_kmh / 3.6  # m/s

    if scenario.ped_direction == "longitudinal":
        # Pedestrian walking same direction: relative speed = vehicle - ped
        relative_speed = v_vehicle - v_ped
    else:
        # Crossing: TTC ≈ distance / vehicle_speed
        # (pedestrian is moving perpendicular, not directly closing gap)
        relative_speed = v_vehicle

    if relative_speed <= 0:
        return float("inf")

    ttc = detection_distance_m / relative_speed
    return ttc


def run_ncap_test(scenario, vehicle_speed_kmh, surface="dry",
                  vehicle_params=None, reaction_time=0.3):
    """
    Execute one Euro NCAP AEB test case.

    Pipeline:
    1. Compute when the system detects the pedestrian
    2. Apply system reaction time (sensor processing + decision delay)
    3. Run vehicle dynamics braking simulation
    4. Determine if vehicle stops before pedestrian location
    5. If not: compute residual impact speed → HIC → injury

    Args:
        scenario: NCAPScenario
        vehicle_speed_kmh: approach speed (km/h)
        surface: road surface condition
        vehicle_params: vehicle specs (default sedan)
        reaction_time: system processing delay (seconds)

    Returns:
        TestResult
    """
    if vehicle_params is None:
        vehicle_params = SEDAN_DEFAULT

    analyzer = ImpactAnalyzer()
    v_ms = vehicle_speed_kmh / 3.6

    # 1. Detection distance
    det_dist = compute_detection_distance(vehicle_speed_kmh, scenario)

    # 2. TTC at detection
    ttc = compute_ttc(vehicle_speed_kmh, scenario.ped_speed_kmh, det_dist, scenario)

    # 3. Reaction distance (system processing delay)
    reaction_dist = v_ms * reaction_time

    # 4. Available braking distance
    available_dist = max(0.0, det_dist - reaction_dist)

    # 5. Run dynamics simulation to get actual braking distance
    brake_dist = braking_distance(vehicle_speed_kmh, surface, vehicle_params=vehicle_params)

    # 6. Determine outcome
    if available_dist >= brake_dist:
        # Vehicle stops in time!
        outcome = "avoided"
        impact_speed = 0.0
    else:
        # Vehicle can't stop — compute impact speed
        impact_speed = analyzer.compute_residual_speed(
            vehicle_speed_kmh, brake_dist, available_dist
        )
        if impact_speed < 1.0:
            outcome = "avoided"
            impact_speed = 0.0
        elif impact_speed < vehicle_speed_kmh * 0.5:
            outcome = "partial"
        else:
            outcome = "collision"

    # 7. Speed reduction
    speed_reduction = ((vehicle_speed_kmh - impact_speed) / vehicle_speed_kmh * 100
                       if vehicle_speed_kmh > 0 else 0)

    # 8. HIC at impact
    hic = analyzer.compute_hic(impact_speed, scenario.ped_height_m)
    severity = analyzer.injury_severity(hic)

    # 9. Score (simplified Euro NCAP scoring)
    if outcome == "avoided":
        score = 1.0
    elif speed_reduction >= 50:
        score = speed_reduction / 100.0
    elif speed_reduction >= 10:
        score = speed_reduction / 200.0
    else:
        score = 0.0

    return TestResult(
        scenario_code=scenario.code,
        vehicle_speed_kmh=vehicle_speed_kmh,
        surface=surface,
        detection_distance_m=det_dist,
        ttc_at_detection_s=ttc,
        reaction_time_s=reaction_time,
        braking_distance_needed_m=brake_dist,
        available_braking_distance_m=available_dist,
        outcome=outcome,
        impact_speed_kmh=impact_speed,
        speed_reduction_pct=speed_reduction,
        hic=hic,
        ais_level=severity["ais_level"],
        injury_description=severity["description"],
        score=score,
    )


def run_full_test_matrix(surface="dry", vehicle_params=None):
    """
    Run ALL 4 scenarios × 9 speeds = 36 test cases.

    Returns:
        list of TestResult
    """
    results = []
    for scenario in ALL_SCENARIOS:
        for speed in scenario.test_speeds_kmh:
            result = run_ncap_test(scenario, speed, surface, vehicle_params)
            results.append(result)
    return results


def print_test_matrix(results):
    """Print formatted test matrix."""
    print(f"\n{'Scenario':<10} {'Speed':>7} {'Surface':>8} {'Detect':>8} {'Brake':>8} "
          f"{'Outcome':>10} {'Impact':>8} {'HIC':>8} {'Score':>7}")
    print("-" * 90)

    current_scenario = None
    for r in results:
        if r.scenario_code != current_scenario:
            if current_scenario is not None:
                print()
            current_scenario = r.scenario_code

        outcome_sym = {"avoided": "✓", "partial": "◐", "collision": "✗"}
        print(f"{r.scenario_code:<10} {r.vehicle_speed_kmh:>5.0f} km/h {r.surface:>8} "
              f"{r.detection_distance_m:>6.1f} m {r.braking_distance_needed_m:>6.1f} m "
              f"{outcome_sym.get(r.outcome, '?')} {r.outcome:>8} "
              f"{r.impact_speed_kmh:>6.1f} km/h {r.hic:>6.0f} {r.score:>6.0%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Euro NCAP AEB Test Runner")
    parser.add_argument("--surface", type=str, default="dry", help="Road surface")
    parser.add_argument("--test", action="store_true", help="Run full 36-test matrix")
    parser.add_argument("--scenario", type=str, help="Run single scenario (e.g., CPFA-50)")
    parser.add_argument("--speed", type=float, default=50.0, help="Test speed (km/h)")
    args = parser.parse_args()

    if args.test:
        print("\n" + "=" * 90)
        print("  Euro NCAP Full Test Matrix — " + args.surface.capitalize())
        print("=" * 90)
        results = run_full_test_matrix(args.surface)
        print_test_matrix(results)

        # Summary
        total_score = np.mean([r.score for r in results])
        avoided = sum(1 for r in results if r.outcome == "avoided")
        partial = sum(1 for r in results if r.outcome == "partial")
        collision = sum(1 for r in results if r.outcome == "collision")
        print(f"\nSummary: {avoided}✓ avoided | {partial}◐ partial | {collision}✗ collision")
        print(f"Overall score: {total_score:.0%}")
    else:
        from src.ncap_testing.ncap_scenarios import get_scenario
        scenario = get_scenario(args.scenario) if args.scenario else ALL_SCENARIOS[0]
        result = run_ncap_test(scenario, args.speed, args.surface)
        print(f"\n{result.scenario_code} at {result.vehicle_speed_kmh} km/h ({result.surface}):")
        print(f"  Detected at: {result.detection_distance_m:.1f} m")
        print(f"  Braking needed: {result.braking_distance_needed_m:.1f} m")
        print(f"  Available: {result.available_braking_distance_m:.1f} m")
        print(f"  Outcome: {result.outcome}")
        print(f"  Impact speed: {result.impact_speed_kmh:.1f} km/h")
        print(f"  HIC: {result.hic:.0f} ({result.injury_description})")
        print(f"  Score: {result.score:.0%}")
