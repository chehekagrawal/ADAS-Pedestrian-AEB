"""
Euro NCAP AEB-Pedestrian Test Scenario Definitions.

Defines the 4 standard Euro NCAP car-to-pedestrian scenarios used to
rate AEB systems worldwide. These are the exact same tests that
every car manufacturer (BMW, Toyota, Tesla, etc.) must pass.

Scenarios:
    CPFA — Car-to-Pedestrian Farside Adult (50% overlap)
    CPNA — Car-to-Pedestrian Nearside Adult (25% overlap)
    CPNC — Car-to-Pedestrian Nearside Child (behind parked cars)
    CPLA — Car-to-Pedestrian Longitudinal Adult (walking ahead)

Reference: Euro NCAP AEB Pedestrian Testing Protocol v4.3 (2023)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class NCAPScenario:
    """Complete specification of one Euro NCAP AEB test scenario."""

    # Identification
    code: str                       # e.g., "CPFA-50"
    full_name: str                  # e.g., "Car-to-Pedestrian Farside Adult 50%"
    description: str                # Human-readable description

    # Pedestrian parameters
    ped_speed_kmh: float            # Walking/running speed (km/h)
    ped_height_m: float             # Pedestrian height (meters)
    ped_direction: str              # "crossing_farside", "crossing_nearside", "longitudinal"

    # Test geometry
    overlap_percent: float          # Vehicle-pedestrian overlap (25% or 50%)
    is_obstructed: bool             # View obstructed by parked car?
    detection_range_m: float        # Max distance at which system can detect ped

    # Test speeds (Euro NCAP standard range)
    test_speeds_kmh: List[float] = field(
        default_factory=lambda: [20, 25, 30, 35, 40, 45, 50, 55, 60]
    )

    # Timing
    ped_trigger_ttc: float = 0.0    # TTC at which pedestrian starts crossing

    @property
    def ped_speed_ms(self):
        return self.ped_speed_kmh / 3.6


# ─── Standard Euro NCAP Scenarios ─────────────────────────────────

CPFA = NCAPScenario(
    code="CPFA-50",
    full_name="Car-to-Pedestrian Farside Adult 50%",
    description=(
        "Adult pedestrian (1.75m) crosses from the far side of the road "
        "at 5 km/h walking speed, with 50% vehicle overlap. "
        "Unobstructed view — the system should detect early."
    ),
    ped_speed_kmh=5.0,
    ped_height_m=1.75,
    ped_direction="crossing_farside",
    overlap_percent=50.0,
    is_obstructed=False,
    detection_range_m=45.0,
)

CPNA = NCAPScenario(
    code="CPNA-25",
    full_name="Car-to-Pedestrian Nearside Adult 25%",
    description=(
        "Adult pedestrian (1.75m) crosses from the near side at 5 km/h, "
        "but with only 25% vehicle overlap. This tests the system's ability "
        "to detect pedestrians at the edge of the vehicle's path."
    ),
    ped_speed_kmh=5.0,
    ped_height_m=1.75,
    ped_direction="crossing_nearside",
    overlap_percent=25.0,
    is_obstructed=False,
    detection_range_m=40.0,
)

CPNC = NCAPScenario(
    code="CPNC-50",
    full_name="Car-to-Pedestrian Nearside Child 50%",
    description=(
        "HARDEST SCENARIO. A child (1.15m) runs out from behind a parked car "
        "at 5 km/h. The system has very little time to react because the child "
        "is hidden until the last moment. Tests AEB under worst-case conditions."
    ),
    ped_speed_kmh=5.0,
    ped_height_m=1.15,
    ped_direction="crossing_nearside",
    overlap_percent=50.0,
    is_obstructed=True,
    detection_range_m=20.0,  # Much shorter — child appears late
)

CPLA = NCAPScenario(
    code="CPLA",
    full_name="Car-to-Pedestrian Longitudinal Adult",
    description=(
        "Adult pedestrian (1.75m) walking in the same direction as the vehicle "
        "at 5 km/h (longitudinal). Tests AEB for rear-end approach scenarios "
        "where the pedestrian is walking away from the vehicle."
    ),
    ped_speed_kmh=5.0,
    ped_height_m=1.75,
    ped_direction="longitudinal",
    overlap_percent=50.0,
    is_obstructed=False,
    detection_range_m=50.0,
)

# All scenarios for batch testing
ALL_SCENARIOS = [CPFA, CPNA, CPNC, CPLA]


def get_scenario(code):
    """Look up a scenario by its code."""
    for s in ALL_SCENARIOS:
        if s.code == code:
            return s
    raise ValueError(f"Unknown scenario code: {code}")


def print_scenario_table():
    """Print a summary of all scenarios."""
    print(f"\n{'Code':<10} {'Ped':>5} {'Speed':>7} {'Overlap':>8} {'Blocked':>8} {'Detect':>8}")
    print("-" * 50)
    for s in ALL_SCENARIOS:
        ped_type = "Child" if s.ped_height_m < 1.30 else "Adult"
        print(f"{s.code:<10} {ped_type:>5} {s.ped_speed_kmh:>5.0f} km/h {s.overlap_percent:>6.0f}% "
              f"{'Yes' if s.is_obstructed else 'No':>8} {s.detection_range_m:>6.0f} m")


if __name__ == "__main__":
    print("=" * 50)
    print("  Euro NCAP AEB-Pedestrian Test Scenarios")
    print("=" * 50)
    print_scenario_table()
    for s in ALL_SCENARIOS:
        print(f"\n{s.code}: {s.description}")
