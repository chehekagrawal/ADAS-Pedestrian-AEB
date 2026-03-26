"""
Pedestrian Impact Biomechanics — Head Injury Criterion (HIC) Model.

Calculates pedestrian injury severity when AEB fails to fully stop the vehicle.
Uses the Head Injury Criterion (HIC), which is THE standard metric used by:
- Euro NCAP (star rating for pedestrian protection)
- NHTSA (US safety standards)
- Every automotive OEM in the world

HIC Formula:
    HIC = max[(t2-t1) × (1/(t2-t1) × ∫a(t)dt)^2.5]

Simplified model uses a half-sine acceleration pulse during head impact.

References:
    - Euro NCAP Pedestrian Testing Protocol v9.0
    - NHTSA FMVSS 208 (Head Injury Criterion)
    - Simms & Wood, "Pedestrian and Cyclist Impact" (Springer)
"""

import numpy as np
from src.safety_analysis.vehicle_front_geometry import VehicleFrontGeometry, SEDAN_GEOMETRY


# ─── Injury Classification (AIS Scale) ───────────────────────────

AIS_THRESHOLDS = [
    (0, 150, 1, "Minor", "green"),
    (150, 500, 2, "Moderate", "yellow"),
    (500, 1000, 3, "Serious", "orange"),
    (1000, 1500, 4, "Severe", "red"),
    (1500, 2500, 5, "Critical", "darkred"),
    (2500, float("inf"), 6, "Fatal", "black"),
]


class ImpactAnalyzer:
    """
    Pedestrian-vehicle impact analysis using biomechanical models.
    """

    def __init__(self, vehicle_geometry=None):
        """
        Args:
            vehicle_geometry: VehicleFrontGeometry instance (default: sedan)
        """
        self.geometry = vehicle_geometry or SEDAN_GEOMETRY

    def compute_residual_speed(self, initial_speed_kmh, braking_distance_m,
                                available_distance_m):
        """
        If AEB brakes but can't stop in time, what impact speed remains?

        Uses energy method:
            v_impact² = v_initial² - 2 × a_avg × d_braked

        Where d_braked = min(available_distance, braking_distance)

        Args:
            initial_speed_kmh: vehicle speed when AEB triggers (km/h)
            braking_distance_m: total distance needed to stop (from dynamics sim)
            available_distance_m: actual distance to pedestrian

        Returns:
            residual_speed_kmh: speed at impact (0 if collision avoided)
        """
        v_init = initial_speed_kmh / 3.6  # m/s

        if available_distance_m >= braking_distance_m:
            return 0.0  # Vehicle stops in time!

        # Distance actually used for braking
        d_braked = available_distance_m

        # Average deceleration from energy method
        # v²=v0²-2*a*d → a = v0²/(2*d_total)
        if braking_distance_m > 0:
            a_avg = (v_init ** 2) / (2 * braking_distance_m)
        else:
            return initial_speed_kmh

        # Residual velocity after partial braking
        v_impact_sq = v_init ** 2 - 2 * a_avg * d_braked
        v_impact_sq = max(0.0, v_impact_sq)
        v_impact = np.sqrt(v_impact_sq)

        return v_impact * 3.6  # convert back to km/h

    def compute_wrap_around_distance(self, ped_height_m, impact_speed_kmh):
        """
        Wrap-Around Distance (WAD): where on the vehicle the pedestrian's
        head lands during impact.

        Simplified model based on Euro NCAP test methodology:
        WAD ≈ pedestrian_height × scaling_factor + speed_correction

        For adults (1.75m): head typically hits hood or windshield
        For children (1.15m): head typically hits front of hood

        Args:
            ped_height_m: pedestrian height in meters
            impact_speed_kmh: impact speed in km/h

        Returns:
            wad_m: wrap-around distance in meters
            impact_zone: str (where the head hits)
        """
        # Empirical WAD model (simplified from research data)
        # WAD increases with both pedestrian height and impact speed
        wad = ped_height_m * 0.72 + impact_speed_kmh * 0.003

        # Clamp to vehicle geometry
        wad = max(self.geometry.bumper_bottom_height, wad)
        wad = min(self.geometry.roof_height, wad)

        impact_zone = self.geometry.impact_zone(wad)

        return wad, impact_zone

    def compute_hic(self, impact_speed_kmh, ped_height_m=1.75):
        """
        Head Injury Criterion (HIC) calculation.

        Uses simplified half-sine acceleration pulse model:
        1. Determine where head hits (WAD → impact zone)
        2. Get local stiffness at that zone
        3. Compute impact duration from energy balance
        4. Compute peak head acceleration
        5. Calculate HIC from the acceleration pulse

        Simplified HIC formula for half-sine pulse:
            HIC = t_impact × (a_peak / g)^2.5

        Where:
            t_impact = contact duration (from stiffness & speed)
            a_peak = peak head acceleration (from impact dynamics)

        Args:
            impact_speed_kmh: vehicle speed at moment of impact
            ped_height_m: pedestrian height (1.75 for adult, 1.15 for child)

        Returns:
            hic_value: float (dimensionless, higher = more severe)
        """
        if impact_speed_kmh <= 0.5:
            return 0.0

        v_impact = impact_speed_kmh / 3.6  # m/s

        # Determine impact zone
        _, impact_zone = self.compute_wrap_around_distance(ped_height_m, impact_speed_kmh)

        # Local stiffness at impact zone
        k = self.geometry.stiffness_at_zone(impact_zone)

        # Head mass (approximate)
        head_mass = 4.5  # kg (average adult head mass)
        if ped_height_m < 1.30:
            head_mass = 3.5  # child

        # Deformation depth during impact
        # Energy: ½mv² = ½kδ² → δ = v√(m/k)
        deformation = v_impact * np.sqrt(head_mass / k)
        deformation = min(deformation, self.geometry.hood_deformation_depth)

        # Contact duration (half-sine model)
        # t_contact = π × √(m/k)
        t_contact = np.pi * np.sqrt(head_mass / k)
        t_contact = max(t_contact, 0.005)  # minimum 5ms

        # Peak head acceleration
        # For half-sine: a_peak = π × v / (2 × t_contact)
        a_peak = np.pi * v_impact / (2 * t_contact)

        # HIC calculation for half-sine pulse
        # HIC = t_contact × (a_peak / g)^2.5
        g = 9.81
        hic = t_contact * (a_peak / g) ** 2.5

        return float(hic)

    def injury_severity(self, hic_value):
        """
        Classify injury severity from HIC value.

        Uses AIS (Abbreviated Injury Scale) and Euro NCAP color zones.

        Returns:
            dict: {
                "hic": float,
                "ais_level": int (1-6),
                "description": str,
                "ncap_color": str,
                "survivable": bool
            }
        """
        for hic_min, hic_max, ais, desc, color in AIS_THRESHOLDS:
            if hic_min <= hic_value < hic_max:
                return {
                    "hic": hic_value,
                    "ais_level": ais,
                    "description": desc,
                    "ncap_color": color,
                    "survivable": ais <= 4,
                }

        return {
            "hic": hic_value,
            "ais_level": 6,
            "description": "Fatal",
            "ncap_color": "black",
            "survivable": False,
        }

    def full_impact_analysis(self, impact_speed_kmh, ped_height_m=1.75):
        """
        Complete impact analysis at a given speed.

        Returns:
            dict with all impact parameters
        """
        wad, zone = self.compute_wrap_around_distance(ped_height_m, impact_speed_kmh)
        hic = self.compute_hic(impact_speed_kmh, ped_height_m)
        severity = self.injury_severity(hic)

        return {
            "impact_speed_kmh": impact_speed_kmh,
            "ped_height_m": ped_height_m,
            "wrap_around_distance_m": wad,
            "impact_zone": zone,
            "hic": hic,
            **severity,
        }


def run_self_test():
    """Verify HIC calculations against expected values."""
    print("\n" + "=" * 60)
    print("  Impact Biomechanics — Self Test")
    print("=" * 60)

    analyzer = ImpactAnalyzer()

    # Test HIC at various speeds
    print(f"\n{'Speed (km/h)':>14} {'HIC':>8} {'AIS':>5} {'Severity':>12} {'Zone':>15}")
    print("-" * 60)

    speeds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    for speed in speeds:
        result = analyzer.full_impact_analysis(speed)
        print(f"{speed:>14} {result['hic']:>8.0f} {result['ais_level']:>5} "
              f"{result['description']:>12} {result['impact_zone']:>15}")

    # Test residual speed calculation
    print("\n--- Residual Speed Test ---")
    print(f"  50 km/h, brake dist=20m, available=30m → {analyzer.compute_residual_speed(50, 20, 30):.1f} km/h (should be 0)")
    print(f"  50 km/h, brake dist=20m, available=10m → {analyzer.compute_residual_speed(50, 20, 10):.1f} km/h (should be ~35)")
    print(f"  50 km/h, brake dist=20m, available=0m  → {analyzer.compute_residual_speed(50, 20, 0):.1f} km/h (should be 50)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pedestrian Impact Biomechanics Analysis")
    parser.add_argument("--speed", type=float, default=50.0, help="Impact speed (km/h)")
    parser.add_argument("--height", type=float, default=1.75, help="Pedestrian height (m)")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    if args.test:
        run_self_test()
    else:
        analyzer = ImpactAnalyzer()
        result = analyzer.full_impact_analysis(args.speed, args.height)
        print(f"\nImpact Analysis at {args.speed} km/h:")
        for k, v in result.items():
            print(f"  {k}: {v}")
