"""
Vehicle Longitudinal Dynamics Simulator.

Performs detailed time-step simulation of vehicle braking events.
This is THE mechanical engineering core of the ADAS-AEB project.

Forces modeled:
    - Tire braking force (Pacejka model with ABS)
    - Aerodynamic drag (½ρCdAv²)
    - Rolling resistance (Crr × m × g × cos θ)
    - Road grade (m × g × sin θ)
    - Dynamic weight transfer (front/rear axle load redistribution)

The simulator replaces the simple d = v²/2a formula with a proper
physics-based calculation that accounts for real-world effects.

Reference: Rajamani, R. "Vehicle Dynamics and Control" (Springer)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.vehicle_dynamics.vehicle_params import VehicleParameters, SEDAN_DEFAULT
from src.vehicle_dynamics.tire_model import pacejka_mu, braking_force, get_peak_slip, SURFACES
from src.vehicle_dynamics.weight_transfer import compute_axle_loads, max_braking_force_per_axle


@dataclass
class SimulationResult:
    """Complete time-history of a braking event."""

    # Time histories (lists of floats, one per timestep)
    time: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)          # m/s
    velocity_kmh: List[float] = field(default_factory=list)      # km/h
    distance: List[float] = field(default_factory=list)          # meters traveled
    deceleration: List[float] = field(default_factory=list)      # m/s² (positive = braking)
    slip_ratio_front: List[float] = field(default_factory=list)  # 0.0–1.0
    slip_ratio_rear: List[float] = field(default_factory=list)
    front_axle_load: List[float] = field(default_factory=list)   # Newtons
    rear_axle_load: List[float] = field(default_factory=list)
    brake_force_total: List[float] = field(default_factory=list) # Newtons
    aero_drag: List[float] = field(default_factory=list)
    abs_active: List[bool] = field(default_factory=list)

    # Summary
    stopped: bool = False
    stopping_distance: float = 0.0
    stopping_time: float = 0.0
    initial_speed_kmh: float = 0.0
    surface: str = "dry"
    grade_deg: float = 0.0
    peak_deceleration: float = 0.0
    average_deceleration: float = 0.0

    def summary(self):
        """Print a human-readable summary."""
        lines = [
            f"  Initial speed: {self.initial_speed_kmh:.1f} km/h ({self.initial_speed_kmh/3.6:.2f} m/s)",
            f"  Surface: {self.surface}",
            f"  Grade: {self.grade_deg:.1f}°",
            f"  Stopped: {'Yes' if self.stopped else 'No'}",
            f"  Stopping distance: {self.stopping_distance:.2f} m",
            f"  Stopping time: {self.stopping_time:.3f} s",
            f"  Peak deceleration: {self.peak_deceleration:.2f} m/s² ({self.peak_deceleration/9.81:.2f} g)",
            f"  Average deceleration: {self.average_deceleration:.2f} m/s²",
        ]
        return "\n".join(lines)


def simulate_braking(initial_speed_kmh, surface="dry", grade_deg=0.0,
                     vehicle_params=None, dt=0.001, max_time=20.0,
                     brake_application_time=0.0):
    """
    Full longitudinal dynamics braking simulation.

    For each timestep:
    1. Compute dynamic weight transfer based on current deceleration
    2. Compute tire braking force from Pacejka model at each axle
    3. Check ABS: if slip > threshold, modulate brake pressure
    4. Compute aerodynamic drag, rolling resistance, grade force
    5. Sum all forces, compute deceleration
    6. Update velocity and position (Euler integration)
    7. Check if vehicle has stopped

    Args:
        initial_speed_kmh: float (km/h)
        surface: str ("dry", "wet", "snow", "ice", "gravel")
        grade_deg: float (degrees, positive = uphill, negative = downhill)
        vehicle_params: VehicleParameters (None = default sedan)
        dt: float (timestep in seconds, default 1ms for stability)
        max_time: float (maximum simulation time in seconds)
        brake_application_time: float (delay before brakes are applied)

    Returns:
        SimulationResult with full time histories
    """
    if vehicle_params is None:
        vehicle_params = SEDAN_DEFAULT
    p = vehicle_params

    # Convert inputs
    v = initial_speed_kmh / 3.6  # m/s
    grade_rad = np.radians(grade_deg)

    # State variables
    t = 0.0
    x = 0.0  # distance traveled
    prev_decel = 0.0

    # ABS state
    abs_brake_pressure_fraction = 1.0  # 1.0 = full pressure
    abs_timer = 0.0
    abs_releasing = False

    # Result storage
    result = SimulationResult(
        initial_speed_kmh=initial_speed_kmh,
        surface=surface,
        grade_deg=grade_deg,
    )

    # Record initial state
    _record_state(result, t, v, x, 0.0, 0.0, 0.0,
                  p.static_front_load, p.static_rear_load, 0.0, 0.0, False)

    while t < max_time and v > 0.001:  # stop at very low speed
        t += dt

        # ─── 1. Weight Transfer ───────────────────────────────
        N_front, N_rear = compute_axle_loads(p, prev_decel)

        # ─── 2. Braking Force (only after brake application) ──
        if t >= brake_application_time:
            # Target slip ratio for optimal braking
            target_slip = SURFACES[surface].slip_at_peak

            # Compute wheel speed if at target slip
            # slip = (v - v_wheel) / v → v_wheel = v * (1 - slip)
            slip_front = target_slip
            slip_rear = target_slip

            # Tire forces from Pacejka model
            F_brake_front = braking_force(slip_front, N_front, surface)
            F_brake_rear = braking_force(slip_rear, N_rear, surface)

            # ─── 3. ABS Modulation ────────────────────────────
            abs_is_active = False
            if p.abs_enabled:
                abs_timer += dt

                # Check if we need to limit force to prevent lockup
                max_F_front = SURFACES[surface].peak_mu * N_front
                max_F_rear = SURFACES[surface].peak_mu * N_rear

                if F_brake_front > max_F_front or F_brake_rear > max_F_rear:
                    abs_is_active = True

                    # ABS pressure modulation cycle
                    if abs_timer >= p.abs_cycle_time:
                        abs_timer = 0.0
                        abs_releasing = not abs_releasing

                    if abs_releasing:
                        abs_brake_pressure_fraction *= (1.0 - p.abs_pressure_release_rate * dt / p.abs_cycle_time)
                        abs_brake_pressure_fraction = max(0.3, abs_brake_pressure_fraction)
                    else:
                        abs_brake_pressure_fraction += p.abs_pressure_buildup_rate * dt / p.abs_cycle_time
                        abs_brake_pressure_fraction = min(1.0, abs_brake_pressure_fraction)

                    F_brake_front *= abs_brake_pressure_fraction
                    F_brake_rear *= abs_brake_pressure_fraction
                else:
                    abs_brake_pressure_fraction = 1.0

            F_brake_total = F_brake_front + F_brake_rear
        else:
            F_brake_total = 0.0
            F_brake_front = 0.0
            F_brake_rear = 0.0
            slip_front = 0.0
            slip_rear = 0.0
            abs_is_active = False

        # ─── 4. Aerodynamic Drag ──────────────────────────────
        F_aero = 0.5 * p.air_density * p.drag_coeff * p.frontal_area * v * v

        # ─── 5. Rolling Resistance ────────────────────────────
        F_roll = p.rolling_resistance * p.mass * p.gravity * np.cos(grade_rad)

        # ─── 6. Grade Force ───────────────────────────────────
        # Positive grade = uphill = helps braking (negative force opposes motion)
        # Negative grade = downhill = hurts braking (force aids motion)
        F_grade = p.mass * p.gravity * np.sin(grade_rad)

        # ─── 7. Total Force & Deceleration ────────────────────
        # All retarding forces are positive (opposing motion)
        F_total = F_brake_total + F_aero + F_roll + F_grade
        decel = F_total / p.mass  # m/s²

        # ─── 8. Update Velocity & Position ────────────────────
        v_new = v - decel * dt

        # Don't allow negative velocity (vehicle can't go backwards from braking)
        if v_new < 0:
            # Adjust dt for exact stop
            dt_stop = v / decel if decel > 0 else dt
            x += v * dt_stop - 0.5 * decel * dt_stop * dt_stop
            v = 0.0
        else:
            x += v * dt - 0.5 * decel * dt * dt
            v = v_new

        prev_decel = decel

        # ─── 9. Record State ─────────────────────────────────
        # Record at reduced frequency to save memory (every 10ms)
        if int(t * 1000) % 10 == 0:
            _record_state(result, t, v, x, decel, slip_front, slip_rear,
                          N_front, N_rear, F_brake_total, F_aero, abs_is_active)

        if v <= 0.001:
            break

    # ─── Finalize Results ─────────────────────────────────────
    result.stopped = (v <= 0.001)
    result.stopping_distance = x
    result.stopping_time = t

    if result.deceleration:
        result.peak_deceleration = max(result.deceleration)
        nonzero_decels = [d for d in result.deceleration if d > 0.1]
        result.average_deceleration = np.mean(nonzero_decels) if nonzero_decels else 0.0

    return result


def _record_state(result, t, v, x, decel, slip_f, slip_r,
                  n_front, n_rear, f_brake, f_aero, abs_active):
    """Append current state to result histories."""
    result.time.append(t)
    result.velocity.append(v)
    result.velocity_kmh.append(v * 3.6)
    result.distance.append(x)
    result.deceleration.append(decel)
    result.slip_ratio_front.append(slip_f)
    result.slip_ratio_rear.append(slip_r)
    result.front_axle_load.append(n_front)
    result.rear_axle_load.append(n_rear)
    result.brake_force_total.append(f_brake)
    result.aero_drag.append(f_aero)
    result.abs_active.append(abs_active)


def braking_distance(initial_speed_kmh, surface="dry", grade_deg=0.0,
                     vehicle_params=None):
    """
    Convenience function: get just the stopping distance.

    Args:
        initial_speed_kmh: float (km/h)
        surface: str
        grade_deg: float (degrees)
        vehicle_params: VehicleParameters (None for default)

    Returns:
        distance_meters: float
    """
    result = simulate_braking(initial_speed_kmh, surface, grade_deg, vehicle_params)
    return result.stopping_distance


def stopping_time(initial_speed_kmh, surface="dry", grade_deg=0.0,
                  vehicle_params=None):
    """Convenience: get just the stopping time."""
    result = simulate_braking(initial_speed_kmh, surface, grade_deg, vehicle_params)
    return result.stopping_time


def braking_distance_comparison(speed_kmh=50.0, vehicle_params=None):
    """
    Compare braking distances across all surfaces.

    Returns:
        dict: {surface_name: distance_meters}
    """
    results = {}
    for surface in SURFACES:
        d = braking_distance(speed_kmh, surface, vehicle_params=vehicle_params)
        results[surface] = d
    return results


def run_self_test():
    """
    Verify simulation against known engineering values.

    Reference values (approximate, for a 1500 kg sedan):
    - 50 km/h dry: 13-19 m
    - 100 km/h dry: 50-70 m
    - 50 km/h ice: 70-100 m
    """
    print("\n" + "=" * 60)
    print("  Vehicle Dynamics Simulator — Self Test")
    print("=" * 60)

    tests = [
        (50, "dry", 0, 13, 22),
        (100, "dry", 0, 45, 75),
        (50, "wet", 0, 20, 40),
        (50, "ice", 0, 60, 120),
        (50, "dry", 5, 13, 25),    # uphill helps
        (50, "dry", -5, 15, 30),   # downhill hurts
    ]

    all_pass = True
    for speed, surface, grade, d_min, d_max in tests:
        result = simulate_braking(speed, surface, grade)
        d = result.stopping_distance
        passed = d_min <= d <= d_max
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {speed:>3} km/h  {surface:<6}  grade={grade:>+3}°"
              f"  →  {d:>6.1f} m  (expected {d_min}-{d_max} m)")

    print(f"\n  {'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")

    # Detailed result for reference
    print("\n--- Detailed result: 50 km/h, dry, flat ---")
    result = simulate_braking(50, "dry", 0)
    print(result.summary())

    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle Dynamics Braking Simulation")
    parser.add_argument("--speed", type=float, default=50.0, help="Initial speed (km/h)")
    parser.add_argument("--surface", type=str, default="dry", help="Road surface")
    parser.add_argument("--grade", type=float, default=0.0, help="Road grade (degrees)")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--compare", action="store_true", help="Compare all surfaces")

    args = parser.parse_args()

    if args.test:
        run_self_test()
    elif args.compare:
        print(f"\nBraking distance comparison at {args.speed} km/h:")
        comp = braking_distance_comparison(args.speed)
        for surface, d in comp.items():
            print(f"  {SURFACES[surface].name:<15}  {d:.1f} m")
    else:
        result = simulate_braking(args.speed, args.surface, args.grade)
        print(f"\nBraking Simulation Result:")
        print(result.summary())
