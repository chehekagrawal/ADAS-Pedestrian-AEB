"""
Weight Transfer During Braking.

When a vehicle brakes, inertial forces cause weight (normal load) to
transfer from the rear axle to the front axle. This changes the
available grip at each axle and determines the maximum braking force.

Key equations:
    N_f = (m*g*l_r + m*a*h) / L     (front axle — load INCREASES)
    N_r = (m*g*l_f - m*a*h) / L     (rear axle — load DECREASES)

Where:
    m = vehicle mass
    g = gravitational acceleration
    l_f, l_r = CG distances to front and rear axles
    h = CG height above ground
    L = wheelbase
    a = deceleration (positive = braking)

Reference: Rajamani, R. "Vehicle Dynamics and Control", Chapter 4
"""

import numpy as np


def compute_axle_loads(vehicle_params, deceleration):
    """
    Compute dynamic axle loads during braking.

    During braking, weight shifts forward:
    - Front axle load INCREASES → more grip available
    - Rear axle load DECREASES → less grip, risk of lockup

    Args:
        vehicle_params: VehicleParameters instance
        deceleration: float (m/s², positive = braking)

    Returns:
        (N_front, N_rear): tuple of normal forces in Newtons
    """
    p = vehicle_params
    decel = abs(deceleration)  # ensure positive

    N_front = (p.mass * p.gravity * p.cg_to_rear +
               p.mass * decel * p.cg_height) / p.wheelbase

    N_rear = (p.mass * p.gravity * p.cg_to_front -
              p.mass * decel * p.cg_height) / p.wheelbase

    # Rear load cannot go negative (rear wheels lift off ground)
    N_rear = max(0.0, N_rear)

    return N_front, N_rear


def max_deceleration_before_rear_lift(vehicle_params):
    """
    Deceleration at which rear wheels completely unload (lift off).

    Occurs when: m*a*h = m*g*l_f
    Therefore: a = g * l_f / h

    This is the theoretical maximum deceleration before the car
    starts to pitch forward onto its nose.

    Returns:
        a_lift: float (m/s²)
    """
    p = vehicle_params
    return p.gravity * p.cg_to_front / p.cg_height


def weight_transfer_fraction(vehicle_params, deceleration):
    """
    What fraction of total weight is on the front axle during braking.

    Static: typically 55-60% front
    Under hard braking: can reach 75-80% front

    Returns:
        front_fraction: float (0.0 to 1.0)
    """
    N_f, N_r = compute_axle_loads(vehicle_params, deceleration)
    total = N_f + N_r
    if total <= 0:
        return 0.5
    return N_f / total


def max_braking_force_per_axle(vehicle_params, deceleration, surface="dry"):
    """
    Maximum braking force each axle can produce, limited by tire grip.

    F_max = μ_peak × N_axle

    The ABS system ensures that the applied brake force never
    exceeds this limit (preventing wheel lockup).

    Args:
        vehicle_params: VehicleParameters
        deceleration: float (current deceleration for weight transfer calc)
        surface: str (road surface type)

    Returns:
        (F_max_front, F_max_rear): max braking force in Newtons per axle
    """
    from src.vehicle_dynamics.tire_model import SURFACES

    N_front, N_rear = compute_axle_loads(vehicle_params, deceleration)
    mu_peak = SURFACES[surface].peak_mu

    F_max_front = mu_peak * N_front
    F_max_rear = mu_peak * N_rear

    return F_max_front, F_max_rear


def optimal_brake_distribution(vehicle_params, deceleration, surface="dry"):
    """
    Compute the ideal front/rear brake force distribution.

    For maximum deceleration, brake force should be distributed
    proportional to the dynamic axle loads.

    In reality, most vehicles have a fixed bias (e.g., 65% front).
    This function computes what the IDEAL distribution would be.

    Returns:
        (front_fraction, rear_fraction): ideal brake bias
    """
    N_f, N_r = compute_axle_loads(vehicle_params, deceleration)
    total = N_f + N_r
    if total <= 0:
        return 0.65, 0.35

    return N_f / total, N_r / total


def print_weight_transfer_table(vehicle_params):
    """Print weight transfer at different deceleration levels."""
    print(f"\n{'Deceleration':>14} {'Front Load':>12} {'Rear Load':>12} {'Front %':>10} {'Rear Lift?':>12}")
    print("-" * 65)

    a_lift = max_deceleration_before_rear_lift(vehicle_params)

    for decel in [0, 2, 4, 6, 8, 10]:
        N_f, N_r = compute_axle_loads(vehicle_params, decel)
        frac = weight_transfer_fraction(vehicle_params, decel)
        lift = "YES!" if decel >= a_lift else "No"
        print(f"{decel:>12.1f} m/s² {N_f:>10.0f} N {N_r:>10.0f} N {frac:>9.1%} {lift:>12}")

    print(f"\nRear lift-off deceleration: {a_lift:.2f} m/s² ({a_lift / 9.81:.2f} g)")


if __name__ == "__main__":
    from src.vehicle_dynamics.vehicle_params import VehicleParameters

    params = VehicleParameters()
    print("=" * 65)
    print("  Weight Transfer Analysis — Default Sedan")
    print("=" * 65)
    print_weight_transfer_table(params)
