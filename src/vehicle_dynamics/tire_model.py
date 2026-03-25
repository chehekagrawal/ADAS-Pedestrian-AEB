"""
Pacejka "Magic Formula" Tire Model — Longitudinal (Braking) Forces.

The Pacejka Magic Formula is THE industry-standard tire model used in
CarSim, Adams, IPG CarMaker, and every major vehicle dynamics simulator.

This implements the simplified longitudinal-only version for braking force
as a function of slip ratio and road surface conditions.

Reference:
    Pacejka, H.B. "Tire and Vehicle Dynamics" (Butterworth-Heinemann)
    Formula: F = D * sin(C * arctan(B*s - E*(B*s - arctan(B*s))))
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SurfaceParameters:
    """Tire-road friction parameters for a specific surface condition."""
    name: str
    peak_mu: float          # Peak friction coefficient
    B: float                # Stiffness factor
    C: float                # Shape factor
    D_scale: float          # Peak factor (= peak_mu for normalized)
    E: float                # Curvature factor
    slip_at_peak: float     # Approximate slip ratio at peak force


# ─── Surface Definitions ──────────────────────────────────────────
# Parameters calibrated to match published tire test data

SURFACES = {
    "dry": SurfaceParameters(
        name="Dry Asphalt",
        peak_mu=0.85, B=10.0, C=1.9, D_scale=0.85, E=0.97,
        slip_at_peak=0.12
    ),
    "wet": SurfaceParameters(
        name="Wet Asphalt",
        peak_mu=0.55, B=12.0, C=2.3, D_scale=0.55, E=1.0,
        slip_at_peak=0.14
    ),
    "snow": SurfaceParameters(
        name="Packed Snow",
        peak_mu=0.25, B=5.0, C=2.0, D_scale=0.25, E=1.0,
        slip_at_peak=0.20
    ),
    "ice": SurfaceParameters(
        name="Ice",
        peak_mu=0.12, B=4.0, C=2.0, D_scale=0.12, E=1.0,
        slip_at_peak=0.08
    ),
    "gravel": SurfaceParameters(
        name="Gravel",
        peak_mu=0.35, B=6.0, C=2.0, D_scale=0.35, E=1.0,
        slip_at_peak=0.18
    ),
}


def pacejka_mu(slip_ratio, surface="dry"):
    """
    Compute friction coefficient using Pacejka Magic Formula.

    The Magic Formula:
        μ(s) = D * sin(C * arctan(B*s - E*(B*s - arctan(B*s))))

    Where:
        s = slip ratio (0.0 = free rolling, 1.0 = locked wheel)
        B = stiffness factor (controls initial slope)
        C = shape factor (controls peak width)
        D = peak factor (= peak friction coefficient)
        E = curvature factor (controls post-peak drop)

    Args:
        slip_ratio: float or array (0.0 to 1.0)
        surface: str key from SURFACES dict

    Returns:
        mu: friction coefficient (same shape as slip_ratio)
    """
    if surface not in SURFACES:
        raise ValueError(f"Unknown surface '{surface}'. Available: {list(SURFACES.keys())}")

    p = SURFACES[surface]
    s = np.abs(np.asarray(slip_ratio, dtype=float))

    # Clamp slip to avoid numerical issues
    s = np.clip(s, 0.0, 1.0)

    Bs = p.B * s
    mu = p.D_scale * np.sin(p.C * np.arctan(Bs - p.E * (Bs - np.arctan(Bs))))

    return np.abs(mu)


def braking_force(slip_ratio, normal_force, surface="dry"):
    """
    Compute tire braking force in Newtons.

    F_brake = μ(slip) × N

    Args:
        slip_ratio: float (0.0 to 1.0)
        normal_force: float (Newtons, vertical load on tire)
        surface: str surface type

    Returns:
        force: float (Newtons, braking force)
    """
    mu = pacejka_mu(slip_ratio, surface)
    return float(mu * normal_force)


def get_peak_slip(surface="dry"):
    """
    Find the slip ratio at which maximum braking force occurs.

    This is the optimal operating point for ABS — keep the tire
    at peak slip for maximum braking effectiveness.

    Args:
        surface: str surface type

    Returns:
        (peak_slip, peak_mu): tuple of optimal slip and corresponding friction
    """
    slips = np.linspace(0.001, 0.60, 2000)
    mus = pacejka_mu(slips, surface)
    idx = np.argmax(mus)
    return float(slips[idx]), float(mus[idx])


def locked_wheel_mu(surface="dry"):
    """
    Friction coefficient when wheel is fully locked (slip = 1.0).
    Always lower than peak — this is why ABS prevents wheel lockup.

    Returns:
        mu_locked: float
    """
    return float(pacejka_mu(1.0, surface))


def max_deceleration(surface="dry", gravity=9.81):
    """
    Maximum possible deceleration on a given surface (m/s²).

    a_max = μ_peak × g

    This is the theoretical limit, assuming perfect ABS and
    no weight transfer effects.

    Returns:
        a_max: float (m/s²)
    """
    _, peak_mu = get_peak_slip(surface)
    return peak_mu * gravity


def print_surface_comparison():
    """Print a summary table of all surface parameters."""
    print(f"\n{'Surface':<15} {'Peak μ':>8} {'Slip @peak':>12} {'Locked μ':>10} {'Max Decel':>12}")
    print("-" * 60)
    for key, params in SURFACES.items():
        peak_s, peak_m = get_peak_slip(key)
        locked_m = locked_wheel_mu(key)
        max_a = max_deceleration(key)
        print(f"{params.name:<15} {peak_m:>8.3f} {peak_s:>11.1%} {locked_m:>10.3f} {max_a:>10.2f} m/s²")


if __name__ == "__main__":
    print("=" * 60)
    print("  Pacejka Tire Model — Surface Comparison")
    print("=" * 60)
    print_surface_comparison()

    print("\nPeak slip analysis:")
    for surface in SURFACES:
        peak_s, peak_mu = get_peak_slip(surface)
        print(f"  {surface}: peak at {peak_s:.1%} slip → μ = {peak_mu:.3f}")
