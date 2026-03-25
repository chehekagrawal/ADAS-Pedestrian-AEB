"""
Engineering Plots for Vehicle Dynamics Module.

Generates publication-quality plots for the ADAS project report:
- Velocity vs. time during braking
- Deceleration profile
- ABS slip oscillation
- Pacejka tire curves for all surfaces
- Braking distance comparison across surfaces
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots

# Consistent styling
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
}
plt.rcParams.update(PLOT_STYLE)

SURFACE_COLORS = {
    "dry": "#2ecc71",
    "wet": "#3498db",
    "snow": "#9b59b6",
    "ice": "#e74c3c",
    "gravel": "#e67e22",
}


def plot_velocity_vs_time(result, save_path):
    """
    Plot velocity (km/h) vs time (s) during a braking event.
    Shows the deceleration curve shape.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.time, result.velocity_kmh, "b-", linewidth=2.5)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Velocity (km/h)", fontsize=13)
    ax.set_title(f"Braking from {result.initial_speed_kmh:.0f} km/h — {result.surface.capitalize()} Surface",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    # Annotate stopping point
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"Stopped at {result.stopping_distance:.1f} m\nin {result.stopping_time:.2f} s",
                xy=(result.stopping_time, 0), fontsize=11,
                xytext=(result.stopping_time * 0.5, result.initial_speed_kmh * 0.3),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_deceleration_profile(result, save_path):
    """
    Plot deceleration (m/s²) vs time.
    Shows ABS oscillation pattern and the actual braking force over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.time, result.deceleration, "r-", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Deceleration (m/s²)", fontsize=13)
    ax.set_title(f"Deceleration Profile — {result.surface.capitalize()} Surface", fontsize=14, fontweight="bold")
    ax.set_xlim(left=0)

    # Show peak deceleration
    if result.peak_deceleration > 0:
        ax.axhline(y=result.peak_deceleration, color="orange", linestyle="--", alpha=0.7,
                   label=f"Peak: {result.peak_deceleration:.2f} m/s² ({result.peak_deceleration/9.81:.2f} g)")
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_abs_slip(result, save_path):
    """
    Plot tire slip ratio vs time.
    ABS keeps slip oscillating around the optimal range (10-15%).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.time, [s * 100 for s in result.slip_ratio_front],
            "b-", linewidth=1.5, label="Front Axle", alpha=0.8)
    ax.plot(result.time, [s * 100 for s in result.slip_ratio_rear],
            "r-", linewidth=1.5, label="Rear Axle", alpha=0.8)

    # Optimal slip zone
    from src.vehicle_dynamics.tire_model import SURFACES
    peak_slip = SURFACES[result.surface].slip_at_peak * 100
    ax.axhline(y=peak_slip, color="green", linestyle="--", alpha=0.7,
               label=f"Peak grip slip ({peak_slip:.0f}%)")
    ax.axhspan(peak_slip - 3, peak_slip + 3, alpha=0.1, color="green",
               label="Optimal ABS zone")

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Slip Ratio (%)", fontsize=13)
    ax.set_title(f"Tire Slip Ratio — ABS Active on {result.surface.capitalize()}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 30)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pacejka_curves(save_path):
    """
    Plot the Pacejka Magic Formula: slip ratio vs braking force coefficient
    for all surface conditions on one chart.

    This is THE characteristic tire plot used in automotive engineering.
    """
    from src.vehicle_dynamics.tire_model import pacejka_mu, SURFACES

    fig, ax = plt.subplots(figsize=(10, 7))
    slips = np.linspace(0.001, 0.50, 500)

    for surface_key, params in SURFACES.items():
        mus = pacejka_mu(slips, surface_key)
        color = SURFACE_COLORS.get(surface_key, "gray")
        ax.plot(slips * 100, mus, color=color, linewidth=2.5,
                label=f"{params.name} (μ_peak={params.peak_mu:.2f})")

        # Mark peak
        idx_peak = np.argmax(mus)
        ax.plot(slips[idx_peak] * 100, mus[idx_peak], "o", color=color,
                markersize=8, markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel("Slip Ratio (%)", fontsize=13)
    ax.set_ylabel("Friction Coefficient μ", fontsize=13)
    ax.set_title("Pacejka Magic Formula — Longitudinal Braking Force", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.0)

    # Annotation
    ax.annotate("ABS keeps tires\nin this zone →",
                xy=(12, 0.82), fontsize=10, color="gray", fontstyle="italic",
                ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_braking_distance_comparison(speeds=None, save_path=""):
    """
    Bar chart comparing stopping distance across surfaces at multiple speeds.
    """
    from src.vehicle_dynamics.dynamics_sim import braking_distance
    from src.vehicle_dynamics.tire_model import SURFACES

    if speeds is None:
        speeds = [30, 50, 80, 100]

    surfaces = list(SURFACES.keys())
    fig, ax = plt.subplots(figsize=(12, 7))

    bar_width = 0.15
    x = np.arange(len(speeds))

    for i, surface in enumerate(surfaces):
        distances = [braking_distance(s, surface) for s in speeds]
        color = SURFACE_COLORS.get(surface, "gray")
        bars = ax.bar(x + i * bar_width, distances, bar_width,
                      label=SURFACES[surface].name, color=color, edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, d in zip(bars, distances):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{d:.0f}m", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Initial Speed (km/h)", fontsize=13)
    ax.set_ylabel("Stopping Distance (m)", fontsize=13)
    ax.set_title("Braking Distance Comparison — All Surfaces", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width * (len(surfaces) - 1) / 2)
    ax.set_xticklabels([f"{s} km/h" for s in speeds], fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_weight_transfer(result, save_path):
    """Plot front and rear axle loads over time during braking."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(result.time, [n / 1000 for n in result.front_axle_load],
            "b-", linewidth=2, label="Front Axle Load")
    ax.plot(result.time, [n / 1000 for n in result.rear_axle_load],
            "r-", linewidth=2, label="Rear Axle Load")

    # Static loads
    from src.vehicle_dynamics.vehicle_params import SEDAN_DEFAULT
    p = SEDAN_DEFAULT
    ax.axhline(y=p.static_front_load / 1000, color="blue", linestyle="--", alpha=0.4, label="Static Front")
    ax.axhline(y=p.static_rear_load / 1000, color="red", linestyle="--", alpha=0.4, label="Static Rear")

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Axle Load (kN)", fontsize=13)
    ax.set_title("Dynamic Weight Transfer During Braking", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_plots(save_dir="results/vehicle_dynamics/"):
    """Generate all engineering plots for the report."""
    from src.vehicle_dynamics.dynamics_sim import simulate_braking

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating vehicle dynamics plots in {save_dir}...")

    # 1. Pacejka curves (no simulation needed)
    plot_pacejka_curves(os.path.join(save_dir, "pacejka_curves.png"))

    # 2. Braking distance comparison
    plot_braking_distance_comparison(
        speeds=[30, 50, 80, 100],
        save_path=os.path.join(save_dir, "braking_distance_comparison.png")
    )

    # 3. Detailed plots for 50 km/h on dry
    result_dry = simulate_braking(50, "dry")
    plot_velocity_vs_time(result_dry, os.path.join(save_dir, "velocity_vs_time_dry.png"))
    plot_deceleration_profile(result_dry, os.path.join(save_dir, "deceleration_dry.png"))
    plot_abs_slip(result_dry, os.path.join(save_dir, "abs_slip_dry.png"))
    plot_weight_transfer(result_dry, os.path.join(save_dir, "weight_transfer_dry.png"))

    # 4. Wet surface for comparison
    result_wet = simulate_braking(50, "wet")
    plot_velocity_vs_time(result_wet, os.path.join(save_dir, "velocity_vs_time_wet.png"))

    # 5. Ice for extreme comparison
    result_ice = simulate_braking(50, "ice")
    plot_velocity_vs_time(result_ice, os.path.join(save_dir, "velocity_vs_time_ice.png"))

    print(f"\nAll vehicle dynamics plots generated in {save_dir}")


if __name__ == "__main__":
    generate_all_plots()
