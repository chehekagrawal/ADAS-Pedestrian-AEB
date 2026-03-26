"""
Safety Analysis Plots.

Generates publication-quality figures for the pedestrian safety analysis:
- HIC vs. impact speed curve
- AEB effectiveness comparison chart
- Injury severity zones (Euro NCAP color coding)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.safety_analysis.impact_model import ImpactAnalyzer, AIS_THRESHOLDS
from src.safety_analysis.aeb_effectiveness import evaluate_aeb_benefit, compute_hic_reduction


def plot_hic_vs_speed(save_path):
    """
    Plot HIC value vs. impact speed with injury severity zones.
    This is a key figure for the report.
    """
    analyzer = ImpactAnalyzer()
    speeds = np.linspace(0, 80, 200)
    hics = [analyzer.compute_hic(s) for s in speeds]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color zones for injury severity
    zone_colors = [
        (0, 150, "#2ecc71", "Minor (AIS 1)"),
        (150, 500, "#f1c40f", "Moderate (AIS 2)"),
        (500, 1000, "#e67e22", "Serious (AIS 3)"),
        (1000, 1500, "#e74c3c", "Severe (AIS 4)"),
        (1500, 2500, "#c0392b", "Critical (AIS 5)"),
    ]

    max_hic = max(hics) * 1.1
    for hic_min, hic_max, color, label in zone_colors:
        if hic_min < max_hic:
            ax.axhspan(hic_min, min(hic_max, max_hic), alpha=0.15, color=color, label=label)

    # Main curve
    ax.plot(speeds, hics, "k-", linewidth=3, label="HIC (Adult, 1.75m)")

    # Key reference points
    for ref_speed in [20, 40, 60]:
        hic_val = analyzer.compute_hic(ref_speed)
        ax.plot(ref_speed, hic_val, "ko", markersize=8)
        ax.annotate(f"HIC={hic_val:.0f}", xy=(ref_speed, hic_val),
                    xytext=(ref_speed + 3, hic_val + max_hic * 0.05),
                    fontsize=10, fontweight="bold")

    ax.set_xlabel("Impact Speed (km/h)", fontsize=13)
    ax.set_ylabel("Head Injury Criterion (HIC)", fontsize=13)
    ax.set_title("HIC vs. Impact Speed — Pedestrian Head Injury Risk", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, max_hic)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_aeb_effectiveness(initial_speed=50.0, save_path=""):
    """
    Bar chart comparing HIC at different TTC thresholds.
    Shows how earlier AEB intervention reduces injury severity.
    """
    results = evaluate_aeb_benefit(initial_speed)

    labels = [r["label"] for r in results]
    hics = [r["hic"] for r in results]
    colors_list = [r["ncap_color"] for r in results]

    # Map ncap colors to matplotlib colors
    color_map = {
        "green": "#2ecc71", "yellow": "#f1c40f", "orange": "#e67e22",
        "red": "#e74c3c", "darkred": "#c0392b", "black": "#2c3e50"
    }
    bar_colors = [color_map.get(c, "#95a5a6") for c in colors_list]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(labels)), hics, color=bar_colors,
                  edgecolor="black", linewidth=0.8, width=0.6)

    # Value labels
    for bar, hic_val, result in zip(bars, hics, results):
        label_text = f"HIC: {hic_val:.0f}"
        if result["collision_avoided"]:
            label_text = "AVOIDED"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(hics) * 0.02,
                label_text, ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Impact speed below bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f"{result['impact_speed_kmh']:.0f} km/h",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Severity threshold lines
    ax.axhline(y=650, color="orange", linestyle="--", alpha=0.5, label="Euro NCAP limit (HIC=650)")
    ax.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="Serious injury (HIC=1000)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, rotation=15)
    ax.set_ylabel("Head Injury Criterion (HIC)", fontsize=13)
    ax.set_title(f"AEB Effectiveness at {initial_speed:.0f} km/h — Injury Severity Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_hic_reduction_chart(initial_speed=50.0, save_path=""):
    """
    Bar chart showing % HIC reduction at each TTC threshold.
    """
    results = evaluate_aeb_benefit(initial_speed)
    reductions = compute_hic_reduction(results)

    if not reductions:
        return

    labels = [f"TTC={r['ttc_threshold']:.1f}s" for r in reductions]
    values = [r['hic_reduction_pct'] for r in reductions]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 50 else "#f39c12" if v < 80 else "#2ecc71" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("HIC Reduction (%)", fontsize=13)
    ax.set_title(f"HIC Reduction vs. No AEB — {initial_speed:.0f} km/h Approach",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3, label="Full avoidance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_surface_impact(save_path):
    """
    Compare AEB effectiveness across different road surfaces.
    Shows how wet/icy roads reduce AEB effectiveness.
    """
    surfaces = ["dry", "wet", "snow", "ice"]
    surface_labels = ["Dry", "Wet", "Snow", "Ice"]
    ttc = 1.5  # fixed TTC threshold

    fig, ax = plt.subplots(figsize=(10, 6))

    speeds = [30, 40, 50, 60]
    x = np.arange(len(speeds))
    width = 0.18

    for i, (surface, label) in enumerate(zip(surfaces, surface_labels)):
        impact_speeds = []
        for speed in speeds:
            results = evaluate_aeb_benefit(speed, surface=surface)
            # Find TTC=1.5s result
            for r in results:
                if r["ttc_threshold"] == ttc:
                    impact_speeds.append(r["impact_speed_kmh"])
                    break

        colors = {"dry": "#2ecc71", "wet": "#3498db", "snow": "#9b59b6", "ice": "#e74c3c"}
        ax.bar(x + i * width, impact_speeds, width, label=label,
               color=colors.get(surface, "gray"), edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Approach Speed (km/h)", fontsize=13)
    ax.set_ylabel("Residual Impact Speed (km/h)", fontsize=13)
    ax.set_title(f"Impact Speed After AEB (TTC={ttc}s) — Surface Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s} km/h" for s in speeds], fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_plots(save_dir="results/safety_analysis/"):
    """Generate all safety analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating safety analysis plots in {save_dir}...")

    plot_hic_vs_speed(os.path.join(save_dir, "hic_vs_speed.png"))
    plot_aeb_effectiveness(50.0, os.path.join(save_dir, "aeb_effectiveness_50kmh.png"))
    plot_aeb_effectiveness(80.0, os.path.join(save_dir, "aeb_effectiveness_80kmh.png"))
    plot_hic_reduction_chart(50.0, os.path.join(save_dir, "hic_reduction_50kmh.png"))
    plot_surface_impact(os.path.join(save_dir, "surface_impact_comparison.png"))

    print(f"\nAll safety plots generated.")


if __name__ == "__main__":
    generate_all_plots()
