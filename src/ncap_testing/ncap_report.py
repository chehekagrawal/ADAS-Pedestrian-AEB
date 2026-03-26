"""
Euro NCAP Report & Plot Generation.

Produces publication-quality visualizations of Euro NCAP test results:
- Test matrix heatmap (scenario × speed → pass/fail)
- Score vs. speed line chart
- Overall scorecard figure
- Test track layout diagram
- Dry vs. wet surface comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches

from src.ncap_testing.ncap_runner import run_full_test_matrix, TestResult
from src.ncap_testing.ncap_scenarios import ALL_SCENARIOS
from src.ncap_testing.ncap_scorer import overall_score, score_single_test


def generate_test_matrix_heatmap(results, surface_label="Dry", save_path=""):
    """
    Heatmap: rows = scenarios, columns = speeds.
    Green = avoided, yellow = partial, red = collision.
    """
    scenarios = [s.code for s in ALL_SCENARIOS]
    speeds = ALL_SCENARIOS[0].test_speeds_kmh

    # Build matrix
    matrix = np.zeros((len(scenarios), len(speeds)))
    annotations = [[None] * len(speeds) for _ in scenarios]

    for r in results:
        if r.scenario_code in scenarios:
            row = scenarios.index(r.scenario_code)
            if r.vehicle_speed_kmh in speeds:
                col = speeds.index(r.vehicle_speed_kmh)
                matrix[row][col] = score_single_test(r)
                annotations[row][col] = r.outcome[0].upper()  # A/P/C

    fig, ax = plt.subplots(figsize=(12, 5))

    # Custom colormap: red → yellow → green
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
    cmap = LinearSegmentedColormap.from_list("ncap", colors, N=100)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f"{int(s)}" for s in speeds], fontsize=12)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=12)
    ax.set_xlabel("Vehicle Speed (km/h)", fontsize=13)
    ax.set_ylabel("Scenario", fontsize=13)
    ax.set_title(f"Euro NCAP AEB Test Matrix — {surface_label} Surface", fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(scenarios)):
        for j in range(len(speeds)):
            score = matrix[i][j]
            letter = annotations[i][j] or "?"
            text_color = "white" if score < 0.5 else "black"
            ax.text(j, i, f"{score:.0%}\n{letter}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_score_vs_speed(results, save_path=""):
    """Line chart: score vs. speed for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    for scenario, color in zip(ALL_SCENARIOS, colors):
        scenario_results = [r for r in results if r.scenario_code == scenario.code]
        scenario_results.sort(key=lambda r: r.vehicle_speed_kmh)

        speeds = [r.vehicle_speed_kmh for r in scenario_results]
        scores = [score_single_test(r) * 100 for r in scenario_results]

        ax.plot(speeds, scores, "o-", color=color, linewidth=2.5,
                markersize=8, label=scenario.code)

    ax.set_xlabel("Vehicle Speed (km/h)", fontsize=13)
    ax.set_ylabel("Test Score (%)", fontsize=13)
    ax.set_title("Euro NCAP Score vs. Vehicle Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(15, 65)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_scorecard_figure(results, surface_label="Dry", save_path=""):
    """Visual scorecard as a matplotlib figure."""
    score_data = overall_score(results)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "Euro NCAP AEB-Pedestrian Scorecard", fontsize=16,
            fontweight="bold", ha="center", va="top", transform=ax.transAxes)

    # Stars
    stars = score_data["star_rating"]
    star_str = "★" * stars + "☆" * (5 - stars)
    ax.text(0.5, 0.85, star_str, fontsize=30, ha="center", va="center",
            transform=ax.transAxes, color="#f1c40f")

    # Overall score
    ax.text(0.5, 0.75, f"Overall: {score_data['overall_percent']:.0f}% (Grade {score_data['grade']})",
            fontsize=18, ha="center", va="center", transform=ax.transAxes,
            fontweight="bold")

    ax.text(0.5, 0.68, f"Surface: {surface_label}",
            fontsize=12, ha="center", va="center", transform=ax.transAxes, color="gray")

    # Per-scenario bars
    y_start = 0.55
    bar_height = 0.08
    for i, (code, sc) in enumerate(score_data["scenarios"].items()):
        y = y_start - i * (bar_height + 0.04)
        pct = sc["score"]

        # Background bar
        ax.barh(y, 1.0, height=bar_height, left=0.15,
                color="#ecf0f1", edgecolor="gray", linewidth=0.5,
                transform=ax.transAxes)

        # Score bar
        color = "#2ecc71" if pct >= 0.7 else "#f39c12" if pct >= 0.4 else "#e74c3c"
        ax.barh(y, pct * 0.7, height=bar_height, left=0.15,
                color=color, edgecolor="none",
                transform=ax.transAxes)

        # Label
        ax.text(0.10, y, code, fontsize=11, ha="right", va="center",
                transform=ax.transAxes, fontweight="bold")
        ax.text(0.88, y, f"{pct * 100:.0f}%", fontsize=11, ha="left", va="center",
                transform=ax.transAxes, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_surface_comparison(save_path=""):
    """Compare overall scores across dry, wet, and icy surfaces."""
    surfaces = ["dry", "wet", "ice"]
    labels = ["Dry Asphalt", "Wet Asphalt", "Ice"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    scores = []
    for surface in surfaces:
        results = run_full_test_matrix(surface)
        score_data = overall_score(results)
        scores.append(score_data["overall_percent"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score:.0f}%", ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel("Overall Score (%)", fontsize=13)
    ax.set_title("Euro NCAP Score — Surface Condition Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_test_track_diagram(save_path=""):
    """
    Engineering drawing of the Euro NCAP test track layout.
    Shows vehicle approach path, pedestrian crossing, trigger line.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Road
    road_y = 2.0
    road_width = 3.5
    ax.axhspan(road_y - road_width / 2, road_y + road_width / 2,
               color="#95a5a6", alpha=0.3, label="Road")

    # Lane markings
    ax.axhline(y=road_y, color="white", linestyle="--", linewidth=2, alpha=0.8)

    # Vehicle path
    vehicle_x = np.linspace(0, 40, 50)
    ax.plot(vehicle_x, [road_y] * len(vehicle_x), "b-", linewidth=3, alpha=0.6)

    # Vehicle (rectangle)
    from matplotlib.patches import FancyBboxPatch
    car = FancyBboxPatch((2, road_y - 0.4), 2.0, 0.8,
                         boxstyle="round,pad=0.1", facecolor="#3498db",
                         edgecolor="black", linewidth=1.5)
    ax.add_patch(car)
    ax.text(3, road_y, "EGO", ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

    # Pedestrian crossing path
    ped_x = 30
    ax.plot([ped_x, ped_x], [road_y - 3, road_y + 3], "r--", linewidth=2, alpha=0.7)

    # Pedestrian (circle)
    ped_circle = plt.Circle((ped_x, road_y + 2), 0.3, color="#e74c3c",
                            edgecolor="black", linewidth=1.5)
    ax.add_patch(ped_circle)
    ax.annotate("PEDESTRIAN", xy=(ped_x, road_y + 2),
                xytext=(ped_x + 3, road_y + 3),
                arrowprops=dict(arrowstyle="->"), fontsize=10, fontweight="bold")
    ax.annotate("↓ 5 km/h", xy=(ped_x, road_y + 1.5),
                fontsize=9, ha="center", color="red")

    # Detection zone
    ax.axvline(x=15, color="green", linestyle=":", linewidth=2, alpha=0.7)
    ax.text(15, road_y + 3.5, "Detection\nRange", ha="center", fontsize=9,
            color="green", fontweight="bold")

    # AEB trigger zone
    ax.axvline(x=22, color="orange", linestyle=":", linewidth=2, alpha=0.7)
    ax.text(22, road_y - 3, "AEB\nTrigger", ha="center", fontsize=9,
            color="orange", fontweight="bold")

    # Distances
    ax.annotate("", xy=(15, road_y - 2.5), xytext=(30, road_y - 2.5),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(22.5, road_y - 2.8, "Braking Distance", ha="center", fontsize=9, color="gray")

    ax.set_xlim(0, 42)
    ax.set_ylim(-2, 6)
    ax.set_xlabel("Distance (m)", fontsize=13)
    ax.set_title("Euro NCAP AEB Test Track Layout — CPFA Scenario", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_full_report(save_dir="results/ncap_testing/"):
    """Generate all Euro NCAP report visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating Euro NCAP report in {save_dir}...")

    # Run tests on dry surface
    results_dry = run_full_test_matrix("dry")

    generate_test_matrix_heatmap(results_dry, "Dry",
                                 os.path.join(save_dir, "test_matrix_dry.png"))
    generate_score_vs_speed(results_dry,
                            os.path.join(save_dir, "score_vs_speed.png"))
    generate_scorecard_figure(results_dry, "Dry",
                              os.path.join(save_dir, "scorecard_dry.png"))
    generate_test_track_diagram(os.path.join(save_dir, "test_track_layout.png"))
    generate_surface_comparison(os.path.join(save_dir, "surface_comparison.png"))

    # Wet surface for comparison
    results_wet = run_full_test_matrix("wet")
    generate_test_matrix_heatmap(results_wet, "Wet",
                                 os.path.join(save_dir, "test_matrix_wet.png"))
    generate_scorecard_figure(results_wet, "Wet",
                              os.path.join(save_dir, "scorecard_wet.png"))

    print(f"\nAll Euro NCAP reports generated.")


if __name__ == "__main__":
    generate_full_report()
