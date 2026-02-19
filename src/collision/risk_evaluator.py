import json
import os
import matplotlib.pyplot as plt


def plot_risk():
    # Load simulation logs from collision module
    with open("results/collision/logs/run_01.json", "r") as f:
        logs = json.load(f)

    distances = [log["distance"] for log in logs]
    ttcs = [log["ttc"] for log in logs]

    plt.figure()
    plt.plot(distances, label="Distance")
    plt.plot(ttcs, label="TTC")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()

    # Create output directory if not exists
    os.makedirs("results/collision/plots", exist_ok=True)

    # Save plot
    plt.savefig("results/collision/plots/risk_plot.png")
    plt.close()

    print("Risk plot saved.")


if __name__ == "__main__":
    plot_risk()
